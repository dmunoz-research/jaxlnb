"""Implementation of Online Linear Neuron Boosting.

This implementation assumes that for an input tensor to the neuron, the first
axis corresponds to the feature dimension (e.g., a 1-D vector) and all others
correspond to spatial dimensions (e.g., CHW order in a CNN). This axis
assumption is noted in the code via "AA".
"""

import functools
import itertools
from typing import Any, Callable, NamedTuple, Optional, Tuple, TypeAlias, Union

import chex
import jax
import jax.numpy as jnp
import jax.scipy as jsp
import optax
from optax._src import base
from optax._src import combine
from optax._src import utils
import optax.tree_utils as otu

# The parameters of the neuron.
Weight: TypeAlias = chex.Array  # shape=(d_out, d_in, ...)
Bias: TypeAlias = chex.Array  # shape=(d_out,) or shape=(d_out, 1, ..., 1)

# Returns its weight tensor when called on a node in the pytree that is a neuron; otherwise None.
GetWeight: TypeAlias = Callable[[base.PyTree], Optional[Weight]]
# Similarly, returns the neuron's bias tensor, if present, when called on a node in the pytree.
GetBias: TypeAlias = Callable[[base.PyTree], Optional[Bias]]

# Returns true if the node in the pytree represents a linear neuron.
IsNeuron: TypeAlias = Callable[[base.PyTree], bool]

# Matrix-vector-product that returns a pytree with the same structure.
MVP: TypeAlias = Callable[[base.PyTree], base.PyTree]

Vector: TypeAlias = chex.Array  # shape=(d_in,)


def _get_neurons(pytree: base.PyTree, is_neuron: IsNeuron) -> list[base.PyTree]:
    """Returns the neurons in pytree, in leaf traversal order."""
    return list(filter(is_neuron, jax.tree_util.tree_leaves(pytree, is_leaf=is_neuron)))


def _set_value(
    node: base.PyTree, getter: Union[GetBias, GetWeight], new_value: chex.Array
) -> base.PyTree:
    """Sets node.getter(node) = new_value."""
    query_id = id(getter(node))
    leaves, treedef = jax.tree_util.tree_flatten(node)
    idx = next(i for i, leaf in enumerate(leaves) if id(leaf) == query_id)
    leaves[idx] = new_value
    return jax.tree_util.tree_unflatten(treedef, leaves)


def _set_neurons(
    pytree: base.PyTree, new_neurons: list[base.PyTree], is_neuron: IsNeuron
) -> base.PyTree:
    """Replaces all neurons in pytree with new_neurons."""
    # Create a mapping of the neuron ids to replace
    old_neurons = _get_neurons(pytree, is_neuron)
    old2new = {id(old): new for old, new in zip(old_neurons, new_neurons, strict=True)}

    # Replace the old neurons, while keeping the non-neuron leaves.
    leaves, treedef = jax.tree_util.tree_flatten(pytree, is_leaf=is_neuron)
    new_leaves = [old2new.get(id(leaf), leaf) for leaf in leaves]
    return jax.tree_util.tree_unflatten(treedef, new_leaves)


def _compute_mu(xs: chex.Array) -> Vector:
    """Returns the mean feature vector across the spatial dimensions."""
    return jnp.mean(xs, axis=(0, *tuple(range(2, xs.ndim))))  # AA


def _shrink(diag: Vector, shrinkage: float) -> Vector:
    """See https://scikit-learn.org/stable/modules/covariance.html#basic-shrinkage."""
    assert 0.0 <= shrinkage
    assert shrinkage < 1.0
    return (1.0 - shrinkage) * diag + (shrinkage * jnp.sum(diag) / jnp.size(diag))


def make_pvp(
    mu: Vector, nu: Vector, shrinkage: float, get_weight: GetWeight, get_bias: GetBias
) -> MVP:
    """Returns the neuron's preconditioner-vector-product for conj. gradient.

    The preconditioner is the incomplete Cholesky factorization defined in
    Section 3.5.

    Args:
      mu: mean input features
      nu: mean squared input features
      shrinkage: covariance shinkage in [0, 1)
      get_weight: see type definition
      get_bias: see type definition
    """
    chex.assert_rank(mu, 1)
    chex.assert_equal_shape((mu, nu))
    # These represent the diagonal.
    covariance: Vector = jnp.maximum(0.0, nu - jnp.multiply(mu, mu))
    precision = 1.0 / _shrink(covariance, shrinkage)
    inv_nu = 1.0 / _shrink(nu, shrinkage)

    # PVP for one component in the parameters.
    def _pvp_bias(w: chex.Array, b: chex.Array) -> tuple[chex.Array, chex.Array]:
        # Computes the inner product with the same mu_x over all axes, i.e., it
        # assumes the translational equivariance over those (spatial) axes.
        chex.assert_size(b, 1)  # For CNNs this might not be rank 1
        bcast_shape = mu.shape + (1,) * (w.ndim - 1)
        mu_bcast = jnp.reshape(mu, bcast_shape)
        w_opt = jnp.reshape(precision, bcast_shape) * (w - mu_bcast * b)
        b_opt = jnp.sum(-mu_bcast * w_opt, keepdims=True)[0] + b
        chex.assert_shape([w_opt, b_opt], [w.shape, b.shape])
        return w_opt, b_opt

    def _pvp_nobias(w: chex.Array) -> chex.Array:
        bcast_shape = inv_nu.shape + (1,) * (w.ndim - 1)
        w_opt = jnp.reshape(inv_nu, bcast_shape) * w
        chex.assert_equal_shape([w_opt, w])
        return w_opt

    # PVP for the neuron's entire parameter vector.
    def pvp(v: base.PyTree) -> base.PyTree:
        weight = get_weight(v)
        chex.assert_axis_dimension(weight, 1, mu.size)  # AA
        bias = get_bias(v)
        if bias is None:
            weight = jax.vmap(_pvp_nobias)(weight)
            v = _set_value(v, get_weight, weight)
        else:
            weight, bias = jax.vmap(_pvp_bias)(weight, bias)
            v = _set_value(v, get_weight, weight)
            v = _set_value(v, get_bias, bias)
        return v

    return pvp


def project(
    xs: chex.Array,
    grad: base.PyTree,
    init: base.PyTree,
    get_weight: GetWeight,
    get_bias: GetBias,
    ridge: float = 0.0,
    **kwargs: Any,
) -> base.PyTree:
    """Preconditions the gradient vector corresponding to one neuron.

    Conjugate gradient is used to solve the linear system.

    Args:
      xs: the input tensors to the neuron
      grad: the component of the gradient vector for the neuron
      init: the initialization for conjugate gradient; this should be a
        Callable pytree that promises to be linear in its parameters
      get_weight: see type definition
      get_bias: see type definition
      ridge: ridge reglarization; only applied to the weights
      kwargs: forwards to jax.scipy.sparse.linalg.cg

    Returns the preconditioned gradient vector.
    """

    def batch_predict(neuron: base.PyTree) -> chex.Array:
        return jax.vmap(neuron)(xs)

    # Because init is linear, the linearization point doesn't matter.
    ys, jvp = jax.linearize(batch_predict, init)
    vjp = jax.linear_transpose(jvp, init)
    num_samples = ys.size / ys.shape[1]  # AA

    def mvp(v: base.PyTree) -> base.PyTree:
        return otu.tree_scalar_mul(1.0 / num_samples, vjp(jvp(v))[0])  # J'Jv

    # Create a pytree with same shape as the neuron and use scalar broadcasting
    # to implement ridge regression while ignoring bias, if present.
    ridge_neuron = _set_value(init, get_weight, ridge)
    if get_bias(ridge_neuron) is not None:
        ridge_neuron = _set_value(ridge_neuron, get_bias, 0.0)

    def mvp_ridge(v: base.PyTree) -> base.PyTree:
        return otu.tree_add(mvp(v), otu.tree_mul(ridge_neuron, v))

    _mvp = mvp_ridge if ridge > 0.0 else mvp
    return jsp.sparse.linalg.cg(_mvp, grad, x0=init, **kwargs)[0]


class ScaleByLNBState(NamedTuple):
    """State for the online Linear Neuron Boosting algorithm."""

    # Each state is a list of length number of neurons, in leaf traversal order
    h_neurons: list[base.PyTree]  # The previous cg solution; must be callable
    mu_state: optax.EmaState  # Mean input features
    nu_state: optax.EmaState  # Mean squared input features


def scale_by_lnb(
    b_mu: float = 0.9,
    b_nu: float = 0.999,
    pvp_shrinkage: float = 0.1,
    cg_ridge: float = 0.1,
    cg_maxiter: int = 2,
    min_norm: float = 1e-2,
    *,
    get_weight: GetWeight = lambda node: getattr(node, "weight", None),
    get_bias: GetBias = lambda node: getattr(node, "bias", None),
    pvp_only: bool = False,
    accumulator_dtype: Optional[Any] = None,
) -> base.GradientTransformationExtraArgs:
    r"""Applies an online Linear Neuron Boosting update.

    Implements lines 5-11 and 13-14 in Algorithm 2.

    Each neuron in the parameters pytree must be a callable function that has a
    single input and output tensor with feature dimensions in the first axis.

    Args:
      b_mu: EMA decay for mean of input features
      b_nu: EMA decay for mean of squared input features
      pvp_shrinkage: covariance shrinkage for the preconditioner, in [0, 1)
      cg_ridge: ridge regularizer for conj. gradient (\gamma in Section 3.4)
      cg_maxiter: number of conj. gradient iterations
      min_norm: the minimum norm to use to avoid dividing by zero
      get_weight: see type definition
      get_bias: see type definition
      pvp_only: precondition the gradient vector with the Cholesky factorization
      accumulator_dtype: accumulator dtype

    Returns a GradientTransformationExtraArgs with update_fn that expects the
    `xs_neurons` kwarg is a list of input tensors to each neuron, respectively
    in leaf traversal order.
    """
    accumulator_dtype = utils.canonicalize_dtype(accumulator_dtype)
    mu_ema = optax.ema(b_mu, debias=True, accumulator_dtype=accumulator_dtype)
    nu_ema = optax.ema(b_nu, debias=True, accumulator_dtype=accumulator_dtype)
    _make_pvp = functools.partial(
        make_pvp, shrinkage=pvp_shrinkage, get_weight=get_weight, get_bias=get_bias
    )
    _project = functools.partial(
        project, get_weight=get_weight, get_bias=get_bias, ridge=cg_ridge, maxiter=cg_maxiter
    )

    def _is_neuron(node: base.PyTree) -> bool:
        return get_weight(node) is not None

    def init_fn(params: base.Params) -> ScaleByLNBState:
        def _make_zero_vector(neuron: base.PyTree) -> Vector:  # AA
            return jnp.zeros(get_weight(neuron).shape[1])  # type: ignore

        params_neurons = _get_neurons(params, _is_neuron)
        assert all(map(callable, params_neurons)), "Each neuron must be callable."
        return ScaleByLNBState(
            h_neurons=list(map(otu.tree_zeros_like, params_neurons)),
            mu_state=mu_ema.init(list(map(_make_zero_vector, params_neurons))),
            nu_state=nu_ema.init(list(map(_make_zero_vector, params_neurons))),
        )

    def update_fn(
        updates: base.Updates,
        state: ScaleByLNBState,
        params: base.Params,
        xs_neurons: list[chex.Array],
    ) -> Tuple[base.Updates, ScaleByLNBState]:
        del params
        # Update moments.
        mu_neurons = list(map(_compute_mu, xs_neurons))
        nu_neurons = list(map(_compute_mu, map(jnp.square, xs_neurons)))
        mu_neurons, mu_state = mu_ema.update(mu_neurons, state.mu_state)
        nu_neurons, nu_state = nu_ema.update(nu_neurons, state.nu_state)

        # Construct PVPs for conjugate gradient.
        pvp_neurons = list(itertools.starmap(_make_pvp, zip(mu_neurons, nu_neurons, strict=True)))

        # Precondition the gradient vector and then update its neurons.
        grad_neurons = _get_neurons(updates, _is_neuron)
        if pvp_only:
            h_neurons = [pvp(grad) for grad, pvp in zip(grad_neurons, pvp_neurons, strict=True)]
        else:
            h_neurons = [
                _project(xs, grad, init, M=pvp)
                for xs, grad, init, pvp in zip(
                    xs_neurons, grad_neurons, state.h_neurons, pvp_neurons, strict=True
                )
            ]
        h = _set_neurons(updates, h_neurons, _is_neuron)

        # Rescale under the metric (adaptive step size).
        norm = jnp.sqrt(jnp.maximum(min_norm**2, otu.tree_vdot(h, updates)))
        h_unit = otu.tree_scalar_mul(1.0 / norm, h)
        return h_unit, ScaleByLNBState(h_neurons, mu_state, nu_state)

    return base.GradientTransformationExtraArgs(init_fn, update_fn)


def lnb(
    b_g: float = 0.9,
    b_mu: float = 0.9,
    b_nu: float = 0.999,
    pvp_shrinkage: float = 0.1,
    cg_ridge: float = 0.1,
    cg_maxiter: int = 2,
    min_norm: float = 1e-2,
    weight_decay: float = 1e-4,
    *,
    get_weight: GetWeight = lambda node: getattr(node, "weight", None),
    get_bias: GetBias = lambda node: getattr(node, "bias", None),
    pvp_only: bool = False,
    accumulator_dtype: Optional[Any] = None,
) -> base.GradientTransformationExtraArgs:
    """Implements lines 4-14 in Algorithm 2."""
    accumulator_dtype = utils.canonicalize_dtype(accumulator_dtype)
    return combine.chain(
        optax.ema(b_g, debias=True, accumulator_dtype=accumulator_dtype),
        scale_by_lnb(
            b_mu,
            b_nu,
            pvp_shrinkage,
            cg_ridge,
            cg_maxiter,
            min_norm,
            get_weight=get_weight,
            get_bias=get_bias,
            pvp_only=pvp_only,
            accumulator_dtype=accumulator_dtype,
        ),
        optax.add_decayed_weights(weight_decay),
    )
