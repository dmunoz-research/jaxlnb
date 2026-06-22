"""Implementation of Online Linear Neuron Boosting.

We refer to a "Neuron" as a callable PyTree that is linear in its parameters; it may or may not
contain bias terms. The input tensor to each Neuron must have shape (d_in, ...) and its Weight
parameters must have shape (d_out, d_in, ...). This axis assumption is noted in the code via "AA".
"""

import functools
import itertools
from typing import Any, Callable, NamedTuple, Optional, Tuple, TypeAlias, Union

import jax
import jax.numpy as jnp
import jax.scipy
import jax.tree_util as jtu
import optax
from optax._src import base
from optax._src import combine
from optax._src import utils
import optax.tree_utils as otu

Neuron: TypeAlias = base.PyTree
"""A PyTree that is Callable and linear in its parameters (weights and biases)."""

Weight: TypeAlias = jax.Array
"""Weight array of a Neuron with expected shape=(d_out, d_in, ...)"""

Bias: TypeAlias = jax.Array
"""Bias array of a Neuron with expected shape=(d_out,) or shape=(d_out, 1, ..., 1)"""

IsNeuron: TypeAlias = Callable[[base.PyTree], bool]
"""Returns True if the PyTree node is a Neuron."""

IsWeight: TypeAlias = Callable[[jtu.KeyPath], bool]
"""When called on a Neuron's KeyPath, returns True if the KeyPath is for its Weight."""

IsBias: TypeAlias = Callable[[jtu.KeyPath], bool]
"""When called on a Neuron's KeyPath, returns True if the KeyPath is for its Bias."""

Vector: TypeAlias = jax.Array
"""A 1-D array."""


def _get_array(neuron: Neuron, predicate: Union[IsWeight, IsBias]) -> Optional[jax.Array]:
    """Returns the respective array when the predicate evaluates True on the KeyPath."""
    leaves, _ = jtu.tree_flatten_with_path(neuron)
    for key_path, arr in leaves:
        if predicate(key_path):
            return arr
    return None


def _set_array(neuron: Neuron, predicate: Union[IsWeight, IsBias], new_array: jax.Array) -> Neuron:
    """Assigns `new_array` when the predicate evaluates True on the KeyPath."""

    def _replace(key_path: jtu.KeyPath, curr_array: jax.Array) -> jax.Array:
        return new_array if predicate(key_path) else curr_array

    return jtu.tree_map_with_path(_replace, neuron)


def _last_name_is_str(key_path: jtu.KeyPath, query: str) -> bool:
    """Returns True if the last EntryName in `key_path` matches `query`."""
    last_entry = key_path[-1]
    entry_name = getattr(last_entry, "key", getattr(last_entry, "name", None))
    return entry_name == query


def _compute_mu(xs: jax.Array) -> Vector:
    """Returns the mean feature vector across the spatial dimensions."""
    return jnp.mean(xs, axis=(0, *tuple(range(2, xs.ndim))))  # AA


def _shrink(diag: Vector, shrinkage: float) -> Vector:
    """See https://scikit-learn.org/stable/modules/covariance.html#basic-shrinkage."""
    assert 0.0 <= shrinkage
    assert shrinkage < 1.0
    return (1.0 - shrinkage) * diag + (shrinkage * jnp.sum(diag) / jnp.size(diag))


def default_is_weight(key_path: jtu.KeyPath) -> bool:
    """Returns True if the entry name is `weight`."""
    return _last_name_is_str(key_path, "weight")


def default_is_bias(key_path: jtu.KeyPath) -> bool:
    """Returns True if the entry name is `bias`."""
    return _last_name_is_str(key_path, "bias")


def make_pvp(
    mu: Vector, nu: Vector, shrinkage: float, is_weight: IsWeight, is_bias: IsBias
) -> Callable[[Neuron], Neuron]:
    """Returns the Neuron's preconditioner-vector-product function for conjugate gradient.

    The preconditioner is the incomplete Cholesky factorization defined in Section 3.5.

    Args:
      mu: mean input features
      nu: mean squared input features
      shrinkage: covariance shinkage in [0, 1)
      is_weight: see type definition
      is_bias: see type definition
    """
    assert len(mu.shape) == 1
    assert mu.shape == nu.shape
    variance_shrunk = _shrink(jnp.maximum(0.0, nu - mu**2), shrinkage)

    # PVP for one component of the parameter vector when it contains a bias term.
    def _pvp_bias(weight: jax.Array, bias: jax.Array) -> Tuple[jax.Array, jax.Array]:
        # We use vmap here because `weight` might be an N-D spatial filter and we use scalar
        # broadcasting to equally apply the feature moments over the spatial axes (i.e., assumes
        # translational equivariance).
        weight_out = jax.vmap(lambda w, b, m, v: (w - m * b) / v, in_axes=(0, None, 0, 0))(
            weight, bias, mu, variance_shrunk
        )
        bias_out = bias - jnp.sum(jax.vmap(lambda m, w: m * w)(mu, weight_out))
        assert weight_out.shape == weight.shape
        assert bias_out.shape == bias.shape
        return weight_out, bias_out

    # PVP for one component of the parameter vector when it does not contain a bias term.
    def _pvp_nobias(weight: jax.Array) -> jax.Array:
        nu_shrunk = variance_shrunk + mu**2
        return jax.vmap(lambda w, n: w / n)(weight, nu_shrunk)

    # PVP for all components of the neuron's parameter vector.
    def pvp(v: Neuron) -> Neuron:
        weight = _get_array(v, is_weight)
        assert weight is not None
        assert weight.shape[1] == mu.size  # AA
        bias = _get_array(v, is_bias)
        if bias is None:
            weight = jax.vmap(_pvp_nobias)(weight)
            v = _set_array(v, is_weight, weight)
        else:
            weight, bias = jax.vmap(_pvp_bias)(weight, bias)
            v = _set_array(v, is_weight, weight)
            v = _set_array(v, is_bias, bias)
        return v

    return pvp


def project(
    xs: jax.Array,
    grad: Neuron,
    init: Neuron,
    is_weight: IsWeight,
    is_bias: IsBias,
    ridge: float = 0.0,
    **kwargs: Any,
) -> Neuron:
    """Preconditions the gradient vector corresponding to one neuron.

    Conjugate gradient is used to solve the linear system.

    Args:
      xs: the batched input tensors to the neuron
      grad: the component of the gradient vector for the neuron
      init: initialization for conjugate gradient
      is_weight: see type definition
      is_bias: see type definition
      ridge: ridge reglarization; only applied to the weights
      kwargs: forwards to jax.scipy.sparse.linalg.cg

    Returns the preconditioned gradient vector.
    """

    def jvp(v: Neuron) -> jax.Array:
        return jax.vmap(v)(xs)

    ys = jvp(init)
    num_samples = ys.size / ys.shape[1]  # AA
    del ys
    vjp = jax.linear_transpose(jvp, init)  # The linearization point doesn't matter since linear.

    def mvp(v: Neuron) -> Neuron:
        return otu.tree_scalar_mul(1.0 / num_samples, vjp(jvp(v))[0])  # J'Jv

    # Create a pytree with same shape as the neuron and use scalar broadcasting
    # to implement ridge regression while ignoring bias, if present.
    ridge_neuron = _set_array(init, is_weight, ridge)
    ridge_neuron = _set_array(ridge_neuron, is_bias, 0.0)

    def mvp_ridge(v: Neuron) -> Neuron:
        return otu.tree_add(mvp(v), otu.tree_mul(ridge_neuron, v))

    _mvp = mvp_ridge if ridge > 0.0 else mvp
    return jax.scipy.sparse.linalg.cg(_mvp, grad, x0=init, **kwargs)[0]


class ScaleByLNBState(NamedTuple):
    """State for the online Linear Neuron Boosting algorithm."""

    # Each state is a list of length number of neurons, in leaf traversal order
    h_neurons: list[Neuron]  # The previous conjugate gradient solution
    mu_state: optax.EmaState  # Mean input features (first moment)
    nu_state: optax.EmaState  # Mean squared input features (second moment)


def scale_by_lnb(
    b_mu: float = 0.9,
    b_nu: float = 0.999,
    pvp_shrinkage: float = 0.1,
    cg_ridge: float = 0.1,
    cg_maxiter: int = 2,
    min_norm: float = 1e-2,
    *,
    is_neuron: IsNeuron = lambda node: hasattr(node, "weight"),
    is_weight: IsWeight = default_is_weight,
    is_bias: IsBias = default_is_bias,
    pvp_only: bool = False,
    accumulator_dtype: Optional[Any] = None,
) -> base.GradientTransformationExtraArgs:
    r"""Applies a Linear Neuron Boosting update.

    Each neuron in the model pytree must be a callable that has a
    single input and output tensor with feature dimensions in the first axis.

    Args:
      b_mu: EMA decay for input features' first moment
      b_nu: EMA decay for input features' second momment
      pvp_shrinkage: covariance shrinkage for the preconditioner, in [0, 1)
      cg_ridge: ridge regularizer for conjugate gradient (\gamma in Section 3.4)
      cg_maxiter: number of conjugate gradient iterations
      min_norm: the minimum norm to use to avoid dividing by zero
      is_neuron: see type definition
      is_weight: see type definition
      is_bias: see type definition
      pvp_only: precondition the gradient vector with the Cholesky factorization
      accumulator_dtype: accumulator dtype

    Returns a `GradientTransformationExtraArgs` with an `update_fn` that expects the
    `xs_neurons` kwarg to be a list of batched input tensors to each respective
    neuron, in leaf traversal order as specified by `is_neuron`.
    """
    accumulator_dtype = utils.canonicalize_dtype(accumulator_dtype)
    mu_ema = optax.ema(b_mu, debias=True, accumulator_dtype=accumulator_dtype)
    nu_ema = optax.ema(b_nu, debias=True, accumulator_dtype=accumulator_dtype)
    _make_pvp = functools.partial(
        make_pvp, shrinkage=pvp_shrinkage, is_weight=is_weight, is_bias=is_bias
    )
    _project = functools.partial(
        project, is_weight=is_weight, is_bias=is_bias, ridge=cg_ridge, maxiter=cg_maxiter
    )

    def init_fn(params: base.Params) -> ScaleByLNBState:
        def _make_zero_vector(neuron: Neuron) -> Vector:
            weight = _get_array(neuron, is_weight)
            assert weight is not None
            return jnp.zeros(weight.shape[1])  # AA

        neurons = list(filter(is_neuron, jtu.tree_leaves(params, is_leaf=is_neuron)))
        assert all(map(callable, neurons)), "Each Neuron must be callable."
        return ScaleByLNBState(
            h_neurons=list(map(otu.tree_zeros_like, neurons)),
            mu_state=mu_ema.init(list(map(_make_zero_vector, neurons))),
            nu_state=nu_ema.init(list(map(_make_zero_vector, neurons))),
        )

    def update_fn(
        updates: base.Updates,
        state: ScaleByLNBState,
        params: base.Params,
        xs_neurons: list[jax.Array],
    ) -> Tuple[base.Updates, ScaleByLNBState]:
        del params
        # Update feature moments.
        mu_neurons = list(map(_compute_mu, xs_neurons))
        nu_neurons = list(map(_compute_mu, map(jnp.square, xs_neurons)))
        mu_neurons, mu_state = mu_ema.update(mu_neurons, state.mu_state)
        nu_neurons, nu_state = nu_ema.update(nu_neurons, state.nu_state)

        # Construct PVPs for conjugate gradient using moments.
        pvp_neurons = list(itertools.starmap(_make_pvp, zip(mu_neurons, nu_neurons, strict=True)))

        # Grab components of the entire gradient vector that are neurons.
        leaves, treedef = jtu.tree_flatten(updates, is_leaf=is_neuron)
        idx_neurons, grad_neurons = zip(
            *[(i, node) for i, node in enumerate(leaves) if is_neuron(node)], strict=True
        )

        # Precondition the gradient vector and then update its neurons.
        if pvp_only:
            h_neurons = [pvp(grad) for grad, pvp in zip(grad_neurons, pvp_neurons, strict=True)]
        else:
            h_neurons = [
                _project(xs, grad, init, M=pvp)
                for xs, grad, init, pvp in zip(
                    xs_neurons, grad_neurons, state.h_neurons, pvp_neurons, strict=True
                )
            ]

        # Update only the Neuron components (implies the Identity metric for the other components).
        for idx, neuron in zip(idx_neurons, h_neurons, strict=True):
            leaves[idx] = neuron
        h = jtu.tree_unflatten(treedef, leaves)

        # Rescale under the metric (adaptive step size).
        norm = jnp.maximum(min_norm, jnp.sqrt(otu.tree_vdot(h, updates)))
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
    is_neuron: IsNeuron = lambda node: hasattr(node, "weight"),
    is_weight: IsWeight = default_is_weight,
    is_bias: IsBias = default_is_bias,
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
            is_neuron=is_neuron,
            is_weight=is_weight,
            is_bias=is_bias,
            pvp_only=pvp_only,
            accumulator_dtype=accumulator_dtype,
        ),
        optax.add_decayed_weights(weight_decay),
    )
