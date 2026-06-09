"""Network initialization utility functions."""

from functools import partial
from typing import Any, Callable, TypeAlias

import equinox as eqx
import jax
from jaxtyping import Array, Num, PyTree

Tensor: TypeAlias = Num[Array, "..."]

# Can't use jax.nn.initializers.Initializer due to https://github.com/beartype/beartype/issues/184
JaxInitializer: TypeAlias = Callable[[Any, Any, Any], jax.Array]


def _tree_leaves_attr(pytree: PyTree, attr: str) -> list[PyTree]:
    leaves = jax.tree_util.tree_leaves(pytree, is_leaf=lambda x: hasattr(x, attr))
    # Default with None to ignore random jax objects in the pytree at the leaves.
    return [x for leaf in leaves if (x := getattr(leaf, attr, None)) is not None]


def weights_biases(
    model: eqx.Module,
    weight_init: JaxInitializer,
    bias_init: JaxInitializer = jax.nn.initializers.zeros,
    *,
    key: jax.Array,
) -> eqx.Module:
    """Applies the initializers on the `weights` and `bias` fields for any Node in the eqx.Module.

    Args:
      model: the Module to initialize
      weight_init: the [initializer]
        (https://jax.readthedocs.io/en/latest/jax.nn.initializers.html) to use for the weights
      bias_init: the initializer for the biases, if present.
      key: the random key to split.

    Returns:
      The initialized module.
    """

    def _initialize(
        key_init: jax.Array, arrays: list[Tensor], initializer: JaxInitializer
    ) -> list[Tensor]:
        keys = jax.random.split(key_init, len(arrays))
        return [
            initializer(k, array.shape, array.dtype) for k, array in zip(keys, arrays, strict=True)
        ]

    w_key, b_key = jax.random.split(key, 2)
    get_weights = partial(_tree_leaves_attr, attr="weight")
    get_biases = partial(_tree_leaves_attr, attr="bias")
    model = eqx.tree_at(get_weights, model, _initialize(w_key, get_weights(model), weight_init))
    model = eqx.tree_at(get_biases, model, _initialize(b_key, get_biases(model), bias_init))
    return model
