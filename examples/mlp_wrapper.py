"""MLP Wrapper that saves each neuron's features."""

from typing import Any, Tuple

import equinox as eqx
from jaxtyping import Array


def wrap(model: eqx.nn.MLP) -> eqx.Module:
    """Replaces model's call operator with one that returns the tuple [output, x_neurons].

    Where x_neurons are the inputs to each layer in the MLP.
    """
    # A buffer containing the inputs to each neuron. Follows the example from:
    # https://github.com/patrick-kidger/equinox/issues/186#issuecomment-1233606690
    num_layers = len(model.layers)
    x_neurons = [None] * num_layers

    # Saves the input to the respective location in x_neurons.
    class Neuron(eqx.Module):
        layer: eqx.nn.Linear
        idx: int

        def __call__(self, x: Array, *args: Any, **kwargs: Any) -> Array:
            x_neurons[self.idx] = x
            return self.layer(x, *args, **kwargs)

    # Replace all linear layers.
    for i in range(num_layers):
        model = eqx.tree_at(lambda m, i=i: m.layers[i], model, Neuron(model.layers[i], i))

    # Wrap around the model and return each neuron's input feature, in tree traversal order.
    class Wrapper(eqx.Module):
        mlp: eqx.nn.MLP

        def __call__(self, *args: Any, **kwargs: Any) -> Tuple[Array, list[Array]]:
            return self.mlp(*args, **kwargs), x_neurons

    return Wrapper(model)
