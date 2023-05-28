"""A neuron made up of input weights and a single bias
"""

import random
from typing import Literal

from nn_intro.nn_by_scalar import scalar


class Neuron():
    def __init__(self, 
                 n_inputs: int, 
                 activation_fn: Literal['tanh', 'relu'] = 'tanh'):
        """Create a neuron as a group of Scalars based on number of
        inputs

        Args:
            n_inputs (int): number of input connections from a previous
                layer
            activation_fn ('tanh' or 'relu'): the activation function.
                Defaults to tanh
        """
        self.w = [scalar.Scalar(random.uniform(-1, 1)) for _ in range(n_inputs)]
        self.b = scalar.Scalar(random.uniform(-1, 1))
        self._act_fn = activation_fn

    def __call__(self, x: list[float]) -> scalar.Scalar:
        """Forward pass through the neuron, treated like a function

        Args:
            x (list[float]): compute w*x + b for each w_i and x_i

        Returns:
            scalar.Scalar: output value of the neuron
        """
        # The second argument to sum is the initialization value
        # The output of sum is a Scalar because of Scalar's implementation
        # of add and multiply
        neuron_body = sum((wi*xi for wi, xi in zip(self.w, x)), self.b)
        return neuron_body.tanh() if self._act_fn == 'tanh' else neuron_body.relu()

    def parameters(self) -> list[scalar.Scalar]:
        """Return the Scalar parameters that make up the neuron

        Returns:
            list[scalar.Scalar]: a single list of weight and bias scalars
        """
        return self.w + [self.b]
    
    def __repr__(self):
        return f'Neuron({self._act_fn}, {len(self.w)})'