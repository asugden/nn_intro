"""A fully connected layer of neurons
"""

from nn_intro.nn_by_scalar import neuron, scalar


class Layer():
    def __init__(self, 
                 n_inputs: int, 
                 n_outputs: int, 
                 _last_layer: bool = False, 
                 **kwargs):
        """Create a layer of neurons, each of which has scalars, by the
        number of connections between layers

        Args:
            n_inputs (int): the number of inputs to the layer
            n_outputs (int): the number of outputs from the layer
            _last_layer (bool): if True, return 
        """
        self.neurons = [neuron.Neuron(n_inputs, **kwargs) for _ in range(n_outputs)]
        self._last_layer = _last_layer

    def __call__(self, x: list[float]) -> scalar.Scalar:
        """Forward pass through the layer, treated like a function

        Args:
            x (list[float]): compute w*x + b for each w_i and x_i

        Returns:
            scalar.Scalar: output value of the neuron
        """
        out = [n(x) for n in self.neurons]
        return out[0] if len(self.neurons) == 1 and self._last_layer else out 

    def parameters(self) -> list[scalar.Scalar]:
        """Return the Scalar parameters that make up each neuron in the 
            layer

        Returns:
            list[scalar.Scalar]: all neuron weights and biases
        """
        return [p for n in self.neurons for p in n.parameters()]
    
    def __repr__(self):
        return f'Layer([{", ".join(str(n) for n in self.neurons)}])'