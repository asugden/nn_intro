"""A Multilayer Perceptron, otherwise known as a fully connected
feedforward artificial neural network. The terminology in this space is
imprecise. We will use MLP explicitly because it specifies that there is
at least one hidden layer and that all of the layers are fully connected.
"""

from nn_intro.nn_by_scalar import layer, scalar


class MLP():
    def __init__(self, layer_sizes: list[int], **kwargs):
        """Create a fully connected feedforward (vanilla) neural network
        also known as a multilayer perceptron.

        Args:
            layer_sizes (list[int]): the sizes of all layers, including
                input layers and output layer
        """
        self.layers = [layer.Layer(layer_sizes[i], layer_sizes[i+1], 
                                   _last_layer=(i == len(layer_sizes) - 2), 
                                   **kwargs)
                       for i in range(len(layer_sizes) - 1)]

    def __call__(self, x: list[float]) -> scalar.Scalar:
        """A forward pass of data through each layer sequentially

        Args:
            x (list[float]): input data

        Returns:
            scalar.Scalar: an output scalar value
        """
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self) -> list[scalar.Scalar]:
        """Return the Scalar parameters that make up each neuron in each
            layer

        Returns:
            list[scalar.Scalar]: all neuron weights and biases
        """
        return [p for layer in self.layers for p in layer.parameters()]

    def __repr__(self):
        return f'MLP([{", ".join(str(layer) for layer in self.layers)}])'
