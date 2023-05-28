"""A neural network is a collection of layers. In this case, we're 
going to call it a multilayer perceptron because we're creating
fully connected neural networks just as in nn_by_scalar
"""
from typing import Iterator

from nn_intro.nn_by_tensor import layer, tensor


class MLP():
    def __init__(self, layers: list[layer.Layer]) -> None:
        self.layers = layers

    def forward(self, x: tensor.Tensor) -> tensor.Tensor:
        """Compute a forward pass through the entire network

        Args:
            x (tensor.Tensor): input data to be propagated

        Returns:
            tensor.Tensor: the output of the entire network computation
        """
        for layer in self.layers:
            x = layer.forward(x)
        return x
    
    def backward(self, grad: tensor.Tensor) -> tensor.Tensor:
        """The backward pass from a known gradient (output of loss)

        Args:
            grad (tensor.Tensor): the computed gradient

        Returns:
            tensor.Tensor: the gradient propagated through all layers
        """
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad

    def params_and_grads(self) -> Iterator[tuple[tensor.Tensor, tensor.Tensor]]:
        """Return the weights and biases for every single layer in turn,
        along with their gradients

        Yields:
            Iterator[tuple[tensor.Tensor, tensor.Tensor]]: parameter
            (weight or bias) and its associated gradient
        """
        for layer in self.layers:
            for pair in [(layer.w, layer.grad_w), (layer.b, layer.grad_b)]:
                yield pair

    def zero_parameters(self) -> None:
        """Set all weights and biases to zero"""
        for layer in self.layers:
            layer.grad_w[:] = 0
            layer.grad_b[:] = 0