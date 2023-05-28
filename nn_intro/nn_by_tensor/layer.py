"""Layers of neurons. Note that a layer contains a set of tensors.
Unlike nn_by_scalar where the forward and backward passes are 
implemented at the scalar level, here the logic for forward and
backward passes is at the layer level.
"""
from typing import Callable

import numpy as np

from nn_intro.nn_by_tensor import tensor


class Layer():
    def __init__(self) -> None:
        self.w = tensor.Tensor
        self.b = tensor.Tensor
        self.x = None  # Also can be thought of as inputs
        self.grad_w = 0
        self.grad_b = 0

    def forward(self, x: tensor.Tensor) -> tensor.Tensor:
        """Compute the forward pass of the neurons in a layer

        Args:
            x (tensor.Tensor): input values for the layer

        Returns:
            tensor.Tensor: output computations passing through class
        """
        raise NotImplementedError
    
    def backward(self, grad: tensor.Tensor) -> tensor.Tensor:
        """Backpropagate gradient through this layer

        Args:
            grad (tensor.Tensor): the gradient, via the loss function

        Returns:
            tensor.Tensor: the gradient updated through this layer
        """
        raise NotImplementedError
    

class Linear(Layer):
    """Compute outputs from the inputs updated by a weight and bias
    out = input @ w + b"""
    def __init__(self, input_size: int, output_size: int) -> None:
        """Create a linear layer

        Args:
            input_size (int): the number of input values to the layer
                (batch_size, input_size)
            output_size (int): the number of output values to the next
                layer (or final value)
                (batch_size, output_size)
        """
        super().__init__()
        self.w = np.random.randn(input_size, output_size)
        self.b = np.random.randn(output_size)

    def forward(self, x: tensor.Tensor) -> tensor.Tensor:
        """Computing y = w @ x + b
        where @ is matrix multiplication (equivalent to scalar *)"""
        self.x = x
        return self.x @ self.w + self.b
    
    def backward(self, grad: tensor.Tensor) -> tensor.Tensor:
        """Similarly to how we worked out the derivate for Scalar in 
        nn_by_scalar, we can now compute the combined gradient for 
        X = w*x + b
        y = f(X)
        dy/dw = f'(X)*x
        dy/dx = f'(X)*w
        dy/db = f'(X)

        The new component being added is the tensor form:
        if y = f(X) and X = x @ w + b and f'(X) is grad
        dy/dx = f'(X) @ w.T
        dy/dw = x.T @ f'(X)
        dy/db = f'(X)
        """
        # Sum along the batch dimension
        self.grad_b = np.sum(grad, axis=0)
        self.grad_w = self.x.T @ grad
        return grad @ self.w.T
    

class Activation(Linear):
    """A generic activation layer type. Key is that it applies a
    function elementwise after computing x*w + b
    """
    def __init__(self, 
                 input_size: int, 
                 output_size: int,
                 f: Callable[[tensor.Tensor], tensor.Tensor], 
                 f_prime: Callable[[tensor.Tensor], tensor.Tensor]) -> None:
        """Initialize an activation layer as a generic layer that also
        has a function and its derivative, from which the gradient can
        be computed

        Args:
            input_size (int): the number of input values to the layer
                (batch_size, input_size)
            output_size (int): the number of output values to the next
                layer (or final value)
                (batch_size, output_size)
            f (Callable[[tensor.Tensor], tensor.Tensor]): a 
                differentiable function
            f_prime (Callable[[tensor.Tensor], tensor.Tensor]): the 
                first derivative of f, as a function
        """
        super().__init__(input_size=input_size, output_size=output_size)
        self.f = f
        self.f_prime = f_prime

    def forward(self, x: tensor.Tensor) -> tensor.Tensor:
        self.x = x
        return self.f(super(Activation, self).forward(x))
    
    def backward(self, grad: tensor.Tensor) -> tensor.Tensor:
        """
        If y = f(x) and x = g(z)
        then dy/dz = f'(x) * g'(z)

        Think of f as the computation performed by the rest of the
        neural network and g being the component performed by this layer
        or as we called it in nn_by_scalar, the local derivative.
        """
        grad = super(Activation, self).backward(grad)
        return self.f_prime(self.x)*grad
    

def tanh(x: tensor.Tensor) -> tensor.Tensor:
    """Implement tanh function for tensors"""
    return np.tanh(x)

def tanh_prime(x: tensor.Tensor) -> tensor.Tensor:
    """First derivative of tanh"""
    y = tanh(x)
    return 1 - y**2


class Tanh(Activation):
    def __init__(self, input_size: int, output_size: int):
        super().__init__(input_size, output_size, tanh, tanh_prime)