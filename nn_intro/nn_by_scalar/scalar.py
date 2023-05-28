"""A single scalar value that can be manipulated as a component of a
neural network. It stores its own gradient and is capable of propagating
it via its knowledge of how it combines with otehr scalars.

Inspired by micrograd by Andrej Karpathy
See: https://www.youtube.com/watch?v=VMj-3S1tku0 and
https://github.com/karpathy/micrograd
"""

from __future__ import annotations

import math


class Scalar():
    def __init__(self, 
                 val: float,
                 label: str = '', 
                 _input_scalars: tuple[Scalar]=(),
                 _operation: str = ''):
        """Initialize a new Scalar or subset of a node, keeping track
        of the components and operations that make up the value.

        Args:
            val (float): the current value of the node
            _input_scalars (tuple[Scalar], optional): the inputs to the node. 
                Defaults to ().
            _operation (str, optional): the mathematical operation of the node. 
                Defaults to ''.
            label (str, optional): an optional label for graphing. 
                Defaults to ''.
        """
        self.val = val
        self.label = label
        self.grad = 0
        self._backward_pass = lambda: None
        self._input_scalars = _input_scalars
        self._operation = _operation

    def __repr__(self):
        return f'Scalar({self.val})'
    
    def __add__(self, scalar2: Scalar | float) -> Scalar: 
        """Addition with another Scalar.
        Note that, for the equation
        a + b = c
        The local derivative of a with respect to c, da/dc can be derived
        by taking the limit as step => 0 of the following:
        (f(x + step) - f(x))/step
        (((a + step) + b) - (a + b))/step
        (a + b + step - a - b)/step
        step/step
        1

        To propagate a derivative through, the chain rule says that, to
        determine the derivative of a with respect to the downstream 
        loss function L, dL/da, so long as you know dL/dc
        dL/da = dc/da*dL/dc
        dL/da = 1.0*dL/dc

        Args:
            scalar2 (Scalar): A second Scalar

        Returns:
            Scalar: a child Scalar that is the sum of two input Scalars
        """
        if not isinstance(scalar2, Scalar):
            scalar2 = Scalar(scalar2)

        out = Scalar(self.val + scalar2.val, _input_scalars=(self, scalar2), _operation='+')

        def local_derivative():
            # += is necessary to account for multiple instances of the
            # same value
            self.grad += 1.0*out.grad
            scalar2.grad += 1.0*out.grad
        out._backward_pass = local_derivative

        return out
    
    def __radd__(self, other: float | int) -> Scalar:
        """Account for inverted order of addition with a float or int"""
        return self + other

    def __sub__(self, other: Scalar) -> Scalar:
        "Form subtraction from addition"
        return self + (-other)

    def __rsub__(self, other) -> Scalar:
        """Account for inverted order of subtraction with a float or int"""
        return other + (-self)
    
    def __mul__(self, scalar: Scalar | float) -> Scalar:
        """Multiplication with another Scalar
        Note that for the equation:
        a * b = c
        The derivative of a with respect to c, da/dc can be derived
        by taking the limit as step => 0 of the following:
        (f(x + step) - f(x))/step
        ((a + step)*b - a*b)/step
        (a*b + step*b - a*b)/step
        b

        To propagate a derivative through, the chain rule says that, to
        determine the derivative of a with respect to the downstream 
        loss function L, dL/da, so long as you know dL/dc
        dL/da = dc/da*dL/dc
        dL/da = b*dL/dc

        Args:
            scalar2 (Scalar): A second scalar

        Returns:
            Scalar: a child Scalar that is the product of two input Scalars
        """
        if not isinstance(scalar, Scalar):
            scalar = Scalar(scalar)

        out = Scalar(self.val * scalar.val, _input_scalars=(self, scalar), _operation='*')

        def local_derivative():
            self.grad += scalar.val*out.grad
            scalar.grad += self.val*out.grad
        out._backward_pass = local_derivative

        return out
    
    def __rmul__(self, other: float | int) -> Scalar:
        """Account for inverted order of multiple with a float"""
        return self * other
    
    def __neg__(self) -> Scalar:
        """Negative is required for subtraction"""
        return self * -1

    def __truediv__(self, other: Scalar) -> Scalar:
        """Form addition via multiplication (or exponentiation)"""
        return self * other**-1

    def __rtruediv__(self, other: float | int) -> Scalar:
        """Account for inverted order of division with a float or int"""
        return other * self**-1

    def __pow__(self, other: int | float) -> Scalar:
        """Power is required for being able to compute MSE loss

        Args:
            other (int | float): the power to which the value is raised

        Returns:
            Scalar: an exponentiated output
        """
        assert isinstance(other, (int, float)), 'Powers must be numerical'
        out = Scalar(self.val**other, _input_scalars=(self, ), _operation=f'**{other}')

        def local_derivative():
            self.grad += (other * self.val**(other-1)) * out.grad
        out._backward_pass = local_derivative

        return out
    
    def tanh(self) -> Scalar:
        """The output of the tanh function. This could be produced via
        a set of differentiable "atomic elements" such as addition, 
        multiplication, and exponentiation. Or, it can be produced by
        direct knowledge of the derivative of tanh.

        The first derivative of tanh(x) is 
        1 - tanh^2(x)

        Which can be propagated from local to global gradient as above

        Returns:
            Scalar: an output Scalar after passing through hyperbolic
                tangent function
        """
        x = self.val
        t = (math.exp(2*x) - 1)/(math.exp(2*x) + 1)

        out = Scalar(t, _input_scalars=(self, ), _operation='tanh')

        def local_derivative():
            self.grad += (1 - t**2)*out.grad
        out._backward_pass = local_derivative

        return out
    
    def relu(self) -> Scalar:
        """The output of the rectified linear unit-- the common activation
        function these days. It is nonlinear, but incredibly simply so
        and its derivative is trivial.

        relu(x) is 0 for x <= 0, x for x > 0
        The first derivative of relu(x) is
        0 for x <= 0, 1 for x > 0

        Returns:
            Scalar: an output scalar after passing through relu
        """
        r = 0 if self.val <= 0 else self.val
        out = Scalar(r, _input_scalars=(self, ), _operation='relu')

        def local_derivative():
            self.grad += out.grad if self.val > 0 else 0
        out._backward_pass = local_derivative

        return out
    
    def backward_pass(self) -> None:
        """To propagate loss backwards, one must ensure that each node
        is called only once. This can be done by treating the connections
        as a directed acyclic graph (DAG) and ordering the children so
        that each successor's _backward_pass function is called before
        its predecessor.
        """
        ordered_scalars = []
        visited_scalars = set()
        def recursive_pass(v: Scalar):
            if v not in visited_scalars:
                visited_scalars.add(v)
                for input_v in v._input_scalars:
                    recursive_pass(input_v)
                ordered_scalars.append(v)
        recursive_pass(self)

        self.grad = 1.0
        for v in reversed(ordered_scalars):
            v._backward_pass()
