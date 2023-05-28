"""An optimizer updates the paramters of the layers based on the ouput
of the gradient.
"""
from nn_intro.nn_by_tensor import mlp


class Optimizer:
    def __init__(self, neural_network: mlp.MLP, learning_rate: float = 0.01) -> None:
        """Create an optimizer for a neural network with a known
        (initial) learning rate

        Args:
            neural_network (mlp.MLP): the neural network to operate on
            learning_rate (float): the learning rate. Defaults to 0.01
        """
        self.net = neural_network
        self.lr = learning_rate

    def step(self) -> None:
        """Take a step forward, then backpropagate the error.
        """
        raise NotImplementedError
    

class SGD(Optimizer):
    def step(self) -> None:
        for param, grad in self.net.params_and_grads():
            param -= grad*self.lr