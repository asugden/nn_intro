"""Loss functions used for training a neural network. These measure
the difference between predictions and labels.
"""
import numpy as np

from nn_intro.nn_by_tensor import tensor


class Loss():
    def loss(self, predictions: tensor.Tensor, labels: tensor.Tensor) -> float:
        """The value of the loss function computed from predictions
        output by our neural network operating on features and labels 
        from known data.

        Args:
            predictions (tensor.Tensor): predicted values from nn
            labels (tensor.Tensor): known values

        Returns:
            float: the value of the loss
        """
        raise NotImplementedError
    
    def grad(self, predictions: tensor.Tensor, labels: tensor.Tensor) -> tensor.Tensor:
        """The gradient of the loss function with respect to the 
        predictions

        Args:
            predictions (tensor.Tensor): predicted values from nn
            labels (tensor.Tensor): known values

        Returns:
            tensor.Tensor: of the same size as predictions
        """
        raise NotImplementedError
    

class MSE(Loss):
    """Return the mean-squared error"""
    def loss(self, predictions: tensor.Tensor, labels: tensor.Tensor) -> float:
        return np.mean((predictions - labels)**2)
    
    def grad(self, predictions: tensor.Tensor, labels: tensor.Tensor) -> tensor.Tensor:
        return 2*(predictions - labels)
