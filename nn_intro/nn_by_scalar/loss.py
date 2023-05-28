"""Loss functions that can be used for training a neural network
"""


from nn_intro.nn_by_scalar import scalar


def mse(predictions: list[scalar.Scalar], labels: list[float]) -> scalar.Scalar:
    """Return the mean-squared error loss function of a set of
    predictions or labels

    Args:
        predictions (list[scalar.Scalar]): predictions from neural network
        labels (list[float]): ground truth values being fit

    Returns:
        scalar.Scalar: mean-squared error. 
        IMPORTANTLY: this is also a scalar, for which backward_pass
            is correclty implemented to pass through the entire network
    """
    loss = sum([(p - l)**2 for p, l in zip(predictions, labels)])
    return loss