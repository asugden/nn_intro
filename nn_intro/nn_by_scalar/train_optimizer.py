"""Train a neural network, backpropagate, and update with a learning
rate.
"""


from nn_intro.nn_by_scalar import loss, mlp, scalar


def sgd(nnet: mlp.MLP, 
                     loss_fn, 
                     epochs: int,
                     learning_rate: float,
                     data: list[list[float]], 
                     labels: list[float]) -> scalar.Scalar:
    """Implement stochastic gradient descent optimizer using our simple 
    tools. SGD can subsample data points but does not in this case.

    Args:
        nnet (mlp.MLP): an initialized neural network
        loss_fn (): a function that returns a scalar from predictions and
            labels
        epochs (int): the number of times to iterate
        learning_rate (float): the learning rate of the network, usually
            0.01 for a tiny network like this
        data (list[list[float]]): input data
        labels (list[float]): output labels

    Returns:
        scalar.Scalar: the final value of the loss
    """
    for step in range(epochs):
        # Forward pass + loss
        pred  = [nnet(x) for x in data]
        l = loss_fn(pred, labels)

        # Zero gradients because we use += for the case of scalars used
        # multiple times
        for p in nnet.parameters():
            p.grad = 0

        # Backpropagate and update parameters
        l.backward_pass()
        for p in nnet.parameters():
            p.val += -learning_rate*p.grad

        print(f'Loss of {l} on step {step}')

if __name__ == '__main__':
    nnet = mlp.MLP([3, 4, 4, 1])
    data = [
        [2, 3, -5],
        [3, -5, 0.5],
        [0.5, 1, 1],
        [1, 1, -5],
    ]
    labels = [1, -1, -1, 1]
    sgd(nnet, loss.mse, 50, 0.05, data, labels)