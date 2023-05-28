"""Create the framework for training a neural network.
"""

from nn_intro.nn_by_tensor import batch, loss, mlp, optimizer, tensor


def train(nn: mlp.MLP,
          features: tensor.Tensor,
          labels: tensor.Tensor,
          epochs: int = 5000,
          iterator: batch.DataIterator = batch.BatchIterator(),
          loss: loss.Loss = loss.MSE(),
          optimizer: optimizer.Optimizer = optimizer.SGD,
          learning_rate: float = 0.05):
    """Train a neural network (fully connected feedforward/multilayer
    perceptron)

    Args:
        nn (mlp.MLP): a neural network
        features (tensor.Tensor): feature data
        labels (tensor.Tensor): label data
        epochs (int, optional): the number of training epochs. 
            Defaults to 5000.
        iterator (batch.DataIterator, optional): the batch iterator tool. 
            Defaults to batch.BatchIterator().
        loss (loss.Loss, optional): the loss function. 
            Defaults to loss.MSE().
        optimizer (optimizer.Optimizer, optional): the optimizer to be
            use for training. Defaults to optimizer.SGD.
        learning_rate (float, optional): the initial learning rate. 
            Defaults to 0.01.
    """
    optim = optimizer(nn, learning_rate)
    for e in range(epochs):
        epoch_loss = 0.0
        for batch in iterator(features, labels):
            predictions = nn.forward(batch.features)
            epoch_loss =+ loss.loss(predictions, batch.labels)
            grad = loss.grad(predictions, batch.labels)
            nn.backward(grad)
            optim.step()
            nn.zero_parameters()
        print(f'Epoch {e} has loss {epoch_loss}')


if __name__ == '__main__':
    import numpy as np

    from nn_intro.nn_by_tensor import layer

    # Use XOR as an example because a linear function cannot solve it
    features = np.array([
        [0, 0],
        [1, 0],
        [0, 1],
        [1, 1]
    ])
    labels = np.array([
        [1, 0],
        [0, 1],
        [0, 1],
        [1, 0]
    ])

    network = mlp.MLP([
        layer.Tanh(2, 2),
        layer.Tanh(2, 2),
    ])
    train(network, features, labels)
    print(network.forward(features))
