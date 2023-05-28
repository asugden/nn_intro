"""Create a mechanism for batching data for the nn.
"""
from typing import Iterator, NamedTuple

import numpy as np

from nn_intro.nn_by_tensor import tensor

Batch = NamedTuple('Batch', [('features', tensor.Tensor), ('labels', tensor.Tensor)])


class DataIterator():
    def __call__(self, features: tensor.Tensor, labels: tensor.Tensor) -> Iterator:
        """Batch a set of data of features and labels

        Args:
            features (tensor.Tensor): features
            labels (tensor.Tensor): associated labels

        Yields:
            Iterator: a subset of data to be trained on
        """
        raise NotImplementedError
    

class BatchIterator(DataIterator):
    def __init__(self, batch_size: int = 32, shuffle: bool = True) -> None:
        """Create a new iterator for batching data

        Args:
            batch_size (int, optional): the number of values to batch. 
                Defaults to 32.
            shuffle (bool, optional): whether to shuffle the training 
                data. Defaults to True.
        """
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __call__(self, features: tensor.Tensor, labels: tensor.Tensor) -> Iterator:
        starts = np.arange(0, len(features), self.batch_size)
        if self.shuffle:
            np.random.shuffle(starts)
        
        for start in starts:
            end = start + self.batch_size
            batch_features = features[start:end]
            batch_labels = labels[start:end]
            yield Batch(batch_features, batch_labels)