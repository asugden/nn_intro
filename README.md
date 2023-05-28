# nn_intro

This is a set of introductions to the core components of neural networks.

1. `nn_by_scalar` introduces you to a neural network via individual
   scalar values that back up a single "neuron" within a layer within a
   network.
1. `nn_by_tensor` uses numpy to introduce mechanisms of batching data
   and a demonstration of the slight updates that are required to set
   weights, biases, and gradients within a tensor.

As you go through, pay close attention to the derivative descriptions in
nn_by_scalar.scalar and nn_by_tensor.layer. It is also great to consider
how a neuron is made up of a set of simple operations: multiplcation of
weights and input data, addition with the biases, and then the slightly-
more-complicated activation functions.

### References

These are largely based on projects from Andrej Karpathy and Joel Grus
whose excellent demos I use in my classes. In this case, they have been
updated to use consistent nomenclature with each other and with my teaching.
