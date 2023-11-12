# we will build a neuron using NumPy

import numpy as np

# activation function
def sigmoid(x):
    # this is our activation function: f(x) = 1 / (1 + e^(-x))
    return 1 / (1 + np.exp(-x))

# actual neuron class
class Neuron:
    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias
    # passing inputs forward to get an output
    def feedforward(self, inputs):
        # weight inputs, add bias, and use activation function
        total = np.dot(self.weights, inputs) + self.bias
        return sigmoid(total)

weights = np.array([2, 3]) # w1 = 0, w2 = 1
bias = 8 

n = Neuron(weights, bias)

x = np.array([5, 3]) # x1 = 2, x2 = 3
print(n.feedforward(x)) 

