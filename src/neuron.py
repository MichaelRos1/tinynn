import numpy as np
from activations import relu

class Neuron:
    def __init__(self, input_size, activation):
        self.input_size = input_size
        self.activation = activation
        self._initialize_parameters(input_size)

    def _initialize_parameters(self, input_size):
        self.weights = np.random.uniform(-0.1, 0.1, size=input_size)
        self.bias = 0.0

    def forward(self, inputs):
        output = np.dot(self.weights, inputs) + self.bias
        return relu(output)

    #def backward(self, inputs, grad_output):