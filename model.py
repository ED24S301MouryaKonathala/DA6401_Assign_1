import numpy as np

class feedforwardnn:
    def __init__(self, input_size, hidden_layers, hidden_size, output_size, activation = 'sigmoid'):
        self.input_size = input_size
        self.hidden_layers = hidden_layers
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.activation = activation
        self.weights = self.initialize_weights()
    
    def initialize_weights(self):
        weights = []
        weights.append(np.random.randn(self.input_size,self.hidden_layers))
        for i in range(self.hidden_layers - 1):
            weights.append(np.random.randn(self.hidden_size, self.hidden_layers))
        weights.append(np.random.randn(self.hidden_layers,self.output_size))
        return weights
    def activation_fn(self, z):
        if self.activation == 'sigmoid':
            return 1/(1+np.exp(-z))
        elif self.activation == 'ReLU':
            return np.max(0,z)
        elif self.activation == 'tanh':
            return np.tanh(z)
        else: # Identity function
            return z