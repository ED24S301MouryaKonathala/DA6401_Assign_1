import numpy as np

class feedforwardnn:
    def __init__(self, input_size, num_layers, hidden_size, output_size, activation, weight_init):

        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.activation = activation
        self.weights = []
        self.biases = []
        self.activations = []
        self.z = []
        self.dz = []

        # Initializing weights and biases for layers
        self._initialize_weights(input_size, hidden_size, output_size, weight_init)

    def _activation(self, z):

        if self.activation == 'sigmoid':
            return 1 / (1 + np.exp(-z))
        elif self.activation == 'ReLU':
            return np.maximum(0, z)
        elif self.activation == 'tanh':
            return np.tanh(z)
        elif self.activation == 'identity':
            return z

    def _softmax(self, z):

        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))  # For numerical stability
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)



    def _initialize_weights(self, input_size, hidden_size, output_size, weight_init):

        if weight_init == 'random':
            self.weights.append(np.random.randn(input_size, hidden_size))
            self.biases.append(np.zeros((1, hidden_size)))
            for _ in range(self.num_layers - 1):
                self.weights.append(np.random.randn(hidden_size, hidden_size))
                self.biases.append(np.zeros((1, hidden_size)))
            self.weights.append(np.random.randn(hidden_size, output_size))
            self.biases.append(np.zeros((1, output_size)))

        elif weight_init == 'Xavier':
            # Xavier normal initialization (Gaussina Dist)
            self.weights.append(np.random.randn(input_size, hidden_size) * np.sqrt(2.0 / input_size))
            self.biases.append(np.zeros((1, hidden_size)))
            for _ in range(self.num_layers - 1):
                self.weights.append(np.random.randn(hidden_size, hidden_size) * np.sqrt(2.0 / hidden_size))
                self.biases.append(np.zeros((1, hidden_size)))
            self.weights.append(np.random.randn(hidden_size, output_size) * np.sqrt(2.0 / hidden_size))
            self.biases.append(np.zeros((1, output_size)))


    def forward(self, X):

        self.activations = [X]  # Input activations
        self.z = []  # List to store pre-activation values
        
        # Forward pass through hidden layers
        for i in range(self.num_layers - 1):  # Last layer is output layer, we process only hidden layers
            z = np.dot(self.activations[i], self.weights[i]) + self.biases[i]
            self.z.append(z)
            a = self._activation(z)
            self.activations.append(a)
        
        # Output layer
        z_out = np.dot(self.activations[-1], self.weights[-1]) + self.biases[-1]
        self.z.append(z_out)
        output = self._softmax(z_out)
        return output



