import numpy as np

class FeedforwardNN:
    def __init__(self, input_size, num_layers, hidden_size, output_size, activation, weight_init):
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.activation_name = activation  
        self.weights = []
        self.biases = []
        self.activations = []
        self.z = []

        self._initialize_weights(input_size, hidden_size, output_size, weight_init)

    def _initialize_weights(self, input_size, hidden_size, output_size, weight_init):
        np.random.seed(42)  # For reproducibility

        if weight_init == 'random':
            self.weights.append(np.random.randn(input_size, hidden_size) * 0.01)  # Small weights to prevent large activations early in training
        elif weight_init == 'Xavier':
            self.weights.append(np.random.randn(input_size, hidden_size) * np.sqrt(1.0 / input_size)) # Balances variance for sigmoid/tanh activations
        elif weight_init == 'He':
            self.weights.append(np.random.randn(input_size, hidden_size) * np.sqrt(2.0 / input_size)) # Optimized for ReLU to maintain variance
        self.biases.append(np.zeros((1, hidden_size))) # Biases initialized to zero

        for _ in range(self.num_layers - 1):
            if weight_init == 'random':
                self.weights.append(np.random.randn(hidden_size, hidden_size) * 0.01)
            elif weight_init == 'Xavier':
                self.weights.append(np.random.randn(hidden_size, hidden_size) * np.sqrt(1.0 / hidden_size))
            elif weight_init == 'He':
                self.weights.append(np.random.randn(hidden_size, hidden_size) * np.sqrt(2.0 / hidden_size))
            self.biases.append(np.zeros((1, hidden_size)))

        if weight_init == 'random':
            self.weights.append(np.random.randn(hidden_size, output_size) * 0.01)
        elif weight_init == 'Xavier':
            self.weights.append(np.random.randn(hidden_size, output_size) * np.sqrt(1.0 / hidden_size))
        elif weight_init == 'He':
            self.weights.append(np.random.randn(hidden_size, output_size) * np.sqrt(2.0 / hidden_size))
        self.biases.append(np.zeros((1, output_size)))

    def apply_activation(self, z): 
        if self.activation_name == 'sigmoid':
            return 1 / (1 + np.exp(-z))
        elif self.activation_name == 'ReLU':
            return np.maximum(0, z)
        elif self.activation_name == 'tanh':
            return np.tanh(z)
        elif self.activation_name == 'identity':
            return z

    def activation_derivative(self, z):
        if self.activation_name == 'sigmoid': 
            return self.apply_activation(z) * (1 - self.apply_activation(z))
        elif self.activation_name == 'ReLU':
            return np.where(z > 0, 1, 0)
        elif self.activation_name == 'tanh':
            return 1 - np.tanh(z) ** 2
        elif self.activation_name == 'identity':
            return np.ones_like(z)

    def softmax(self, z):  
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))  # Subtract max to prevent overflow
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)  # Normalize

    def forward(self, X):   
        self.activations = [X]  
        self.z = []  
    
        for i in range(self.num_layers):
            z = np.dot(self.activations[-1], self.weights[i]) + self.biases[i]  # Weighted sum
            self.z.append(z)
            self.activations.append(self.apply_activation(z))  # Changed to use apply_activation

        z_out = np.dot(self.activations[-1], self.weights[self.num_layers]) + self.biases[self.num_layers]  # Output layer
        self.z.append(z_out)
        a_out = self.softmax(z_out)
        self.activations.append(a_out)
        return a_out

    def compute_loss(self, output, y, loss_type='cross_entropy'):
        m = y.shape[0]  # Number of samples in batch
        if loss_type == 'cross_entropy':
            log_likelihood = -np.log(output[range(m), np.argmax(y, axis=1)] + 1e-8) 
            return np.sum(log_likelihood) / m  # Average loss across all samples
        elif loss_type == 'mean_squared_error':
            return np.mean((output - y) ** 2)

    def backward(self, y, loss_type='cross_entropy'):
        if loss_type == 'cross_entropy':
            grad_wrt_output = self.activations[-1] - y
        elif loss_type == 'mean_squared_error':
            grad_wrt_output = 2 * (self.activations[-1] - y) / y.shape[0]

        grad_wrt_weights = []
        grad_wrt_biases = []
        delta = grad_wrt_output

        # Gradients computation for output layer
        grad_wrt_weights.insert(0, np.dot(self.activations[-2].T, delta))
        grad_wrt_biases.insert(0, np.sum(delta, axis=0, keepdims=True))
        
        # Backpropagate through hidden layers
        for i in range(len(self.weights) - 1, 0, -1):
            delta = np.dot(delta, self.weights[i].T) * self.activation_derivative(self.z[i-1])
            grad_wrt_weights.insert(0, np.dot(self.activations[i-1].T, delta))
            grad_wrt_biases.insert(0, np.sum(delta, axis=0, keepdims=True))

        return grad_wrt_weights, grad_wrt_biases