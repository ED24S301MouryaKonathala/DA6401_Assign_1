import numpy as np

class FeedforwardNN:
    def __init__(self, input_size, num_layers, hidden_size, output_size, activation, weight_init):
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.activation = activation
        self.weights = []
        self.biases = []
        self.activations = []
        self.z = []

        # Initialize weights properly
        self._initialize_weights(input_size, hidden_size, output_size, weight_init)

def _initialize_weights(self, input_size, hidden_size, output_size, weight_init):
    np.random.seed(42)  # Ensures reproducibility

    if weight_init == 'random':
        self.weights.append(np.random.randn(input_size, hidden_size) * 0.01)
    elif weight_init == 'Xavier':
        self.weights.append(np.random.randn(input_size, hidden_size) * np.sqrt(1.0 / input_size))
    elif weight_init == 'He':
        self.weights.append(np.random.randn(input_size, hidden_size) * np.sqrt(2.0 / input_size))  # Added He initialization
    self.biases.append(np.zeros((1, hidden_size)))

    for _ in range(self.num_layers - 1):
        if weight_init == 'random':
            self.weights.append(np.random.randn(hidden_size, hidden_size) * 0.01)
        elif weight_init == 'Xavier':
            self.weights.append(np.random.randn(hidden_size, hidden_size) * np.sqrt(1.0 / hidden_size))
        elif weight_init == 'He':
            self.weights.append(np.random.randn(hidden_size, hidden_size) * np.sqrt(2.0 / hidden_size))  # Added He initialization
        self.biases.append(np.zeros((1, hidden_size)))

    if weight_init == 'random':
        self.weights.append(np.random.randn(hidden_size, output_size) * 0.01)
    elif weight_init == 'Xavier':
        self.weights.append(np.random.randn(hidden_size, output_size) * np.sqrt(1.0 / hidden_size))
    elif weight_init == 'He':
        self.weights.append(np.random.randn(hidden_size, output_size) * np.sqrt(2.0 / hidden_size))  # Added He initialization
    self.biases.append(np.zeros((1, output_size)))


    def _activation(self, z):
        if self.activation == 'sigmoid':
            return 1 / (1 + np.exp(-z))
        elif self.activation == 'ReLU':
            return np.maximum(0, z)
        elif self.activation == 'tanh':
            return np.tanh(z)
        elif self.activation == 'identity':
            return z
    
    def activation_derivative(self, z):
        if self.activation == 'sigmoid':
            return self._activation(z) * (1 - self._activation(z))
        elif self.activation == 'ReLU':
            return np.where(z > 0, 1, 0)
        elif self.activation == 'tanh':
            return 1 - np.tanh(z) ** 2
        elif self.activation == 'identity':
            return np.ones_like(z)
    
    def _softmax(self, z):
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def forward(self, X):
        self.activations = [X]
        self.z = []
        
        for i in range(self.num_layers):
            z = np.dot(self.activations[i], self.weights[i]) + self.biases[i]
            self.z.append(z)
            self.activations.append(self._activation(z))
        
        z_out = np.dot(self.activations[-1], self.weights[-1]) + self.biases[-1]
        self.z.append(z_out)
        return self._softmax(z_out)

    def compute_loss(self, output, y, loss_type='cross_entropy'):
        m = y.shape[0]
        if loss_type == 'cross_entropy':
            log_likelihood = -np.log(output[range(m), np.argmax(y, axis=1)] + 1e-8)
            return np.sum(log_likelihood) / m
        elif loss_type == 'mean_squared_error':
            return np.mean((output - y) ** 2)

    def backward(self, X, y):
        grad_wrt_output = self.activations[-1] - y
        grad_wrt_weights = []
        grad_wrt_biases = []
        
        for i in range(self.num_layers, 0, -1):
            grad_wrt_weights.insert(0, np.dot(self.activations[i-1].T, grad_wrt_output))
            grad_wrt_biases.insert(0, np.sum(grad_wrt_output, axis=0, keepdims=True))
            if i > 1:
                grad_wrt_output = np.dot(grad_wrt_output, self.weights[i-1].T) * self.activation_derivative(self.z[i-2])
        return grad_wrt_weights, grad_wrt_biases