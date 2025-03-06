import numpy as np

# Base Optimizer class
class optimizer:
    def __init__(self, lr=0.001, momentum=0.9, beta=0.9, beta1=0.9, beta2=0.99, epsilon=1e-8, weight_decay=0.0001):
        """
        Initializes the optimizer with the learning rate, momentum, beta values, epsilon (for numerical stability),
        weight decay (for regularization), and other optimizer parameters.
        """
        self.lr = lr
        self.momentum = momentum
        self.beta = beta
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        self.t = 0
        
        self.m = None  # Momentum term (for momentum and nesterov)
        self.v = None  # Velocity term (for Adam, RMSprop)
        self.s = None  # Moving average of squared gradients (for Nadam)

    def step(self, model, X, y, batch_size):
        """
        Performs a single optimization step (forward and backward pass, and weight update).
        
        :param model: The neural network model.
        :param X: The input batch of data.
        :param y: The true labels for the batch.
        :param batch_size: The size of the current batch.
        """
        # Forward pass
        output = model.forward(X)
        
        # Backward pass to calculate gradients
        grad_wrt_weights, grad_wrt_biases = model.backward(X, y)
        
        # Apply L2 weight decay (regularization)
        for i in range(len(model.weights)):
            grad_wrt_weights[i] += self.weight_decay * model.weights[i]
        
        # Normalize gradients by batch size
        grad_wrt_weights = [g / batch_size for g in grad_wrt_weights]
        grad_wrt_biases = [g / batch_size for g in grad_wrt_biases]
        
        # Call specific optimizer step method (SGD, Adam, etc.)
        self._apply_update(model, grad_wrt_weights, grad_wrt_biases)

    def _apply_update(self, model, grad_wrt_weights, grad_wrt_biases):
        """
        Applies weight updates using specific optimization algorithm (implemented in subclass).
        
        :param model: The neural network model.
        :param grad_wrt_weights: Gradients with respect to the weights.
        :param grad_wrt_biases: Gradients with respect to the biases.
        """
        raise NotImplementedError("Must be implemented in subclass.")


# SGD Optimizer (Stochastic Gradient Descent)
class SGD(optimizer):
    def __init__(self, lr=0.001, weight_decay=0.0001):
        """
        Initializes the SGD optimizer, which optionally includes momentum.
        """
        super().__init__(lr, weight_decay=weight_decay)
    
    def step(self, model, X, y, batch_size):
        grad_wrt_weights, grad_wrt_biases, output = super().step(model, X, y, batch_size)
        
        # Update weights and biases using SGD
        for i in range(len(model.weights)):
            model.weights[i] -= self.lr * grad_wrt_weights[i]
            model.biases[i] -= self.lr * grad_wrt_biases[i]


# Momentum Optimizer
class Momentum(SGD):
    def __init__(self, lr=0.001, momentum=0.9, weight_decay=0.0001):
        """
        Inherits SGD and uses momentum.
        """
        super().__init__(lr=lr, weight_decay=weight_decay)
        self.momentum = momentum

    def step(self, model, X, y, batch_size):
        grad_wrt_weights, grad_wrt_biases, output = super().step(model, X, y, batch_size)
        
        # Initialize momentum if it is None
        if self.m is None:
            self.m = [np.zeros_like(w) for w in model.weights]
        
        # Update momentum
        self.m = [self.m[i] * self.momentum + grad_wrt_weights[i] for i in range(len(self.m))]
        
        # Update weights and biases using momentum
        for i in range(len(model.weights)):
            model.weights[i] -= self.lr * self.m[i]
            model.biases[i] -= self.lr * grad_wrt_biases[i]


# Nesterov Accelerated Gradient Optimizer (NAG)
class Nesterov(Momentum):
    def __init__(self, lr=0.001, momentum=0.9, weight_decay=0.0001):
        """
        Inherits from Momentum and implements Nesterov Accelerated Gradient (NAG).
        """
        super().__init__(lr=lr, momentum=momentum, weight_decay=weight_decay)

    def step(self, model, X, y, batch_size):
        grad_wrt_weights, grad_wrt_biases, output = super().step(model, X, y, batch_size)
        
        # Compute Nesterov update (use previous momentum term)
        prev_m = [m.copy() for m in self.m]
        self.m = [self.m[i] * self.momentum + grad_wrt_weights[i] for i in range(len(self.m))]
        
        # Update weights and biases using Nesterov Accelerated Gradient
        for i in range(len(model.weights)):
            model.weights[i] -= self.lr * self.m[i]
            model.biases[i] -= self.lr * grad_wrt_biases[i]


# RMSProp Optimizer
class RMSProp(optimizer):
    def __init__(self, lr=0.001, beta=0.99, epsilon=1e-8, weight_decay=0.0001):
        """
        RMSProp optimizer with adaptive learning rates based on moving average of squared gradients.
        """
        super().__init__(lr=lr, beta=beta, epsilon=epsilon, weight_decay=weight_decay)
        self.s = None

    def step(self, model, X, y, batch_size):
        grad_wrt_weights, grad_wrt_biases, output = super().step(model, X, y, batch_size)
        
        # Initialize squared gradient moving average if None
        if self.s is None:
            self.s = [np.zeros_like(w) for w in model.weights]

        # Update moving averages of squared gradients
        for i in range(len(model.weights)):
            self.s[i] = self.beta * self.s[i] + (1 - self.beta) * (grad_wrt_weights[i] ** 2)
            model.weights[i] -= self.lr * grad_wrt_weights[i] / (np.sqrt(self.s[i]) + self.epsilon)
            model.biases[i] -= self.lr * grad_wrt_biases[i]


# Adam Optimizer
class Adam(optimizer):
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, weight_decay=0.0001):
        """
        Adam optimizer combines momentum (beta1) and RMSProp (beta2) for adaptive learning rates.
        """
        super().__init__(lr=lr, beta1=beta1, beta2=beta2, epsilon=epsilon, weight_decay=weight_decay)
        self.m = None  # First moment vector
        self.v = None  # Second moment vector

    def step(self, model, X, y, batch_size):
        grad_wrt_weights, grad_wrt_biases, output = super().step(model, X, y, batch_size)
        
        # Initialize first moment (m) and second moment (v) if None
        if self.m is None:
            self.m = [np.zeros_like(w) for w in model.weights]
            self.v = [np.zeros_like(w) for w in model.weights]
        
        # Update first and second moment estimates
        for i in range(len(model.weights)):
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad_wrt_weights[i]
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (grad_wrt_weights[i] ** 2)

            # Bias correction
            m_hat = self.m[i] / (1 - self.beta1 ** (self.t + 1))
            v_hat = self.v[i] / (1 - self.beta2 ** (self.t + 1))

            # Update weights and biases using Adam
            model.weights[i] -= self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)
            model.biases[i] -= self.lr * grad_wrt_biases[i]

        self.t += 1  # Increment timestep


# Nadam Optimizer (Nesterov + Adam)
class Nadam(Adam, Nesterov):
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, momentum=0.9, epsilon=1e-8, weight_decay=0.0001):
        """
        Nadam combines Nesterov momentum with Adam optimization.
        """
        Adam.__init__(self, lr=lr, beta1=beta1, beta2=beta2, epsilon=epsilon, weight_decay=weight_decay)
        Nesterov.__init__(self, lr=lr, momentum=momentum, weight_decay=weight_decay)

    def step(self, model, X, y, batch_size):
        grad_wrt_weights, grad_wrt_biases, output = super().step(model, X, y, batch_size)
        
        # Nadam updates the weights using both Adam and Nesterov methods
        self.momentum = self.momentum * (1 - self.beta1 ** self.t) + self.momentum

        for i in range(len(model.weights)):
            model.weights[i] -= self.lr * self.momentum
            model.biases[i] -= self.lr * grad_wrt_biases[i]

