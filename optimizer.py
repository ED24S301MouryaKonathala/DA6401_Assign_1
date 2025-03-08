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
        """Performs a single optimization step."""
        # Forward pass
        output = model.forward(X)
        
        # Backward pass
        grad_wrt_weights, grad_wrt_biases = model.backward(X, y)
        
        # Apply L2 weight decay and normalize gradients
        for i in range(len(model.weights)):
            grad_wrt_weights[i] = grad_wrt_weights[i] / batch_size + self.weight_decay * model.weights[i]
            grad_wrt_biases[i] = grad_wrt_biases[i] / batch_size
        
        # Apply updates
        self._apply_update(model, grad_wrt_weights, grad_wrt_biases)
        return output, grad_wrt_weights, grad_wrt_biases

    def _apply_update(self, model, grad_wrt_weights, grad_wrt_biases):
        for i in range(len(model.weights)):
            model.weights[i] -= self.lr * grad_wrt_weights[i]
            model.biases[i] -= self.lr * grad_wrt_biases[i]


# SGD Optimizer (Stochastic Gradient Descent)
class SGD(optimizer):
    def __init__(self, lr=0.001, weight_decay=0.0001):
        """
        Initializes the SGD optimizer, which optionally includes momentum.
        """
        super().__init__(lr, weight_decay=weight_decay)
    
    def step(self, model, X, y, batch_size):
        super().step(model, X, y, batch_size)


# Momentum Optimizer
class Momentum(SGD):
    def __init__(self, lr=0.001, momentum=0.9, weight_decay=0.0001):
        """
        Inherits SGD and uses momentum.
        """
        super().__init__(lr=lr, weight_decay=weight_decay)
        self.momentum = momentum

    def step(self, model, X, y, batch_size):
        # Forward pass
        output = model.forward(X)
        
        # Backward pass
        grad_wrt_weights, grad_wrt_biases = model.backward(X, y)
        
        # Initialize momentum if it is None
        if self.m is None:
            self.m = [np.zeros_like(w) for w in model.weights]
        
        # Update momentum and apply gradients
        for i in range(len(model.weights)):
            self.m[i] = self.momentum * self.m[i] + grad_wrt_weights[i] / batch_size
            model.weights[i] -= self.lr * self.m[i]
            model.biases[i] -= self.lr * grad_wrt_biases[i] / batch_size
        
        return output


# Nesterov Accelerated Gradient Optimizer (NAG)
class Nesterov(Momentum):
    def __init__(self, lr=0.001, momentum=0.9, weight_decay=0.0001):
        super().__init__(lr=lr, momentum=momentum, weight_decay=weight_decay)

    def step(self, model, X, y, batch_size):
        grad_wrt_weights, grad_wrt_biases = model.backward(X, y)

        if self.m is None:
            self.m = [np.zeros_like(w) for w in model.weights]

        # Compute Nesterov update
        for i in range(len(model.weights)):
            prev_m = self.m[i].copy()
            self.m[i] = self.momentum * self.m[i] + self.lr * grad_wrt_weights[i] / batch_size
            model.weights[i] -= self.momentum * prev_m + (1 + self.momentum) * self.m[i]
            model.biases[i] -= self.lr * grad_wrt_biases[i] / batch_size


# RMSProp Optimizer
class RMSProp(optimizer):
    def __init__(self, lr=0.001, beta=0.99, epsilon=1e-8, weight_decay=0.0001):
        """
        RMSProp optimizer with adaptive learning rates based on moving average of squared gradients.
        """
        super().__init__(lr=lr, beta=beta, epsilon=epsilon, weight_decay=weight_decay)
        self.s = None

    def step(self, model, X, y, batch_size):
        output, grad_wrt_weights, grad_wrt_biases = super().step(model, X, y, batch_size)
        
        # Initialize squared gradient moving average if None
        if self.s is None:
            self.s = [np.zeros_like(w) for w in model.weights]

        # Update moving averages of squared gradients
        for i in range(len(model.weights)):
            self.s[i] = self.beta * self.s[i] + (1 - self.beta) * (grad_wrt_weights[i] ** 2)
            model.weights[i] -= self.lr * grad_wrt_weights[i] / (np.sqrt(self.s[i]) + self.epsilon)
            model.biases[i] -= self.lr * grad_wrt_biases[i]
        return output


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
        output, grad_wrt_weights, grad_wrt_biases = super().step(model, X, y, batch_size)
        
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
        return output


# Nadam Optimizer (Nesterov + Adam)
class Nadam(Adam):
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, momentum=0.9, epsilon=1e-8, weight_decay=0.0001):
        super().__init__(lr=lr, beta1=beta1, beta2=beta2, epsilon=epsilon, weight_decay=weight_decay)
        self.momentum = momentum

    def step(self, model, X, y, batch_size):
        grad_wrt_weights, grad_wrt_biases = model.backward(X, y)

        if self.m is None:
            self.m = [np.zeros_like(w) for w in model.weights]
            self.v = [np.zeros_like(w) for w in model.weights]

        self.t += 1

        for i in range(len(model.weights)):
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad_wrt_weights[i]
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (grad_wrt_weights[i] ** 2)

            # Bias correction
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)

            # Nadam momentum correction
            nesterov_term = self.momentum * self.m[i]
            model.weights[i] -= self.lr * (nesterov_term + (1 - self.beta1) * m_hat) / (np.sqrt(v_hat) + self.epsilon)
            model.biases[i] -= self.lr * grad_wrt_biases[i] / batch_size

