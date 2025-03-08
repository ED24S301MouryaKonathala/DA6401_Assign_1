import argparse
import numpy as np
import wandb
import seaborn as sns
import matplotlib.pyplot as plt
from model import FeedforwardNN
from optimizer import SGD, Momentum, Nesterov, RMSProp, Adam, Nadam
from utils import load_data
from sklearn.metrics import confusion_matrix, accuracy_score

# Parse command-line arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Train a Feedforward Neural Network")

    parser.add_argument('-wp', '--wandb_project', type=str, default='myprojectname', help="Project name for Weights & Biases")
    parser.add_argument('-we', '--wandb_entity', type=str, default='myname', help="WandB entity")
    parser.add_argument('-d', '--dataset', type=str, choices=['mnist', 'fashion_mnist'], default='fashion_mnist')
    parser.add_argument('-e', '--epochs', type=int, default=1, help="Number of epochs")
    parser.add_argument('-b', '--batch_size', type=int, default=4, help="Batch size")
    parser.add_argument('-l', '--loss', type=str, choices=['mean_squared_error', 'cross_entropy'], default='cross_entropy')
    parser.add_argument('-o', '--optimizer', type=str, choices=['sgd', 'momentum', 'nag', 'rmsprop', 'adam', 'nadam'], default='sgd')
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.1, help="Learning rate")
    parser.add_argument('-m', '--momentum', type=float, default=0.5, help="Momentum for Momentum/NAG")
    parser.add_argument('-beta', '--beta', type=float, default=0.5, help="Beta for RMSProp")
    parser.add_argument('-beta1', '--beta1', type=float, default=0.5, help="Beta1 for Adam/Nadam")
    parser.add_argument('-beta2', '--beta2', type=float, default=0.5, help="Beta2 for Adam/Nadam")
    parser.add_argument('-eps', '--epsilon', type=float, default=1e-6, help="Epsilon for optimizers")
    parser.add_argument('-w_d', '--weight_decay', type=float, default=0.0, help="Weight decay (L2 regularization)")
    parser.add_argument('-w_i', '--weight_init', type=str, choices=['random', 'Xavier', 'He'], default='random')
    parser.add_argument('-nhl', '--num_layers', type=int, default=1, help="Number of hidden layers")
    parser.add_argument('-sz', '--hidden_size', type=int, default=4, help="Number of neurons per hidden layer")
    parser.add_argument('-a', '--activation', type=str, choices=['identity', 'sigmoid', 'tanh', 'ReLU'], default='sigmoid')

    return parser.parse_args()

# Function to initialize the optimizer
def get_optimizer(config, model):
    if config.optimizer == 'sgd':
        return SGD(lr=config.learning_rate, weight_decay=config.weight_decay)
    elif config.optimizer == 'momentum':
        return Momentum(lr=config.learning_rate, momentum=config.momentum, weight_decay=config.weight_decay)
    elif config.optimizer == 'nesterov':
        return Nesterov(lr=config.learning_rate, momentum=config.momentum, weight_decay=config.weight_decay)
    elif config.optimizer == 'rmsprop':
        return RMSProp(lr=config.learning_rate, beta=config.beta, epsilon=config.epsilon, weight_decay=config.weight_decay)
    elif config.optimizer == 'adam':
        return Adam(lr=config.learning_rate, beta1=config.beta1, beta2=config.beta2, epsilon=config.epsilon, weight_decay=config.weight_decay)
    elif config.optimizer == 'nadam':
        return Nadam(lr=config.learning_rate, beta1=config.beta1, beta2=config.beta2, momentum=config.momentum, epsilon=config.epsilon, weight_decay=config.weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {config.optimizer}")

# Function to plot confusion matrix
def plot_confusion_matrix(y_true, y_pred, class_names):
    """Plot an enhanced confusion matrix."""
    # Save figure instead of showing it during sweep
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')
    plt.close()  # Close the figure to free memory
    
    # Log the figure to wandb
    wandb.log({"confusion_matrix": wandb.Image('confusion_matrix.png')})

# Training function
def train_model(config=None):
    """Training function that handles both command line and sweep configurations"""
    # Initialize wandb if running from command line
    if config is None:
        args = parse_args()
        wandb.init(project='myprojectname', entity='myname', config=vars(args))
        config = wandb.config
    else:
        # When running from sweep, initialize with provided config
        wandb.init(config=config)
        config = wandb.config

    try:
        # Load dataset
        (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_data(dataset=config.dataset)

        # Ensure input data is correctly shaped
        X_train = X_train.reshape(-1, 784)  # Flatten images
        X_val = X_val.reshape(-1, 784)
        X_test = X_test.reshape(-1, 784)

        # One-hot encode labels
        y_train_onehot = np.eye(10)[y_train]
        y_val_onehot = np.eye(10)[y_val]
        y_test_onehot = np.eye(10)[y_test]

        # Initialize model and optimizer
        model = FeedforwardNN(
            input_size=784,
            num_layers=config.num_layers,
            hidden_size=config.hidden_size,
            output_size=10,
            activation=config.activation,
            weight_init=config.weight_init
        )

        optimizer = get_optimizer(config, model)

        # Training loop
        for epoch in range(config.epochs):
            train_loss = 0
            train_acc = 0
            n_batches = 0

            # Shuffle training data
            indices = np.random.permutation(len(X_train))
            X_train_shuffled = X_train[indices]
            y_train_shuffled = y_train_onehot[indices]

            for i in range(0, len(X_train), config.batch_size):
                X_batch = X_train_shuffled[i:i+config.batch_size]
                y_batch = y_train_shuffled[i:i+config.batch_size]
                
                # Forward pass
                batch_output = model.forward(X_batch)
                
                # Optimizer step
                optimizer.step(model, X_batch, y_batch, config.batch_size)
                
                # Calculate batch metrics
                batch_loss = model.compute_loss(batch_output, y_batch, config.loss)
                batch_acc = np.mean(np.argmax(batch_output, axis=1) == np.argmax(y_batch, axis=1))
                
                train_loss += batch_loss
                train_acc += batch_acc
                n_batches += 1

            # Compute validation metrics
            val_output = model.forward(X_val)
            val_loss = model.compute_loss(val_output, y_val_onehot, config.loss)
            val_accuracy = np.mean(np.argmax(val_output, axis=1) == y_val)

            # Log metrics
            wandb.log({
                'epoch': epoch + 1,
                'train_loss': train_loss / n_batches,
                'train_accuracy': train_acc / n_batches,
                'validation_loss': val_loss,
                'validation_accuracy': val_accuracy
            })

        # Evaluate on the test set
        test_output = model.forward(X_test)
        test_accuracy = np.mean(np.argmax(test_output, axis=1) == y_test)
        wandb.log({'test_accuracy': test_accuracy})
        print(f"Test Accuracy: {test_accuracy:.4f}")

        # Generate and log confusion matrix
        class_names = ["T-shirt", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Boot"]
        y_pred = np.argmax(test_output, axis=1)
        plot_confusion_matrix(y_test, y_pred, class_names)

    except Exception as e:
        print(f"Error during training: {str(e)}")
        wandb.log({'error': str(e)})
        wandb.finish(exit_code=1)
        return

    wandb.finish()

# Main execution
if __name__ == "__main__":
    args = parse_args()
    train_model(config=vars(args))
