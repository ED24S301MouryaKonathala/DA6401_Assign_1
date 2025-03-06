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
    elif config.optimizer == 'nag':
        return Nesterov(lr=config.learning_rate, momentum=config.momentum, weight_decay=config.weight_decay)
    elif config.optimizer == 'rmsprop':
        return RMSProp(lr=config.learning_rate, beta=config.beta, epsilon=config.epsilon, weight_decay=config.weight_decay)
    elif config.optimizer == 'adam':
        return Adam(lr=config.learning_rate, beta1=config.beta1, beta2=config.beta2, epsilon=config.epsilon, weight_decay=config.weight_decay)
    elif config.optimizer == 'nadam':
        return Nadam(lr=config.learning_rate, beta1=config.beta1, beta2=config.beta2, momentum=config.momentum, epsilon=config.epsilon, weight_decay=config.weight_decay)
    else:
        raise ValueError("Invalid optimizer choice.")

# Function to plot confusion matrix
def plot_confusion_matrix(y_true, y_pred, class_names):
    """Plot an enhanced confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.show()

# Training function
def train_model(config):
    # Load dataset
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_data(dataset=config.dataset)

    # One-hot encode labels
    y_train_onehot = np.eye(10)[y_train]
    y_val_onehot = np.eye(10)[y_val]
    y_test_onehot = np.eye(10)[y_test]

    # Initialize model
    model = FeedforwardNN(
        input_size=784,
        num_layers=config.num_layers,
        hidden_size=config.hidden_size,
        output_size=10,
        activation=config.activation,
        weight_init=config.weight_init
    )

    # Initialize optimizer
    optimizer = get_optimizer(config, model)

    # Training loop
    for epoch in range(config.epochs):
        for i in range(0, len(X_train), config.batch_size):
            X_batch = X_train[i:i+config.batch_size]
            y_batch = y_train_onehot[i:i+config.batch_size]
            optimizer.step(model, X_batch, y_batch, batch_size=config.batch_size)

        # Compute validation accuracy
        val_output = model.forward(X_val)
        val_loss = model.compute_loss(val_output, y_val_onehot)
        val_accuracy = np.mean(np.argmax(val_output, axis=1) == y_val)

        # Log to wandb
        wandb.log({
            'epoch': epoch + 1,
            'validation_loss': val_loss,
            'validation_accuracy': val_accuracy
        })
        print(f"Epoch {epoch+1}/{config.epochs} - Validation Accuracy: {val_accuracy:.4f}, Loss: {val_loss:.4f}")

    # Evaluate on the test set
    test_output = model.forward(X_test)
    test_accuracy = np.mean(np.argmax(test_output, axis=1) == y_test)
    wandb.log({'test_accuracy': test_accuracy})
    print(f"Test Accuracy: {test_accuracy:.4f}")

    # Plot and display the confusion matrix
    class_names = ["T-shirt", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Boot"]
    y_pred = np.argmax(test_output, axis=1)
    plot_confusion_matrix(y_test, y_pred, class_names)

# Main execution
if __name__ == "__main__":
    args = parse_args()

    # Initialize WandB
    wandb.init(project=args.wandb_project, entity=args.wandb_entity, config=vars(args))

    # Train the model
    train_model(wandb.config)
