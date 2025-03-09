import wandb
from train import train_model
import warnings
import matplotlib
import json

def generate_sweep_name(params):
    """Generates a meaningful sweep name based on the hyperparameters."""
    return f"hl_{params['num_layers']}_bs_{params['batch_size']}_ac_{params['activation']}_opt_{params['optimizer']}"

# Define the sweep configuration
sweep_configuration = {
    'method': 'bayes',  # Bayesian optimization is more efficient than random search as it learns from previous trials to select better hyperparameters
    'metric': {'name': 'validation_accuracy', 'goal': 'maximize'},
    'parameters': {
        'dataset': {'values': ['fashion_mnist']},
        'epochs': {'values': [5, 10]},
        'num_layers': {'values': [3, 4, 5]},
        'hidden_size': {'values': [32, 64, 128]},
        'weight_decay': {'values': [0, 0.0005, 0.0001, 0.5]},
        'learning_rate': {'values': [1e-3, 1e-4]},
        'optimizer': {'values': ['sgd', 'momentum', 'nesterov', 'rmsprop', 'adam', 'nadam']},
        'batch_size': {'values': [16, 32, 64]},
        'weight_init': {'values': ['random', 'Xavier', 'He']},
        'activation': {'values': ['sigmoid', 'tanh', 'ReLU']},
        # Based on values suggested in the class (Best values suggested by practicioners)
        'momentum': {'values': [0.9]},        
        'beta': {'values': [0.9]},            
        'beta1': {'values': [0.9]},           
        'beta2': {'values': [0.999]},         
        'epsilon': {'values': [1e-8]},
        'loss': {'values': ['cross_entropy']}
    }
}

def train_sweep():
    try:
        wandb.init()
        run_name = f"hl_{wandb.config.num_layers}_bs_{wandb.config.batch_size}_ac_{wandb.config.activation}_opt_{wandb.config.optimizer}"
        wandb.run.name = run_name
        wandb.run.save()
        
        # Train the model
        metrics = train_model(wandb.config)
        
        if metrics and 'validation_accuracy' in metrics:
            # Log metrics directly to wandb
            wandb.run.summary['best_val_accuracy'] = metrics['validation_accuracy']
            wandb.run.summary['best_test_accuracy'] = metrics['test_accuracy']
            
    except Exception as e:
        print(f"Error in sweep run: {str(e)}")
        if wandb.run is not None:
            wandb.log({'error': str(e)})
            wandb.finish(exit_code=1)

# Initialize the sweep with project name
sweep_id = wandb.sweep(
    sweep=sweep_configuration, 
    project='Assign_1'  # Make sure this matches your project name
)

try:
    # Start the sweep
    wandb.agent(
        sweep_id, 
        function=train_sweep,
        count=100
    )
except KeyboardInterrupt:
    print("Sweep interrupted by user")
except Exception as e:
    print(f"Error running sweep: {str(e)}")


