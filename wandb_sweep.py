import wandb
from train import train_model
import warnings
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
warnings.filterwarnings('ignore', category=UserWarning)

def generate_sweep_name(params):
    """Generates a meaningful sweep name based on the hyperparameters."""
    return f"hl_{params['num_layers']}_bs_{params['batch_size']}_ac_{params['activation']}_opt_{params['optimizer']}"

# Define the sweep configuration
sweep_configuration = {
    'method': 'random',
    'metric': {'name': 'validation_accuracy', 'goal': 'maximize'},
    'parameters': {
        'dataset': {'values': ['fashion_mnist']},
        'epochs': {'values': [5, 10]},
        'num_layers': {'values': [3, 4, 5]},
        'hidden_size': {'values': [32, 64, 128]},
        'weight_decay': {'values': [0, 0.0005, 0.5]},
        'learning_rate': {'values': [1e-3, 1e-4]},
        'optimizer': {'values': ['sgd', 'momentum', 'nesterov', 'rmsprop', 'adam', 'nadam']},
        'batch_size': {'values': [16, 32, 64]},
        'weight_init': {'values': ['random', 'Xavier', 'He']},
        'activation': {'values': ['sigmoid', 'tanh', 'ReLU']},
        'momentum': {'values': [0.9]},        
        'beta': {'values': [0.9]},            
        'beta1': {'values': [0.9]},           
        'beta2': {'values': [0.999]},         
        'epsilon': {'values': [1e-8]},
        'loss': {'values': ['cross_entropy']}
    }
}

def train_sweep():
    """Wrapper function for wandb agent."""
    try:
        # First initialize wandb
        wandb.init()
        
        # Now we can safely access the config
        run_name = f"hl_{wandb.config.num_layers}_bs_{wandb.config.batch_size}_ac_{wandb.config.activation}_opt_{wandb.config.optimizer}"
        # Update the run name
        wandb.run.name = run_name
        wandb.run.save()
        
        # Run the training
        train_model(wandb.config)
    except Exception as e:
        print(f"Error in sweep run: {str(e)}")
        # Only try to log if wandb is initialized
        if wandb.run is not None:
            wandb.log({'error': str(e)})
            wandb.finish(exit_code=1)

# Initialize the sweep with project name
sweep_id = wandb.sweep(
    sweep=sweep_configuration, 
    project='myprojectname'  # Make sure this matches your project name
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


