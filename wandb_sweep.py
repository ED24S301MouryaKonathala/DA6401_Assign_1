import wandb

def generate_sweep_name(params):
    """Generates a meaningful sweep name based on the hyperparameters."""
    return f"hl_{params['num_layers']}_bs_{params['batch_size']}_ac_{params['activation']}"

# Define the sweep configuration
sweep_configuration = {
    'method': 'grid',
    'name': generate_sweep_name({
        'num_layers': 3,  # Default, will be updated per sweep
        'batch_size': 32,
        'activation': 'ReLU'
    }),
    'metric': {'name': 'validation_accuracy','goal': 'maximize',},
    'parameters': {
        'epochs': {'values': [5, 10]},
        'num_layers': {'values': [3, 4, 5]},
        'hidden_size': {'values': [32, 64, 128]},
        'weight_decay': {'values': [0, 0.0005, 0.5]},
        'learning_rate': {'values': [1e-3, 1e-4]},
        'optimizer': {'values': ['sgd', 'momentum', 'nesterov', 'rmsprop', 'adam', 'nadam']},
        'batch_size': {'values': [16, 32, 64]},
        'weight_init': {'values': ['random', 'Xavier', 'He']},
        'activation': {'values': ['sigmoid', 'tanh', 'ReLU']},
    }
}

# Initialize the sweep with wandb
wandb.sweep(sweep=sweep_configuration, project='myprojectname')
