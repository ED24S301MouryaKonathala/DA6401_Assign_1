# All the files can be found in Master branch

## [Wandb Report Link](https://wandb.ai/mourya001-indian-institute-of-technology-madras/Assignment_1/reports/DA6401-Assignment-1--VmlldzoxMTgxOTA4MQ?accessToken=tws4gkiyn7erv0shjbaiocogsgaebhmyfv8s0j47zvdw017dpygw6strebs5azfy)

# Fashion MNIST Neural Network

This project implements a feedforward neural network for classifying Fashion MNIST images using various optimizers and architectures.

## Project Structure

```
.
├── model.py           # Neural network implementation
├── optimizer.py       # Optimization algorithms
├── train.py          # Training script
├── utils.py          # Utility functions
├── wandb_sweep.py    # Hyperparameter optimization
└── requirements.txt   # Project dependencies
```

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
```

2. Activate the virtual environment:
```bash
# Windows
.\venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Login to Weights & Biases:
```bash
wandb login
```

## Training

### Single Model Training

Train a model with best parameters:

```bash
python train.py

```

Train a model with specific parameters:

```bash
python train.py -d fashion_mnist -e 10 -b 32 -o nadam -lr 0.001 -nhl 4 -sz 128 -a ReLU -w_i He -l cross_entropy
```

Key parameters:
- `-d`: Dataset ('fashion_mnist')
- `-e`: Number of epochs
- `-b`: Batch size
- `-o`: Optimizer ('sgd', 'momentum', 'nesterov', 'rmsprop', 'adam', 'nadam')
- `-lr`: Learning rate
- `-nhl`: Number of hidden layers
- `-sz`: Hidden layer size
- `-a`: Activation function ('ReLU', 'sigmoid', 'tanh')
- `-w_i`: Weight initialization ('random', 'Xavier', 'He')
- `-l`: Loss function ('cross_entropy', 'mean_squared_error')

### Hyperparameter Sweep

Run hyperparameter optimization:

```bash
python wandb_sweep.py
```

This will:
1. Initialize a Bayesian optimization sweep
2. Test various hyperparameter combinations
3. Log results to Weights & Biases

## Model Evaluation

### Best Model Parameters

The best performing model configuration:
```python
{
    'optimizer': 'nadam',
    'learning_rate': 0.001,
    'num_layers': 4,
    'hidden_size': 128,
    'activation': 'ReLU',
    'weight_init': 'He',
    'batch_size': 32,
    'epochs': 10
}
```

### Visualizations

The training process generates:
- Loss curves
- Confusion matrix
- Sample images
- Training metrics

Access visualizations in:
1. Local files:
   - `confusion_matrix.png`
   - `loss_curve_cross_entropy.png`
   - `sample_images.png`

2. Weights & Biases dashboard:
   - Training curves
   - Model performance metrics
   - Confusion matrix
   - Sample predictions

## Results

To reproduce the best results:

1. Train the model:
```bash
python train.py -d fashion_mnist -e 10 -b 32 -o nadam -lr 0.001 -nhl 4 -sz 128 -a ReLU -w_i He -l cross_entropy
```

2. View results:
   - Check terminal output for metrics
   - Open Weights & Biases dashboard for visualizations
   - Review generated plots in project directory

## Notes

- All metrics and visualizations are automatically logged to Weights & Biases
- Confusion matrix shows per-class performance
- Loss curves help analyze training convergence

