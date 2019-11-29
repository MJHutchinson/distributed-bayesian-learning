import os
import sys

sys.path.append(os.getcwd())

from experiments.experiment_pvi import run_pvi_experiment

data_config = {
    'dataset': 'MNIST',
    'num_shards': 10,
    'data_root': '/data/hemispingus/mhutchin/Projects/distributed-bayesian-learning/data',
    'num_validation': 0,
}

model_config = {
    'type': 'pvi',
    'data_model': {
        'type': 'bnn',
        'hyperparameters': {
            'in_features': 28*28,
            'hidden_sizes': [200],
            'out_features': 10,
            'train_samples': 10,
            'test_samples': 10,
            'batch_size': 200,
            'type': 'classification',
            'lr': 5e-3,
            'N_sync': 100,
            'device': 'cpu'
        }
    },
    'server_config': {
        'type': 'syncronous',
        'hyperparameters': {
            'damping_factor': 0.95,
            'damping_decay': 1.718e-3
        }
    }
}

training_config = {
    'epochs': 100,
    'record_freq': 1,
}

run_pvi_experiment('cpu', 'results/pvi_test', 3, data_config, model_config, training_config)

