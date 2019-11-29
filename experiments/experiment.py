import os
import sys
import time
import json
import torch

sys.path.append(os.getcwd())

from experiments.experiment_pvi import run_pvi_experiment

def run_experiments(device, path_to_config):

    with open(path_to_config) as config_file:
        config = json.load(config_file)

    timestamp = time.strftime("%Y-%m-%d_%H-%M")
    base_results_dir = os.path.join(config['results_dir'], f'results_{path_to_config.split(".")[-2]}_{timestamp}')

    data_configs = config['experiment_config']['data_configs']
    model_configs = config['experiment_config']['model_configs']
    training_config = config['experiment_config']['training_config']

    for i, data_config in enumerate(data_configs):
        for j, model_config in enumerate(model_configs):
            if model_config['type'] == 'pvi':
                run_pvi_experiment(
                    device, 
                    os.path.join(base_results_dir, f'{i}_{j}'), 
                    training_config['seeds'],
                    data_config,
                    model_config,
                    training_config
                )
