import os
import sys
import time
import json
import torch

sys.path.append(os.getcwd())

from experiments.experiment_pvi import run_pvi_experiment

def run_experiment(device, path_to_config):

    with open(path_to_config) as config_file:
        config = json.load(config_file)

    results_dir = os.path.expandvars(config['results_dir'])
    seeds = config['seeds']

    data_config = config['experiment_config']['data_config']
    data_config['data_root'] = os.path.expandvars(data_config['data_root'])
    model_config = config['experiment_config']['model_config']
    training_config = config['experiment_config']['training_config']

    if model_config['type'] == 'pvi':
        run_pvi_experiment(
            device, 
            results_dir,
            seeds,
            data_config,
            model_config,
            training_config
        )
