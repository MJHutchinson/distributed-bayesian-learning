import os
import sys
import json
import argparse

import numpy as np

sys.path.append(os.getcwd())

from viz.plot_classification_model_evaluation import *
from data.generate_shards import genereate_shards


parser = argparse.ArgumentParser()

parser.add_argument('-r', '--results_dir', type=str, required=True)
parser.add_argument('-n', '--model-num', type=int, default=0)

args = parser.parse_args()

results_dir = args.results_dir
model_num = args.model_num

with open(os.path.join(results_dir, 'config.json'), 'r') as f:
    config = json.read(f)

model_config = config['model_config']['data_model']
data_config = config['data_config']


# Load the correct model 
# TODO: abstract the general model loading process?
if model_config['type'] == 'bnn':
    if config['model_config']['type'] == 'pvi':
        from src.methods.pvi.model_bnn import BNNModel
        model = BNNModel.load_model(os.path.join(results_dir, f'best_model_{model_num}'))


# Load the relevant data
# In sample sets, out of sample sets.
if data_config['dataset'] == 'MNIST':
    train_shards, validation_shard, test_shard = genereate_shards(data_config, seed=model_num, device=model.device)

problem_type = data_config['type']

if problem_type =='classification':
    from src.evaluate.evaluate_classification import *
    bin_edges = np.linspace(0,1.,num=11)