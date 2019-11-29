import os
import sys
import torch
sys.path.append(os.getcwd())

from experiments.experiment import run_experiment

# if torch.cuda.is_available():
#     device = torch.device('cuda')
# else:
#     device = torch.device('cpu')

# Get config file from command line arguments
if len(sys.argv) != 3:
    raise(RuntimeError("Wrong arguments, use python main_experiment.py <path_to_config>"))
device = sys.argv[1]
config_path = sys.argv[2]

run_experiment(device, config_path)