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

if os.path.isdir(config_path):
    while len(os.listdir()) > 0:
        config_file = os.listdir().pop()
        config_file = os.path.join(config_path, config_file)
        os.remove(config_file)
        run_experiment(device, config_file)
else:
    run_experiment(device, config_path)