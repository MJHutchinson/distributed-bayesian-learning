import os
import sys
import json
import argparse

sys.path.append(os.getcwd())

from viz.plot_runs import *

parser = argparse.ArgumentParser()

parser.add_argument('-r', '--results-dir', type=str, required=True)
parser.add_argument('-t', '--type', type=str, required=True)

args = parser.parse_args()

results_dir = args.results_dir
model_type = args.type

results = []
configs = []
names = []

for fname in os.listdir(results_dir):
    path = os.path.join(results_dir, fname)
    if os.path.isdir(path) and (fname != 'configs'):
        with open(path + '/results.json', 'r') as f:
            r = json.load(f)
        with open(path + '/config.json', 'r') as f:
            c = json.load(f)
        
        results.append(r)
        configs.append(c)
        names.append(fname)


test_nlls = np.array([
    -np.array(r['test_log_losses'])
    for r
    in results
])

test_regs = np.array([
    np.array(r['test_reg_losses'])
    for r
    in results
])

if model_type == 'classification':
    test_regs = 1 - test_regs

test_epochs = np.array([
    np.array(r['test_epochs'])
    for r
    in results
])

test_times = np.array([
    np.array(r['test_times'])
    for r
    in results
])

labels = []


if model_type == 'classification':
    reg_loss_name = '% error'
elif model_type == 'regression':
    reg_loss_name = 'mse'

plot_runs_confidence(test_epochs, test_nlls, 'epochs', 'nlls', labels=names, same_color=False, loglog=True, savefig=results_dir + '/epoch_nll')
plot_runs_confidence(test_epochs, test_regs, 'epochs', reg_loss_name, labels=names, same_color=False, loglog=True, savefig=results_dir + '/epoch_reg')
plot_runs_confidence(test_times, test_nlls, 'time (s)', 'nlls', labels=names, same_color=False, loglog=True, savefig=results_dir + '/time_nll')
plot_runs_confidence(test_times, test_regs, 'time (s)', reg_loss_name, labels=names, same_color=False, loglog=True, savefig=results_dir + '/time_reg')
