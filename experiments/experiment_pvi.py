import os
import sys
import time
import json
import torch
import logging
import datetime
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data.dataloader import DataLoader
from torchvision import datasets, transforms

sys.path.append(os.getcwd())

from data.generate_shards import genereate_shards
from src.methods.pvi.client import PVIClient
import src.methods.exp_family.diagonal_gaussian as dg
import src.methods.exp_family.utils as utils

def run_pvi_experiment(device, results_dir, seeds, data_config={}, model_config={}, training_config={}):

    test_log_losses = []
    test_reg_losses = []
    test_epochs = []
    test_times = []

    for seed in range(seeds):
        print(f'Running seed {seed+1}/{seeds}. Saving in {results_dir}')
        train_shards, validation_shard, test_shard = genereate_shards(data_config, seed, device)

        num_train_shards = len(train_shards)

        torch.manual_seed(seed)

        if model_config['data_model']['type'] == 'bnn':
            from src.methods.pvi.model_bnn import BNNModel
            model_class = BNNModel
        else:
            raise ValueError(f"Unrecognised model type {model_config['data_model']['type']}")

        model_hyperparameters = model_config['data_model']['hyperparameters']
        model_hyperparameters['device'] = device

        # Initialise a model with the parameters.
        # We might train this a bit to precondition, or use it 
        # as a template for building priors and intialisations
        # for the clients from 
        init_data_model = model_class(
            parameters=None,
            hyperparameters=model_hyperparameters
        )

        # Grab the models initial parameters
        global_ti = init_data_model.get_parameters()
        # set the prior to be a unit gaussian
        prior = dg.list_uniform_prior(global_ti, 0., 1.)
        # Divide down the initial t_i so that the initial model has
        # similar initialisation to that of a single global model.
        # This initialisation can massivly impact the model training
        # performance
        client_ti = utils.list_const_div(num_train_shards, global_ti)

        clients = [
            PVIClient(
                model_class,
                train_shards[i],
                t_i=client_ti,
                model_parameters=None,
                model_hyperparameters=model_hyperparameters,
                hyperparameters=None,
                metadata=None
            )
            for i in range(num_train_shards)
        ]

        parameters = utils.list_add(*[client.t_i for client in clients], prior)

        if model_config['server_config']['type'] == 'syncronous':
            from src.methods.pvi.server import SyncronousPVIParameterServer
            server = SyncronousPVIParameterServer(
                BNNModel,
                prior,
                clients,
                model_parameters=parameters,
                model_hyperparameters=model_hyperparameters,
                hyperparameters=model_config['server_config']['hyperparameters']
            )
        else:
            raise ValueError(f"Unrecognised server type {model_config['server_config']['type']}")

        
        test_log_loss = []
        test_reg_loss = []
        test_epoch = []
        test_time = []

        best_tll = -1000

        start_time = time.time()

        for epoch in range(training_config['epochs']):
            server.tick()

            if epoch % training_config['record_freq'] == 0:
                tll, trl = server.evaluate(
                    test_shard
                )   

                if tll > best_tll:
                    server.model.save_model(os.path.join(results_dir, f'best_model_{seed}'))

                curr_time = time.time() - start_time

                test_log_loss.append(tll)
                test_reg_loss.append(trl)
                test_epoch.append(epoch+1)
                test_time.append(time.time() - start_time)
                best_tll = max(test_log_loss)
                print(f'Epoch {epoch}: \t log loss {tll: 0.6f}, accuracy {trl*100:0.3f}, damping {server.current_damping_factor:0.4f} , time {datetime.timedelta(seconds=curr_time)}')


        test_log_losses.append(test_log_loss)
        test_reg_losses.append(test_reg_loss)
        test_epochs.append(test_epoch)
        test_times.append(test_time)

    # Save results
    os.makedirs(results_dir, exist_ok=True)

    with open(os.path.join(results_dir, 'results.json'), 'w') as f:
        json.dump({
            'test_log_losses' : test_log_losses,
            'test_reg_losses' : test_reg_losses,
            'test_epochs' : test_epochs,
            'test_times' : test_times,
        }, f)

    with open(os.path.join(results_dir, 'config.json'), 'w') as f:
        json.dump({
            'data_config' : data_config,
            'model_config' : model_config,
            'training_config' : training_config,
        }, f)

    # plot some results
    from viz.plot_runs import plot_runs
    import numpy as np

    test_nlls = -np.array(test_log_losses)

    test_regs = np.array(test_reg_losses)

    if data_config['type'] == 'classification':
        test_regs = 1 - test_regs

    test_epochs = np.array(test_epochs)

    test_times = np.array(test_times)

    if data_config['type'] == 'classification':
        reg_loss_name = '% error'
    elif data_config['type'] == 'regression':
        reg_loss_name = 'mse'

    labels = list(range(seeds))

    plot_runs(test_epochs, test_nlls, 'epochs', 'nlls', labels=labels, same_color=False, loglog=True, savefig=results_dir + '/epoch_nll')
    plot_runs(test_epochs, test_regs, 'epochs', reg_loss_name, labels=labels, same_color=False, loglog=True, savefig=results_dir + '/epoch_reg')
    plot_runs(test_times, test_nlls, 'time (s)', 'nlls', labels=labels, same_color=False, loglog=True, savefig=results_dir + '/time_nll')
    plot_runs(test_times, test_regs, 'time (s)', reg_loss_name, labels=labels, same_color=False, loglog=True, savefig=results_dir + '/time_reg')