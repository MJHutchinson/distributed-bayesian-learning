import os
import sys
import time
import json
import torch
import logging
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
        train_shards, validation_shard, test_shard = genereate_shards(data_config, seed)

        num_train_shards = len(train_shards)

        torch.manual_seed(seed)

        if model_config['data_model']['type'] == 'bnn':
            from src.methods.pvi.model_bnn import BNNModel
            model_class = BNNModel
        else:
            raise ValueError(f"Unrecognised model type {model_config['data_model']['type']}")

        model_hyperparameters = model_config['data_model']['hyperparameters']

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

                test_log_loss.append(tll)
                test_reg_loss.append(trl)
                test_epoch.append(test_epoch)
                test_time.append(time.time() - start_time)
                best_tll = max(test_log_loss)
                print(f'Epoch {epoch}: \t log loss {tll: 0.6f}, accuracy {trl*100:0.3f}, damping {server.current_damping_factor}')


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

    plt.loglog(test_epochs, test_log_losses)
    plt.xlabel('epochs/server communications')
    plt.ylabel('test log likelihood')
    plt.savefig(os.path.join(results_dir, 'epochs_ll.pdf'))
    plt.savefig(os.path.join(results_dir, 'epochs_ll.png'))
    plt.close()

    plt.loglog(test_epochs, test_reg_losses)
    plt.xlabel('epochs/server communications')
    plt.ylabel('test regular loss')
    plt.savefig(os.path.join(results_dir, 'epochs_reg.pdf'))
    plt.savefig(os.path.join(results_dir, 'epochs_reg.png'))
    plt.close()

    plt.loglog(test_times, test_log_losses)
    plt.xlabel('time(s)')
    plt.ylabel('test log likelihood')
    plt.savefig(os.path.join(results_dir, 'time_ll.pdf'))
    plt.savefig(os.path.join(results_dir, 'time_ll.png'))
    plt.close()

    plt.loglog(test_times, test_reg_losses)
    plt.xlabel('tims(s)')
    plt.ylabel('test regular loss')
    plt.savefig(os.path.join(results_dir, 'time_reg.pdf'))
    plt.savefig(os.path.join(results_dir, 'time_reg.png'))
    plt.close()