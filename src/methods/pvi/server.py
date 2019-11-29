import logging
from abc import ABC, abstractmethod
from collections import defaultdict

import numpy as np
import torch

from src.methods.base import server
from src.methods.exp_family import utils

logger = logging.getLogger(__name__)


class SyncronousPVIParameterServer(server.ParameterServer):

    def __init__(
        self,
        model_class,
        prior, 
        clients,
        hyperparameters=None, 
        metadata=None,
        model_parameters=None,
        model_hyperparameters=None,
    ):
        super(SyncronousPVIParameterServer, self).__init__(
            model_class,
            prior, 
            clients,
            hyperparameters, 
            metadata,
            model_parameters,
            model_hyperparameters,
        )

        self.current_damping_factor = self.hyperparameters['damping_factor']

    def tick(self):
        # Loop thorough clients
        # Grab update from each
        lambda_old = self.parameters

        self.current_damping_factor = self.hyperparameters["damping_factor"] * np.exp(
            -self.iterations * self.hyperparameters["damping_decay"])

        logger.debug('Getting client updates for iteration {self.iterations}')
        delta_is = []

        # Go through the clients sending them all the current params
        for i, client in enumerate(self.clients):
            logger.debug(f' Updating client {i}')
            client.set_hyperparameters({
                'damping_factor': self.current_damping_factor
            })
            update = client.get_update(
                model_parameters=lambda_old,
                model_hyperparameters=None,
            )
            if update is not None:
                delta_is.append(update)

        logger.debug('Got all updates for clients on iteration {self.iterations}')

        # Combine the updates
        delta = utils.list_add(*delta_is)
        # delta = utils.const_list_mul(self.current_damping_factor, delta)
        lambda_new = utils.list_add(lambda_old, delta)
        self.parameters = lambda_new
        # self.parameters = utils.list_add(self.prior, self.clients[0].t_i)
        self.model.set_parameters(self.parameters)  

        # prior = utils.list_sub(lambda_new, utils.list_add(*[client.t_i for client in self.clients]))
        # print([p.to_moment_params() for p in prior])

        self.iterations += 1

    def evaluate(self, dataset):
        return self.model.evaluate(dataset, self.parameters)      

    def get_default_hyperparameters(self):
        return {
            'damping_factor' : 1.,
            'damping_decay' : 0.
        }

    def get_default_metadata(self):
        return {}