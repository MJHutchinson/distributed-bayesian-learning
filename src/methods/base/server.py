import logging
from abc import ABC, abstractmethod
from collections import defaultdict

import numpy as np
import torch

logger = logging.getLogger(__name__)

class ParameterServer(ABC):
    """Basic class tha defines how a parameter server should function.
    """
    
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

        if hyperparameters is None:
            hyperparameters = {}

        if metadata is None:
            metadata = {}

        self.hyperparameters = self.get_default_hyperparameters()
        self.metadata = self.get_default_metadata()

        self.set_hyperparameters(hyperparameters)
        self.set_metadata(metadata)

        # Set our clients to be the ones passed
        self.set_clients(clients)
        # create a server instance of the model
        self.model = model_class(parameters=model_parameters, hyperparameters=model_hyperparameters)
        self.prior = prior

        # Set the server parameters to be what the current models are.
        self.parameters = self.model.get_parameters()

        self.iterations = 0


    def set_hyperparameters(self, hyperparameters):
        self.hyperparameters = {**self.hyperparameters, **hyperparameters}

    def set_metadata(self, metadata):
        self.metadata = {**self.metadata, **metadata}

    @abstractmethod
    def get_default_hyperparameters(self):
        return {}

    @abstractmethod
    def get_default_metadata(self):
        return {}

    @abstractmethod
    def tick(self):
        '''
        Defines what the Parameter Server should do on each update round. Could be all client synchronous updates,
        async updates, might check for new clients etc
        '''
        pass

    def set_clients(self, clients):
        if clients is None:
            self.clients = []
        else:
            self.clients = clients

    def get_clients(self):
        return self.clients

    def get_client_sacred_logs(self):
        client_sacred_logs = [client.log_sacred() for client in self.get_clients()]
        return client_sacred_logs

    def add_client(self, client):
        self.clients.append(client)

    def get_num_iterations(self):
        return self.iterations

    def get_parameters(self):
        return self.parameters