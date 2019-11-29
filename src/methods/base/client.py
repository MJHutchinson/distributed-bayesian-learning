import logging
from abc import ABC, abstractmethod
from collections import defaultdict

import numpy as np
import torch

logger = logging.getLogger(__name__)

class Client(ABC):

    def __init__(self, model_class, data, model_parameters=None, model_hyperparameters=None, hyperparameters=None,
                 metadata=None):
        if hyperparameters is None:
            hyperparameters = {}

        if metadata is None:
            metadata = {}

        self.hyperparameters = self.get_default_hyperparameters()
        self.metadata = self.get_default_metadata()

        self.set_hyperparameters(hyperparameters)
        self.set_metadata(metadata)

        self.data = data
        self.model = model_class(model_parameters, model_hyperparameters)
        # self.log = defaultdict(list)
        self.times_updated = 0

    def set_hyperparameters(self, hyperparameters):
        self.hyperparameters = {**self.hyperparameters, **hyperparameters}

    def set_metadata(self, metadata):
        self.metadata = {**self.metadata, **metadata}

    @classmethod
    def get_default_hyperparameters(cls):
        return {}

    @classmethod
    def get_default_metadata(cls):
        return {}

    def get_update(self, model_parameters=None, model_hyperparameters=None, update_ti=True):
        """ Method to wrap the update and then logging process.
        :param model_parameters: New model parameters from the server
        :param model_hyperparameters: New model hyperparameters from the server.
        :return:
        """

        update = self.compute_update(model_parameters=model_parameters, model_hyperparameters=model_hyperparameters)

        # self.log_update()

        return update

    @abstractmethod
    def compute_update(self, model_parameters=None, model_hyperparameters=None):
        """ Abstract method for computing the update itself.
        :return: The update step to return to the server
        """
        if model_parameters is not None:
            self.model.set_parameters(model_parameters)
        if model_hyperparameters is not None:
            self.model.set_hyperparameters(model_hyperparameters)

    def can_update(self):
        """
        A check to see if this client can indeed update. Examples of reasons one may not be is simulated unavailability
        or a client had expended all of its privacy.
        :return:
        """
        return True
