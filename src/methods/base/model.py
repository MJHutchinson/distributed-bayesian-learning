import logging
from abc import ABC, abstractmethod
from collections import defaultdict

import numpy as np
import torch

logger = logging.getLogger(__name__)

class Model(ABC):
    def __init__(self, parameters=None, hyperparameters=None):

        if parameters is None:
            parameters = {}

        if hyperparameters is None:
            hyperparameters = {}

        self.parameters = self.get_default_parameters()
        self.hyperparameters = self.get_default_hyperparameters()

        self.set_parameters(parameters)
        self.set_hyperparameters(hyperparameters)

    def set_parameters(self, parameters):
        self.parameters = {**self.parameters, **parameters}

    def get_parameters(self):
        '''
        In case you need to repackage parameters somehow from a form other than the dictionary.
        :return: a dictionary of the parameters
        '''
        return self.parameters

    @classmethod
    def get_default_parameters(cls):
        '''
        :return: A default set of parameters for the model. These might be all zero. Mostly used to get the shape that
        the parameters should be to make parameter server code more general.
        '''
        pass

    def set_hyperparameters(self, hyperparameters):
        if hyperparameters is not None:
            self.hyperparameters = {**self.hyperparameters, **hyperparameters}

    @classmethod
    def get_default_hyperparameters(cls):
        '''
        :return: A default set of hyperparameters( for the model. These might be all zero. Mostly used to get the shape that
        the hyperparameters should be to make parameter server code more general and easier to remeber what could go in here.
        '''
        pass

    @abstractmethod
    def fit(self, dataloader, t_i, parameters=None, hyperparameters=None):
        '''
        :param data: The local data to refine the model with
        :param t_i: The local contribution of the client
        :param parameters: optinal updated model parameters
        :param hyperparameters: optional updated hyperparameters
        :return: lambda_i_new, t_i_new, the new model parameters and new local contribution
        '''

        if parameters is not None:
            self.set_parameters(parameters)

        if hyperparameters is not None:
            self.set_hyperparameters(hyperparameters)

    @abstractmethod
    def predict(self, x, parameters=None, hyperparameters=None):
        '''
        :param x: The data to make predictions about
        :param parameters: optinal updated model parameters
        :param hyperparameters: optional updated hyperparameters
        :return: the model's predictions of the data
        '''

        if parameters is not None:
            self.set_parameters(**parameters)

        if hyperparameters is not None:
            self.set_hyperparameters(**hyperparameters)

    def sample(self, x, parameters=None, hyperparameters=None):
        '''
        Sample from the output of the model. Useful for generating data
        :param x: The data to make predictions about
        :param parameters: optinal updated model parameters
        :param hyperparameters: optional updated hyperparameters
        :return: the model's predictions of the data
        '''
        pass

    @abstractmethod
    def save_model(self, path):
        '''
        Save the model to a specified directory
        :param path: The path to save the model to
        '''
        return NotImplementedError

    @classmethod
    def load_model(cls, path):
        '''
        Load a model from the specified path and return an instance of it.
        '''
        return NotImplementedError

