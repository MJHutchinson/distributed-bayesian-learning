import os
import pickle
from copy import deepcopy

import torch
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader

import src.methods.exp_family.diagonal_gaussian as dg
import src.methods.exp_family.utils as utils
from src.methods.pvi.model import Model
from src.models.bnn.model import BayesianMLPReparam, BayesianLinearReparam


class BNNModel(Model):

    parameters_file = 'parameters.pkl'
    hyperparameters_file = 'hyperparameters.pkl'

    def __init__(self, parameters=None, hyperparameters=None):
        super().__init__(parameters, hyperparameters)

        self.model = BayesianMLPReparam(
            self.in_features,
            self.hidden_sizes,
            self.out_features,
        )
        self.model.to(self.device)

        if self.type =='classification':
            self.log_loss = lambda y, y_hat: -F.cross_entropy(y_hat, y, reduction='sum')
            self.reg_loss = lambda y, y_hat: torch.eq(y, y_hat).sum()

        self.set_parameters(parameters)

        return

    def fit(self, dataset, t_i, parameters=None, hyperparameters=None):
        self.set_parameters(parameters)
        self.set_hyperparameters(hyperparameters)

        if parameters is None:
            parameters = self.get_parameters()

        dataloader = DataLoader(
            dataset,
            self.batch_size,
            num_workers=0,
            shuffle=True
        )

        cavity = utils.list_sub(parameters, t_i)
        self.set_prior(cavity)

        optimiser = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        N_left = self.N_sync

        # while N_left > 0:
        for data, target in dataloader:
            data, target = data.to(self.device), target.to(self.device)
            num_batch, _ = data.shape
            optimiser.zero_grad()

            output = self.model(data, num_samples=self.train_samples)

            target = target.unsqueeze(1).repeat_interleave(self.train_samples, dim=1)

            ll = self.log_loss(
                target.view(-1), 
                output.view(-1, target.shape[-1])
            )

            kl = self.model.kl() 
            loss = -ll / (num_batch * self.train_samples) + (kl / len(dataset))

            loss.backward()
            optimiser.step()

            # N_left -= 1
            # if N_left <= 0:
            #     break

        return self.get_parameters()

    def evaluate(self, dataset, parameters=None, hyperparameters=None):
        self.set_parameters(parameters)
        self.set_hyperparameters(hyperparameters)

        dataloader = DataLoader(
            dataset,
            self.batch_size,
            num_workers=0,
            shuffle=False
        )

        test_log_loss = 0
        test_reg_loss = 0

        with torch.no_grad():
            for data, target in dataloader:
                data, target = data.to(self.device), target.to(self.device)
                num_batch, _ = data.shape

                output = self.model(data, num_samples=self.test_samples)
                target_rep = target.unsqueeze(1).repeat_interleave(self.test_samples, dim=1)

                ll = self.log_loss(
                    target_rep.view(-1), 
                    output.view(-1, target_rep.shape[-1])
                ).sum()

                if self.type == 'classification':
                    pred = output.mean(dim=1).argmax(dim=1)
                    reg_loss = self.reg_loss(target, pred).sum()
                
                test_log_loss += ll
                test_reg_loss += reg_loss

        test_log_loss = test_log_loss / (self.test_samples * len(dataset))
        test_reg_loss = test_reg_loss / float(len(dataset))

        return float(test_log_loss.cpu()), float(test_reg_loss.cpu())

    def predict(self, x, parameters=None, hyperparameters=None):
        return None

    def set_parameters(self, parameters):
        parameters = deepcopy(parameters)
        if parameters is not None:
            if hasattr(self, 'model'):
                layers = [
                    module 
                    for 
                    module 
                    in
                    self.model.modules()
                    if isinstance(module, BayesianLinearReparam)
                ]

                for layer in layers:
                    weight_param = parameters.pop(0).to_moment_params()
                    layer.weights_mean.data = weight_param.mean
                    layer.weights_logvar.data = torch.log(weight_param.variance)
                    if layer.bias:
                        bias_param = parameters.pop(0).to_moment_params()
                        layer.bias_mean.data = bias_param.mean
                        layer.bias_logvar.data = torch.log(bias_param.variance)

    def set_prior(self, priors):
        if priors is not None:
            if hasattr(self, 'model'):
                layers = [
                    module 
                    for 
                    module 
                    in
                    self.model.modules()
                    if isinstance(module, BayesianLinearReparam)
                ]

                for layer in layers:
                    weights_prior = priors.pop(0).to_moment_params()
                    layer.weights_prior_mean.data = weights_prior.mean
                    layer.weights_prior_std.data = torch.sqrt(weights_prior.variance)
                    if layer.bias:
                        bias_prior = priors.pop(0).to_moment_params()
                        layer.bias_prior_mean.data = bias_prior.mean
                        layer.bias_prior_std.data = torch.sqrt(bias_prior.variance)


    def get_parameters(self):
        '''
        In case you need to repackage parameters somehow from a form other than the dictionary.
        :return: a dictionary of the parameters
        '''
        if hasattr(self, 'model'):
            layers = [
                module 
                for 
                module 
                in
                self.model.modules()
                if isinstance(module, BayesianLinearReparam)
            ]

            parameters = []

            for layer in layers:
                parameters.append(
                    dg.DiagGaussianMomentParams(
                        mean=layer.weights_mean.data,
                        variance=torch.exp(layer.weights_logvar.data)
                    ).to_nat_params()
                )
                if layer.bias:
                    parameters.append(
                        dg.DiagGaussianMomentParams(
                            mean=layer.bias_mean.data,
                            variance=torch.exp(layer.bias_logvar.data)
                        ).to_nat_params()
                    )

            return parameters
        else:
            return None

    @classmethod
    def get_default_parameters(cls):
        '''
        :return: A default set of parameters for the model. These might be all zero. Mostly used to get the shape that
        the parameters should be to make parameter server code more general.
        '''
        return [
            dg.DiagGaussianMomentParams(0., 1.).to_nat_params()
        ]

    def set_hyperparameters(self, hyperparameters):
        super().set_hyperparameters(hyperparameters)

        self.in_features = self.hyperparameters['in_features']
        self.hidden_sizes = self.hyperparameters['hidden_sizes']
        self.out_features = self.hyperparameters['out_features']
        self.train_samples = self.hyperparameters['train_samples']
        self.test_samples = self.hyperparameters['test_samples']
        self.batch_size = self.hyperparameters['batch_size']
        self.type = self.hyperparameters['type']
        self.lr = self.hyperparameters['lr']
        self.N_sync = self.hyperparameters['N_sync']
        self.device = self.hyperparameters['device']

    @classmethod
    def get_default_hyperparameters(cls):
        '''
        :return: A default set of hyperparameters( for the model. These might be all zero. Mostly used to get the shape that
        the hyperparameters should be to make parameter server code more general and easier to remeber what could go in here.
        '''
        return {
            'in_features': 1,
            'hidden_sizes': [],
            'out_features': 1,
            'train_samples': 10,
            'test_samples': 10,
            'batch_size': 64,
            'type': 'classification',
            'lr': 1e-3,
            'N_sync': 100,
            'device': 'cpu'
        }

    def save_model(self, path):
        '''
        Save the model to a specified directory
        :param path: The path to save the model to
        '''
        os.makedirs(path, exist_ok=True)

        hyperparameters = self.hyperparameters
        parameters = utils.list_to_numpy(self.get_parameters())

        with open(os.path.join(path, self.hyperparameters_file), 'wb') as f:
            pickle.dump(hyperparameters, f)

        with open(os.path.join(path, self.parameters_file), 'wb') as f:
            pickle.dump(parameters, f)


    @classmethod
    def load_model(cls, path, device=None):
        '''
        Load a model from the specified path and return an instance of it.
        '''
        with open(os.path.join(path, cls.hyperparameters_file), 'rb') as f:
            hyperparameters = pickle.load(f)

        model = cls(parameters=None, hyperparameters=hyperparameters)

        with open(os.path.join(path, cls.parameters_file), 'rb') as f:
            parameters = pickle.load(f)

        # If no specified device, send to the device that the model was on before
        if device is None:
            device = hyperparameters['device']

        model.model.to(device)
        parameters = utils.list_to_torch(parameters, device)
        model.set_parameters(parameters)

        return model