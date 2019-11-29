import torch

from src.methods.base.client import Client
import src.methods.exp_family.utils as utils

class PVIClient(Client):
    def __init__(self, 
                model_class, 
                data, 
                t_i,
                model_parameters=None, 
                model_hyperparameters=None,
                hyperparameters=None,
                metadata=None
                ):

        super().__init__(model_class, data, model_parameters, model_hyperparameters, hyperparameters, metadata)

        self.t_i = t_i
        # for key in model_parameters.keys():
        #     self.t_i[key] = self.t_i_init_func(model_parameters[key])

        self.model.set_parameters(model_parameters)

    # @classmethod
    # def create_factory(cls, model_class, data, t_i, model_parameters=None, model_hyperparameters=None, hyperparameters=None,
    #                    metadata=None):

    #     return lambda: cls(model_class, data, t_i, model_parameters, model_hyperparameters, hyperparameters, metadata)

    def compute_update(self, model_parameters=None, model_hyperparameters=None, update_ti=True):
        # Set the models parameters to be the current global ones
        super().compute_update(model_parameters, model_hyperparameters)

        t_i_old = self.t_i
        lambda_old = self.model.get_parameters()

        # find the new optimal parameters for this clients data, with the 
        # models current local contirbution removed.
        lambda_new = self.model.fit(
            self.data,
            t_i_old
        )

        delta_lambda_i = utils.list_sub(lambda_new, lambda_old)

        delta_lambda_i = utils.list_const_mul(1. - self.damping_factor, delta_lambda_i)

        # compute the new
        lambda_new = utils.list_add(lambda_old, delta_lambda_i)

        t_i_new = utils.list_add(t_i_old, delta_lambda_i)
        self.t_i = t_i_new

        return delta_lambda_i

    def set_hyperparameters(self, hyperparameters):
        super().set_hyperparameters(hyperparameters)

        self.damping_factor = self.hyperparameters['damping_factor']

    def set_metadata(self, metadata):
        super().set_metadata(metadata)

    @classmethod
    def get_default_hyperparameters(cls):
        default_hyperparameters = {
            **super().get_default_hyperparameters(),
            **{
                "damping_factor": 0.,
            }
        }
        return default_hyperparameters

    @classmethod
    def get_default_metadata(cls):
        return {
            **super().get_default_metadata(),
            **{
                'global_iteration': 0,
            }
        }