import os
import sys
sys.path.append(os.getcwd())

import torch
import numpy as np

import src.methods.exp_family.exp_family as exp_family


class DiagGaussianMomentParams(exp_family.MomentsParams):

    def __init__(self, mean, variance):
        self.mean = mean
        self.variance = variance
    
    def to_mean_params(self):
        return DiagGaussianMeanParams.from_moment_params(self)

    def to_nat_params(self):
        return DiagGaussianNatParams.from_moment_params(self)

    def to_moment_params(self):
        return self

    @classmethod
    def from_mean_params(cls, mean_params):
        if isinstance(mean_params, DiagGaussianMeanParams):
            return DiagGaussianMomentParams(
                mean = mean_params.mu_1,
                variance = 2 * mean_params.mu_2 - mean_params.mu_1**2
            )
        else:
            raise ValueError(f'Cannot construct {cls.__name__} from {type(mean_params).__name__}.')

    @classmethod
    def from_nat_params(cls, nat_params):
        if isinstance(nat_params, DiagGaussianNatParams):
            return DiagGaussianMomentParams(
                mean = - nat_params.nu_1 / nat_params.nu_2,
                variance = - 1 / nat_params.nu_2
            )
        else:
            raise ValueError(f'Cannot construct {cls.__name__} from {type(nat_params).__name__}.')

    @classmethod
    def from_moment_params(cls, moment_params):
        return moment_params

    def to_numpy(self):
        # weak check that its not already numpy
        if isinstance(self.mean, np.ndarray):
            return self
        # otherwise assume its a torch tensor
        else:
            return DiagGaussianMomentParams(
                mean = self.mean.detach().cpu().numpy(),
                variance = self.variance.detach().cpu().numpy()
            )

    def to_torch(self, device='cpu'):
        # weak check that its not already torch
        if isinstance(self.mean, torch.Tensor):
            return self
        # otherwise assume its a torch tensor
        else:
            return DiagGaussianMomentParams(
                mean = torch.tensor(self.mean).to(device),
                variance = torch.tensor(self.variance).to(device)
            )

    def __repr__(self):
        return f'{type(self).__name__} with \n \t mean {self.mean} \n \t variance {self.variance}'



class DiagGaussianMeanParams(exp_family.MeanParams):

    def __init__(self, mu_1, mu_2):
        self.mu_1 = mu_1
        self.mu_2 = mu_2

    def to_nat_params(self):
        return DiagGaussianNatParams.from_mean_params(self)

    def to_moment_params(self):
        return DiagGaussianMomentParams.from_mean_params(self)

    def to_mean_params(self):
        return self

    @classmethod
    def from_nat_params(cls, nat_params):
        if isinstance(nat_params, DiagGaussianNatParams):
            return DiagGaussianMeanParams(
                mu_1 = - nat_params.nu_1 / nat_params.nu_2,
                mu_2 = (1 / 2) * ( (nat_params.nu_1**2 / nat_params.nu_2**2) - (1 / nat_params.nu_2) )
            )
        else:
            raise ValueError(f'Cannot construct {cls.__name__} from {type(nat_params).__name__}.')

    @classmethod
    def from_moment_params(cls, moment_params):
        if isinstance(moment_params, DiagGaussianMomentParams):
            return DiagGaussianMeanParams(
                mu_1 = moment_params.mean,
                mu_2 = (1 / 2) * (moment_params.mean ** 2 + moment_params.variance)
            )
        else:
            raise ValueError(f'Cannot construct {cls.__name__} from {type(moment_params).__name__}.')

    @classmethod
    def from_mean_params(cls, mean_params):
        return mean_params

    def to_numpy(self):
        # weak check that its not already numpy
        if isinstance(self.mu_1, np.ndarray):
            return self
        # otherwise assume its a torch tensor
        else:
            return DiagGaussianMeanParams(
                mu_1 = self.mu_1.detach().cpu().numpy(),
                mu_2 = self.mu_2.detach().cpu().numpy()
            )

    def to_torch(self, device='cpu'):
        # weak check that its not already torch
        if isinstance(self.mu_1, torch.Tensor):
            return self
        # otherwise assume its a torch tensor
        else:
            return DiagGaussianMeanParams(
                mu_1 = torch.tensor(self.mu_1).to(device),
                mu_2 = torch.tensor(self.mu_2).to(device)
            )

    def __repr__(self):
        return f'{type(self).__name__} with \n \t mu_1 {self.mu_1} \n \t mu_2 {self.mu_2}\n'


class DiagGaussianNatParams(exp_family.NatParams):
    def __init__(self, nu_1, nu_2):
        self.nu_1 = nu_1
        self.nu_2 = nu_2

    def to_mean_params(self):
        return DiagGaussianMeanParams.from_nat_params(self)

    def to_moment_params(self):
        return DiagGaussianMomentParams.from_nat_params(self)

    def to_nat_params(self):
        return self

    @classmethod
    def from_mean_params(cls, mean_params):
        if isinstance(mean_params, DiagGaussianMeanParams):
            return DiagGaussianNatParams(
                nu_1 =  mean_params.mu_1 / (2 * mean_params.mu_2 - mean_params.mu_1**2) ,
                nu_2 = - 1 / (2 * mean_params.mu_2 - mean_params.mu_1**2)
            )
        else:
            raise ValueError(f'Cannot construct {cls.__name__} from {type(mean_params).__name__}.')

    @classmethod
    def from_moment_params(cls, moment_params):
        if isinstance(moment_params, DiagGaussianMomentParams):
            return DiagGaussianNatParams(
                nu_1 =  moment_params.mean / moment_params.variance,
                nu_2 = - 1 / moment_params.variance
            )
        else:
            raise ValueError(f'Cannot construct {cls.__name__} from {type(moment_params).__name__}.')

    @classmethod
    def from_nat_params(cls, nat_params):
        return nat_params

    def to_numpy(self):
        # weak check that its not already numpy
        if isinstance(self.nu_1, np.ndarray):
            return self
        # otherwise assume its a torch tensor
        else:
            return DiagGaussianNatParams(
                nu_1 = self.nu_1.detach().cpu().numpy(),
                nu_2 = self.nu_2.detach().cpu().numpy()
            )

    def to_torch(self, device='cpu'):
        # weak check that its not already torch
        if isinstance(self.nu_1, torch.Tensor):
            return self
        # otherwise assume its a torch tensor
        else:
            return DiagGaussianNatParams(
                nu_1 = torch.tensor(self.nu_1).to(device),
                nu_2 = torch.tensor(self.nu_2).to(device)
            )

    def enforce_precision_negativity(self, delta=-1e-5):
        errors = self.nu_2[self.nu_2 >= delta]
        if errors.shape[0] > 0:
            print(f'woops, having to clip precisions... of {errors.shape} values, mean {errors.mean()}')
            self.nu_2[self.nu_2 >= delta] = delta
        return self

    def __add__(self, other):
        return DiagGaussianNatParams(self.nu_1 + other.nu_1, self.nu_2 + other.nu_2)

    def __radd__(self, other):
        if other == 0:
            return self
        else:
            return self.__add__(other)

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return other + (-self)

    def __mul__(self, other):
        return DiagGaussianNatParams(self.nu_1 * other, self.nu_2 * other)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        return DiagGaussianNatParams(self.nu_1 / other, self.nu_2 / other)

    def __pos__(self):
        return self

    def __neg__(self):
        return DiagGaussianNatParams(-self.nu_1, -self.nu_2)

    def __repr__(self):
        return f'{type(self).__name__} with \n \t nu_1 {self.nu_1} \n \t nu_2 {self.nu_2}'


# class DiagGaussianParameterList(object):

#     def __init__(self, initial_list):
#         self.list = initial_list

#     def list_uniform_prior(self, prior_mean, prior_var):
#         self.list = [
#             parameter.to_moment_params()
#             for parameter
#             in self.list
#         ]

#         self.list =  [
#             DiagGaussianMomentParams(
#                 mean = torch.ones_like(parameter.mean) * prior_mean,
#                 variance = torch.ones_like(parameter.variance) * prior_var
#             ).to_nat_params()
#             for parameter
#             in parameters
#         ]

#         return self

#     def list_uninformative_prior(parameters):
#         parameters = [
#             parameter.to_moment_params()
#             for parameter
#             in parameters
#         ]

#         return [
#             DiagGaussianNatParams(
#                 nu_1 = torch.zeros_like(parameter.mean),
#                 nu_2 = torch.zeros_like(parameter.variance)
#             ).to_nat_params()
#             for parameter
#             in parameters
#         ]

def list_uniform_prior(parameters, prior_mean, prior_var):
    parameters = [
        parameter.to_moment_params()
        for parameter
        in parameters
    ]

    return [
        DiagGaussianMomentParams(
            mean = torch.ones_like(parameter.mean) * prior_mean,
            variance = torch.ones_like(parameter.variance) * prior_var
        ).to_nat_params()
        for parameter
        in parameters
    ]

def list_uninformative_prior(parameters):
    parameters = [
        parameter.to_moment_params()
        for parameter
        in parameters
    ]

    return [
        DiagGaussianNatParams(
            nu_1 = torch.zeros_like(parameter.mean),
            nu_2 = torch.zeros_like(parameter.variance)
        ).to_nat_params()
        for parameter
        in parameters
    ]


if __name__ =='__main__':   
    import torch
    a = DiagGaussianMomentParams(torch.tensor(0.4643), torch.tensor(1.322))
    print(a)
    print(a.to_mean_params())
    print(a.to_nat_params())
    print('///')
    print(a.to_nat_params())
    print(a.to_mean_params().to_nat_params())
    print('///')
    print(a.to_mean_params())
    print(a.to_nat_params().to_mean_params())
    print('///')
    print(a.to_mean_params().to_nat_params().to_moment_params())
    print(a.to_nat_params().to_mean_params().to_moment_params())
    print('///')

    