import torch
import torch.nn as nn

from src.models.bnn.layers import BayesianLinearReparam


class BayesianMLPReparam(nn.Module):
    """Implements a Bayesian MLP netowrk with the local 
    reparametisation trick applied to the forward pass.
    """

    def __init__(self, input_dim, hidden_sizes, output_dim, activation=nn.ReLU(), init_mean_variance=0.1, init_logvar=-11.):
        super(BayesianMLPReparam, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        hidden_sizes = [input_dim] + hidden_sizes

        layers = []

        for i in range(len(hidden_sizes) - 1):
            layers.append(BayesianLinearReparam(hidden_sizes[i], hidden_sizes[i+1], init_mean_variance=init_mean_variance, init_logvar=init_logvar))
            layers.append(activation)

        layers.append(BayesianLinearReparam(hidden_sizes[-1], output_dim, init_mean_variance=init_mean_variance, init_logvar=init_logvar))

        self.model = nn.Sequential(*layers)

    def forward(self, input, num_samples=10):
        """Compute the forward pass for the model. Takes multiple samples
        
        Arguments:
            input {torch.Tensor} -- Shape(num_batches, )
        
        Keyword Arguments:
            num_samples {int} -- number of samples to take from the model (default: {10})

        Returns:
            torch.Tensor: Shape(num_batches, num_samples, out_dim)
        """

        input = input.unsqueeze(1).repeat_interleave(num_samples, dim=1)

        output = self.model(input)

        return output


    def kl(self):
        """Computes the KL of all the weights in the model.
        
        Arguments:
            prior_mean {float} -- the mean of the prior distribution
            prior_std {float} -- the standard deviation of the prior distribution 
        """

        kl = 0
        # print(list(self.modules()))
        for module in self.model.modules():
            if hasattr(module, 'kl'):
                kl = kl + module.kl().sum()

        return kl
    