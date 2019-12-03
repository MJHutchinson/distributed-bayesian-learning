import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from torch.distributions import Normal, kl_divergence

class BayesianLinearReparam(nn.Module):
    """ Impements a Linear layer for Bayesian Neural Networks using a 
    mean field Gaussian Vairiational approximation for the weights.
    Uses the loacl reparametisaiton trick.
    """
    def __init__(
                    self, 
                    in_features, 
                    out_features,
                    prior_mean=0.,
                    prior_var=1.,
                    init_mean_variance=0.1, 
                    init_logvar=-11., 
                    bias=True
                ):
        super(BayesianLinearReparam, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias

        self.init_mean_var = init_mean_variance
        self.init_logvar = init_logvar

        self.weights_mean = nn.Parameter(torch.Tensor(in_features, out_features))
        self.weights_logvar = nn.Parameter(torch.Tensor(in_features, out_features))

        self.weights_prior_mean = torch.tensor(prior_mean)
        self.weights_prior_std = torch.sqrt((torch.tensor(prior_var)))

        if self.bias:
            self.bias_mean = nn.Parameter(torch.Tensor(out_features))
            self.bias_logvar = nn.Parameter(torch.Tensor(out_features))

            self.bias_prior_mean = torch.tensor(prior_mean)
            self.bias_prior_std = torch.sqrt(torch.tensor(prior_var))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_normal_(self.weights_mean)
        nn.init.ones_(self.weights_logvar)
        self.weights_logvar.data = self.weights_logvar.data * self.init_logvar

        if self.bias:
            nn.init.zeros_(self.bias_mean)
            nn.init.ones_(self.bias_logvar)
            self.bias_logvar.data = self.bias_logvar.data * self.init_logvar

    def forward(self, input):
        """ Computes the forward pass over a batch of inputs
        
        Arguments:
            input {torch.Tensor} -- Shape(num_batches, num_samples, in_features)
        """
        num_batches, num_samples, _ = input.shape

        # Compute the reparametisation trick for the weights forward pass
        # Compute the mean and var of the output of the weights

        m_W_pre = torch.einsum('bsi,io->bso', input, self.weights_mean)
        v_W_pre = torch.einsum('bsi,io->bso', input ** 2, torch.exp(self.weights_logvar))
        eps_W_pre = torch.rand_like(m_W_pre)

        # w = mu + sigma*z
        pre = m_W_pre + torch.sqrt(1e-9 + v_W_pre) * eps_W_pre

        # If we have a bia, add it on. We have no reparametisation trick for the 
        # biases
        if self.bias:
            eps_b_pre = torch.randn_like(self.bias_mean)
            pre = pre + self.bias_mean + eps_b_pre * torch.exp(0.5 * self.bias_logvar)

        return pre

    def kl(self):
        """Computes the KL diveregence between the current weights and a
        specified prior distribution.
        
        Arguments:
            prior_mean {float} -- the mean of the prior distribution
            prior_std {float} -- the standard deviation of the prior distribution 
        """
        weigths_prior_dist = Normal(self.weights_prior_mean, self.weights_prior_std)
        weights_dist = Normal(self.weights_mean, torch.exp(0.5 * self.weights_logvar))

        kl = kl_divergence(weights_dist, weigths_prior_dist)

        if self.bias:
            bias_prior_dist = Normal(self.bias_prior_mean, self.bias_prior_std)
            bias_dist = Normal(self.bias_mean, torch.exp(0.5 *  self.bias_logvar))
            kl = torch.cat((kl, kl_divergence(bias_dist, bias_prior_dist).unsqueeze(0)))

        return kl
