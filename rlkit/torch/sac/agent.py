import numpy as np

import torch
import copy
from torch import nn as nn
import torch.nn.functional as F

import rlkit.torch.pytorch_util as ptu
import pdb


def _product_of_gaussians(mus, sigmas_squared):
    '''
    compute mu, sigma of product of gaussians
    '''
    sigmas_squared = torch.clamp(sigmas_squared, min=1e-7)
    sigma_squared = 1. / torch.sum(torch.reciprocal(sigmas_squared), dim=0)
    mu = sigma_squared * torch.sum(mus / sigmas_squared, dim=0)
    return mu, sigma_squared


def _mean_of_gaussians(mus, sigmas_squared):
    '''
    compute mu, sigma of mean of gaussians
    '''
    mu = torch.mean(mus, dim=0)
    sigma_squared = torch.mean(sigmas_squared, dim=0)
    return mu, sigma_squared


def _natural_to_canonical(n1, n2):
    ''' convert from natural to canonical gaussian parameters '''
    mu = -0.5 * n1 / n2
    sigma_squared = -0.5 * 1 / n2
    return mu, sigma_squared


def _canonical_to_natural(mu, sigma_squared):
    ''' convert from canonical to natural gaussian parameters '''
    n1 = mu / sigma_squared
    n2 = -0.5 * 1 / sigma_squared
    return n1, n2


class PEARLAgent(nn.Module):

    def __init__(self,
                 latent_dim,
                 context_encoder,
                 uncertainty_mlp,
                 policy,
                 **kwargs
    ):
        super().__init__()
        self.latent_dim = latent_dim

        self.context_encoder = context_encoder
        self.uncertainty_mlp = uncertainty_mlp
        self.policy = policy

        self.recurrent = kwargs['recurrent']
        self.sparse_rewards = kwargs['sparse_rewards']
        self.use_next_obs_in_context = kwargs['use_next_obs_in_context']
        self.z_strategy = kwargs['z_strategy'] # ['mean', 'min', 'weighted', 'quantile']

        if kwargs['separate_train'] and kwargs['pretrain']:
            self.z_strategy = 'mean'

        # initialize buffers for z dist and z
        # use buffers so latent context can be saved along with model weights
        self.register_buffer('z', torch.zeros(1, latent_dim))
        self.register_buffer('z_means', torch.zeros(1, latent_dim))
        self.register_buffer('z_mins', torch.zeros(1, latent_dim))
        self.register_buffer('z_vars', torch.zeros(1, latent_dim))

        self.clear_z()

    def clear_z(self, num_tasks=1):
        '''
        reset q(z|c) to the prior
        sample a new z from the prior
        '''
        # reset distribution over z to the prior
        mu = ptu.zeros(num_tasks, self.latent_dim)
        var = ptu.zeros(num_tasks, self.latent_dim)
        self.z_mins = mu
        self.z_means = mu
        self.z_weighted = mu
        self.z_quantile = mu
        self.z_vars = var
        # sample a new z from the prior
        self.sample_z()
        # reset the context collected so far
        self.context = None
        # reset any hidden state in the encoder network (relevant for RNN)
        #self.context_encoder.reset(num_tasks)
    
    def clear_context(self):
        self.context = None

    def detach_z(self):
        ''' disable backprop through z '''
        self.z = self.z.detach()
        if self.recurrent:
            self.context_encoder.hidden = self.context_encoder.hidden.detach()

    def set_z(self, means, vars):
        self.z_means = means
        self.z_vars = vars
        self.sample_z()

    def update_context(self, inputs):
        ''' append single transition to the current context '''
        o, a, r, no, d, info = inputs
        if self.sparse_rewards:
            r = info['sparse_reward']
        o = ptu.from_numpy(o[None, None, ...])
        a = ptu.from_numpy(a[None, None, ...])
        if len(r.shape) == 0:
            r = ptu.from_numpy(np.array([r])[None, None, ...])
        else:
            r = ptu.from_numpy(r[None, None, ...])
        no = ptu.from_numpy(no[None, None, ...])
        if self.use_next_obs_in_context:
            data = torch.cat([o, a, r, no], dim=2)
        else:
            data = torch.cat([o, a, r], dim=2)
        if self.context is None:
            self.context = data
        else:
            self.context = torch.cat([self.context, data], dim=1)
 
    def update_context_dict(self, batch_dict, env):
        ''' append context dictionary containing single/multiple transitions to the current context '''
        o = ptu.from_numpy(batch_dict['observations'][None, ...])
        a = ptu.from_numpy(batch_dict['actions'][None, ...])
        next_o = ptu.from_numpy(batch_dict['next_observations'][None, ...])
        if callable(getattr(env, "sparsify_rewards", None)) and self.sparse_rewards:
            r = batch_dict['rewards']
            sr = []
            for r_entry in r:
                sr.append(env.sparsify_rewards(r_entry))
            r = ptu.from_numpy(np.array(sr)[None, ...])
        else:
            r = ptu.from_numpy(batch_dict['rewards'][None, ...])
        if self.use_next_obs_in_context:
            data = torch.cat([o, a, r, next_o], dim=2)
        else:
            data = torch.cat([o, a, r], dim=2)
        if self.context is None:
            self.context = data
        else:
            self.context = torch.cat([self.context, data], dim=1)

    def compute_kl_div(self):
        ''' compute KL( q(z|c) || r(z) ) '''
        prior = torch.distributions.Normal(ptu.zeros(self.latent_dim), 0.05*ptu.ones(self.latent_dim))
        posteriors = [torch.distributions.Normal(mu, torch.sqrt(var)) for mu, var in zip(torch.unbind(self.z_means), torch.unbind(self.z_vars))]
        kl_divs = [torch.distributions.kl.kl_divergence(post, prior) for post in posteriors]
        kl_div_sum = torch.sum(torch.stack(kl_divs))
        return kl_div_sum

    def infer_posterior(self, context, task_indices=None):
        ''' compute q(z|c) as a function of input context and sample new z from it'''
        params = self.context_encoder(context)
        params = params.view(context.size(0), -1, self.context_encoder.output_size)
        heterodastic_var = F.softplus(self.uncertainty_mlp(params).detach())
        if task_indices is None:
            self.task_indices = np.zeros((context.size(0),))
        elif not hasattr(task_indices, '__iter__'):
            self.task_indices = np.array([task_indices])
        else:
            self.task_indices = np.array(task_indices)

        # z_mean
        self.z_means = torch.mean(params, dim=1)

        # z_hvar_min
        min_index = torch.argmin(heterodastic_var, dim=1).flatten()
        batch_indices = torch.arange(min_index.size(0))
        self.z_mins = params[batch_indices, min_index]

        # z_havar_weighted 
        # softmax(e^-x)
        # weights = F.softmax(torch.exp(-heterodastic_var), dim=1)
        weights = F.softmax(-heterodastic_var, dim=1)
        self.z_weighted = torch.sum(weights * params, dim=1) # [task, latent_dim]
        # 1 - softmax    ：weighted_1-softmax
        # weights = 1 - F.softmax(heterodastic_var, dim=1)
        # self.z_weighted = torch.sum(weights * params, dim=1) / torch.sum(weights, dim=1)  # [task, latent_dim]
        # softmax(-x)    : Softmax
        # weights = F.softmax(-heterodastic_var, dim=1)
        # self.z_weighted = torch.sum(weights * params, dim=1)  # [task, latent_dim]

        # z_quantile_mean 5-10
        sorted_indices = torch.argsort(heterodastic_var, dim=1)
        lower_bound = int(heterodastic_var.size(1) * 0.05)
        upper_bound = int(heterodastic_var.size(1) * 0.20) # 计算5%-20%范围的indices
        selected_indices = sorted_indices[:, lower_bound:upper_bound].squeeze(dim=-1)  # [task, selected_range][10, 563]
        batch_indices = torch.arange(heterodastic_var.size(0)).unsqueeze(-1).expand(-1, selected_indices.size(-1))  # [10, 563]
        selected_params = params[batch_indices, selected_indices]  # [task, selected_range, latent_dim] [10, 1024, 20]
        self.z_quantile = selected_params.mean(dim=1)  # [task, latent_dim]

        self.z_vars = torch.std(params, dim=1)
        self.sample_z()
        
        return heterodastic_var

    def encode_no_mean(self, context):
        params = self.context_encoder(context)
        params = params.view(context.size(0), -1, self.context_encoder.output_size)
        return params

    def sample_z(self):
        if self.z_strategy == 'mean':
            self.z = self.z_means
        elif self.z_strategy == 'min':
            self.z = self.z_mins
        elif self.z_strategy == 'weighted':
            self.z = self.z_weighted
        elif self.z_strategy == 'quantile':
            self.z = self.z_quantile

    def get_action(self, obs, deterministic=False):
        ''' sample action from the policy, conditioned on the task embedding '''
        z = self.z
        obs = ptu.from_numpy(obs[None])
        in_ = torch.cat([obs, z], dim=1)
        return self.policy.get_action(in_, deterministic=deterministic)

    def set_num_steps_total(self, n):
        self.policy.set_num_steps_total(n)

    def forward(self, obs, context, task_indices=None):
        ''' given context, get statistics under the current policy of a set of observations '''
        self.infer_posterior(context, task_indices=task_indices)
        self.sample_z()

        task_z = self.z

        # self.meta_batch * self.batch_size * dim(obs)
        t, b, _ = obs.size()
        obs = obs.view(t * b, -1)
        task_z = [z.repeat(b, 1) for z in task_z]
        task_z = torch.cat(task_z, dim=0)
        # run policy, get log probs and new actions
        in_ = torch.cat([obs, task_z.detach()], dim=1)
        policy_outputs = self.policy(t, b, in_, reparameterize=True, return_log_prob=True)

        task_z_vars = [z.repeat(b, 1) for z in self.z_vars]
        task_z_vars = torch.cat(task_z_vars, dim=0)
        return policy_outputs, task_z, task_z_vars

    def log_diagnostics(self, eval_statistics):
        for i in range(len(self.z_means[0])):
            z_mean = ptu.get_numpy(self.z_means[0][i])
            name = 'Z mean eval' + str(i)
            eval_statistics[name] = z_mean
        z_sig = np.mean(ptu.get_numpy(self.z_vars[0]))
        eval_statistics['Z variance eval'] = z_sig

        eval_statistics['task_idx'] = self.task_indices[0]

    @property
    def networks(self):
        return [self.context_encoder, self.uncertainty_mlp, self.policy]