import os
from turtle import position
import torch
import torch.optim as optim
import numpy as np
import rlkit.torch.pytorch_util as ptu
from torch import nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from itertools import product
from rlkit.core.eval_util import create_stats_ordered_dict
from rlkit.core.rl_algorithm import OfflineMetaRLAlgorithm
from rlkit.torch.brac import divergences
from rlkit.torch.brac import utils
import pdb
from line_profiler import LineProfiler
import atexit

profile = LineProfiler()
atexit.register(profile.print_stats)

# Contrastive Reconstruction Objective for Offline Meta Reinforcement Learning
class CLASSIFIERSoftActorCritic(OfflineMetaRLAlgorithm):
    def __init__(
            self,
            env,
            train_tasks,
            eval_tasks,
            latent_dim,
            nets,
            goal_radius=1,
            optimizer_class=optim.Adam,
            plotter=None,
            render_eval_paths=False,
            **kwargs
    ):
        super().__init__(
            env=env,
            agent=nets[0],
            train_tasks=train_tasks,
            eval_tasks=eval_tasks,
            goal_radius=goal_radius,
            **kwargs
        )

        self.latent_dim                     = latent_dim
        self.soft_target_tau                = kwargs['soft_target_tau']
        self.policy_mean_reg_weight         = kwargs['policy_mean_reg_weight']
        self.policy_std_reg_weight          = kwargs['policy_std_reg_weight']
        self.policy_pre_activation_weight   = kwargs['policy_pre_activation_weight']
        self.recurrent                      = kwargs['recurrent']
        self.kl_lambda                      = kwargs['kl_lambda']
        self._divergence_name               = kwargs['divergence_name']
        self.sparse_rewards                 = kwargs['sparse_rewards']
        self.use_next_obs_in_context        = kwargs['use_next_obs_in_context']
        self.use_brac                       = kwargs['use_brac']
        self.use_value_penalty              = kwargs['use_value_penalty']
        self.alpha_max                      = kwargs['alpha_max']
        self._c_iter                        = kwargs['c_iter']
        self.train_alpha                    = kwargs['train_alpha']
        self._target_divergence             = kwargs['target_divergence']
        self.alpha_init                     = kwargs['alpha_init']
        self.alpha_lr                       = kwargs['alpha_lr']
        self.policy_lr                      = kwargs['policy_lr']
        self.qf_lr                          = kwargs['qf_lr']
        self.vf_lr                          = kwargs['vf_lr']
        self.c_lr                           = kwargs['c_lr']
        self.context_lr                     = kwargs['context_lr']
        self.z_loss_weight                  = kwargs['z_loss_weight']
        self.max_entropy                    = kwargs['max_entropy']
        self.allow_backward_z               = kwargs['allow_backward_z']
        self.use_relabel                    = kwargs['use_relabel']
        self.use_pretrained                 = kwargs['use_pretrained']
        self.train_z0_policy                = kwargs['train_z0_policy']
        self.use_hvar                       = kwargs['use_hvar']
        self.hvar_punish_w                  = kwargs['hvar_punish_w']
        self.loss                           = {}
        self.plotter                        = plotter
        self.render_eval_paths              = render_eval_paths
        self.qf_criterion                   = nn.MSELoss()
        self.vf_criterion                   = nn.MSELoss()
        self.vib_criterion                  = nn.MSELoss()
        self.l2_reg_criterion               = nn.MSELoss()
        self.club_criterion                 = nn.MSELoss()
        self.cross_entropy_loss             = nn.CrossEntropyLoss()

        self.qf1, self.qf2, self.vf, self.c, self.classifier = nets[1:]
        self.target_vf                      = self.vf.copy()

        self.policy_optimizer               = optimizer_class(self.agent.policy.parameters(), lr=self.policy_lr)
        self.qf1_optimizer                  = optimizer_class(self.qf1.parameters(), lr=self.qf_lr)
        self.qf2_optimizer                  = optimizer_class(self.qf2.parameters(), lr=self.qf_lr)
        self.vf_optimizer                   = optimizer_class(self.vf.parameters(),  lr=self.vf_lr)
        self.c_optimizer                    = optimizer_class(self.c.parameters(),   lr=self.c_lr)
        self.context_encoder_optimizer      = optimizer_class(self.agent.context_encoder.parameters(), lr=self.context_lr)
        self.uncertainty_mlp_optimizer      = optimizer_class(self.agent.uncertainty_mlp.parameters(), lr=self.context_lr)
        self.classifier_optimizer           = optimizer_class(self.classifier.parameters(), lr=self.context_lr)

        self._num_steps                     = 0
        self._visit_num_steps_train         = 10
        self._alpha_var                     = torch.tensor(1.)
        self.heterodatastic_var_thresh      = None

    ###### Torch stuff #####
    @property
    def networks(self):
        return self.agent.networks + [self.qf1, self.qf2, self.vf, self.target_vf, self.c, self.classifier]

    @property
    def get_alpha(self):
        return utils.clip_v2(
            self._alpha_var, 0.0, self.alpha_max)

    def training_mode(self, mode):
        for net in self.networks:
            net.train(mode)

    def to(self, device=None):
        if device == None:
            device = ptu.device
        for net in self.networks:
            net.to(device)
        if self.train_alpha:
            self._alpha_var = torch.tensor(self.alpha_init, device=ptu.device, requires_grad=True)
        self._divergence = divergences.get_divergence(name=self._divergence_name, c=self.c, device=ptu.device)

    def print_networks(self, net):
        print('---------- Networks initialized -------------')
        num_params = 0
        for param in net.parameters():
            num_params += param.numel()
        #print(net)
        print('[Network] Total number of parameters : %.3f M' % (num_params / 1e6))
        print('-----------------------------------------------')

    ##### Data handling #####
    def unpack_batch(self, batch, sparse_reward=False):
        ''' unpack a batch and return individual elements '''
        o = batch['observations'][None, ...]
        a = batch['actions'][None, ...]
        if sparse_reward:
            r = batch['sparse_rewards'][None, ...]
        else:
            r = batch['rewards'][None, ...]
        no = batch['next_observations'][None, ...]
        t = batch['terminals'][None, ...]
        return [o, a, r, no, t]

    def sample_sac(self, indices):
        ''' sample batch of training data from a list of tasks for training the actor-critic '''
        # this batch consists of transitions sampled randomly from replay buffer
        # rewards are always dense
        batches = [ptu.np_to_pytorch_batch(self.replay_buffer.random_batch(idx, batch_size=self.batch_size)) for idx in indices]
        unpacked = [self.unpack_batch(batch) for batch in batches]
        # group like elements together
        unpacked = [[x[i] for x in unpacked] for i in range(len(unpacked[0]))]
        unpacked = [torch.cat(x, dim=0) for x in unpacked]
        return unpacked

    def sample_context(self, indices):
        ''' sample batch of context from a list of tasks from the replay buffer '''
        # make method work given a single task index
        if not hasattr(indices, '__iter__'):
            indices = [indices]
        batches = [ptu.np_to_pytorch_batch(self.replay_buffer.random_batch(idx, batch_size=self.embedding_batch_size, sequence=self.recurrent)) for idx in indices]
        context = [self.unpack_batch(batch, sparse_reward=self.sparse_rewards) for batch in batches]
        # group like elements together
        context = [[x[i] for x in context] for i in range(len(context[0]))]
        context = [torch.cat(x, dim=0) for x in context] # 5 * self.meta_batch * self.embedding_batch_size * dim(o, a, r, no, t)
        # full context consists of [obs, act, rewards, next_obs, terms]
        # if dynamics don't change across tasks, don't include next_obs
        # don't include terminals in context
        if self.use_next_obs_in_context:
            context = torch.cat(context[:-1], dim=2)
        else:
            context = torch.cat(context[:-2], dim=2)

        return context

    def get_relabel_output(self, s_a, task_indices):
        with torch.no_grad():
            obs = s_a[..., :self.obs_dim]
            actions = s_a[..., self.obs_dim:]
            relabel_output, relabel_std = self.task_dynamics.step(obs, actions, task_indices, return_std=True)
        return relabel_output, relabel_std

    def make_relabel(self, indices, context):
        mb, b, f = context.shape
        relabel_s_a = []
        for i in range(len(indices)):
            task_relabel_s_a = torch.cat([context[:i, :, :self.obs_dim + self.action_dim], context[i+1:, :, :self.obs_dim+self.action_dim]], dim=0).reshape(-1, self.obs_dim + self.action_dim)
            relabel_s_a.append(task_relabel_s_a)
        relabel_s_a = torch.cat(relabel_s_a, dim=0)
        relabel_output, relabel_std = self.get_relabel_output(relabel_s_a, task_indices=indices)
        relabel_context = torch.cat([relabel_s_a, relabel_output], dim=-1).reshape(mb, -1, f)
        sorted_ind = torch.argsort(relabel_std, dim=-1)
        sorted_relabel_context = torch.cat([relabel_context[i, sorted_ind[i], :] for i in range(mb)]).reshape(mb, -1, f)
        return sorted_relabel_context

    ##### Training #####
    # @ptu.profile
    def _do_training(self, indices):
        mb_size = self.embedding_mini_batch_size # NOTE: not meta batch!

        # sample context batch
        context_batch = self.sample_context(indices)
        if self.use_relabel:
            # relabel
            relabel_context = self.make_relabel(indices, context_batch)
            relabel_ration = 0.5
            relabel_size = int(self.embedding_batch_size*relabel_ration)
            original_size = self.embedding_batch_size - relabel_size
            # context_batch = torch.cat([context_batch[:, :original_size, :], relabel_context[:, :relabel_size, :]], dim=1)
            context_batch = torch.cat([context_batch, relabel_context], dim=1)

        # zero out context and hidden encoder state
        self.agent.clear_z(num_tasks=len(indices))

        z_means_lst = []
        z_vars_lst = []
        # do this in a loop so we can truncate backprop in the recurrent encoder
        num_updates = context_batch.shape[1] // mb_size
        for i in range(num_updates):
            context = context_batch[:, i * mb_size: i * mb_size + mb_size, :]
            self.loss['step'] = self._num_steps
            z_means, z_vars = self._take_step(indices, context)
            self._num_steps += 1
            z_means_lst.append(z_means[None, ...])
            z_vars_lst.append(z_vars[None, ...])
            # stop backprop
            self.agent.detach_z()
        z_means = np.mean(np.concatenate(z_means_lst), axis=0)
        z_vars = np.mean(np.concatenate(z_vars_lst), axis=0)
        return z_means, z_vars

    def _min_q(self, obs, actions, task_z):
        q1 = self.qf1(obs, actions, task_z.detach())
        q2 = self.qf2(obs, actions, task_z.detach())
        min_q = torch.min(q1, q2)
        return min_q

    def _update_target_network(self):
        ptu.soft_update_from_to(self.vf, self.target_vf, self.soft_target_tau)

    def _optimize_c(self, indices, context):
        # data is (task, batch, feat)
        obs, actions, rewards, next_obs, terms = self.sample_sac(indices)

        # run inference in networks
        policy_outputs, task_z, task_z_vars = self.agent(obs, context, task_indices=indices)
        new_actions, policy_mean, policy_log_std, log_pi = policy_outputs[:4]

        # flattens out the task dimension
        t, b, _ = obs.size()
        obs = obs.view(t * b, -1)
        actions = actions.view(t * b, -1)
        next_obs = next_obs.view(t * b, -1)

        # optimize for c network (which computes dual-form divergences)
        c_loss = self._divergence.dual_critic_loss(obs, new_actions.detach(), actions, task_z.detach())
        self.c_optimizer.zero_grad()
        c_loss.backward(retain_graph=True)
        self.c_optimizer.step()

    def FOCAL_loss(self, indices, context, epsilon=1e-3):
        latent_z = torch.mean(self.agent.encode_no_mean(context), dim=1)
        heteroscedastic_var = torch.mean(F.softplus(self.agent.uncertainty_mlp(latent_z.detach())))
        pos_z_loss = 0.
        neg_z_loss = 0.
        pos_cnt = 0
        neg_cnt = 0
        for i in range(len(indices)):
            for j in range(i+1, len(indices)):
                if indices[i] == indices[j]:
                    pos_z_loss += torch.sqrt(torch.mean((latent_z[i] - latent_z[j]) ** 2) + epsilon)
                    pos_cnt += 1
                else:
                    neg_z_loss += 1/(torch.mean((latent_z[i] - latent_z[j]) ** 2) + epsilon * 100)
                    neg_cnt += 1
        return pos_z_loss/(pos_cnt + epsilon) +  neg_z_loss/(neg_cnt + epsilon), heteroscedastic_var

    def CLASSIFY_loss(self, indices, context, epsilon=1e-3):
        target_indices = torch.tensor([np.where(self.train_tasks == task_id)[0] for task_id in indices]).to(ptu.device)
        target_one_hot = F.one_hot(target_indices, num_classes=len(self.train_tasks)).float()
        target_one_hot = target_one_hot.expand(-1, context.shape[1], -1).reshape(-1, len(self.train_tasks))
        # 使用交叉熵损失函数
        latent_z = self.agent.encode_no_mean(context).view(-1, self.latent_dim)
        heteroscedastic_var = torch.mean(F.softplus(self.agent.uncertainty_mlp(latent_z.detach())))
        logits = self.classifier(latent_z)
        cross_entropy_loss = torch.nn.CrossEntropyLoss()
        loss = cross_entropy_loss(logits, target_one_hot).view(-1, 1)
        return loss, heteroscedastic_var

    def _take_step(self, indices, context):
        # ---------------------------------------------------------
        # Update context encoder, heterodastic net and classifier
        # ---------------------------------------------------------
        self.context_encoder_optimizer.zero_grad()
        self.classifier_optimizer.zero_grad()
        self.uncertainty_mlp_optimizer.zero_grad()

        if self.use_FOCAL_cl:
            loss, heteroscedastic_var = self.FOCAL_loss(indices, context)
            self.loss["FOCAL_loss"] = loss.item()
        else:
            loss, heteroscedastic_var= self.CLASSIFY_loss(indices, context)
            self.loss["classify_loss"] = loss.item()

        if self.use_hvar:
            # L_h = L/(2*sigma^2) + log(sigma)
            heteroscedastic_loss = max(1e-4, loss)/(2*heteroscedastic_var)+torch.log(heteroscedastic_var**0.5)
        else:
            heteroscedastic_loss = loss
        heteroscedastic_loss.backward()

        self.loss["heteroscedastic_var"] = heteroscedastic_var.item()
        self.loss["heteroscedastic_loss"] = heteroscedastic_loss.item()
        # self.loss["focal loss"] = focal_z_loss.item()
        if not self.use_pretrained:
            self.uncertainty_mlp_optimizer.step()
            self.classifier_optimizer.step()
            self.context_encoder_optimizer.step()

        # ---------------------------------------------------------
        # Update policy and qf networks
        # ---------------------------------------------------------

        num_tasks = len(indices)
        # data is (task, batch, feat)
        obs, actions, rewards, next_obs, terms = self.sample_sac(indices)

        policy_outputs, task_z, task_z_vars= self.agent(obs, context, task_indices=indices)

        # flattens out the task dimension
        t, b, _ = obs.size()
        obs = obs.view(t * b, -1)
        actions = actions.view(t * b, -1)
        rewards = rewards.view(t * b, -1)
        next_obs = next_obs.view(t * b, -1)
        terms = terms.view(t * b, -1)

        new_actions, policy_mean, policy_log_std, log_pi = policy_outputs[:4]

        if self.train_z0_policy:
            # get \pi(a|s,z0)
            z0 = ptu.zeros_like(task_z)
            z0_in = torch.cat([obs, z0], dim=-1)
            z0_actions, z0_policy_mean, z0_log_std, z0_log_pi = self.agent.policy(t, b, z0_in, reparameterize=True, return_log_prob=True)[:4]

            new_actions = torch.cat([new_actions, z0_actions], dim=0)
            policy_mean = torch.cat([policy_mean, z0_policy_mean], dim=0)
            policy_log_std = torch.cat([policy_log_std, z0_log_std], dim=0)
            log_pi = torch.cat([log_pi, z0_log_pi], dim=0)

            # cat z0 policy
            obs = obs.repeat(2, 1)
            rewards = rewards.repeat(2, 1)
            actions = actions.repeat(2, 1)
            next_obs = next_obs.repeat(2, 1)
            terms = terms.repeat(2, 1)
            task_z = torch.cat([task_z, z0], dim=0)

        ### Q and V networks
        q1_pred = self.qf1(t, b, obs, actions, task_z.detach())
        q2_pred = self.qf2(t, b, obs, actions, task_z.detach())
        v_pred = self.vf(t, b, obs, task_z.detach())
        # get targets for use in V and Q updates
        # BRAC:
        # div_estimate = self._divergence.dual_estimate(
        #     s2, a2_p, a2_b, self._c_fn)
        
        c_loss = self._divergence.dual_critic_loss(obs, new_actions.detach(), actions, task_z.detach()) # BRAC 的行为正则项
        self.c_optimizer.zero_grad()
        c_loss.backward(retain_graph=True)
        self.c_optimizer.step()
        for _ in range(self._c_iter - 1):
            self._optimize_c(indices=indices, context=context)
        self.loss["c_loss"] = c_loss.item()

        div_estimate = self._divergence.dual_estimate(
            obs, new_actions, actions, task_z.detach())
        self.loss["div_estimate"] = torch.mean(div_estimate).item()

        with torch.no_grad():
            if self.use_brac and self.use_value_penalty:
                target_v_values = self.target_vf(t, b, next_obs, task_z) - self.get_alpha * div_estimate
            else:
                target_v_values = self.target_vf(t, b, next_obs, task_z)
        self.loss["target_v_values"] = torch.mean(target_v_values).item()

        # KL constraint on z if probabilistic

        # qf and encoder update (note encoder does not get grads from policy or vf)
        self.qf1_optimizer.zero_grad()
        self.qf2_optimizer.zero_grad()
        # scale rewards for Bellman update
        rewards = rewards * self.reward_scale
        if self.train_z0_policy:
            if self.use_next_obs_in_context:
                transitions = torch.cat([obs, actions, rewards, next_obs], dim=-1)
            else:
                transitions = torch.cat([obs, actions, rewards], dim=-1)
            transition_heterodastic_var = F.softplus(self.agent.uncertainty_mlp(self.agent.encode_no_mean(transitions))).detach()

            transition_heterodastic_var_flat = transition_heterodastic_var.view(self.batch_size * num_tasks, -1)
            var0 = ptu.zeros_like(transition_heterodastic_var_flat)
            transition_heterodastic_var_flat = torch.cat([var0, transition_heterodastic_var_flat], dim=0)
            q_target = rewards + (1. - terms) * self.discount * target_v_values - self.hvar_punish_w*transition_heterodastic_var_flat
        else:
            q_target = rewards + (1. - terms) * self.discount * target_v_values
        qf_loss = torch.mean((q1_pred - q_target) ** 2) + torch.mean((q2_pred - q_target) ** 2)
        qf_loss.backward(retain_graph=True)
        self.loss["qf_loss"] = qf_loss.item()
        self.loss["q_target"] = torch.mean(q_target).item()
        self.loss["q1_pred"] = torch.mean(q1_pred).item()
        self.loss["q2_pred"] = torch.mean(q2_pred).item()

        self.qf1_optimizer.step()
        self.qf2_optimizer.step()

        # compute min Q on the new actions
        min_q_new_actions = torch.min(self.qf1(t, b, obs, new_actions, task_z.detach()),
                                        self.qf2(t, b, obs, new_actions, task_z.detach()))

        # vf update
        if self.max_entropy:
            v_target = min_q_new_actions - log_pi
        else:
            v_target = min_q_new_actions
        vf_loss = self.vf_criterion(v_pred, v_target.detach())
        self.vf_optimizer.zero_grad()
        vf_loss.backward(retain_graph=True)
        self.vf_optimizer.step()
        self._update_target_network()
        self.loss["vf_loss"] = vf_loss.item()
        self.loss["v_target"] = torch.mean(v_target).item()
        self.loss["v_pred"] = torch.mean(v_pred).item()

        ### policy update
        # n.b. policy update includes dQ/da
        log_policy_target = min_q_new_actions

        # BRAC:
        if self.use_brac:
            if self.max_entropy:
                policy_loss = (log_pi - log_policy_target + self.get_alpha.detach() * div_estimate).mean()
            else:
                policy_loss = (-log_policy_target + self.get_alpha.detach() * div_estimate).mean()
        else:
            if self.max_entropy:
                policy_loss = (log_pi - log_policy_target).mean()
            else:
                policy_loss = - log_policy_target.mean()

        mean_reg_loss = self.policy_mean_reg_weight * (policy_mean**2).mean()
        std_reg_loss = self.policy_std_reg_weight * (policy_log_std**2).mean()
        pre_tanh_value = policy_outputs[-1]
        pre_activation_reg_loss = self.policy_pre_activation_weight * (
            (pre_tanh_value**2).sum(dim=-1).mean()
        )
        policy_reg_loss = mean_reg_loss + std_reg_loss + pre_activation_reg_loss
        policy_loss = policy_loss + policy_reg_loss

        self.policy_optimizer.zero_grad()
        policy_loss.backward(retain_graph=True)
        self.policy_optimizer.step()

        self.loss["policy_loss"] = policy_loss.item()
        # optimize for c network (which computes dual-form divergences)
        # BRAC for training alpha:
        a_loss = -torch.mean(self._alpha_var * (div_estimate - self._target_divergence).detach())
        a_loss.backward()
        with torch.no_grad():
            self._alpha_var -= self.alpha_lr * self._alpha_var.grad
            # Manually zero the gradients after updating weights
            self._alpha_var.grad.zero_()
        self.loss["a_loss"] = a_loss.item()
        if self._num_steps % self._visit_num_steps_train == 0:
            print(self.loss)
        # save some statistics for eval
        if self.eval_statistics is None:
            # eval should set this to None.
            # this way, these statistics are only computed for one batch.
            self.eval_statistics = OrderedDict()

            # z_mean = np.mean(np.abs(ptu.get_numpy(self.agent.z_means[0])))
            for i in range(len(self.agent.z_means[0])):
                z_mean = ptu.get_numpy(self.agent.z_means[0][i])
                name = 'Z mean train' + str(i)
                self.eval_statistics[name] = z_mean
                
            #z_mean1 = ptu.get_numpy(self.agent.z_means[0][0])
            #z_mean2 = ptu.get_numpy(self.agent.z_means[0][1])
            #z_mean3 = ptu.get_numpy(self.agent.z_means[0][2])
            #z_mean4 = ptu.get_numpy(self.agent.z_means[0][3])
            #z_mean5 = ptu.get_numpy(self.agent.z_means[0][4])

            z_sig = np.mean(ptu.get_numpy(self.agent.z_vars[0]))
            #self.eval_statistics['Z mean train1'] = z_mean1
            #self.eval_statistics['Z mean train2'] = z_mean2
            #self.eval_statistics['Z mean train3'] = z_mean3
            #self.eval_statistics['Z mean train4'] = z_mean4
            #self.eval_statistics['Z mean train5'] = z_mean5

            self.eval_statistics['Z variance train'] = z_sig
            self.eval_statistics['task idx'] = indices[0]

            self.eval_statistics['Heteroscedastic Var'] = np.mean(ptu.get_numpy(heteroscedastic_var))
            if self.use_FOCAL_cl:
                self.eval_statistics['FOCAL Loss'] = np.mean(ptu.get_numpy(loss))
            else:
                self.eval_statistics['CLASSIFY Loss'] = np.mean(ptu.get_numpy(loss))
            self.eval_statistics['Heteroscedastic Loss'] = np.mean(ptu.get_numpy(heteroscedastic_loss))
            self.eval_statistics['QF Loss'] = np.mean(ptu.get_numpy(qf_loss))
            self.eval_statistics['VF Loss'] = np.mean(ptu.get_numpy(vf_loss))
            self.eval_statistics['Policy Loss'] = np.mean(ptu.get_numpy(
                policy_loss
            ))
            if self.use_brac:
                self.eval_statistics['Dual Critic Loss'] = np.mean(ptu.get_numpy(c_loss))
            self.eval_statistics.update(create_stats_ordered_dict('Q Predictions',  ptu.get_numpy(q1_pred)))
            self.eval_statistics.update(create_stats_ordered_dict('V Predictions',  ptu.get_numpy(v_pred)))
            self.eval_statistics.update(create_stats_ordered_dict('Log Pis',        ptu.get_numpy(log_pi)))
            self.eval_statistics.update(create_stats_ordered_dict('Policy mu',      ptu.get_numpy(policy_mean)))
            self.eval_statistics.update(create_stats_ordered_dict('Policy log std', ptu.get_numpy(policy_log_std)))
            self.eval_statistics.update(create_stats_ordered_dict('alpha',          ptu.get_numpy(self._alpha_var).reshape(-1)))
            self.eval_statistics.update(create_stats_ordered_dict('div_estimate',   ptu.get_numpy(div_estimate)))
        return ptu.get_numpy(self.agent.z_means), ptu.get_numpy(self.agent.z_vars)

    def get_epoch_snapshot(self, epoch):
        # NOTE: overriding parent method which also optionally saves the env
        snapshot = OrderedDict(
            qf1=self.qf1.state_dict(),
            qf2=self.qf2.state_dict(),
            policy=self.agent.policy.state_dict(),
            vf=self.vf.state_dict(),
            target_vf=self.target_vf.state_dict(),
            context_encoder=self.agent.context_encoder.state_dict(),
            uncertainty_mlp=self.agent.uncertainty_mlp.state_dict(),
            c=self.c.state_dict(),
            classifier=self.classifier.state_dict(),
            )
        return snapshot