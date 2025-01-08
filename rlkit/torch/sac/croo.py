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

# Contrastive Reconstruction Objective for Offline Meta Reinforcement Learning
class CROOSoftActorCritic(OfflineMetaRLAlgorithm):
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

        # 基本配置
        self.latent_dim                     = latent_dim
        self.recurrent                      = kwargs['recurrent']
        self.use_next_obs_in_context        = kwargs['use_next_obs_in_context']
        
        # 基本训练参数
        # self.kl_lambda                      = kwargs['kl_lambda']
        self._divergence_name               = kwargs['divergence_name']
        self.use_brac                       = kwargs['use_brac']
        self.soft_target_tau                = kwargs['soft_target_tau']
        self.allow_backward_z               = kwargs['allow_backward_z']

        # 奖励设置相关
        self.sparse_rewards                 = kwargs['sparse_rewards']
        self.max_entropy                    = kwargs['max_entropy']

        # 策略更新相关
        self.policy_mean_reg_weight         = kwargs['policy_mean_reg_weight']
        self.policy_std_reg_weight          = kwargs['policy_std_reg_weight']
        self.policy_pre_activation_weight   = kwargs['policy_pre_activation_weight']
        
        self.alpha_max                      = kwargs['alpha_max']
        self.train_alpha                    = kwargs['train_alpha']
        self.alpha_init                     = kwargs['alpha_init']
        self._target_divergence             = kwargs['target_divergence']
        self._c_iter                        = kwargs['c_iter']
        self.use_value_penalty              = kwargs['use_value_penalty']

        # 更新步长
        self.alpha_lr                       = kwargs['alpha_lr']
        self.policy_lr                      = kwargs['policy_lr']
        self.qf_lr                          = kwargs['qf_lr']
        self.vf_lr                          = kwargs['vf_lr']
        self.c_lr                           = kwargs['c_lr']
        self.context_lr                     = kwargs['context_lr']

        # loss
        self.use_recon_loss                 = kwargs['use_recon_loss']
        self.recon_loss_weight              = kwargs['recon_loss_weight']

        self.use_focal_loss                 = kwargs['use_focal_loss']
        self.focal_loss_weight              = kwargs['focal_loss_weight']
        
        self.use_club_loss                  = kwargs['use_club_loss']
        self.club_loss_weight               = kwargs['club_loss_weight']

        self.use_croo_loss                  = kwargs['use_croo_loss']
        self.croo_loss_weight               = kwargs['croo_loss_weight']

        self.use_infoNCE_loss               = kwargs['use_infoNCE_loss']
        self.infoNCE_loss_weight            = kwargs['infoNCE_loss_weight']
        self.infoNCE_temp                   = kwargs['infoNCE_temp']

        # self.z_loss_weight                  = kwargs['z_loss_weight']
        # self.use_latent_projection          = kwargs['use_latent_projection']

        self.loss                           = {}
        self.plotter                        = plotter
        self.render_eval_paths              = render_eval_paths
        self.qf_criterion                   = nn.MSELoss()
        self.vf_criterion                   = nn.MSELoss()
        self.vib_criterion                  = nn.MSELoss()
        self.l2_reg_criterion               = nn.MSELoss()
        self.club_criterion                 = nn.MSELoss()
        self.cross_entropy_loss             = nn.CrossEntropyLoss()

        self.qf1, self.qf2, self.vf, self.c, self.behavior_encoder, self.context_decoder= nets[1:]
        self.target_vf                      = self.vf.copy()

        self.policy_optimizer               = optimizer_class(self.agent.policy.parameters(), lr=self.policy_lr)
        self.qf1_optimizer                  = optimizer_class(self.qf1.parameters(), lr=self.qf_lr)
        self.qf2_optimizer                  = optimizer_class(self.qf2.parameters(), lr=self.qf_lr)
        self.vf_optimizer                   = optimizer_class(self.vf.parameters(),  lr=self.vf_lr)
        self.c_optimizer                    = optimizer_class(self.c.parameters(),   lr=self.c_lr)
        self.context_encoder_optimizer      = optimizer_class(self.agent.context_encoder.parameters(), lr=self.context_lr)
        self.behavior_encoder_optimizer     = optimizer_class(self.behavior_encoder.parameters(), lr=self.context_lr)
        self.context_decoder_optimizer      = optimizer_class(self.context_decoder.parameters(), lr=self.context_lr)

        self._num_steps                     = 0
        self._visit_num_steps_train         = 10
        self._alpha_var                     = torch.tensor(1.)

        for net in nets:
            self.print_networks(net)

    ###### Torch stuff #####
    @property
    def networks(self):
        return self.agent.networks + [self.qf1, self.qf2, self.vf, self.target_vf, self.c, self.behavior_encoder, self.context_decoder]

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

    ##### Training #####
    def _do_training(self, indices):
        mb_size = self.embedding_mini_batch_size # NOTE: not meta batch!
        num_updates = self.embedding_batch_size // mb_size

        # sample context batch
        context_batch = self.sample_context(indices)

        # zero out context and hidden encoder state
        self.agent.clear_z(num_tasks=len(indices))

        z_means_lst = []
        z_vars_lst = []
        # do this in a loop so we can truncate backprop in the recurrent encoder
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

    def CROO_loss(self, indices, context, task_z, epsilon=1e-8):
        mb, b, _ = context.shape
        # task0 z0 b f
        #       z1 b f 
        # task1 z0 b f 
        #       z1 b f 
        task_z = task_z.unsqueeze(1).unsqueeze(0).expand(mb, -1, b, -1) # (mb, mb, b, z_dim)
        # task0 c0 b f
        #       c0 b f
        # task1 c1 b f 
        #       c1 b f 
        context = context.unsqueeze(1).expand(-1, mb, -1, -1) # (mb, mb, b, feat_dim)
        r_ns = context[..., self.obs_dim+self.action_dim:]
        # get behavior z probs
        with torch.no_grad():
            behavior_z_param = self.behavior_encoder(context[..., :self.obs_dim+self.action_dim])
            behacior_z_mean = behavior_z_param[..., :self.latent_dim]
            behavior_z_var = F.softplus(behavior_z_param[..., self.latent_dim:])
            behavior_z_probs = torch.exp(-(task_z-behacior_z_mean)**2/behavior_z_var+epsilon)/(behavior_z_var**0.5+epsilon)
            # 检查 behavior_z_probs 
            assert not torch.isnan(behavior_z_probs).any(), "behavior_z_probs contains NaN values"
            assert not torch.isinf(behavior_z_probs).any(), "behavior_z_probs contains infinite values"
        # construct (s,z,a)
        s_z_a = torch.cat([context[..., :self.obs_dim], task_z, context[..., self.obs_dim:self.obs_dim+self.action_dim]], dim=-1)
        pre_r_ns_param = self.context_decoder(s_z_a)
        split_size = int(self.context_decoder.output_size/2)
        pre_r_ns_mean = pre_r_ns_param[..., :split_size]
        pre_r_ns_var = F.softplus(pre_r_ns_param[..., split_size:])
        # 检查 pre_r_ns_param, pre_r_ns_mean 和 pre_r_ns_var
        assert not torch.isnan(pre_r_ns_param).any(), "pre_r_ns_param contains NaN values"
        assert not torch.isinf(pre_r_ns_param).any(), "pre_r_ns_param contains infinite values"
        assert not torch.isnan(pre_r_ns_mean).any(), "pre_r_ns_mean contains NaN values"
        assert not torch.isinf(pre_r_ns_mean).any(), "pre_r_ns_mean contains infinite values"
        assert not torch.isnan(pre_r_ns_var).any(), "pre_r_ns_var contains NaN values"
        assert not torch.isinf(pre_r_ns_var).any(), "pre_r_ns_var contains infinite values"
        assert (pre_r_ns_var >= 0).all(), "pre_r_ns_var contains negative values"
        # 计算重建损失
        recon_loss = 0
        neg_recon_loss = 0
        for i in range(len(indices)):
            neg_idxs = [j for j in range(len(indices)) if indices[j] != indices[i]]
            recon_loss += torch.mean((r_ns[i, i] - pre_r_ns_mean[i, i])**2)
            neg_recon_loss += torch.mean((r_ns[i, neg_idxs] - pre_r_ns_mean[i, neg_idxs])**2)
        recon_loss /= len(indices)
        neg_recon_loss /= len(indices)
        # 计算高斯
        prob = torch.exp(-(r_ns-pre_r_ns_mean)**2/(pre_r_ns_var+epsilon))/(pre_r_ns_var**0.5+epsilon)
        # 检查 prob
        assert not torch.isnan(prob).any(), "prob contains NaN values"
        assert not torch.isinf(prob).any(), "prob contains infinite values"
        croo_loss = 0
        for i in range(len(indices)):
            neg_idxs = [j for j in range(len(indices)) if indices[j] != indices[i]]
            # neg_idxs = [j for j in range(len(indices))]
            if len(neg_idxs) == 0:
                continue
            positive_prob = prob[i, i].unsqueeze(0)
            negative_prob = prob[i, neg_idxs]
            negative_behavior_z_prob = behavior_z_probs[i, neg_idxs]
            if self.use_croo_max:
                sum_neg_prob = torch.max(negative_prob, dim=0, keepdim=True)[0] * len(neg_idxs)
            else:
                sum_neg_prob = torch.sum(negative_prob*negative_behavior_z_prob, dim=0, keepdim=True)
            # 检查 sum_neg_prob
            assert not torch.isnan(sum_neg_prob).any(), "sum_neg_prob contains NaN values"
            assert not torch.isinf(sum_neg_prob).any(), "sum_neg_prob contains infinite values"
            assert (sum_neg_prob + epsilon).min() > 0, "sum_neg_prob + epsilon contains non-positive values"
        
            croo_loss += torch.log((positive_prob/(sum_neg_prob+epsilon))+epsilon)
            # 检查 croo_loss
            assert not torch.isnan(croo_loss).any(), f"{i} croo_loss contains NaN values"
            assert not torch.isinf(croo_loss).any(), f"{i} croo_loss contains infinite values"
        croo_loss = -torch.mean(croo_loss)
        return croo_loss, recon_loss, neg_recon_loss

    def Recon_loss(self, context, task_z, epsilon=1e-8):
        # construct (s,z,a)
        r_ns = context[..., self.obs_dim+self.action_dim:]
        task_z = task_z.unsqueeze(1).expand(-1, context.shape[1], -1) # (mb, b, z_dim)
        s_z_a = torch.cat([context[..., :self.obs_dim], task_z, context[..., self.obs_dim:self.obs_dim+self.action_dim]], dim=-1)
        pre_r_ns_param = self.context_decoder(s_z_a)
        split_size = int(self.context_decoder.output_size/2)
        pre_r_ns_mean = pre_r_ns_param[..., :split_size]
        pre_r_ns_var = F.softplus(pre_r_ns_param[..., split_size:])
        # 计算高斯
        log_probs = -(r_ns-pre_r_ns_mean)**2/(pre_r_ns_var+epsilon) - torch.log(pre_r_ns_var**0.5+epsilon)
        return -torch.mean(log_probs)
    
    def FOCAL_z_loss(self, indices, task_z, task_z_vars, b, epsilon=1e-3, threshold=0.999):
        pos_z_loss = 0.
        neg_z_loss = 0.
        pos_cnt = 0
        neg_cnt = 0
        for i in range(len(indices)):
            idx_i = i * b # index in task * batch dim
            for j in range(i+1, len(indices)):
                idx_j = j * b # index in task * batch dim
                if indices[i] == indices[j]:
                    pos_z_loss += torch.sqrt(torch.mean((task_z[idx_i] - task_z[idx_j]) ** 2) + epsilon)
                    pos_cnt += 1
                else:
                    neg_z_loss += 1/(torch.mean((task_z[idx_i] - task_z[idx_j]) ** 2) + epsilon * 100)
                    neg_cnt += 1
        return pos_z_loss/(pos_cnt + epsilon) +  neg_z_loss/(neg_cnt + epsilon)

    def CLUB_loss(self, context, epsilon=1e-8):
        z_target = self.agent.encode_no_mean(context)
        with torch.no_grad():
            z_param = self.behavior_encoder(context[...,:self.behavior_encoder.input_size])
            z_mean = z_param[..., :self.latent_dim]
            z_var = F.softplus(z_param[..., self.latent_dim:])
        z_t, z_b, _ = z_mean.size()
        position = - ((z_target-z_mean)**2/z_var).mean()
        z_mean_expand = z_mean[:, :, None, :].expand(-1, -1, z_b, -1).reshape(z_t, z_b**2, -1)
        z_var_expand = z_var[:, :, None, :].expand(-1, -1, z_b, -1).reshape(z_t, z_b**2, -1)
        z_target_repeat = z_target.repeat(1, z_b, 1)
        negative = - ((z_target_repeat-z_mean_expand)**2/z_var_expand).mean()
        club_loss = position - negative
        return club_loss

    # InfoNCE
    # Unlike the infoNCE in CORRO which the x^* = (s,a,r*,s*), we use x^* = (s*,a*,r*,s*)
    # This would not relabel the (s,a), so it can not decrease the mutual information between (s,a) and z
    def infoNCE_loss(self, indices, context, epsilon=1e-8):
        mb, b, _ = context.shape
        query_context = context.reshape(mb*b, 1, -1) # (b, 1, feat_dim)
        key_context = self.sample_context(indices).reshape(mb*b, 1, -1)  # (b, 1, feat_dim)
        neg_context = []
        for pos_task_id in indices:
            neg_tasks = [task_id for task_id in self.train_tasks if task_id != pos_task_id]
            neg_indices = np.random.choice(neg_tasks, self.meta_batch-1, replace=self.mb_replace)
            neg_context.append(self.sample_context(neg_indices).transpose(0,1)) # (b, N, feat_dim)
        neg_context = torch.cat(neg_context) # (b, N, feat_dim)

        q = self.agent.encode_no_mean(query_context) # (b, 1, z_dim)
        k = self.agent.encode_no_mean(key_context) # (b, 1, z_dim)
        neg = self.agent.encode_no_mean(neg_context) # (b, N, z_dim)

        N = neg.shape[1]
        l_pos = torch.bmm(q, k.transpose(1,2)) # (b,1,1)
        l_neg = torch.bmm(q, neg.transpose(1,2)) # (b,1,N)
        logits = torch.cat([l_pos.view(-1, 1), l_neg.view(-1, N)], dim=1)
        
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(ptu.device)
        loss = self.cross_entropy_loss(logits/self.infoNCE_temp , labels)
        return loss

    def _take_step(self, indices, context):
        obs_dim = int(np.prod(self.env.observation_space.shape))
        action_dim = int(np.prod(self.env.action_space.shape))
        reward_in_context = context[:, :, obs_dim + action_dim].cpu().numpy()
        self.loss["non_sparse_ratio"] = len(reward_in_context[np.nonzero(reward_in_context)]) / np.size(reward_in_context)

        num_tasks = len(indices)
        # data is (task, batch, feat)
        obs, actions, rewards, next_obs, terms = self.sample_sac(indices)

        policy_outputs, task_z, task_z_vars= self.agent(obs, context, task_indices=indices)

        if self.use_club_loss or self.use_croo_loss:
            # update behavor encoder
            self.behavior_encoder_optimizer.zero_grad()
            z_target = self.agent.encode_no_mean(context).detach()
            z_param = self.behavior_encoder(context[...,:self.behavior_encoder.input_size])
            z_mean = z_param[..., :self.latent_dim]
            z_var = F.softplus(z_param[..., self.latent_dim:])
            behavior_encoder_loss = self.club_model_loss_weight*((z_target - z_mean)**2/(2*z_var) + torch.log(torch.sqrt(z_var))).mean()
            behavior_encoder_loss.backward()
            self.loss["behavior_encoder_loss"] = behavior_encoder_loss.item()
            self.behavior_encoder_optimizer.step()

        # update context encoder and decoder
        self.context_encoder_optimizer.zero_grad()
        self.context_decoder_optimizer.zero_grad()
        with torch.autograd.set_detect_anomaly(True):
            total_loss = 0
            if self.use_club_loss:
                club_loss = self.CLUB_loss(context)
                total_loss += self.club_loss_weight*club_loss
            if self.use_focal_loss:
                focal_z_loss = self.FOCAL_z_loss(indices=indices, task_z=task_z, task_z_vars=task_z_vars, b=obs.shape[1])
                total_loss += self.focal_loss_weight*focal_z_loss
            if self.use_recon_loss:
                recon_loss = self.Recon_loss(context, self.agent.z)
                total_loss += self.recon_loss_weight*recon_loss
            if self.use_croo_loss:
                croo_loss, croo_recon_loss, croo_neg_recon_loss = self.CROO_loss(indices, context, self.agent.z)
                total_loss += self.croo_loss_weight*croo_loss
            if self.use_infoNCE_loss:
                infoNCE_loss = self.infoNCE_loss(indices, context)
                total_loss += self.infoNCE_loss_weight*infoNCE_loss
            total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.agent.context_encoder.parameters(), max_norm=100)
        torch.nn.utils.clip_grad_norm_(self.context_decoder.parameters(), max_norm=100)
        self.context_decoder_optimizer.step()
        self.context_encoder_optimizer.step()

        # flattens out the task dimension
        t, b, _ = obs.size()
        obs = obs.view(t * b, -1)
        actions = actions.view(t * b, -1)
        next_obs = next_obs.view(t * b, -1)

        new_actions, policy_mean, policy_log_std, log_pi = policy_outputs[:4]

        # Q and V networks
        # encoder will only get gradients from Q nets
        if self.allow_backward_z:
            q1_pred = self.qf1(t, b, obs, actions, task_z)
            q2_pred = self.qf2(t, b, obs, actions, task_z)
            v_pred = self.vf(t, b, obs, task_z.detach())
        else:
            q1_pred = self.qf1(t, b, obs, actions, task_z.detach())
            q2_pred = self.qf2(t, b, obs, actions, task_z.detach())
            v_pred = self.vf(t, b, obs, task_z.detach())
        # get targets for use in V and Q updates
        # BRAC:
        # div_estimate = self._divergence.dual_estimate(
        #     s2, a2_p, a2_b, self._c_fn)
        
        c_loss = self._divergence.dual_critic_loss(obs, new_actions.detach(), actions, task_z.detach())
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
        rewards_flat = rewards.view(self.batch_size * num_tasks, -1)
        # scale rewards for Bellman update
        rewards_flat = rewards_flat * self.reward_scale
        terms_flat = terms.view(self.batch_size * num_tasks, -1)
        q_target = rewards_flat + (1. - terms_flat) * self.discount * target_v_values
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

        # policy update
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
                
            z_sig = np.mean(ptu.get_numpy(self.agent.z_vars[0]))

            if self.use_club_loss:
                self.eval_statistics['CLUB Loss'] = np.mean(ptu.get_numpy(club_loss))
            if self.use_focal_loss:
                self.eval_statistics['FOCAL Loss'] = np.mean(ptu.get_numpy(focal_z_loss))
            if self.use_recon_loss:
                self.eval_statistics['Recon Loss'] = np.mean(ptu.get_numpy(recon_loss))
            if self.use_croo_loss:
                self.eval_statistics['CROO Loss'] = np.mean(ptu.get_numpy(croo_loss))
                self.eval_statistics['CROO Recon Loss'] = np.mean(ptu.get_numpy(croo_recon_loss))
                self.eval_statistics['CROO NegRecon Loss'] = np.mean(ptu.get_numpy(croo_neg_recon_loss))
            if self.use_infoNCE_loss:
                self.eval_statistics['infoNCE Loss'] = np.mean(ptu.get_numpy(infoNCE_loss))
            if self.use_club_loss or self.use_croo_loss:
                self.eval_statistics['Behavior Encoder Loss'] = np.mean(ptu.get_numpy(behavior_encoder_loss))
            self.eval_statistics['Total Loss'] = np.mean(ptu.get_numpy(total_loss))

            self.eval_statistics['Z variance train'] = z_sig
            self.eval_statistics['task idx'] = indices[0]

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
            c=self.c.state_dict(),
            context_decoder=self.context_decoder.state_dict(),
            )
        return snapshot