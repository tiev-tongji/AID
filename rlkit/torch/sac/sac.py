import os
from pathlib import Path
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

class CSROSoftActorCritic(OfflineMetaRLAlgorithm):
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
            seed=0,
            algo_type = 'CSRO',
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

        ## 基本配置
        self.seed                           = seed
        self.env_name                       = kwargs['env_name']
        self.algo_type                      = algo_type
        self.latent_dim                     = latent_dim
        self.separate_train                 = kwargs['separate_train']
        self.pretrain                       = kwargs['pretrain']
        self.train_z0_policy                = kwargs['train_z0_policy']
        self.use_hvar                       = kwargs['use_hvar'] # 是否使用异方差
        self.hvar_punish_w                  = kwargs['hvar_punish_w']
        self.policy_update_strategy         = kwargs['policy_update_strategy'] # 策略更新 BRAC or TD3BC

        self.recurrent                      = kwargs['recurrent'] # 是否使用循环编码器
        self.use_relabel                    = kwargs['use_relabel'] # 是否使用重标记
        self.use_next_obs_in_context        = kwargs['use_next_obs_in_context'] # 是否使用下一个观测

        ## 训练参数
        self.soft_target_tau                = kwargs['soft_target_tau'] # 软目标更新
        self.allow_backward_z               = kwargs['allow_backward_z'] # 是否允许梯度通过z流动
        self.num_ensemble                   = kwargs['num_ensemble'] # IDAQ

        # BRAC参数
        self.use_brac                       = kwargs['use_brac'] # 是否使用BRAC
        self.train_alpha                    = kwargs['train_alpha'] # 是否训练alpha
        self.use_value_penalty              = kwargs['use_value_penalty'] # 是否使用值惩罚
        self.alpha_max                      = kwargs['alpha_max'] # alpha的最大值
        self.alpha_init                     = kwargs['alpha_init'] # alpha的初始值
        self._target_divergence             = kwargs['target_divergence'] # 训练alpha自适应的目标散度
        self._c_iter                        = kwargs['c_iter'] # 每次迭代的双重评论者步数
        self._divergence_name               = kwargs['divergence_name'] # BRAC算法的散度类型
        # self.kl_lambda                      = kwargs['kl_lambda']
        self.policy_mean_reg_weight         = kwargs['policy_mean_reg_weight']
        self.policy_std_reg_weight          = kwargs['policy_std_reg_weight']
        self.policy_pre_activation_weight   = kwargs['policy_pre_activation_weight']

        # TD3BC参数
        self.step_to_update_policy          = kwargs['step_to_update_policy'] # 更新策略的步数
        self.delay_frequency                = kwargs['delay_frequency'] # 延迟频率
        self.bc_weight                      = kwargs['bc_weight'] # 行为克隆权重
        self.gamma                          = kwargs['gamma'] # 折扣因子

        # 奖励设置
        self.sparse_rewards                 = kwargs['sparse_rewards'] # 是否稀疏奖励
        self.max_entropy                    = kwargs['max_entropy'] # 是否在值函数中包含最大熵项

        # 更新步长
        self.alpha_lr                       = kwargs['alpha_lr']
        self.policy_lr                      = kwargs['policy_lr']
        self.qf_lr                          = kwargs['qf_lr']
        self.vf_lr                          = kwargs['vf_lr']
        self.c_lr                           = kwargs['c_lr']
        self.context_lr                     = kwargs['context_lr']

        # loss
        params = kwargs[algo_type]
        self.use_recon_loss                 = params['use_recon_loss']
        self.recon_loss_weight              = params['recon_loss_weight']

        self.use_focal_loss                 = params['use_focal_loss']
        self.focal_loss_weight              = params['focal_loss_weight']
        
        self.use_club_loss                  = params['use_club_loss']
        self.club_loss_weight               = params['club_loss_weight']
        self.club_model_loss_weight         = params['club_model_loss_weight']

        self.use_croo_loss                  = params['use_croo_loss']
        self.croo_loss_weight               = params['croo_loss_weight']
        self.use_croo_max                   = params['use_croo_max']

        self.use_classify_loss              = params['use_classify_loss']
        self.classify_loss_weight           = params['classify_loss_weight']

        self.use_infoNCE_loss               = params['use_infoNCE_loss']
        self.infoNCE_loss_weight            = params['infoNCE_loss_weight']
        self.infoNCE_temp                   = params['infoNCE_temp']

        self.loss                           = {}
        self.plotter                        = plotter
        self.render_eval_paths              = render_eval_paths
        self.qf_criterion                   = nn.MSELoss()
        self.vf_criterion                   = nn.MSELoss()
        self.vib_criterion                  = nn.MSELoss()
        self.l2_reg_criterion               = nn.MSELoss()
        self.club_criterion                 = nn.MSELoss()
        self.pred_loss                      = nn.MSELoss()
        self.hvar_loss                      = nn.MSELoss()
        self.cross_entropy_loss             = nn.CrossEntropyLoss()

        self.qf1, self.qf2, _, _, self.club_model, self.context_decoder, self.classifier, self.reward_models, self.dynamic_models = nets[1:]
        if self.policy_update_strategy == 'BRAC':
            self.vf                             = nets[3]
            self.target_vf                      = self.vf.copy()
            self.c                              = nets[4]
        elif self.policy_update_strategy == 'TD3BC':
            self.target_qf1                     = self.qf1.copy()
            self.target_qf2                     = self.qf2.copy()
        else:
            raise NotImplementedError

        # 分开训练的训练部分，不更新
        if self.separate_train and not self.pretrain:
            print('*************************Using pretrained agent************************')
            agent_path = Path(kwargs[algo_type]['pretrained_agent_path'])
            if not agent_path.exists():
                raise ValueError(f"separate_train is True but pretrained_agent_path does not exist: {agent_path}")
            if agent_path.is_dir():
                agent_path = agent_path / 'agent.pth'
            if not agent_path.exists():
                agent_path = agent_path.parent / f'seed{seed}/agent.pth'
            if not agent_path.exists():
                raise ValueError(f"separate_train is True but pretrained_agent_path does not exist: {agent_path}")
            agent_ckpt = torch.load(str(agent_path))
            self.agent.context_encoder.load_state_dict(agent_ckpt['context_encoder'])
            self.agent.uncertainty_mlp.load_state_dict(agent_ckpt['uncertainty_mlp'])
            if algo_type == 'CSRO':
                self.club_model.load_state_dict(agent_ckpt['club_model'])
            if algo_type == 'CLASSIFIER':
                self.classifier.load_state_dict(agent_ckpt['classifier'])
            if algo_type == 'CROO':
                self.context_decoder.load_state_dict(agent_ckpt['context_decoder'])
                self.club_model.load_state_dict(agent_ckpt['club_model'])
            if algo_type == 'UNICORN':
                self.context_decoder.load_state_dict(agent_ckpt['context_decoder'])

        self.qf1_optimizer                  = optimizer_class(self.qf1.parameters(), lr=self.qf_lr)
        self.qf2_optimizer                  = optimizer_class(self.qf2.parameters(), lr=self.qf_lr)
        if self.policy_update_strategy == 'BRAC':
            self.vf_optimizer                   = optimizer_class(self.vf.parameters(),  lr=self.vf_lr)
            self.c_optimizer                    = optimizer_class(self.c.parameters(),   lr=self.c_lr)
        self.club_model_optimizer           = optimizer_class(self.club_model.parameters(), lr=self.context_lr)
        # self.behavior_encoder_optimizer     = optimizer_class(self.behavior_encoder.parameters(), lr=self.context_lr)
        self.context_decoder_optimizer      = optimizer_class(self.context_decoder.parameters(), lr=self.context_lr)
        self.classifier_optimizer           = optimizer_class(self.classifier.parameters(), lr=self.context_lr)
        self.reward_models_optimizer = optimizer_class(self.reward_models.parameters(), lr=self.qf_lr)
        self.dynamic_models_optimizer = optimizer_class(self.dynamic_models.parameters(), lr=self.qf_lr)
        
        self.policy_optimizer               = optimizer_class(self.agent.policy.parameters(), lr=self.policy_lr)
        self.uncertainty_mlp_optimizer      = optimizer_class(self.agent.uncertainty_mlp.parameters(), lr=self.context_lr)
        self.context_encoder_optimizer      = optimizer_class(self.agent.context_encoder.parameters(), lr=self.context_lr)

        self._num_steps                     = 0
        self._visit_num_steps_train         = 10
        self._alpha_var                     = torch.tensor(1.)

    ###### Torch stuff #####
    @property
    def networks(self):
        if self.policy_update_strategy == 'BRAC':
            return self.agent.networks + [self.qf1, self.qf2, self.vf, self.target_vf, self.c, self.club_model, self.context_decoder, self.classifier]
        elif self.policy_update_strategy == 'TD3BC':
            return self.agent.networks + [self.qf1, self.qf2, self.target_qf1, self.target_qf2, self.club_model, self.context_decoder, self.classifier]

    @property
    def get_alpha(self):
        return utils.clip_v2(
            self._alpha_var, 0.0, self.alpha_max)

    def training_mode(self, mode):
        if not self.separate_train or mode is False:
            for net in self.networks:
                net.train(mode)
        elif self.separate_train and self.pretrain:
            self.club_model.train(mode)
            self.context_decoder.train(mode)
            self.classifier.train(mode)
            self.agent.uncertainty_mlp.train(mode)
            self.agent.context_encoder.train(mode)
        elif self.separate_train and not self.pretrain:
            for net in self.networks:
                net.train(mode)
            self.club_model.eval()
            self.context_decoder.eval()
            self.classifier.eval()
            self.agent.uncertainty_mlp.eval()
            self.agent.context_encoder.eval()

    def to(self, device=None):
        if device == None:
            device = ptu.device
        for net in self.networks:
            net.to(device)
        if self.train_alpha and self.policy_update_strategy == 'BRAC':
            self._alpha_var = torch.tensor(self.alpha_init, device=ptu.device, requires_grad=True)
        if self.policy_update_strategy == 'BRAC':
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

    def _min_q(self, t, b, obs, actions, task_z):
        q1 = self.qf1(t, b, obs, actions, task_z.detach())
        q2 = self.qf2(t, b, obs, actions, task_z.detach())
        min_q = torch.min(q1, q2)
        return min_q

    def _update_target_network(self):
        if self.policy_update_strategy == 'BRAC':
            ptu.soft_update_from_to(self.vf, self.target_vf, self.soft_target_tau)
        elif self.policy_update_strategy == 'TD3BC':
            ptu.soft_update_from_to(self.qf1, self.target_qf1, self.soft_target_tau)
            ptu.soft_update_from_to(self.qf2, self.target_qf2, self.soft_target_tau)

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
    
    def CLUB_loss(self, context, epsilon=1e-8):
        latent_z = self.agent.encode_no_mean(context)
        with torch.no_grad():
            z_param = self.club_model(context[...,:self.club_model.input_size])
            z_mean = z_param[..., :self.latent_dim]
            z_var = F.softplus(z_param[..., self.latent_dim:])
        z_t, z_b, _ = z_mean.size()
        position = - ((latent_z - z_mean)**2/z_var).mean()
        z_mean_expand = z_mean[:, :, None, :].expand(-1, -1, z_b, -1).reshape(z_t, z_b**2, -1)
        z_var_expand = z_var[:, :, None, :].expand(-1, -1, z_b, -1).reshape(z_t, z_b**2, -1)
        z_target_repeat = latent_z.repeat(1, z_b, 1)
        negative = - ((z_target_repeat-z_mean_expand)**2/z_var_expand).mean()
        club_loss = position - negative
        return club_loss

    def CROO_loss(self, indices, context, epsilon=1e-8):
        mb, b, _ = context.shape
        # task0 z0 b f
        #       z1 b f
        # task1 z0 b f 
        #       z1 b f 
        # task_z = task_z.unsqueeze(1).unsqueeze(0).expand(mb, -1, b, -1) # (mb, mb, b, z_dim) [10, 2560, 1024, 20]
        latent_z = self.agent.encode_no_mean(context).expand(mb, -1, b, -1) # [10, 10, 1024, 20]
        # task0 c0 b f
        #       c0 b f
        # task1 c1 b f 
        #       c1 b f 
        context = context.unsqueeze(1).expand(-1, mb, -1, -1) # (mb, mb, b, feat_dim) [10, 10, 1024, 5]
        r_ns = context[..., self.obs_dim+self.action_dim:]
        # get behavior z probs
        with torch.no_grad():
            behavior_z_param = self.club_model(context[..., :self.obs_dim+self.action_dim])
            behacior_z_mean = behavior_z_param[..., :self.latent_dim]
            behavior_z_var = F.softplus(behavior_z_param[..., self.latent_dim:]) # [10, 10, 1024, 20]
            behavior_z_probs = torch.exp(-(latent_z-behacior_z_mean)**2/behavior_z_var+epsilon)/(behavior_z_var**0.5+epsilon)
            # 检查 behavior_z_probs 
            assert not torch.isnan(behavior_z_probs).any(), "behavior_z_probs contains NaN values"
            assert not torch.isinf(behavior_z_probs).any(), "behavior_z_probs contains infinite values"
        # construct (s,z,a)
        s_z_a = torch.cat([context[..., :self.obs_dim], latent_z, context[..., self.obs_dim:self.obs_dim+self.action_dim]], dim=-1)
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

    def Recon_loss(self, context, epsilon=1e-8):
        mb, b, _ = context.shape
        # construct (s,z,a)
        r_ns = context[..., self.obs_dim+self.action_dim:]
        # task_z = task_z.unsqueeze(1).expand(-1, context.shape[1], -1) # (mb, b, z_dim) [2560, 20], b = 1024
        latent_z = self.agent.encode_no_mean(context).expand(mb, b, -1) # [10, 1024, 20]
        s_z_a = torch.cat([context[..., : self.obs_dim], latent_z, context[..., self.obs_dim : self.obs_dim + self.action_dim]], dim=-1) # [10, 1024, *]
        pre_r_ns_param = self.context_decoder(s_z_a)
        split_size = int(self.context_decoder.output_size/2)
        pre_r_ns_mean = pre_r_ns_param[..., :split_size]
        pre_r_ns_var = F.softplus(pre_r_ns_param[..., split_size:])
        # 计算高斯
        probs = -(r_ns-pre_r_ns_mean)**2/(pre_r_ns_var+epsilon) - torch.log(pre_r_ns_var**0.5+epsilon)
        return -torch.mean(probs)

    def FOCAL_z_loss(self, indices, context, epsilon=1e-3):
        latent_z = self.agent.encode_no_mean(context).mean(dim=1)
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
        focal_z_loss = pos_z_loss/(pos_cnt + epsilon) +  neg_z_loss/(neg_cnt + epsilon)
        return focal_z_loss
    
    def CLASSIFY_loss(self, indices, context):
        target_indices = torch.tensor([np.where(self.train_tasks == task_id)[0] for task_id in indices]).to(ptu.device)
        target_one_hot = F.one_hot(target_indices, num_classes=len(self.train_tasks)).float()
        target_one_hot = target_one_hot.expand(-1, context.shape[1], -1).reshape(-1, len(self.train_tasks))
        # 使用交叉熵损失函数
        latent_z = self.agent.encode_no_mean(context).view(-1, self.latent_dim)
        logits = self.classifier(latent_z)
        cross_entropy_loss = torch.nn.CrossEntropyLoss()
        loss = cross_entropy_loss(logits, target_one_hot).view(-1, 1)
        return loss

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

    def Reward_Prediction_loss(self, indices, task_z):
        model_loss = None
        num_tasks = len(indices)
        for i in range(self.num_ensemble):
            obs1, actions1, rewards1, next_obs1, terms1 = self.sample_sac(indices)
            t, b, _ = obs1.size()
            obs1 = obs1.view(t * b, -1)
            actions1 = actions1.view(t * b, -1)
            next_obs1 = next_obs1.view(t * b, -1)
            pred_rewardss1 = rewards1.view(self.batch_size * num_tasks, -1)

            rew_pred1 = self.reward_models[i].forward(0, 0, task_z.detach(), obs1, actions1)
            next_obs_pred1=self.dynamic_models[i].forward(0, 0, task_z.detach(), obs1, actions1)


            rew_loss = self.pred_loss(pred_rewardss1, rew_pred1)
            dynamic_loss = self.pred_loss(next_obs1, next_obs_pred1)

            if model_loss is None:
                model_loss = rew_loss
            else:
                model_loss = model_loss + rew_loss + dynamic_loss
        return model_loss

    def HeteroscedasticLoss(self, context, loss):
        latent_z = self.agent.encode_no_mean(context).view(-1, self.latent_dim)
        heteroscedastic_var = torch.mean(F.softplus(self.agent.uncertainty_mlp(latent_z.detach())))
        # heteroscedastic_var = torch.mean(F.softplus(self.agent.uncertainty_mlp(latent_z)))
        heteroscedastic_loss = max(1e-4, loss)/(2 * heteroscedastic_var) + torch.log(heteroscedastic_var**0.5)
        return heteroscedastic_loss, heteroscedastic_var
    
    def HeteroscedasticLoss2(self, context, loss):
        latent_z = self.agent.encode_no_mean(context).view(-1, self.latent_dim)
        heteroscedastic_var = torch.mean(F.softplus(self.agent.uncertainty_mlp(latent_z.detach())))
        heteroscedastic_loss = self.hvar(loss, heteroscedastic_var)
        return heteroscedastic_loss, heteroscedastic_var

    def _take_step(self, indices, context):
        # ---------------------------------------------------------
        # Update context encoder, heterodastic net, classifier net, behavior encoder and club model
        # ---------------------------------------------------------
        if self.separate_train and not self.pretrain:
            pass
        else:
            self._update_context_encoder(indices, context)

        # ---------------------------------------------------------
        # Update policy and qf networks
        # ---------------------------------------------------------
        if self.separate_train and self.pretrain:
            pass # 分开训练的预训练部分不更新
        else:
            if self.policy_update_strategy == 'BRAC':
                self._update_policy_use_BRAC(indices, context)
            elif self.policy_update_strategy == 'TD3BC':
                self._update_policy_use_TD3BC(indices, context)

        if self._num_steps % self._visit_num_steps_train == 0:
            print(self.loss)

        # ---------------------------------------------------------
        # save some statistics for eval
        # ---------------------------------------------------------
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

            self.eval_statistics['Z variance train'] = z_sig
            self.eval_statistics['task idx'] = indices[0]
            print(len(self.loss.items()))
            if self.loss.get("heteroscedastic_var") is not None:
                self.eval_statistics['Heteroscedastic Var'] = self.loss["heteroscedastic_var"]
            if self.loss.get("heteroscedastic_loss") is not None:
                self.eval_statistics['Heteroscedastic Loss'] = self.loss["heteroscedastic_loss"]
            if self.loss.get("focal_loss") is not None:
                self.eval_statistics['FOCAL Loss'] = self.loss["focal_loss"]
            if self.loss.get("classify_loss") is not None:
                self.eval_statistics['CLASSIFY Loss'] = self.loss["classify_loss"]
            if self.loss.get("club_loss") is not None:
                self.eval_statistics['CLUB Loss'] = self.loss["club_loss"]
            if self.loss.get("croo_loss") is not None:
                self.eval_statistics['CROO Loss'] = self.loss["croo_loss"]
            if self.loss.get("recon_loss") is not None:
                self.eval_statistics['Recon Loss'] = self.loss["recon_loss"]
            if self.loss.get("infoNCE_loss") is not None:
                self.eval_statistics['infoNCE Loss'] = self.loss["infoNCE_loss"]

            if self.separate_train and self.pretrain:
                return ptu.get_numpy(self.agent.z_means), ptu.get_numpy(self.agent.z_vars)
            
            if self.policy_update_strategy == 'BRAC':
                self.eval_statistics['QF Loss'] = self.loss["qf_loss"]
                self.eval_statistics['VF Loss'] = self.loss["vf_loss"]
                self.eval_statistics['Policy Loss'] = self.loss["policy_loss"]
                self.eval_statistics['Dual Critic Loss'] = self.loss["c_loss"]
                self.eval_statistics.update(create_stats_ordered_dict('V Predictions',  self.loss["v_pred"]))
                self.eval_statistics.update(create_stats_ordered_dict('div_estimate',   self.loss["div_estimate"]))
                self.eval_statistics.update(create_stats_ordered_dict('alpha',          ptu.get_numpy(self._alpha_var).reshape(-1)))

            self.eval_statistics['QF Loss'] = self.loss["qf_loss"]
            if self.loss.get("policy_loss") is not None:
                self.eval_statistics['Policy Loss'] = self.loss["policy_loss"]
            if self.loss.get('reward_prediction_loss') is not None:
                self.eval_statistics['Reward Prediction Loss'] = self.loss['reward_prediction_loss']
            self.eval_statistics.update(create_stats_ordered_dict('Q Predictions',  self.loss["q1_pred"]))
            self.eval_statistics.update(create_stats_ordered_dict('Log Pis',        self.loss["log_pi"]))
            self.eval_statistics.update(create_stats_ordered_dict('Policy mu',      self.loss["policy_mean"]))
            self.eval_statistics.update(create_stats_ordered_dict('Policy log std', self.loss["policy_log_std"]))
        return ptu.get_numpy(self.agent.z_means), ptu.get_numpy(self.agent.z_vars)

    def _update_context_encoder(self, indices, context):
        """
        更新上下文编码器、异方差网络和分类器等
        """

        # update club model
        if self.use_club_loss or self.use_croo_loss:
            self.club_model_optimizer.zero_grad()
            z_target = self.agent.encode_no_mean(context).detach()
            z_param = self.club_model(context[...,:self.club_model.input_size])
            z_mean = z_param[..., :self.latent_dim]
            z_var = F.softplus(z_param[..., self.latent_dim:])
            club_model_loss = self.club_model_loss_weight * ((z_target - z_mean)**2/(2*z_var) + torch.log(torch.sqrt(z_var))).mean()
            club_model_loss.backward()
            self.loss["club_model_loss"] = club_model_loss.item()
            if self.separate_train and not self.pretrain:
                pass
            else:
                self.club_model_optimizer.step()

        # 清除梯度
        self.classifier_optimizer.zero_grad()
        self.uncertainty_mlp_optimizer.zero_grad()
        self.context_decoder_optimizer.zero_grad()
        self.context_encoder_optimizer.zero_grad()

        # 计算损失函数
        total_loss = 0
        if self.use_club_loss:
            club_loss = self.CLUB_loss(context)
            self.loss["club_loss"] = club_loss.item()
            total_loss += self.club_loss_weight * club_loss
        if self.use_focal_loss:
            focal_z_loss = self.FOCAL_z_loss(indices, context)
            self.loss["focal_loss"] = focal_z_loss.item()
            total_loss += self.focal_loss_weight * focal_z_loss
        if self.use_recon_loss:
            recon_loss = self.Recon_loss(context)
            self.loss["recon_loss"] = recon_loss.item()
            total_loss += self.recon_loss_weight * recon_loss
        if self.use_classify_loss:
            classify_loss = self.CLASSIFY_loss(indices, context)
            self.loss["classify_loss"] = classify_loss.item()
            total_loss += self.classify_loss_weight * classify_loss
        if self.use_croo_loss:
            croo_loss, recon_loss, neg_recon_loss = self.CROO_loss(indices, context)
            self.loss["croo_loss"] = croo_loss.item()
            self.loss["recon_loss"] = recon_loss.item()
            self.loss["neg_recon_loss"] = neg_recon_loss.item()
            total_loss += self.croo_loss_weight * croo_loss
        if self.use_infoNCE_loss:
            infoNCE_loss = self.infoNCE_loss(indices, context)
            self.loss["infoNCE_loss"] = infoNCE_loss.item()
            total_loss += self.infoNCE_loss_weight*infoNCE_loss

        # 计算异方差损失
        if self.use_hvar:
            heteroscedastic_loss, heteroscedastic_var = self.HeteroscedasticLoss(context, total_loss)
            self.loss["heteroscedastic_var"] = heteroscedastic_var.item()
        else:
            heteroscedastic_loss = total_loss
        heteroscedastic_loss.backward()
        self.loss["heteroscedastic_loss"] = heteroscedastic_loss.item()

        # 更新优化器 分开训练的训练部分不更新
        if self.separate_train and not self.pretrain:
            pass
        else:
            self.uncertainty_mlp_optimizer.step()
            self.context_encoder_optimizer.step()
            if self.algo_type == 'CLASSIFIER': # CLASSIFIER
                self.classifier_optimizer.step()
            if self.algo_type == 'CROO' or self.algo_type == 'UNICORN': # CROO UNICORN
                self.context_decoder_optimizer.step()

    def _update_policy_use_BRAC(self, indices, context):
        """
        更新策略网络和 Q 网络
        使用 BRAC 算法
        """
        obs, actions, rewards, next_obs, terms = self.sample_sac(indices)
        policy_outputs, task_z, task_z_vars = self.agent(obs, context, task_indices=indices)
        new_actions, policy_mean, policy_log_std, log_pi = policy_outputs[:4]

        # flattens out the task dimension
        t, b, _ = obs.size()
        obs = obs.view(t * b, -1)
        actions = actions.view(t * b, -1)
        rewards = rewards.view(t * b, -1)
        next_obs = next_obs.view(t * b, -1)
        terms = terms.view(t * b, -1)

        # ---------------------------------------------------------
        # z0 policy
        # ---------------------------------------------------------

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
        self.loss["log_pi"] = torch.mean(log_pi).item()
        self.loss["policy_mean"] = torch.mean(policy_mean).item()
        self.loss["policy_log_std"] = torch.mean(policy_log_std).item()

        # ---------------------------------------------------------
        # update Q and V networks
        # ---------------------------------------------------------

        # 行为正则项
        c_loss = self._divergence.dual_critic_loss(obs, new_actions.detach(), actions, task_z.detach())
        self.c_optimizer.zero_grad()
        c_loss.backward(retain_graph=True)
        self.c_optimizer.step()
        for _ in range(self._c_iter - 1):
            self._optimize_c(indices=indices, context=context)
        self.loss["c_loss"] = c_loss.item()
        
        # 策略行为的分布偏差
        div_estimate = self._divergence.dual_estimate(obs, new_actions, actions, task_z.detach())
        self.loss["div_estimate"] = torch.mean(div_estimate).item()

        #
        with torch.no_grad():
            if self.use_brac and self.use_value_penalty: # 使用 BRAC 和 value penalty
                target_v_values = self.target_vf(t, b, next_obs, task_z) - self.get_alpha * div_estimate
            else:
                target_v_values = self.target_vf(t, b, next_obs, task_z)
        self.loss["target_v_values"] = torch.mean(target_v_values).item()

        # Q 函数目标值
        rewards = rewards * self.reward_scale
        if self.train_z0_policy:
            if self.use_next_obs_in_context:
                transitions = torch.cat([obs, actions, rewards, next_obs], dim=-1)
            else:
                transitions = torch.cat([obs, actions, rewards], dim=-1)
            transition_heterodastic_var = F.softplus(self.agent.uncertainty_mlp(self.agent.encode_no_mean(transitions))).detach()

            transition_heterodastic_var_flat = transition_heterodastic_var.view(self.batch_size * len(indices), -1)
            var0 = ptu.zeros_like(transition_heterodastic_var_flat)
            transition_heterodastic_var_flat = torch.cat([var0, transition_heterodastic_var_flat], dim=0)
            q_target = rewards + (1. - terms) * self.discount * target_v_values - self.hvar_punish_w * transition_heterodastic_var_flat
        else:
            q_target = rewards + (1. - terms) * self.discount * target_v_values

        # Q 函数的更新
        self.qf1_optimizer.zero_grad()
        self.qf2_optimizer.zero_grad()
        q1_pred = self.qf1(t, b, obs, actions, task_z.detach())
        q2_pred = self.qf2(t, b, obs, actions, task_z.detach())
        qf_loss = torch.mean((q1_pred - q_target) ** 2) + torch.mean((q2_pred - q_target) ** 2)
        qf_loss.backward(retain_graph=True)
        self.qf1_optimizer.step()
        self.qf2_optimizer.step()
        self.loss["qf_loss"] = qf_loss.item()
        self.loss["q_target"] = torch.mean(q_target).item()
        self.loss["q1_pred"] = torch.mean(q1_pred).item()
        self.loss["q2_pred"] = torch.mean(q2_pred).item()

        # if self.algo_type == 'IDAQ': # TODO 是放在这里还是放在预训练部分
        #     model_loss = self.Reward_Prediction_loss(indices, task_z)
        #     self.loss["reward_prediction_loss"] = torch.mean(model_loss).item()
        #     self.reward_models_optimizer.zero_grad()
        #     self.dynamic_models_optimizer.zero_grad()
        #     model_loss.backward()
        #     self.reward_models_optimizer.step()
        #     self.dynamic_models_optimizer.step()

        # V 函数目标值
        min_q_new_actions = self._min_q(t, b, obs, new_actions, task_z.detach())
        if self.max_entropy:
            v_target = min_q_new_actions - log_pi
        else:
            v_target = min_q_new_actions

        # V 函数的更新
        self.vf_optimizer.zero_grad()
        v_pred = self.vf(t, b, obs, task_z.detach()) #  [5120, 2] [5120, 20]
        vf_loss = self.vf_criterion(v_pred, v_target.detach())
        vf_loss.backward(retain_graph=True)
        self.vf_optimizer.step()
        self._update_target_network()
        self.loss["vf_loss"] = vf_loss.item()
        self.loss["v_target"] = torch.mean(v_target).item()
        self.loss["v_pred"] = torch.mean(v_pred).item()

        # n.b. policy update includes dQ/da
        log_policy_target = min_q_new_actions

        # ---------------------------------------------------------
        # Update policy
        # ---------------------------------------------------------

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

    def _update_policy_use_TD3BC(self, indices, context):
        """
        更新策略网络和 Q 网络
        使用 TD3BC 算法
        """
        obs, actions, rewards, next_obs, terms = self.sample_sac(indices)

        policy_outputs, task_z, task_z_vars = self.agent(obs, context, task_indices=indices)
        new_actions, policy_mean, policy_log_std, log_pi = policy_outputs[:4]

        next_policy_outputs, next_task_z, next_task_z_vars = self.agent(next_obs, context, task_indices=indices)
        next_new_actions, _, _, _ = next_policy_outputs[:4]

        # flattens out the task dimension
        t, b, _ = obs.size()
        obs = obs.view(t * b, -1)
        actions = actions.view(t * b, -1)
        rewards = rewards.view(t * b, -1)
        next_obs = next_obs.view(t * b, -1)
        terms = terms.view(t * b, -1)

        # ---------------------------------------------------------
        # z0 policy
        # ---------------------------------------------------------

        if self.train_z0_policy:
            # get \pi(a|s,z0)
            z0 = ptu.zeros_like(task_z)
            z0_in = torch.cat([obs, z0], dim=-1)
            z0_actions, z0_policy_mean, z0_log_std, z0_log_pi = self.agent.policy(t, b, z0_in, reparameterize=True, return_log_prob=True)[:4]

            new_actions = torch.cat([new_actions, z0_actions], dim=0)
            policy_mean = torch.cat([policy_mean, z0_policy_mean], dim=0)
            policy_log_std = torch.cat([policy_log_std, z0_log_std], dim=0)
            log_pi = torch.cat([log_pi, z0_log_pi], dim=0)

            next_new_actions = torch.cat([next_new_actions, z0_actions], dim=0)

            # cat z0 policy
            obs = obs.repeat(2, 1)
            rewards = rewards.repeat(2, 1)
            actions = actions.repeat(2, 1)
            next_obs = next_obs.repeat(2, 1)
            terms = terms.repeat(2, 1)
            task_z = torch.cat([task_z, z0], dim=0)
        self.loss["log_pi"] = torch.mean(log_pi).item()
        self.loss["policy_mean"] = torch.mean(policy_mean).item()
        self.loss["policy_log_std"] = torch.mean(policy_log_std).item()

        # ---------------------------------------------------------
        # update Q networks
        # ---------------------------------------------------------
        
        noise = (torch.randn_like(next_new_actions)*0.2).clamp(-0.5, 0.5)
        next_new_actions = (next_new_actions + noise).clamp(-1.0, 1.0)

        # Q 函数目标值
        q1_next = self.target_qf1(t, b, next_obs, next_new_actions, task_z.detach())
        q2_next = self.target_qf2(t, b, next_obs, next_new_actions, task_z.detach())
        min_next_q_target = torch.min(q1_next, q2_next)
        q_target = rewards + (1. - terms) * self.gamma * min_next_q_target

        # Q 函数的更新
        self.qf1_optimizer.zero_grad()
        self.qf2_optimizer.zero_grad()
        q1_pred = self.qf1(t, b, obs, actions, task_z.detach())
        q2_pred = self.qf2(t, b, obs, actions, task_z.detach())
        qf1_loss = torch.mean((q1_pred - q_target) ** 2) # TD error
        qf2_loss = torch.mean((q2_pred - q_target) ** 2)
        qf1_loss.backward(retain_graph=True)
        qf2_loss.backward(retain_graph=True)
        self.loss["qf_loss"] = qf1_loss.item()
        self.loss["q_target"] = torch.mean(q_target).item()
        self.loss["q1_pred"] = torch.mean(q1_pred).item()
        self.loss["q2_pred"] = torch.mean(q2_pred).item()
        self.qf1_optimizer.step()
        self.qf2_optimizer.step()
        self._update_target_network()

        # ---------------------------------------------------------
        # Update policy
        # ---------------------------------------------------------

        self.step_to_update_policy -= 1
        if self.step_to_update_policy < 1:
            self.step_to_update_policy = self.delay_frequency
            # bug fixed: this should be after the q function update
            # compute min Q on the new actions
            min_q_new_actions = self._min_q(t, b, obs, new_actions, task_z.detach())
            lmbda = self.bc_weight/min_q_new_actions.abs().mean().detach()
            bc_loss = F.mse_loss(new_actions, actions)
            policy_loss = -lmbda * min_q_new_actions.mean() + bc_loss

            # update policy network
            self.policy_optimizer.zero_grad()
            policy_loss.backward(retain_graph=True)
            self.policy_optimizer.step()
            self.loss["policy_loss"] = policy_loss.item()

    def get_epoch_snapshot(self, epoch):
        # NOTE: overriding parent method which also optionally saves the env
        if self.policy_update_strategy == 'BRAC':
            snapshot = OrderedDict(
                qf1=self.qf1.state_dict(),
                qf2=self.qf2.state_dict(),
                vf=self.vf.state_dict(),
                target_vf=self.target_vf.state_dict(),
                c=self.c.state_dict(),
                club_model=self.club_model.state_dict(),
                context_decoder=self.context_decoder.state_dict(),
                classifier=self.classifier.state_dict(),
                policy=self.agent.policy.state_dict(),
                uncertainty_mlp=self.agent.uncertainty_mlp.state_dict(),
                context_encoder=self.agent.context_encoder.state_dict(),

                )
        elif self.policy_update_strategy == 'TD3BC':
            snapshot = OrderedDict(
                qf1=self.qf1.state_dict(),
                qf2=self.qf2.state_dict(),
                target_qf1=self.target_qf1.state_dict(),
                target_qf2=self.target_qf2.state_dict(),
                club_model=self.club_model.state_dict(),
                context_decoder=self.context_decoder.state_dict(),
                classifier=self.classifier.state_dict(),
                policy=self.agent.policy.state_dict(),
                uncertainty_mlp=self.agent.uncertainty_mlp.state_dict(),
                context_encoder=self.agent.context_encoder.state_dict(),
                reward_models=[reward_model.state_dict() for reward_model in self.reward_models],
                dynamic_models=[dynamic_model.state_dict() for dynamic_model in self.dynamic_models],
                )
        else:
            raise NotImplementedError
        return snapshot