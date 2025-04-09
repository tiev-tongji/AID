import abc
from collections import OrderedDict
import time
import os
import glob
import gtimer as gt
import numpy as np
import random

from rlkit.core import logger, eval_util
from rlkit.data_management.env_replay_buffer import MultiTaskReplayBuffer
from rlkit.data_management.path_builder import PathBuilder
from rlkit.samplers.in_place import InPlacePathSampler, OfflineInPlacePathSampler
from rlkit.torch import pytorch_util as ptu
import pdb
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.lines import Line2D
import seaborn as sns
import torch
import torch.nn.functional as F
from pathlib import Path

from line_profiler import LineProfiler
import atexit

profile = LineProfiler()
atexit.register(profile.print_stats)

class OfflineMetaRLAlgorithm(metaclass=abc.ABCMeta):
    def __init__(
            self,
            env,
            agent,
            train_tasks,
            eval_tasks,
            goal_radius,
            eval_deterministic=False,
            render=False,
            render_eval_paths=False,
            plotter=None,
            **kwargs
    ):
        """
        :param env: training env
        :param agent: agent that is conditioned on a latent variable z that rl_algorithm is responsible for feeding in
        :param train_tasks: list of tasks used for training
        :param eval_tasks: list of tasks used for eval
        :param goal_radius: reward threshold for defining sparse rewards

        see default experiment config file for descriptions of the rest of the arguments
        """
        self.env                             = env
        self.obs_dim = int(np.prod(env.observation_space.shape))
        self.action_dim = int(np.prod(env.action_space.shape))
        self.agent                           = agent
        self.train_tasks                     = train_tasks
        self.eval_tasks                      = eval_tasks
        self.n_tasks                         = len(train_tasks) + len(eval_tasks)
        self.goal_radius                     = goal_radius

        self.offline_data_quality            = kwargs['offline_data_quality']
        self.meta_batch                      = kwargs['meta_batch']
        self.batch_size                      = kwargs['batch_size']
        self.num_iterations                  = kwargs['num_iterations']
        self.num_train_steps_per_itr         = kwargs['num_train_steps_per_itr']
        self.num_tasks_sample                = kwargs['num_tasks_sample']
        self.num_evals                       = kwargs['num_evals']
        self.num_steps_per_eval              = kwargs['num_steps_per_eval']
        self.embedding_batch_size            = kwargs['embedding_batch_size']
        self.embedding_mini_batch_size       = kwargs['embedding_mini_batch_size']
        self.max_path_length                 = kwargs['max_path_length']
        self.discount                        = kwargs['discount']

        self.replay_buffer_size              = kwargs['replay_buffer_size']
        self.reward_scale                    = kwargs['reward_scale']
        self.update_post_train               = kwargs['update_post_train']
        self.num_exp_traj_eval               = kwargs['num_exp_traj_eval']
        self.save_replay_buffer              = kwargs['save_replay_buffer']
        self.save_algorithm                  = kwargs['save_algorithm']
        self.save_environment                = kwargs['save_environment']
        self.data_dir                        = kwargs['data_dir']
        self.train_epoch                     = kwargs['train_epoch']
        self.eval_epoch                      = kwargs['eval_epoch']
        self.sample                          = kwargs['sample']
        self.n_trj                           = kwargs['n_trj']
        self.allow_eval                      = kwargs['allow_eval']
        self.mb_replace                      = kwargs['mb_replace']
        # self.use_FOCAL_cl                    = kwargs['use_FOCAL_cl']
        # self.use_club                        = kwargs['use_club']
        # self.club_model_loss_weight          = kwargs['club_model_loss_weight']
        # self.club_loss_weight                = kwargs['club_loss_weight']
        self.train_z0_policy                 = kwargs['train_z0_policy']
        self.separate_train                  = kwargs['separate_train']
        self.pretrain                        = kwargs['pretrain']


        self.is_onlineadapt_thres            = kwargs['is_onlineadapt_thres'] # True  return-based
        self.is_onlineadapt_max              = kwargs['is_onlineadapt_max']   # False
        self.r_thres                         = kwargs['r_thres']              # 0.3
        self.onlineadapt_max_num_candidates  = kwargs['onlineadapt_max_num_candidates']

        self.eval_deterministic              = eval_deterministic
        if kwargs.get('eval_deterministic') is not None:
            eval_deterministic = kwargs.get('eval_deterministic')
        self.render                          = render
        self.eval_statistics                 = None
        self.render_eval_paths               = render_eval_paths
        self.plotter                         = plotter
        
        self.eval_buffer       = MultiTaskReplayBuffer(self.replay_buffer_size, env, self.eval_tasks,  self.goal_radius)
        self.replay_buffer     = MultiTaskReplayBuffer(self.replay_buffer_size, env, self.train_tasks, self.goal_radius)
        # offline sampler which samples from the train/eval buffer
        self.offline_sampler   = OfflineInPlacePathSampler(env=env, policy=agent, max_path_length=self.max_path_length)
        # online sampler for evaluation (if collect on-policy context, for offline context, use self.offline_sampler)
        self.sampler           = InPlacePathSampler(env=env, policy=agent, max_path_length=self.max_path_length)

        self._n_env_steps_total     = 0
        self._n_train_steps_total   = 0
        self._n_rollouts_total      = 0
        self._do_train_time         = 0
        self._epoch_start_time      = None
        self._algo_start_time       = None
        self._old_table_keys        = None
        self.tb_writer              = None
        self._current_path_builder  = PathBuilder()
        self._exploration_paths     = []
        self.init_buffer()

    def init_buffer(self):
        train_trj_paths = []
        eval_trj_paths = []
        # trj entry format: [obs, action, reward, new_obs]
        for i in range(self.n_tasks):
            goal_i_dir = Path(self.data_dir) / f"goal_idx{i}"
            quality_steps = np.array(sorted(list(set([int(trj_path.stem.split('step')[-1]) for trj_path in goal_i_dir.rglob('trj_evalsample*_step*.npy')]))))
            low_quality_steps, mid_quality_steps, high_quality_steps = np.array_split(quality_steps, 3)
            if self.offline_data_quality == 'low':
                training_date_steps = low_quality_steps
            elif self.offline_data_quality == 'mid':
                training_date_steps = mid_quality_steps
            elif self.offline_data_quality == 'expert':
                training_date_steps = high_quality_steps[-1:]
            else:
                training_date_steps = quality_steps
            for j in training_date_steps:
                for k in range(self.n_trj):
                    train_trj_paths += [os.path.join(self.data_dir, f"goal_idx{i}", f"trj_evalsample{k}_step{j}.npy")]
                    eval_trj_paths += [os.path.join(self.data_dir, f"goal_idx{i}", f"trj_evalsample{k}_step{j}.npy")]
        
        train_paths = [train_trj_path for train_trj_path in train_trj_paths if
                       int(train_trj_path.split('/')[-2].split('goal_idx')[-1]) in self.train_tasks]
        train_task_idxs = [int(train_trj_path.split('/')[-2].split('goal_idx')[-1]) for train_trj_path in train_trj_paths if
                       int(train_trj_path.split('/')[-2].split('goal_idx')[-1]) in self.train_tasks]
        eval_paths = [eval_trj_path for eval_trj_path in eval_trj_paths if
                      int(eval_trj_path.split('/')[-2].split('goal_idx')[-1]) in self.eval_tasks]
        eval_task_idxs = [int(eval_trj_path.split('/')[-2].split('goal_idx')[-1]) for eval_trj_path in eval_trj_paths if
                          int(eval_trj_path.split('/')[-2].split('goal_idx')[-1]) in self.eval_tasks]

        obs_train_lst = []
        action_train_lst = []
        reward_train_lst = []
        next_obs_train_lst = []
        terminal_train_lst = []
        task_train_lst = []
        obs_eval_lst = []
        action_eval_lst = []
        reward_eval_lst = []
        next_obs_eval_lst = []
        terminal_eval_lst = []
        task_eval_lst = []

        for train_path, train_task_idx in zip(train_paths, train_task_idxs):
            trj_npy = np.load(train_path, allow_pickle=True)
            obs, action, reward, next_obs = np.array_split(trj_npy, [self.obs_dim, self.obs_dim+self.action_dim, -self.obs_dim], axis=-1)
            obs_train_lst += list(obs)
            action_train_lst += list(action)
            reward_train_lst += list(reward)
            next_obs_train_lst += list(next_obs)
            terminal = [0 for _ in range(trj_npy.shape[0])]
            terminal[-1] = 1
            terminal_train_lst += terminal
            task_train = [train_task_idx for _ in range(trj_npy.shape[0])]
            task_train_lst += task_train
        for eval_path, eval_task_idx in zip(eval_paths, eval_task_idxs):
            trj_npy = np.load(eval_path, allow_pickle=True)
            obs, action, reward, next_obs = np.array_split(trj_npy, [self.obs_dim, self.obs_dim+self.action_dim, -self.obs_dim], axis=-1)
            obs_eval_lst += list(obs)
            action_eval_lst += list(action)
            reward_eval_lst += list(reward)
            next_obs_eval_lst += list(next_obs)
            terminal = [0 for _ in range(trj_npy.shape[0])]
            terminal[-1] = 1
            terminal_eval_lst += terminal
            task_eval = [eval_task_idx for _ in range(trj_npy.shape[0])]
            task_eval_lst += task_eval


        # load training buffer
        for i, (
                task_train,
                obs,
                action,
                reward,
                next_obs,
                terminal,
        ) in enumerate(zip(
            task_train_lst,
            obs_train_lst,
            action_train_lst,
            reward_train_lst,
            next_obs_train_lst,
            terminal_train_lst,
        )):
            self.replay_buffer.add_sample(
                task_train,
                obs,
                action,
                reward,
                terminal,
                next_obs,
                **{'env_info': {}},
            )

        # load evaluation buffer
        for i, (
                task_eval,
                obs,
                action,
                reward,
                next_obs,
                terminal,
        ) in enumerate(zip(
            task_eval_lst,
            obs_eval_lst,
            action_eval_lst,
            reward_eval_lst,
            next_obs_eval_lst,
            terminal_eval_lst,
        )):
            self.eval_buffer.add_sample(
                task_eval,
                obs,
                action,
                reward,
                terminal,
                next_obs,
                **{'env_info': {}},
            )

    def _try_to_eval(self, epoch):
        if self._can_evaluate():
            self.evaluate(epoch)
            table_keys = logger.get_table_key_set()
            # if self._old_table_keys is not None:
            #     assert table_keys == self._old_table_keys, (
            #         "Table keys cannot change from iteration to iteration."
            #     )
            self._old_table_keys = table_keys
            logger.record_tabular("Number of train steps total", self._n_train_steps_total)
            logger.record_tabular("Number of env steps total",   self._n_env_steps_total)
            logger.record_tabular("Number of rollouts total",    self._n_rollouts_total)

            times_itrs  = gt.get_times().stamps.itrs
            train_time  = times_itrs['train'][-1]
            eval_time   = times_itrs['eval'][-1] if epoch > 0 else 0
            epoch_time  = train_time + eval_time
            total_time  = gt.get_times().total

            logger.record_tabular('Train Time (s)', train_time)
            logger.record_tabular('(Previous) Eval Time (s)', eval_time)
            logger.record_tabular('Epoch Time (s)', epoch_time)
            logger.record_tabular('Total Train Time (s)', total_time)

            logger.record_tabular("Epoch", epoch)
            logger.dump_tabular(with_prefix=False, with_timestamp=False, tb_writer = self.tb_writer)
        else:
            logger.log("Skipping eval for now.")

    def _can_evaluate(self):
        """
        One annoying thing about the logger table is that the keys at each
        iteration need to be the exact same. So unless you can compute
        everything, skip evaluation.

        A common example for why you might want to skip evaluation is that at
        the beginning of training, you may not have enough data for a
        validation and training set.

        :return:
        """
        # eval collects its own context, so can eval any time
        return True
    
    def _can_train(self):
        return all([self.replay_buffer.num_steps_can_sample(idx) >= self.batch_size for idx in self.train_tasks])

    def _get_action_and_info(self, agent, observation):
        """
        Get an action to take in the environment.
        :param observation:
        :return:
        """
        agent.set_num_steps_total(self._n_env_steps_total)
        return agent.get_action(observation,)

    def _start_epoch(self, epoch):
        self._epoch_start_time = time.time()
        self._exploration_paths = []
        self._do_train_time = 0
        logger.push_prefix('Iteration #%d | ' % epoch)

    def _end_epoch(self):
        logger.log("Epoch Duration: {0}".format(
            time.time() - self._epoch_start_time
        ))
        logger.log("Started Training: {0}".format(self._can_train()))
        logger.pop_prefix()

    ##### Snapshotting utils #####
    def get_epoch_snapshot(self, epoch):
        data_to_save = dict(
            epoch=epoch,
            exploration_policy=self.exploration_policy,
        )
        if self.save_environment:
            data_to_save['env'] = self.training_env
        return data_to_save

    def get_extra_data_to_save(self, epoch):
        """
        Save things that shouldn't be saved every snapshot but rather
        overwritten every time.
        :param epoch:
        :return:
        """
        if self.render:
            self.training_env.render(close=True)
        data_to_save = dict(
            epoch=epoch,
        )
        if self.save_environment:
            data_to_save['env'] = self.training_env
        if self.save_replay_buffer:
            data_to_save['replay_buffer'] = self.replay_buffer
        if self.save_algorithm:
            data_to_save['algorithm'] = self
        return data_to_save

    def _do_eval(self, indices, epoch, buffer, num_shots=2):
        online_final_returns = []
        online_final_returns_per_shot = [[] for _ in range(num_shots)]
        online_all_return = []
        for idx in indices:
            all_rets = []  # 存放所有评估轮次的平均返回（多轨迹的平均值）
            all_rets_per_shot = [[] for _ in range(num_shots)]  # 每条轨迹单独存放返回
            for r in range(self.num_evals):
                paths = self.collect_online_paths(idx, epoch, r, buffer, num_shots = num_shots)
                shot_returns = [eval_util.get_average_returns([p]) for p in paths]
                all_rets.append([np.mean(shot_returns)])
                for i, ret in enumerate(shot_returns):
                    all_rets_per_shot[i].append([ret])
            online_final_returns.append(np.mean([a[-1] for a in all_rets]))
            for i in range(num_shots):
                online_final_returns_per_shot[i].append(np.mean([a[-1] for a in all_rets_per_shot[i]]))
            # record all returns for the first n trajectories
            n = min([len(a) for a in all_rets])
            all_rets = [a[:n] for a in all_rets]
            online_all_return.append(np.mean(np.stack(all_rets), axis=0))
        n = min([len(t) for t in online_all_return])
        online_all_return = [t[:n] for t in online_all_return]

        online_random_final_returns = []
        online_random_final_returns_per_shot = [[] for _ in range(num_shots)]
        online_random_all_return = []
        for idx in indices:
            all_rets = []
            all_rets_per_shot = [[] for _ in range(num_shots)]
            for r in range(self.num_evals):
                paths = self.collect_random_online_paths(idx, epoch, r, buffer, num_shots = num_shots)
                shot_returns = [eval_util.get_average_returns([p]) for p in paths]
                all_rets.append([np.mean(shot_returns)])
                for i, ret in enumerate(shot_returns):
                    all_rets_per_shot[i].append([ret])
            online_random_final_returns.append(np.mean([a[-1] for a in all_rets]))
            for i in range(num_shots):
                online_random_final_returns_per_shot[i].append(np.mean([a[-1] for a in all_rets_per_shot[i]]))
            n = min([len(a) for a in all_rets])
            all_rets = [a[:n] for a in all_rets]
            online_random_all_return.append(np.mean(np.stack(all_rets), axis=0))
        n = min([len(t) for t in online_random_all_return])
        online_random_all_return = [t[:n] for t in online_random_all_return]

        offline_final_returns = []
        offline_all_return = []
        for idx in indices:
            all_rets = []
            for r in range(self.num_evals):
                paths = self.collect_offline_paths(idx, epoch, r, buffer)
                all_rets.append([eval_util.get_average_returns([p]) for p in paths])
            offline_final_returns.append(np.mean([a[-1] for a in all_rets]))
            # record all returns for the first n trajectories
            n = min([len(a) for a in all_rets])
            all_rets = [a[:n] for a in all_rets]
            all_rets = np.mean(np.stack(all_rets), axis=0) # avg return per nth rollout
            offline_all_return.append(all_rets)
        n = min([len(t) for t in offline_all_return])
        offline_all_return = [t[:n] for t in offline_all_return]

        return (online_final_returns, online_final_returns_per_shot, online_all_return,
                    online_random_final_returns, online_random_final_returns_per_shot, online_random_all_return,
                    offline_final_returns, offline_all_return)
    
    def _do_eval_z_random_switch(self, indices, epoch, buffer):
        online_final_returns = []
        online_final_returns_first = []
        online_final_returns_second = []
        online_all_return = []
        for idx in indices:
            all_rets = []
            all_rets_first = []
            all_rets_second = []
            for r in range(self.num_evals):
                paths = self.collect_z_random_switch_online_paths(idx, epoch, r, buffer)
                paths_first, paths_second = paths[0], paths[1]
                # all_rets.append([eval_util.get_average_returns([p]) for p in paths])
                all_rets_first.append([eval_util.get_average_returns([paths_first])])
                all_rets_second.append([eval_util.get_average_returns([paths_second])])
                all_rets.append([(eval_util.get_average_returns([paths_first]) + eval_util.get_average_returns([paths_second]))/2])
            online_final_returns.append(np.mean([a[-1] for a in all_rets]))
            online_final_returns_first.append(np.mean([a[-1] for a in all_rets_first]))
            online_final_returns_second.append(np.mean([a[-1] for a in all_rets_second]))
            # record all returns for the first n trajectories
            n = min([len(a) for a in all_rets])
            all_rets = [a[:n] for a in all_rets]
            all_rets = np.mean(np.stack(all_rets), axis=0) # avg return per nth rollout
            online_all_return.append(all_rets)
        n = min([len(t) for t in online_all_return])
        online_all_return = [t[:n] for t in online_all_return]

        return online_final_returns, [online_final_returns_first, online_final_returns_second], online_all_return
    
    # @profile
    def train(self, tb_writer):
        '''
        meta-training loop
        '''
        self.tb_writer = tb_writer
        params = self.get_epoch_snapshot(-1)
        logger.save_itr_params(-1, params)
        gt.reset()
        gt.set_def_unique(False)
        self._current_path_builder = PathBuilder()

        # at each iteration, we first collect data from tasks, perform meta-updates, then try to evaluate
        for it_ in gt.timed_for(range(self.num_iterations), save_itrs=True):
            self._start_epoch(it_)
            self.training_mode(True)
            
            indices_lst = []
            z_means_lst = []
            z_vars_lst = []
            # Sample train tasks and compute gradient updates on parameters.
            for train_step in range(self.num_train_steps_per_itr):
                indices = np.random.choice(self.train_tasks, self.meta_batch, replace=self.mb_replace)
                z_means, z_vars = self._do_training(indices)
                indices_lst.append(indices)
                z_means_lst.append(z_means)
                z_vars_lst.append(z_vars)
                self._n_train_steps_total += 1

            indices = np.concatenate(indices_lst)
            z_means = np.concatenate(z_means_lst)
            z_vars = np.concatenate(z_vars_lst)
            data_dict = self.data_dict(indices, z_means, z_vars)
            gt.stamp('train')
            self.training_mode(False)
            # eval
            params = self.get_epoch_snapshot(it_)
            logger.save_itr_params(it_, params)

            if self.allow_eval:
                self._try_to_eval(it_)
                gt.stamp('eval')
            self._end_epoch()

    def data_dict(self, indices, z_means, z_vars):
        data_dict = {}
        data_dict['task_idx'] = indices
        for i in range(z_means.shape[1]):
            data_dict['z_means%d' %i] = list(z_means[:, i])
        for i in range(z_vars.shape[1]):
            data_dict['z_vars%d' % i] = list(z_vars[:, i])
        return data_dict

    def evaluate(self, epoch):
        if self.eval_statistics is None:
            self.eval_statistics = OrderedDict()
        
        ### prepare z of training tasks
        self.trained_z = {}
        self.trained_z_sample = {}
        for idx in self.train_tasks:
            context = self.sample_context(idx)
            self.agent.infer_posterior(context)
            self.trained_z[idx] = (self.agent.z_means, self.agent.z_vars)
            self.trained_z_sample[idx] = self.agent.z

        ### test tasks
        eval_util.dprint('evaluating on {} test tasks'.format(len(self.eval_tasks)))
        test_online_final_returns_avg, test_online_final_returns, test_online_all_returns,   test_online_random_final_returns_avg, test_online_random_final_returns, test_online_random_all_returns,   test_offline_final_returns, test_offline_all_returns = self._do_eval(self.eval_tasks, epoch, buffer=self.eval_buffer)
        train_online_final_returns_avg, train_online_final_returns, train_online_all_returns,   train_online_random_final_returns_avg, train_online_random_final_returns, train_online_random_all_returns,   train_offline_final_returns, train_offline_all_returns = self._do_eval(self.train_tasks, epoch, buffer=self.replay_buffer)
        eval_util.dprint('test online all returns')
        eval_util.dprint(test_online_all_returns)
        eval_util.dprint('test online_random all returns')
        eval_util.dprint(test_online_random_all_returns)
        eval_util.dprint('test offline all returns')
        eval_util.dprint(test_offline_all_returns)

        # save the final posterior
        self.agent.log_diagnostics(self.eval_statistics)
        
        test_online_final_returns_fir, test_online_final_returns_sec = test_online_final_returns
        avg_test_online_final_return = np.mean(test_online_final_returns_avg)
        fir_test_online_final_return = np.mean(test_online_final_returns_fir)
        sec_test_online_final_return = np.mean(test_online_final_returns_sec)
        test_online_random_final_returns_fir, test_online_random_final_returns_sec = test_online_random_final_returns
        avg_test_online_random_final_return = np.mean(test_online_random_final_returns_avg)
        fir_test_online_random_final_return = np.mean(test_online_random_final_returns_fir)
        sec_test_online_random_final_return = np.mean(test_online_random_final_returns_sec)
        avg_test_offline_final_return = np.mean(test_offline_final_returns)

        train_online_final_returns_fir, train_online_final_returns_sec = train_online_final_returns
        avg_train_online_final_return = np.mean(train_online_final_returns_avg)
        fir_train_online_final_return = np.mean(train_online_final_returns_fir)
        sec_train_online_final_return = np.mean(train_online_final_returns_sec)
        train_online_random_final_returns_fir, train_online_random_final_returns_sec = train_online_random_final_returns
        avg_train_online_random_final_return = np.mean(train_online_random_final_returns_avg)
        fir_train_online_random_final_return = np.mean(train_online_random_final_returns_fir)
        sec_train_online_random_final_return = np.mean(train_online_random_final_returns_sec)
        avg_train_offline_final_return = np.mean(train_offline_final_returns)
        
        avg_test_online_all_return = np.mean(np.stack(test_online_all_returns), axis=0)
        avg_test_online_random_all_return = np.mean(np.stack(test_online_random_all_returns), axis=0)
        avg_test_offline_all_return = np.mean(np.stack(test_offline_all_returns), axis=0)
        
        avg_train_online_all_return = np.mean(np.stack(train_online_all_returns), axis=0)
        avg_train_online_random_all_return = np.mean(np.stack(train_online_random_all_returns), axis=0)
        avg_train_offline_all_return = np.mean(np.stack(train_offline_all_returns), axis=0)
            
        self.eval_statistics['first_OnlineReturn_all_test_tasks'] = fir_test_online_final_return
        self.eval_statistics['second_OnlineReturn_all_test_tasks'] = sec_test_online_final_return
        self.eval_statistics['first_OnlineReturn_all_test_tasks_random'] = fir_test_online_random_final_return
        self.eval_statistics['second_OnlineReturn_all_test_tasks_random'] = sec_test_online_random_final_return
        self.eval_statistics['Average_OfflineReturn_all_test_tasks'] = avg_test_offline_final_return

        self.eval_statistics['first_OnlineReturn_all_train_tasks'] = fir_train_online_final_return
        self.eval_statistics['second_OnlineReturn_all_train_tasks'] = sec_train_online_final_return
        self.eval_statistics['first_OnlineReturn_all_train_tasks_random'] = fir_train_online_random_final_return
        self.eval_statistics['second_OnlineReturn_all_train_tasks_random'] = sec_train_online_random_final_return
        self.eval_statistics['Average_OfflineReturn_all_train_tasks'] = avg_train_offline_final_return

        self.loss['avg_test_online_final_return'] = avg_test_online_final_return
        self.loss['avg_test_online_random_final_return'] = avg_test_online_random_final_return
        self.loss['avg_test_offline_final_return'] = avg_test_offline_final_return
        self.loss['avg_train_online_final_return'] = avg_train_online_final_return  
        self.loss['avg_train_online_random_final_return'] = avg_train_online_random_final_return
        self.loss['avg_train_offline_final_return'] = avg_train_offline_final_return

        self.loss['avg_test_online_all_return'] = np.mean(avg_test_online_all_return)
        self.loss['avg_test_offline_all_return'] = np.mean(avg_test_offline_all_return)
        self.loss['avg_test_online_random_all_return'] = np.mean(avg_test_online_random_all_return)
        self.loss['avg_train_online_all_return'] = np.mean(avg_train_online_all_return)
        self.loss['avg_train_online_random_all_return'] = np.mean(avg_train_online_random_all_return)
        self.loss['avg_train_offline_all_return'] = np.mean(avg_train_offline_all_return)

        for key, value in self.eval_statistics.items():
            logger.record_tabular(key, value)
        self.eval_statistics = None

        if self.plotter:
            self.plotter.draw()
        # Plot T-SNE at the end of training
        if epoch != (self.num_iterations - 1):
            return
        if self.separate_train and not self.pretrain:
            if 'point-robot' in self.env_name:
                self.draw_path(epoch, logger._snapshot_dir)
        if self.seed == 0:
            self.draw_tsne(epoch, logger._snapshot_dir)
    
    ##### draw #####
    def draw_tsne(self, epoch, logdir=None):
        # sample offline context zs
        fig_save_dir = logdir + '/figures'
        if not os.path.exists(fig_save_dir):
            os.makedirs(fig_save_dir)
        n_points = 100
        offline_train_zs, offline_test_zs, online_train_zs, online_test_zs = [], [], [], []
        for i in range(n_points):
            print(f'Clollect {i} traj...')
            offline_train_zs.append(self.collect_offline_zs(self.train_tasks, self.replay_buffer))
            offline_test_zs.append(self.collect_offline_zs(self.eval_tasks, self.eval_buffer))
            online_train_zs.append(self.collect_online_zs(self.train_tasks))
            online_test_zs.append(self.collect_online_zs(self.eval_tasks))
        offline_train_zs = np.stack(offline_train_zs, axis=1)
        offline_test_zs = np.stack(offline_test_zs, axis=1)
        online_train_zs = np.stack(online_train_zs, axis=1)
        online_test_zs = np.stack(online_test_zs, axis=1)
            
        self.vis_task_embeddings(save_dir = fig_save_dir, fig_name=f'offline_train_zs_{epoch}.png', zs=[offline_train_zs], subplot_title_lst=[f'offline_train_zs_{epoch}'])
        self.vis_task_embeddings(save_dir = fig_save_dir, fig_name=f'offline_test_zs_{epoch}.png', zs=[offline_test_zs], subplot_title_lst=[f'offline_test_zs_{epoch}'])
        self.vis_task_embeddings(save_dir = fig_save_dir, fig_name=f'online_train_zs_{epoch}.png', zs=[online_train_zs], subplot_title_lst=[f'online_train_zs_{epoch}'])
        self.vis_task_embeddings(save_dir = fig_save_dir, fig_name=f'online_test_zs_{epoch}.png', zs=[online_test_zs], subplot_title_lst=[f'online_test_zs_{epoch}'])
        # self.vis_task_embeddings(save_dir = fig_save_dir, fig_name=f'online_train_np_zs_{epoch}.png', zs=[online_train_np_zs], subplot_title_lst=[f'online_train_np_zs_{epoch}'])
        # self.vis_task_embeddings(save_dir = fig_save_dir, fig_name=f'online_test_np_zs_{epoch}.png', zs=[online_test_np_zs], subplot_title_lst=[f'online_test_np_zs_{epoch}'])

    def draw_heatmap(self, matrix, task_labels, title, save_path, figsize=(16, 12)):
        """
        绘制热力图并保存图片

        参数：
            matrix: 需要绘制热力图的数据矩阵
            task_labels: 热力图的行和列标签
            title: 热力图标题
            save_path: 图片保存路径
            figsize: 图片大小（默认为 (16,12)）
        """
        plt.figure(figsize=figsize)
        ax = sns.heatmap(matrix, annot=True, fmt=".2f", cmap='viridis',
                         xticklabels=task_labels, yticklabels=task_labels,
                         annot_kws={"size": 8})
        plt.title(title, fontsize=12)
        plt.xlabel('Latent Dimension', fontsize=10)
        plt.ylabel('Task', fontsize=10)
        plt.xticks(fontsize=8)
        plt.yticks(fontsize=8)
        plt.tight_layout()
        plt.savefig(save_path, dpi=400, bbox_inches='tight')
        plt.close()

    def compute_distance_matrix(self, z_array, distance_metric='euclidean'):
        """
        计算给定 z_array 的距离矩阵
        参数：
            z_array: numpy 数组，形状为 (n_tasks, latent_dim)
            distance_metric: 距离计算方法，可选 'euclidean', 'manhattan', 'cosine'
        返回：
            dist_matrix: 形状为 (n_tasks, n_tasks) 的距离矩阵
        """
        n_tasks = z_array.shape[0]
        dist_matrix = np.zeros((n_tasks, n_tasks))

        for i in range(n_tasks):
            for j in range(n_tasks):
                if distance_metric == 'euclidean':
                    dist_matrix[i, j] = np.linalg.norm(z_array[i] - z_array[j])
                elif distance_metric == 'manhattan':
                    dist_matrix[i, j] = np.sum(np.abs(z_array[i] - z_array[j]))
                elif distance_metric == 'cosine':
                    dot_val = np.dot(z_array[i], z_array[j])
                    norm_i = np.linalg.norm(z_array[i])
                    norm_j = np.linalg.norm(z_array[j])
                    cos_sim = dot_val / (norm_i * norm_j + 1e-8)
                    dist_matrix[i, j] = 1.0 - cos_sim  # 余弦距离
                else:
                    raise ValueError(f"Unknown distance metric: {distance_metric}")

        return dist_matrix

    def draw_z_trajectory(self, task_labels, z_weighted_list, z_mean_list, latent_dim, save_path, figsize=(16, 10)):
        """
        绘制 z_weighted 和 z_mean 随任务变化的曲线图，并保存图像

        参数:
            task_labels: 各任务标签列表
            z_weighted_list: 每个任务对应的 z_weighted 值列表，列表内元素为数组（形状：[latent_dim]）
            z_mean_list: 每个任务对应的 z_mean 值列表，列表内元素为数组（形状：[latent_dim]）
            latent_dim: 隐变量的维度
            fig_save_dir: 图像保存的文件夹路径
            epoch: 当前训练的 epoch
            algo_type: 算法类型（用于文件命名）
            figsize: 图像尺寸，默认 (16, 10)
        """
        # 定义颜色映射，确保每个维度都有不同的颜色
        colors = plt.cm.tab20(np.linspace(0, 1, latent_dim, endpoint=False))
        # 将列表转换为 numpy 数组，便于按维度索引
        z_weighted_array = np.array(z_weighted_list)
        z_mean_array = np.array(z_mean_list)

        plt.figure(figsize=figsize)
        for dim in range(latent_dim):
            color = colors[dim]
            # 绘制 z_weighted：虚线 + 圆形 marker
            plt.plot(task_labels, z_weighted_array[:, dim],
                     linestyle='--', marker='o', color=color,
                     label=f'Dim {dim} z_weighted')
            # 绘制 z_mean：实线 + 正方形 marker
            plt.plot(task_labels, z_mean_array[:, dim],
                     linestyle='-', marker='s', color=color,
                     label=f'Dim {dim} z_mean')
        plt.xlabel('task', fontsize=10)
        plt.ylabel('value', fontsize=10)
        plt.xticks(fontsize=8)
        plt.yticks(fontsize=8)
        plt.title('z_weighted and z_mean', fontsize=12)
        plt.legend(fontsize=8, ncol=2)
        plt.grid(True)
        plt.tight_layout()

        plt.savefig(save_path, dpi=400, bbox_inches='tight')
        plt.close()

    def draw_path_plot(self, paths, heterodastic_vars, task, save_path, min_5=None, max_95=None, is_half_circle=True):
        """
        绘制路径图，包括轨迹、散点以及目标和起始点标记，同时添加色条和图例

        参数:
            paths: 包含路径数据的列表，每个元素是一个字典，至少包含 'observations' 和 'goal'
            heterodastic_vars: 与路径对应的不确定性数据列表，每个元素为与 observations 长度相同的数组
            task: 当前任务编号或标签（用于标题及文件名）
            save_path: 保存图像的路径
            min_5: 可选，定义散点颜色映射的下限
            max_95: 可选，定义散点颜色映射的上限
            is_half_circle: 是否绘制半圆（True）还是全圆（False）
        """
        fig = plt.figure(figsize=(20, 12))
        sns.set_theme(style="white", font_scale=2.0)
        ax = plt.gca()

        # 绘制背景圆/半圆
        if is_half_circle:
            circle = patches.Arc((0, 0), 2, 2, angle=0, theta1=0, theta2=180, color=(180./255., 180./255., 180./255.), linewidth=2)
        else:
            circle = patches.Arc((0, 0), 2, 2, angle=0, theta1=0, theta2=360, color=(180./255., 180./255., 180./255.), linewidth=2)
        ax.add_artist(circle)

        # 设置坐标轴范围和比例
        ax.set_xlim(-1.1, 1.1)  # x 轴范围 [-1.1, 1.1]
        if is_half_circle:
            ax.set_ylim(-0.35, 1.2)   # y 轴范围 [0, 1]
        else:
            ax.set_ylim(-1.35, 1.2)  # y 轴范围 [-1.1, 1.1]
        ax.set_aspect('equal', 'box')
        
        ymin, ymax = ax.get_ylim()
        ax.set_ylim(ymin + 0.3, ymax + 0.3)

        # 定义颜色列表，用于不同路径
        colors = [(117./255., 196./255., 175./255.), (249./255., 116./255., 70./255.)]
        marker = ['o', '^', 'v', '<', '>', 'D', 'P', 'X', 'H'] # 分别为圆形、三角形、倒三角形、左三角、右三角、菱形、五角星、X、六角星
        lines_for_legend = []  # 存储图例中需要显示的线条
        save_data = []  # 若需要保存数据，可以使用
        # 根据 heterodastic_vars 的整体值范围确定色条范围
        vmin = min_5 if min_5 is not None else np.min(np.concatenate(heterodastic_vars))
        vmax = max_95 if max_95 is not None else np.max(np.concatenate(heterodastic_vars))

        rewards = []

        # 绘制前两条路径
        for i, path in enumerate(paths[:2]):
            observations = np.array(path['observations'])
            reward = eval_util.get_average_returns([path])
            rewards.append(reward)
                

            # 绘制散点，使用色条表示不确定性
            sc = plt.scatter(observations[:, 0], observations[:, 1],
                             c=heterodastic_vars[i], cmap='coolwarm', vmin=vmin, vmax=vmax,
                             marker=marker[i], label=f'episode {i + 1}', linewidth=4, zorder=2)
            ########
            # lines_for_legend.append(Line2D([0], [0], marker=marker[i], color=(67/255,93/255,200/255),
            #                      linestyle='None', markersize=10,
            #                      label=f'episode {i + 1}'))
            
            save_data.append({
                'observations': observations.tolist(),
                'heterodastic_var': np.array(heterodastic_vars[i]).tolist(),
                'goal': path['goal']
            })
            ########
            # if (i == 0):
            #     continue
            # 绘制轨迹线
            line, = plt.plot(observations[:, 0], observations[:, 1],
                         color=colors[i % len(colors)], linestyle='-',
                         label=f'episode Line {i + 1}', linewidth=4, zorder=1)
            lines_for_legend.append(line)

        # 绘制目标点
        goal = paths[0]['goal']
        goal_point = plt.scatter(
            goal[0], goal[1],
            edgecolors=(0.9, 0.2, 0.4),  # 边框颜色
            facecolors='none',           # 空心标记
            marker='*',
            label='Goal',
            s=300,                       # 标记大小
            linewidths=3,
            zorder=3
        )
        lines_for_legend.append(goal_point)

        # 绘制起始点
        start = (0., 0.)
        start_point = plt.scatter(
            start[0], start[1],
            edgecolors=(0.9, 0.2, 0.4),
            facecolors='none',
            marker='o',
            label='Start',
            s=250,
            linewidths=3,
            zorder=3
        )
        lines_for_legend.append(start_point)

        # 添加色条
        cbar = plt.colorbar(sc, label='Uncertainty', fraction=0.05, pad=0.1,
                            shrink=1.0, aspect=20, orientation='horizontal')
        cbar.ax.tick_params(labelsize=20)

        # 隐藏坐标轴
        plt.axis('off')
        # 添加图例
        plt.legend(handles=lines_for_legend, loc='upper right', fontsize=16)

        # 将第一条和第二条路径的 reward 添加到标题中
        title_text = f"Return of Episode 1: {rewards[0]:.1f}, Return of Episode 2: {rewards[1]:.1f}"
        plt.title(title_text, fontsize=20)

        plt.savefig(save_path, dpi=400, bbox_inches='tight')
        plt.close()

    def draw_z(self, epoch, logdir, is_z_random_switch=False, distance_metric='euclidean'):
        z_weighted_list = []
        z_mean_list = []
        task_labels = []
        fig_save_dir = logdir + f'/figures_itr_{epoch}'
        # fig_save_dir = logdir + '/figures'
        if not os.path.exists(fig_save_dir):
            os.makedirs(fig_save_dir)
    
        for idx in list(self.eval_tasks):
            if is_z_random_switch:
                _, _, z_weighted, z_mean = self.collect_z_random_switch_online_paths(idx, 0, 0, None, return_heterodastic_var=True)
            else:
                _, _, z_weighted, z_mean = self.collect_online_paths(idx, 0, 0, None, return_heterodastic_var=True)
            z_weighted_list.append(z_weighted) # [task, latent_dim]
            z_mean_list.append(z_mean)
            task_labels.append(f'Task {idx}')

        ############################
        # z 可视化
        ############################
        self.draw_z_trajectory(
            task_labels=task_labels,
            z_weighted_list=z_weighted_list,
            z_mean_list=z_mean_list,
            latent_dim=self.latent_dim,
            save_path=os.path.join(fig_save_dir, f'traj_{epoch}_{self.algo_type}_z_weighted_mean.pdf'),
        )
        
        # 计算并绘制 z_weighted 的热力图
        z_weighted_array = np.array(z_weighted_list)
        z_mean_array = np.array(z_mean_list)
        dist_matrix_weighted = self.compute_distance_matrix(z_weighted_array, distance_metric)
        self.draw_heatmap(
            matrix=dist_matrix_weighted,
            task_labels=task_labels,
            title=f'Heatmap of z_weighted across tasks (Epoch {epoch})',
            save_path=os.path.join(fig_save_dir, f'z_weighted_heatmap_{epoch}_{self.algo_type}.pdf')
        )
        # 计算并绘制 z_mean 的热力图
        dist_matrix_mean = self.compute_distance_matrix(z_mean_array, distance_metric)
        self.draw_heatmap(
            matrix=dist_matrix_mean,
            task_labels=task_labels,
            title=f'Heatmap of z_mean across tasks (Epoch {epoch})',
            save_path=os.path.join(fig_save_dir, f'z_mean_heatmap_{epoch}_{self.algo_type}.pdf')
        )

    def draw_path(self, epoch, logdir, min_5 = None, max_95 = None, is_z_random_switch=False):
        fig_save_dir = logdir + f'/figures_itr_{epoch}'
        # fig_save_dir = logdir + '/figures'
        if not os.path.exists(fig_save_dir):
            os.makedirs(fig_save_dir)

        for idx in list(self.eval_tasks):
            if is_z_random_switch:
                paths, heterodastic_vars = self.collect_z_random_switch_online_paths(idx, 0, 0, None, return_heterodastic_var=True)
            else:
                paths, heterodastic_vars = self.collect_online_paths(idx, 0, 0, None, return_heterodastic_var=True)
            # fig, axs = plt.subplots(1, 2, figsize=(24, 10))
            
            save_path = f'{fig_save_dir}/traj_{epoch}_{self.algo_type}_{idx}.pdf'
            self.draw_path_plot(
                paths=paths,
                heterodastic_vars=heterodastic_vars,
                task=idx,        # 当前任务编号或标签
                save_path=save_path,  # 图像保存的文件路径
                min_5=min_5,
                max_95=max_95,
                is_half_circle=True,
            )

    def draw_neg_path(self, epoch, logdir, min_5 = None, max_95 = None, is_z_random_switch=False):
        fig_save_dir = logdir + '/figures_neg'
        if not os.path.exists(fig_save_dir):
            os.makedirs(fig_save_dir)
        for idx in np.arange(0, 4).tolist():
            if is_z_random_switch:
                paths, heterodastic_vars = self.collect_z_random_switch_online_paths(idx, 0, 0, None, return_heterodastic_var=True, neg_task=True)
            else:
                paths, heterodastic_vars = self.collect_online_paths(idx, 0, 0, None, return_heterodastic_var=True, neg_task=True)
            save_path = f'{fig_save_dir}/traj_{epoch}_{self.algo_type}_{idx}_neg.pdf'
            self.draw_path_plot(
                paths=paths,
                heterodastic_vars=heterodastic_vars,
                task=idx,        # 当前任务编号或标签
                save_path=save_path,  # 图像保存的文件路径
                min_5=min_5,
                max_95=max_95,
                is_half_circle=False,
            )
    
    def draw_manual_path(self, epoch, logdir, min_5 = None, max_95 = None, is_z_random_switch=False, distance_metric='euclidean'): 
        fig_save_dir = logdir + '/figures_'
        if not os.path.exists(fig_save_dir):
            os.makedirs(fig_save_dir)
    
        manual_goals = [[np.cos(np.pi/4), np.sin(np.pi/4)],[np.cos(np.pi*3/4), np.sin(np.pi*3/4)],[np.cos(np.pi*3/2), np.sin(np.pi*3/2)]]
        for idx_goal, manual_goal in enumerate(manual_goals):
            z_weighted_list = []
            z_mean_list = []
            task_labels = []
            for idx_task, task in enumerate(list(self.eval_tasks) + list([0, 1, 2, 3])):
                paths, heterodastic_vars, z_weighted, z_mean = self.collect_manual_paths(task, 0, 0, None, return_heterodastic_var=True, manual_goal=manual_goal, neg_task=(idx_task >= len(self.eval_tasks)))
                fig = plt.figure(figsize=(20, 12))
                
                z_weighted_list.append(z_weighted) # [task, latent_dim]
                z_mean_list.append(z_mean)
                if (idx_task < len(self.eval_tasks)):
                    task_labels.append(f'Task {idx_task} Goal {idx_goal}')
                else:
                    task_labels.append(f'Neg Task {idx_task} Goal {idx_goal}')

                ############################
                # Path 可视化
                ############################
                if (idx_task < len(self.eval_tasks)):
                    save_path = f'{fig_save_dir}/traj_{epoch}_{self.algo_type}_task{task}_goal{idx_goal}.pdf'
                else:
                    save_path = f'{fig_save_dir}/traj_{epoch}_{self.algo_type}_task{task}_goal{idx_goal}_neg.pdf'
                
                self.draw_path_plot(
                    paths=paths,
                    heterodastic_vars=heterodastic_vars,
                    task=task,        # 当前任务编号或标签
                    save_path=save_path,  # 图像保存的文件路径
                    min_5=min_5,
                    max_95=max_95,
                    is_half_circle=False,
                )
            
            ############################
            # z 可视化
            ############################
            self.draw_z_trajectory(
                task_labels=task_labels,
                z_weighted_list=z_weighted_list,
                z_mean_list=z_mean_list,
                latent_dim=self.latent_dim,
                save_path=os.path.join(fig_save_dir, f'traj_{epoch}_{self.algo_type}_goal{idx_goal}_z_weighted_mean.pdf'),
            )

            # 计算并绘制 z_weighted 的热力图
            z_weighted_array = np.array(z_weighted_list)
            z_mean_array = np.array(z_mean_list)
            dist_matrix_weighted = self.compute_distance_matrix(z_weighted_array, distance_metric)
            self.draw_heatmap(
                matrix=dist_matrix_weighted,
                task_labels=task_labels,
                title=f'Heatmap of z_weighted across tasks (Epoch {epoch})',
                save_path=os.path.join(fig_save_dir, f'z_weighted_heatmap_{epoch}_{self.algo_type}_goal{idx_goal}.pdf')
            )
            # 计算并绘制 z_mean 的热力图
            dist_matrix_mean = self.compute_distance_matrix(z_mean_array, distance_metric)
            self.draw_heatmap(
                matrix=dist_matrix_mean,
                task_labels=task_labels,
                title=f'Heatmap of z_mean across tasks (Epoch {epoch})',
                save_path=os.path.join(fig_save_dir, f'z_mean_heatmap_{epoch}_{self.algo_type}_goal{idx_goal}.pdf')
            )

    def draw_in_and_out_of_distribution_path(self, epoch, logdir, min_5 = None, max_95 = None, is_z_random_switch=False, distance_metric='euclidean', path=None):
        fig_save_dir = logdir + '/figures_in_out'
        if not os.path.exists(fig_save_dir):
            os.makedirs(fig_save_dir)
            
        first_path = []
    
        for i, idx in enumerate(list(self.eval_tasks)):
            paths, heterodastic_vars = self.collect_in_and_out_of_distribution_paths(idx, epoch, 0, self.eval_buffer, return_heterodastic_var=True, first_path= [path[i]] if path is not None else None)

            save_path = f'{fig_save_dir}/traj_{epoch}_{self.algo_type}_{idx}.pdf'
                
            self.draw_path_plot(
                paths=paths,
                heterodastic_vars=heterodastic_vars,
                task=idx,        # 当前任务编号或标签
                save_path=save_path,  # 图像保存的文件路径
                min_5=min_5,
                max_95=max_95,
                is_half_circle=False,
            )
            
            first_path.append(paths[0])
        return first_path
        
    def show_return(self, tb_writer, epoch, is_last=False):
        if self.tb_writer is None:
            self.tb_writer = tb_writer
        if self.eval_statistics is None:
            self.eval_statistics = OrderedDict()
        test_online_final_returns_avg, test_online_final_returns, test_online_all_returns,   test_online_random_final_returns_avg, test_online_random_final_returns, test_online_random_all_returns,   test_offline_final_returns, test_offline_all_returns = self._do_eval(self.eval_tasks, epoch, buffer=self.eval_buffer)
        train_online_final_returns_avg, train_online_final_returns, train_online_all_returns,   train_online_random_final_returns_avg, train_online_random_final_returns, train_online_random_all_returns,   train_offline_final_returns, train_offline_all_returns = self._do_eval(self.train_tasks, epoch, buffer=self.replay_buffer)
        
        test_online_final_returns_fir, test_online_final_returns_sec = test_online_final_returns
        fir_test_online_final_return = np.mean(test_online_final_returns_fir)
        sec_test_online_final_return = np.mean(test_online_final_returns_sec)
        test_online_random_final_returns_fir, test_online_random_final_returns_sec = test_online_random_final_returns
        fir_test_online_random_final_return = np.mean(test_online_random_final_returns_fir)
        sec_test_online_random_final_return = np.mean(test_online_random_final_returns_sec)
        avg_test_offline_final_return = np.mean(test_offline_final_returns)

        train_online_final_returns_fir, train_online_final_returns_sec = train_online_final_returns
        fir_train_online_final_return = np.mean(train_online_final_returns_fir)
        sec_train_online_final_return = np.mean(train_online_final_returns_sec)
        train_online_random_final_returns_fir, train_online_random_final_returns_sec = train_online_random_final_returns
        fir_train_online_random_final_return = np.mean(train_online_random_final_returns_fir)
        sec_train_online_random_final_return = np.mean(train_online_random_final_returns_sec)
        avg_train_offline_final_return = np.mean(train_offline_final_returns)
        
        self.eval_statistics['first_OnlineReturn_all_test_tasks'] = fir_test_online_final_return
        self.eval_statistics['second_OnlineReturn_all_test_tasks'] = sec_test_online_final_return
        self.eval_statistics['first_OnlineReturn_all_test_tasks_random'] = fir_test_online_random_final_return
        self.eval_statistics['second_OnlineReturn_all_test_tasks_random'] = sec_test_online_random_final_return
        self.eval_statistics['Average_OfflineReturn_all_test_tasks'] = avg_test_offline_final_return

        self.eval_statistics['first_OnlineReturn_all_train_tasks'] = fir_train_online_final_return
        self.eval_statistics['second_OnlineReturn_all_train_tasks'] = sec_train_online_final_return
        self.eval_statistics['first_OnlineReturn_all_train_tasks_random'] = fir_train_online_random_final_return
        self.eval_statistics['second_OnlineReturn_all_train_tasks_random'] = sec_train_online_random_final_return
        self.eval_statistics['Average_OfflineReturn_all_train_tasks'] = avg_train_offline_final_return

        for key, value in self.eval_statistics.items():
            logger.record_tabular(key, value)
        self.eval_statistics = None

        logger.record_tabular("Epoch", epoch)
        logger.dump_tabular(with_prefix=False, with_timestamp=False, tb_writer = self.tb_writer)

        if is_last and 'point-robot' in self.env_name:
            print('draw path')
            self.draw_path(epoch, logger._snapshot_dir+'/tensorboard_' + str(self.first_path_len))

    def show_few_shot_return(self, tb_writer, epoch, num_shots=5, is_last=False):
        if self.tb_writer is None:
            self.tb_writer = tb_writer
        if self.eval_statistics is None:
            self.eval_statistics = OrderedDict()
        (test_online_final_returns_avg, test_online_final_returns, test_online_all_returns, 
         test_online_random_final_returns_avg, test_online_random_final_returns, test_online_random_all_returns, 
         test_offline_final_returns, test_offline_all_returns) = self._do_eval(self.eval_tasks, epoch, buffer=self.eval_buffer, num_shots=num_shots)
        (train_online_final_returns_avg, train_online_final_returns, train_online_all_returns,
         train_online_random_final_returns_avg, train_online_random_final_returns, train_online_random_all_returns,
         train_offline_final_returns, train_offline_all_returns) = self._do_eval(self.train_tasks, epoch, buffer=self.replay_buffer, num_shots=num_shots)
    
        for i in range(num_shots):
            self.eval_statistics[f'test_OnlineReturn_shot{i+1}'] = np.mean(test_online_final_returns[i])
            self.eval_statistics[f'test_OnlineReturn_random_shot{i+1}'] = np.mean(test_online_random_final_returns[i])
            self.eval_statistics[f'train_OnlineReturn_shot{i+1}'] = np.mean(train_online_final_returns[i])
            self.eval_statistics[f'train_OnlineReturn_random_shot{i+1}'] = np.mean(train_online_random_final_returns[i])
        
        self.eval_statistics['Average_OfflineReturn_all_test_tasks'] = np.mean(test_offline_final_returns)
        self.eval_statistics['Average_OfflineReturn_all_train_tasks'] = np.mean(train_offline_final_returns)

        for key, value in self.eval_statistics.items():
            logger.record_tabular(key, value)
        self.eval_statistics = None

        logger.record_tabular("Epoch", epoch)
        logger.dump_tabular(with_prefix=False, with_timestamp=False, tb_writer=self.tb_writer)

        # if is_last and 'point-robot' in self.env_name:
        #     print('draw path')
        #     self.draw_path(epoch, logger._snapshot_dir+'/tensorboard_few_shot')

    def show_z_random_switch(self, tb_writer, epoch, is_last=False):
        if self.tb_writer is None:
            self.tb_writer = tb_writer
        if self.eval_statistics is None:
            self.eval_statistics = OrderedDict()
        
        test_online_z_random_switch_returns_avg, test_online_z_random_switch_returns, test_online_all_returns = self._do_eval_z_random_switch(self.eval_tasks, epoch, buffer=self.eval_buffer)
        train_online_z_random_switch_returns_avg, train_online_z_random_switch_returns, train_online_all_returns = self._do_eval_z_random_switch(self.train_tasks, epoch, buffer=self.replay_buffer)

        test_online_z_random_switch_returns_fir, test_online_z_random_switch_returns_sec = test_online_z_random_switch_returns
        # avg_test_online_z_random_switch_return = np.mean(test_online_z_random_switch_returns_avg)
        fir_test_online_z_random_switch_return = np.mean(test_online_z_random_switch_returns_fir)
        sec_test_online_z_random_switch_return = np.mean(test_online_z_random_switch_returns_sec)

        train_online_z_random_switch_returns_fir, train_online_z_random_switch_returns_sec = train_online_z_random_switch_returns
        # avg_train_online_z_random_switch_return = np.mean(train_online_z_random_switch_returns_avg)
        fir_train_online_z_random_switch_return = np.mean(train_online_z_random_switch_returns_fir)
        sec_train_online_z_random_switch_return = np.mean(train_online_z_random_switch_returns_sec)
        
        self.eval_statistics['first_OnlineReturn_all_test_tasks'] = fir_test_online_z_random_switch_return
        self.eval_statistics['second_OnlineReturn_all_test_tasks'] = sec_test_online_z_random_switch_return

        self.eval_statistics['first_OnlineReturn_all_train_tasks'] = fir_train_online_z_random_switch_return
        self.eval_statistics['second_OnlineReturn_all_train_tasks'] = sec_train_online_z_random_switch_return

        for key, value in self.eval_statistics.items():
            logger.record_tabular(key, value)
        self.eval_statistics = None

        logger.record_tabular("Epoch", epoch)
        logger.dump_tabular(with_prefix=False, with_timestamp=False, tb_writer = self.tb_writer)

        if is_last and 'point-robot' in self.env_name:
            self.draw_path(epoch, logger._snapshot_dir+'/tensorboard_z_random_switch', is_z_random_switch=True)

    def collect_offline_paths(self, idx, epoch, run, buffer):
        self.task_idx = idx
        self.env.reset_task(idx)

        self.agent.clear_z()
        paths = []
        num_transitions = 0
        # while num_transitions < self.num_steps_per_eval:
        for i in range(1):
            path, num = self.offline_sampler.obtain_samples(
                buffer=buffer,
                deterministic=self.eval_deterministic,
                max_samples=self.num_steps_per_eval - num_transitions,
                max_trajs=1,
                accum_context=True,
                rollout=True,
                max_path_length=self.first_path_len if i == 0 else None)
            paths += path
            num_transitions += num

        if self.sparse_rewards:
            for p in paths:
                sparse_rewards = np.stack([e['sparse_reward'] for e in p['env_infos']]).reshape(-1, 1)
                p['rewards'] = sparse_rewards

        goal = self.env._goal
        for path in paths:
            path['goal'] = goal # goal

        return paths

    def adapt_draw_one_task_from_prior(self):
        candidates = np.random.choice(self.train_tasks, self.onlineadapt_max_num_candidates)
        ans = -1e8
        idx = -1
        for can_idx in candidates:
            task_z = self.trained_z_sample[can_idx]
            min_dis = 1e8
            for past_z in self.adapt_sampled_z_list:
                dis = torch.mean((task_z - past_z) ** 2).detach().cpu().numpy()
                min_dis = min(min_dis, dis)
            if min_dis > ans:
                ans = min_dis
                idx = can_idx
        assert idx != -1
        return idx

    def collect_online_paths(self, idx, epoch, run, buffer, num_models=12, return_heterodastic_var=False, neg_task=False, num_shots=2):
        self.task_idx = idx
        if neg_task:
            self.env.reset_neg_task(idx)
        else:
            self.env.reset_task(idx)

        self.agent.clear_z()
        paths = []
        num_transitions = 0
        num_trajs = 0
        self.adapt_sampled_z_list = []
        # while num_transitions < self.num_steps_per_eval:
        if self.algo_type == 'IDAQ':
            for i in range(num_shots):
                if num_trajs < self.num_exp_traj_eval or type(self.agent.context) == type(None): # num_exp_traj_eval = 10 ? 1 TODO
                    sampled_idx = np.random.choice(self.train_tasks)
                    sampled_idx = self.adapt_draw_one_task_from_prior()
                    self.agent.clear_z()
                    self.agent.set_z(self.trained_z[sampled_idx][0], self.trained_z[sampled_idx][1])
                path, num = self.sampler.obtain_samples(
                    deterministic=self.eval_deterministic,
			        max_samples=np.inf, 
                    max_trajs=1,
			        accum_context=True,
			        is_select=True,
			        r_thres=self.r_thres,
			        is_onlineadapt_max=self.is_onlineadapt_max,
			        is_sparse_reward=self.sparse_rewards,
			    	reward_models=self.reward_models[:num_models],
                    dynamic_models=self.dynamic_models[:num_models],
                    update_score=(num_trajs < self.num_exp_traj_eval),
                    max_path_length=self.first_path_len if i == 0 else None)
                paths += path
                num_transitions += num
                num_trajs += 1
                if num_transitions >= self.num_exp_traj_eval * self.max_path_length and self.agent.context is not None:
                    heterodastic_var = self.agent.infer_posterior(self.agent.context).cpu().numpy()
        else:
            for i in range(num_shots):
                path, num = self.sampler.obtain_samples(
                    deterministic=self.eval_deterministic,
                    max_samples=np.inf,
                    max_trajs=1,
                    accum_context=True,
                    max_path_length=self.first_path_len if i == 0 else None)
                paths += path
                num_transitions += num
                heterodastic_var = self.agent.infer_posterior(self.agent.context).squeeze().cpu().numpy()

        if self.sparse_rewards:
            for p in paths:
                sparse_rewards = np.stack([e['sparse_reward'] for e in p['env_infos']]).reshape(-1, 1)
                p['rewards'] = sparse_rewards

        goal = self.env._goal
        for path in paths:
            path['goal'] = goal # goal
        
        if return_heterodastic_var:
            if self.first_path_len is None:
                heterodastic_var_list = np.array_split(heterodastic_var, 2)
            else:
                heterodastic_var_list = [heterodastic_var[:self.first_path_len], heterodastic_var[self.first_path_len:]]
            return paths, heterodastic_var_list
        else:
            return paths

    def collect_manual_paths(self, idx, epoch, run, buffer, num_models=12, return_heterodastic_var=False, neg_task=False, manual_goal=None):
        self.task_idx = idx
        if neg_task:
            self.env.reset_neg_task(idx)
        else:
            self.env.reset_task(idx)

        self.agent.clear_z()
        paths = []
        num_transitions = 0
        num_trajs = 0
        self.adapt_sampled_z_list = []
        # path 1
        path, num = self.sampler.manual_samples(
            deterministic=self.eval_deterministic,
            max_samples=np.inf,
            max_trajs=1,
            accum_context=True,
            max_path_length=self.first_path_len,
            manual_goal=manual_goal)
        paths += path
        num_transitions += num
        self.agent.infer_posterior(self.agent.context)
        
        path, num = self.sampler.obtain_samples(
            deterministic=self.eval_deterministic,
            max_samples=np.inf,
            max_trajs=1,
            accum_context=True,
            max_path_length=None)
        paths += path
        num_transitions += num
        heterodastic_var = self.agent.infer_posterior(self.agent.context).squeeze().cpu().numpy()

        if self.sparse_rewards:
            for p in paths:
                sparse_rewards = np.stack([e['sparse_reward'] for e in p['env_infos']]).reshape(-1, 1)
                p['rewards'] = sparse_rewards

        goal = self.env._goal
        for path in paths:
            path['goal'] = goal # goal
        
        if return_heterodastic_var:
            if self.first_path_len is None:
                heterodastic_var_list = np.array_split(heterodastic_var, 2)
            else:
                heterodastic_var_list = [heterodastic_var[:self.first_path_len], heterodastic_var[self.first_path_len:]]
            return paths, heterodastic_var_list
        else:
            return paths

    def collect_in_and_out_of_distribution_paths(self, idx, epoch, run, buffer, num_models=12, return_heterodastic_var=False, in_distribution_ratio=0.25, first_path=None):
        self.task_idx = idx
        self.env.reset_task(idx)

        self.agent.clear_z()
        paths = []
        if first_path is None:
            path_i_d, num_i_d = self.offline_sampler.obtain_samples(
                buffer=buffer,
                deterministic=self.eval_deterministic,
                max_samples=np.inf,
                max_trajs=1,
                accum_context=True,
                rollout=True,
                max_path_length=int(self.max_path_length * in_distribution_ratio))

            path_o_d, num_o_d = self.sampler.manual_samples(
                deterministic=self.eval_deterministic,
                max_samples=np.inf,
                max_trajs=1,
                accum_context=True,
                max_path_length=self.max_path_length - int(self.max_path_length * in_distribution_ratio))

            first_path = [{}]
            for j in range(len(path_o_d)):
                first_path[j]['observations'] = np.concatenate([path_i_d[j]['observations'], path_o_d[j]['observations']], axis=0)
                first_path[j]['observations'] = np.concatenate([path_i_d[j]['observations'], path_o_d[j]['observations']], axis=0)
                first_path[j]['actions'] = np.concatenate([path_i_d[j]['actions'], path_o_d[j]['actions']], axis=0)
                first_path[j]['rewards'] = np.concatenate([path_i_d[j]['rewards'], path_o_d[j]['rewards']], axis=0)
                first_path[j]['next_observations'] = np.concatenate([path_i_d[j]['next_observations'], path_o_d[j]['next_observations']], axis=0)
                first_path[j]['terminals'] = np.concatenate([path_i_d[j]['terminals'], path_o_d[j]['terminals']], axis=0)
                first_path[j]['agent_infos'] = path_i_d[j]['agent_infos'] + path_o_d[j]['agent_infos']
                first_path[j]['env_infos'] = path_i_d[j]['env_infos'] + path_o_d[j]['env_infos']
                first_path[j]['context'] = np.array([np.concatenate([path_i_d[j]['context'][0], path_o_d[j]['context'][0]], axis=0)])
        else:
            self.agent.update_context_dict(first_path[0], env=self.env)
        paths += first_path
        self.agent.infer_posterior(self.agent.context).squeeze().cpu().numpy()

        path, num = self.sampler.obtain_samples(
            deterministic=self.eval_deterministic,
            max_samples=np.inf,
            max_trajs=1,
            accum_context=True,
            max_path_length=None)
        paths += path
        heterodastic_var = self.agent.infer_posterior(self.agent.context).squeeze().cpu().numpy()

        if self.sparse_rewards:
            for p in paths:
                sparse_rewards = np.stack([e['sparse_reward'] for e in p['env_infos']]).reshape(-1, 1)
                p['rewards'] = sparse_rewards

        goal = self.env._goal
        for path in paths:
            path['goal'] = goal # goal

        if return_heterodastic_var:
            heterodastic_var_list = np.array_split(heterodastic_var, 2)
            return paths, heterodastic_var_list
        else:
            return paths

    def collect_random_online_paths(self, idx, epoch, run, buffer, num_models=12, return_heterodastic_var=False, neg_task=False, num_shots=2):
        self.task_idx = idx
        if neg_task:
            self.env.reset_neg_task()
        else:
            self.env.reset_task(idx)

        self.agent.clear_z()
        paths = []
        num_transitions = 0
        self.adapt_sampled_z_list = []
        for i in range(num_shots):
            path, num = self.sampler.obtain_samples(
                deterministic=self.eval_deterministic,
                max_samples=np.inf,
                max_trajs=1,
                accum_context=True,
                np_online_collect=(i==0),
                max_path_length=self.first_path_len if i == 0 else None)
            paths += path
            num_transitions += num
            # if num_transitions >= self.num_exp_traj_eval * self.max_path_length and self.agent.context is not None:
            heterodastic_var = self.agent.infer_posterior(self.agent.context).squeeze().cpu().numpy()

        if self.sparse_rewards:
            for p in paths:
                sparse_rewards = np.stack([e['sparse_reward'] for e in p['env_infos']]).reshape(-1, 1)
                p['rewards'] = sparse_rewards

        goal = self.env._goal
        for path in paths:
            path['goal'] = goal # goal
        
        if return_heterodastic_var:
            if self.first_path_len is None:
                heterodastic_var_list = np.array_split(heterodastic_var, 2)
            else:
                heterodastic_var_list = [heterodastic_var[:self.first_path_len], heterodastic_var[self.first_path_len:]]
                print(heterodastic_var_list)
            return paths, heterodastic_var_list
        else:
            return paths
    
    def collect_z_random_switch_online_paths(self, idx, epoch, run, buffer, num_models=12, return_heterodastic_var=False, neg_task=False):
        self.task_idx = idx
        if neg_task:
            self.env.reset_neg_task()
        else:
            self.env.reset_task(idx)

        self.agent.clear_z()
        paths = []
        num_transitions = 0
        num_trajs = 0
        self.adapt_sampled_z_list = []
        for i in range(2):
            path, num = self.sampler.obtain_samples(
                deterministic=self.eval_deterministic,
                max_samples=np.inf,
                max_trajs=1,
                accum_context=True,
                z_random=(i==0),
                max_path_length=self.first_path_len if i == 0 else None)
            paths += path
            num_transitions += num
            heterodastic_var = self.agent.infer_posterior(self.agent.context).squeeze().cpu().numpy()

        if self.sparse_rewards:
            for p in paths:
                sparse_rewards = np.stack([e['sparse_reward'] for e in p['env_infos']]).reshape(-1, 1)
                p['rewards'] = sparse_rewards

        goal = self.env._goal
        for path in paths:
            path['goal'] = goal # goal
        
        if return_heterodastic_var:
            if self.first_path_len is None:
                heterodastic_var_list = np.array_split(heterodastic_var, 2)
            else:
                heterodastic_var_list = np.split(heterodastic_var, [self.first_path_len])
            return paths, heterodastic_var_list
        else:
            return paths

    def epsilon_decay(self, steps):
        if steps < self.num_steps_per_eval*0.5:
            alpha=1
        else:
            alpha=0
        # alpha=1
        return alpha

    def collect_np_online_paths(self, idx, epoch, run, buffer):
        self.task_idx = idx
        if idx == -1:
            self.env.reset_neg_task()
        else:
            self.env.reset_task(idx)

        self.agent.clear_z()
        paths = []
        num_transitions = 0
        np_online_all_num = 0
        while num_transitions < self.num_steps_per_eval:
            np_online_path, np_online_num = self.sampler.obtain_samples(
                deterministic=self.eval_deterministic,
                max_samples=self.max_path_length,
                max_trajs=np.inf,
                accum_context=True,
                update_z_per_step=False,
                np_online_collect=True,
                use_np_online_decay=True, 
                init_num=num_transitions, 
                decay_function=self.epsilon_decay)
            
            # self.agent.infer_posterior(self.agent.context)

            path, num = self.sampler.obtain_samples(
                    deterministic=self.eval_deterministic,
                    max_samples=self.num_steps_per_eval - num_transitions,
                    max_trajs=1,
                    accum_context=False,
                    update_z_per_step=False)

            paths += path
            num_transitions += num
            np_online_all_num += np_online_num

            if num_transitions >= self.num_exp_traj_eval * self.max_path_length:
                self.agent.infer_posterior(self.agent.context)

        if self.sparse_rewards:
            for p in paths:
                sparse_rewards = np.stack([e['sparse_reward'] for e in p['env_infos']]).reshape(-1, 1)
                p['rewards'] = sparse_rewards

        goal = self.env._goal
        for path in paths:
            path['goal'] = goal # goal

        return paths

    @abc.abstractmethod
    def training_mode(self, mode):
        """
        Set training mode to `mode`.
        :param mode: If True, training will happen (e.g. set the dropout
        probabilities to not all ones).
        """
        pass

    @abc.abstractmethod
    def _do_training(self):
        """
        Perform some update, e.g. perform one gradient step.
        :return:
        """
        pass

    def collect_offline_zs(self, indices, buffer):
        batches = [ptu.np_to_pytorch_batch(buffer.random_batch(idx, batch_size=self.embedding_batch_size, sequence=False)) for idx in indices]
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
        self.agent.clear_z()
        self.agent.infer_posterior(context, indices)
        self.agent.detach_z()
        return ptu.get_numpy(self.agent.z)

    def collect_online_zs(self, indices, np_online=False):
        zs = []
        for idx in indices:
            if np_online:
                paths = self.collect_np_online_paths(idx, 0, 0, None)
            else:
                paths = self.collect_online_paths(idx, 0, 0, None)
            zs.append(paths[-1]['context'])
        return np.concatenate(zs, axis=0)

    @staticmethod
    def vis_task_embeddings(save_dir, fig_name, zs, rows=1, cols=1, n_figs=1,
                subplot_title_lst = ["train_itr_0"],  goals_name_lst=None, figsize=[12, 6], fontsize=15):
        if figsize is None:
            fig = plt.figure(figsize=(8, 10))
        else:
            fig = plt.figure(figsize=figsize)
        fig.subplots_adjust(hspace=0.5, wspace=0.3)
        if goals_name_lst is None:
            goals_name_lst = [None]*n_figs
            legend = False
        else:
            legend = True

        for z, n_fig, subplot_title, goals_name in zip(zs, range(1, n_figs+1), subplot_title_lst, goals_name_lst):
            n_tasks, n_points, _ = z.shape
            proj = TSNE(n_components=2).fit_transform(X=z.reshape(n_tasks*n_points, -1))
            ax = fig.add_subplot(rows, cols, n_fig)
            for task_idx in range(n_tasks):
                idxs = np.arange(task_idx*n_points, (task_idx+1)*n_points)
                if goals_name is None:
                    ax.scatter(proj[idxs, 0], proj[idxs, 1], s=2, alpha=0.3, cmap=plt.cm.Spectral)
                else:
                    ax.scatter(proj[idxs, 0], proj[idxs, 1], s=2, alpha=0.3, cmap=plt.cm.Spectral, label=goals_name[task_idx])

            ax.set_title(subplot_title, fontsize=fontsize)
            ax.set_xlabel('t-SNE dimension 1', fontsize=fontsize)
            ax.set_ylabel('t-SNE dimension 2', fontsize=fontsize)
            if legend:
                ax.legend(loc='best')
        
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, fig_name), dpi=400)
        plt.close()
