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

    def _do_eval(self, indices, epoch, buffer):
        online_final_returns = []
        online_final_returns_first = []
        online_final_returns_second = []
        online_all_return = []
        for idx in indices:
            all_rets = []
            all_rets_first = []
            all_rets_second = []
            for r in range(self.num_evals):
                paths = self.collect_online_paths(idx, epoch, r, buffer)
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

        online_random_final_returns = []
        online_random_final_returns_first = []
        online_random_final_returns_second = []
        online_random_all_return = []
        for idx in indices:
            all_rets = []
            all_rets_first = []
            all_rets_second = []
            for r in range(self.num_evals):
                paths = self.collect_random_online_paths(idx, epoch, r, buffer)
                paths_first, paths_second = paths[0], paths[1]
                # all_rets.append([eval_util.get_average_returns([p]) for p in paths])
                all_rets_first.append([eval_util.get_average_returns([paths_first])])
                all_rets_second.append([eval_util.get_average_returns([paths_second])])
                all_rets.append([(eval_util.get_average_returns([paths_first]) + eval_util.get_average_returns([paths_second]))/2])
            online_random_final_returns.append(np.mean([a[-1] for a in all_rets]))
            online_random_final_returns_first.append(np.mean([a[-1] for a in all_rets_first]))
            online_random_final_returns_second.append(np.mean([a[-1] for a in all_rets_second]))
            # record all returns for the first n trajectories
            n = min([len(a) for a in all_rets])
            all_rets = [a[:n] for a in all_rets]
            all_rets = np.mean(np.stack(all_rets), axis=0) # avg return per nth rollout
            online_random_all_return.append(all_rets)
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

        return online_final_returns, [online_final_returns_first, online_final_returns_second], online_all_return, online_random_final_returns, [online_random_final_returns_first, online_random_final_returns_second], online_random_all_return, offline_final_returns, offline_all_return
    
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
        if epoch == (self.num_iterations - 1):
            if self.separate_train and not self.pretrain:
                if 'point-robot' in self.env_name:
                    self.draw_path(epoch, logger._snapshot_dir)
            if self.seed != 0:
                return
            print('---------T-SNE: Collecting Zs----------')
            # sample offline context zs
            fig_save_dir = logger._snapshot_dir + '/figures'
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
            
            print('---------T-SNE: Plotting------------')
            self.vis_task_embeddings(save_dir = fig_save_dir, fig_name=f'offline_train_zs_{epoch}.png', zs=[offline_train_zs], subplot_title_lst=[f'offline_train_zs_{epoch}'])
            self.vis_task_embeddings(save_dir = fig_save_dir, fig_name=f'offline_test_zs_{epoch}.png', zs=[offline_test_zs], subplot_title_lst=[f'offline_test_zs_{epoch}'])
            self.vis_task_embeddings(save_dir = fig_save_dir, fig_name=f'online_train_zs_{epoch}.png', zs=[online_train_zs], subplot_title_lst=[f'online_train_zs_{epoch}'])
            self.vis_task_embeddings(save_dir = fig_save_dir, fig_name=f'online_test_zs_{epoch}.png', zs=[online_test_zs], subplot_title_lst=[f'online_test_zs_{epoch}'])
            # self.vis_task_embeddings(save_dir = fig_save_dir, fig_name=f'online_train_np_zs_{epoch}.png', zs=[online_train_np_zs], subplot_title_lst=[f'online_train_np_zs_{epoch}'])
            # self.vis_task_embeddings(save_dir = fig_save_dir, fig_name=f'online_test_np_zs_{epoch}.png', zs=[online_test_np_zs], subplot_title_lst=[f'online_test_np_zs_{epoch}'])

    def draw_path(self, epoch, logdir, is_baseline=False, min_5 = None, max_95 = None):
        import matplotlib.patches as patches
        import seaborn as sns

        for idx in self.eval_tasks:
            paths, heterodastic_vars = self.collect_online_paths(idx, 0, 0, None, return_heterodastic_var=True)

            fig = plt.figure(figsize=(20, 12))
            sns.set_theme(style="white", font_scale=2.0)

            ax = plt.gca()

            half_circle = patches.Arc((0, 0), 2, 2, angle=0, theta1=0, theta2=180, color=(180./255., 180./255., 180./255.), linewidth=2)
            ax.add_artist(half_circle)

            ax.set_xlim(-1.1, 1.1)  # x 轴范围 [-1, 1]
            ax.set_ylim(-0.2, 1.2)   # y 轴范围 [0, 1]
            ax.set_aspect('equal', 'box')

            # markers = ['^', 's', 'o', 'x', 'D', '*']  # 三角形、方块、圆点、叉叉、菱形、星号
            colors = [(117./255., 196./255., 175./255.), (249./255., 116./255., 70./255.)]
            # linestyles = ['-', '--', '-.']
            lines_for_legend = []  # 存储用于显示在图例中的线条
            save_data = []
            vmin = min_5 if min_5 is not None else np.min(heterodastic_vars)
            vmax = max_95 if max_95 is not None else np.max(heterodastic_vars)
            for i, path in enumerate(paths[:2]):
                observations = np.array(path['observations'])
                heterodastic_var = np.array(heterodastic_vars[i])
                line, = plt.plot(observations[:, 0], observations[:, 1], color=colors[i % len(colors)], linestyle='-', label=f'episode Line {i + 1}', linewidth=4, zorder=1)
                sc = plt.scatter(observations[:, 0], observations[:, 1], c=heterodastic_var, cmap='coolwarm', vmin=vmin, vmax=vmax, marker='o', label=f'episode {i + 1}', linewidth=4, zorder=2)
                save_data.append({
                    'observations': observations.tolist(),
                    'heterodastic_var': heterodastic_var.tolist(),
                    'goal': path['goal']
                })
                lines_for_legend.append(line)

            goal = path['goal']
            plt.axis('off')
            goal_point = plt.scatter(
                goal[0], goal[1],
                edgecolors=(0.9, 0.2, 0.4),  # 设置边框颜色
                facecolors='none',           # 空心
                marker='*',
                label='Goal',
                s=500,                       # 控制圆圈的大小，增大半径
                linewidths=3,                # 设置边框线宽
                zorder=3                     # 控制绘制顺序
            )
            lines_for_legend.append(goal_point)

            start = (0.,0.)
            start_point = plt.scatter(
                start[0], start[1],
                edgecolors=(0.9, 0.2, 0.4),  # 设置边框颜色
                facecolors='none',
                marker='o',
                label='Start',
                s=450,                       # 控制圆圈的大小，增大半径
                linewidths=3,                # 设置边框线宽
                zorder=3                     # 控制绘制顺序
            )
            lines_for_legend.append(start_point)
            cbar = plt.colorbar(sc, label='Uncertainty', fraction=0.05, pad=0.1, shrink=1.0, aspect=20, orientation='horizontal')
            cbar.ax.tick_params(labelsize=20)
            # if is_baseline:
            #     plt.title(f'Point-Robot {self.algo_type}', pad=20, fontsize=20)
            # else:
            #     plt.title(f'Point-Robot {self.algo_type} + Ours', pad=20, fontsize=20)
            plt.xlabel('X')
            plt.ylabel('Y')
            # plt.legend(loc='upper right')
            plt.legend(handles=lines_for_legend, loc='upper right')

            fig_save_dir = logdir + '/figures'
            if not os.path.exists(fig_save_dir):
                os.makedirs(fig_save_dir)
            plt.savefig(f'{fig_save_dir}/traj_{epoch}_{self.algo_type}_{idx}.pdf', dpi=400, bbox_inches='tight')
            plt.close()
    
    def show_return(self, tb_writer, epoch):
        if self.tb_writer is None:
            self.tb_writer = tb_writer
        self.evaluate(epoch)
        logger.record_tabular("Epoch", epoch)
        logger.dump_tabular(with_prefix=False, with_timestamp=False, tb_writer = self.tb_writer)

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
                rollout=True)
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

    def collect_online_paths(self, idx, epoch, run, buffer, num_models=12, return_heterodastic_var=False):
        self.task_idx = idx
        self.env.reset_task(idx)

        self.agent.clear_z()
        paths = []
        num_transitions = 0
        num_trajs = 0
        self.adapt_sampled_z_list = []
        # while num_transitions < self.num_steps_per_eval:
        if self.algo_type == 'IDAQ':
            for _ in range(2):
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
                    update_score=(num_trajs < self.num_exp_traj_eval))
                paths += path
                num_transitions += num
                num_trajs += 1
                if num_transitions >= self.num_exp_traj_eval * self.max_path_length and self.agent.context is not None:
                    heterodastic_var = self.agent.infer_posterior(self.agent.context).cpu().numpy()
        else:
            clear = []
            for i in range(2):
                if i in clear:
                    self.agent.clear_context()
                path, num = self.sampler.obtain_samples(
                    deterministic=self.eval_deterministic,
                    max_samples=np.inf,
                    max_trajs=1,
                    accum_context=True)
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
            heterodastic_var_list = np.array_split(heterodastic_var, 2)
            return paths, heterodastic_var_list
        else:
            return paths

    def collect_random_online_paths(self, idx, epoch, run, buffer, num_models=12, return_heterodastic_var=False):
        self.task_idx = idx
        self.env.reset_task(idx)

        self.agent.clear_z()
        paths = []
        num_transitions = 0
        self.adapt_sampled_z_list = []
        for i in range(2):
            path, num = self.sampler.obtain_samples(
                deterministic=self.eval_deterministic,
                max_samples=np.inf,
                max_trajs=1,
                accum_context=True,
                np_online_collect=(i==0))
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
            heterodastic_var_list = np.array_split(heterodastic_var, 2)
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
