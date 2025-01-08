import os
import numpy as np
import click
import json, time
import torch
import random
import multiprocessing as mp
from itertools import product
import glob, ast

from rlkit.envs import ENVS
from rlkit.envs.wrappers import NormalizedBoxEnv
from rlkit.torch.multi_task_dynamics import MultiTaskDynamics

import rlkit.torch.pytorch_util as ptu
from configs.default import default_config
from numpy.random import default_rng
from rlkit.data_management.env_replay_buffer import MultiTaskReplayBuffer
from pathlib import Path

rng = default_rng()

def global_seed(seed=0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def experiment(variant, seed=None):
    env = NormalizedBoxEnv(ENVS[variant['env_name']](**variant['env_params']))
    
    if seed is not None:
        global_seed(seed)
        env.seed(seed)

    tasks = env.get_all_task_idx()
    obs_dim = int(np.prod(env.observation_space.shape))
    action_dim = int(np.prod(env.action_space.shape))
    reward_dim = 1

    os.environ['CUDA_VISIBLE_DEVICES'] = str(variant['util_params']['gpu_id'])
    ptu.set_gpu_mode(variant['util_params']['use_gpu'], variant['util_params']['gpu_id'])
    DEBUG = variant['util_params']['debug']
    os.environ['DEBUG'] = str(int(DEBUG))

    net_size = variant['net_size']
    use_next_obs_in_context = variant['algo_params']['use_next_obs_in_context']
    
    if use_next_obs_in_context:
        task_dynamics =  MultiTaskDynamics(num_tasks=len(tasks), 
                                     hidden_size=net_size, 
                                     num_hidden_layers=3, 
                                     action_dim=action_dim, 
                                     obs_dim=obs_dim,
                                     reward_dim=1,
                                     use_next_obs_in_context=use_next_obs_in_context,
                                     ensemble_size=variant['algo_params']['ensemble_size'],
                                     dynamics_weight_decay=[2.5e-5, 5e-5, 7.5e-5, 7.5e-5])
    else:
        task_dynamics = MultiTaskDynamics(num_tasks=len(tasks), 
                                     hidden_size=net_size, 
                                     num_hidden_layers=2, 
                                     action_dim=action_dim, 
                                     obs_dim=obs_dim,
                                     reward_dim=1,
                                     use_next_obs_in_context=use_next_obs_in_context,
                                     ensemble_size=variant['algo_params']['ensemble_size'],
                                     dynamics_weight_decay=[2.5e-5, 5e-5, 7.5e-5])
    train_tasks = list(tasks)
    train_buffer = MultiTaskReplayBuffer(variant['algo_params']['replay_buffer_size'], env, train_tasks, 1)

    data_dir = variant['algo_params']['data_dir']
    offline_data_quality = variant['algo_params']['offline_data_quality']
    n_trj = variant['algo_params']['n_trj']
    train_trj_paths = []
    for i in range(len(train_tasks)):
        goal_i_dir = Path(data_dir) / f"goal_idx{i}"
        quality_steps = np.array(sorted(list(set([int(trj_path.stem.split('step')[-1]) for trj_path in goal_i_dir.rglob('trj_evalsample*_step*.npy')]))))
        low_quality_steps, mid_quality_steps, high_quality_steps = np.array_split(quality_steps, 3)
        if offline_data_quality == 'low':
            training_date_steps = low_quality_steps
        elif offline_data_quality == 'mid':
            training_date_steps = mid_quality_steps
        elif offline_data_quality == 'expert':
            training_date_steps = high_quality_steps[-1:]
        else:
            training_date_steps = quality_steps
        for j in training_date_steps:
            for k in range(n_trj):
                train_trj_paths += [os.path.join(data_dir, f"goal_idx{i}", f"trj_evalsample{k}_step{j}.npy")]
    train_paths = [train_trj_path for train_trj_path in train_trj_paths if
                   int(train_trj_path.split('/')[-2].split('goal_idx')[-1]) in train_tasks]
    train_task_idxs = [int(train_trj_path.split('/')[-2].split('goal_idx')[-1]) for train_trj_path in train_trj_paths if
                   int(train_trj_path.split('/')[-2].split('goal_idx')[-1]) in train_tasks]
    obs_train_lst = []
    action_train_lst = []
    reward_train_lst = []
    next_obs_train_lst = []
    terminal_train_lst = []
    task_train_lst = []
    for train_path, train_task_idx in zip(train_paths, train_task_idxs):
        trj_npy = np.load(train_path, allow_pickle=True)
        obs, action, reward, next_obs = np.array_split(trj_npy, [obs_dim, obs_dim+action_dim, -obs_dim], axis=-1)
        obs_train_lst += list(obs)
        action_train_lst += list(action)
        reward_train_lst += list(reward)
        next_obs_train_lst += list(next_obs)
        terminal = [0 for _ in range(trj_npy.shape[0])]
        terminal[-1] = 1
        terminal_train_lst += terminal
        task_train = [train_task_idx for _ in range(trj_npy.shape[0])]
        task_train_lst += task_train
    # load training buffer
    for i, (task_train,
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
        train_buffer.add_sample(
            task_train,
            obs,
            action,
            reward,
            terminal,
            next_obs,
            **{'env_info': {}},
        )

    for task_idx in train_tasks:
        data = train_buffer.get_all_data(task_idx)
        task_dynamics.set_task_idx(task_idx)
        task_dynamics.train(data)
        print(f"Task {task_idx} finished training")
    
    save_dir = os.path.join(variant['algo_params']['data_dir'], 'dynamics')
    os.makedirs(save_dir, exist_ok=True)
    task_dynamics.save(save_dir)


def deep_update_dict(fr, to):
    ''' update dict of dicts with new values '''
    # assume dicts have same keys
    for k, v in fr.items():
        if type(v) is dict:
            deep_update_dict(v, to[k])
        else:
            to[k] = v
    return to

@click.command()
@click.argument('config', default=None)
@click.option('--gpu', default=0)
@click.option('--seed', default=0)
def main(config, gpu, seed):
    variant = default_config
    if config:
        with open(os.path.join(config)) as f:
            exp_params = json.load(f)
        variant = deep_update_dict(exp_params, variant)
    variant['util_params']['gpu_id'] = gpu
    experiment(variant, seed=seed)

if __name__ == "__main__":
    main()



