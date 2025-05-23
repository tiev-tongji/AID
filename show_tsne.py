"""
Visualize the t-SNE plot of task embeddings.
"""
import os
import pathlib
import numpy as np
import click
import json
import torch
import random
import multiprocessing as mp
from pathlib import Path
from itertools import product
import torch.nn.functional as F

from rlkit.envs import ENVS
from rlkit.envs.wrappers import NormalizedBoxEnv
from rlkit.torch.sac.policies import TanhGaussianPolicy
from rlkit.torch.multi_task_dynamics import MultiTaskDynamics
from rlkit.torch.networks import FlattenMlp, MlpEncoder, RecurrentEncoder, MlpDecoder
from rlkit.core.rl_algorithm import OfflineMetaRLAlgorithm
from rlkit.torch.sac.certain import CERTAINSoftActorCritic
from rlkit.torch.sac.croo import CROOSoftActorCritic
from rlkit.torch.sac.unicorn import UNICORNSoftActorCritic
from rlkit.torch.sac.classifier import CLASSIFIERSoftActorCritic 
from rlkit.torch.sac.agent import PEARLAgent
from rlkit.launchers.launcher_util import setup_logger
from rlkit.data_management.env_replay_buffer import MultiTaskReplayBuffer
import rlkit.torch.pytorch_util as ptu
from configs.default import default_config
from tqdm import tqdm

def global_seed(seed=0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def show_tsne(variant, gpu_id, seed):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    ptu.set_gpu_mode(True, gpu_id)
    # create multi-task environment and sample tasks, normalize obs if provided with 'normalizer.npz'
    if 'normalizer.npz' in os.listdir(variant['algo_params']['data_dir']):
        obs_absmax = np.load(os.path.join(variant['algo_params']['data_dir'], 'normalizer.npz'))['abs_max']
        env = NormalizedBoxEnv(ENVS[variant['env_name']](**variant['env_params']), obs_absmax=obs_absmax)
    else:
        env = NormalizedBoxEnv(ENVS[variant['env_name']](**variant['env_params']))

    if seed is not None:
        global_seed(seed)
        env.seed(seed)

    tasks = env.get_all_task_idx()
    obs_dim = int(np.prod(env.observation_space.shape))
    action_dim = int(np.prod(env.action_space.shape))
    reward_dim = 1

    # instantiate networks
    latent_dim = variant['latent_size']
    context_encoder_input_dim = 2 * obs_dim + action_dim + reward_dim if variant['algo_params']['use_next_obs_in_context'] else obs_dim + action_dim + reward_dim
    context_encoder_output_dim = latent_dim * 2 if variant['algo_params']['use_information_bottleneck'] else latent_dim
    net_size = variant['net_size']
    recurrent = variant['algo_params']['recurrent']
    encoder_model = RecurrentEncoder if recurrent else MlpEncoder
    
    context_encoder = encoder_model(
        hidden_sizes=[200, 200, 200],
        input_size=context_encoder_input_dim,
        output_size=context_encoder_output_dim,
        output_activation=torch.tanh,
        layer_norm=variant['algo_params']['layer_norm'] if 'layer_norm' in variant['algo_params'].keys() else False
    )
    uncertainty_mlp = MlpDecoder(
        hidden_sizes=[net_size],
        input_size=latent_dim,
        output_size=1,
    )

    classifier = MlpDecoder(
        hidden_sizes=[net_size],
        input_size=context_encoder_output_dim,
        output_size=variant['n_train_tasks'],
        layer_norm=variant['algo_params']['layer_norm'] if 'layer_norm' in variant['algo_params'].keys() else False
    )

    exp_name = variant['util_params']['exp_name']
    base_log_dir = variant['util_params']['base_log_dir']
    exp_prefix = variant['env_name']
    log_dir = Path(os.path.join(base_log_dir, exp_prefix.replace("_", "-"), exp_name, f"seed{seed}"))
    agent_path = log_dir/"agent.pth"
    if not agent_path.exists():
        exit(f"agent path {str(agent_path)} does not exist")
    agent_ckpt = torch.load(str(agent_path))
    print(agent_ckpt.keys())
    context_encoder.load_state_dict(agent_ckpt['context_encoder'])
    uncertainty_mlp.load_state_dict(agent_ckpt['uncertainty_mlp'])
    classifier.load_state_dict(agent_ckpt['classifier'])
    context_encoder.to(ptu.device)
    uncertainty_mlp.to(ptu.device)
    classifier.to(ptu.device)

    # Setting up tasks
    if 'randomize_tasks' in variant.keys() and variant['randomize_tasks']:
        train_tasks = np.random.choice(len(tasks), size=variant['n_train_tasks'], replace=False)
    elif 'interpolation' in variant.keys() and variant['interpolation']:
        step = len(tasks)/variant['n_train_tasks']
        train_tasks = np.array([tasks[int(i*step)] for i in range(variant['n_train_tasks'])])
    eval_tasks = np.array(list(set(range(len(tasks))).difference(train_tasks)))

    # Load dataset
    train_trj_paths = []
    eval_trj_paths = []
    # trj entry format: [obs, action, reward, new_obs]
    n_tasks = len(train_tasks) + len(eval_tasks)
    data_dir = variant['algo_params']['data_dir']
    offline_data_quality = variant['algo_params']['offline_data_quality']
    n_trj = variant['algo_params']['n_trj']
    for i in range(n_tasks):
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
                eval_trj_paths += [os.path.join(data_dir, f"goal_idx{i}", f"trj_evalsample{k}_step{j}.npy")]
        
    train_paths = [train_trj_path for train_trj_path in train_trj_paths if
                   int(train_trj_path.split('/')[-2].split('goal_idx')[-1]) in train_tasks]
    train_task_idxs = [int(train_trj_path.split('/')[-2].split('goal_idx')[-1]) for train_trj_path in train_trj_paths if
                   int(train_trj_path.split('/')[-2].split('goal_idx')[-1]) in train_tasks]
    eval_paths = [eval_trj_path for eval_trj_path in eval_trj_paths if
                  int(eval_trj_path.split('/')[-2].split('goal_idx')[-1]) in eval_tasks]
    eval_task_idxs = [int(eval_trj_path.split('/')[-2].split('goal_idx')[-1]) for eval_trj_path in eval_trj_paths if
                      int(eval_trj_path.split('/')[-2].split('goal_idx')[-1]) in eval_tasks]

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
    for eval_path, eval_task_idx in zip(eval_paths, eval_task_idxs):
        trj_npy = np.load(eval_path, allow_pickle=True)
        obs, action, reward, next_obs = np.array_split(trj_npy, [obs_dim, obs_dim+action_dim, -obs_dim], axis=-1)
        obs_eval_lst += list(obs)
        action_eval_lst += list(action)
        reward_eval_lst += list(reward)
        next_obs_eval_lst += list(next_obs)
        terminal = [0 for _ in range(trj_npy.shape[0])]
        terminal[-1] = 1
        terminal_eval_lst += terminal
        task_eval = [eval_task_idx for _ in range(trj_npy.shape[0])]
        task_eval_lst += task_eval

    train_context = ptu.from_numpy(np.concatenate([np.array(obs_train_lst), np.array(action_train_lst), np.array(reward_train_lst), np.array(next_obs_train_lst)], axis=-1))
    eval_context = ptu.from_numpy(np.concatenate([np.array(obs_eval_lst), np.array(action_eval_lst), np.array(reward_eval_lst), np.array(next_obs_eval_lst)], axis=-1))
    train_z = context_encoder(train_context[..., :context_encoder_input_dim]).detach().cpu().numpy().reshape(len(train_tasks), -1, latent_dim)
    eval_z = context_encoder(eval_context[..., :context_encoder_input_dim]).detach().cpu().numpy().reshape(len(eval_tasks), -1, latent_dim)
    train_sample_zs = np.concatenate([np.mean(train_z[:, np.random.choice(train_z.shape[1], size=variant['algo_params']['embedding_batch_size'], replace=False)], axis=1, keepdims=True) for _ in range(100)], axis=1)
    eval_sample_zs = np.concatenate([np.mean(eval_z[:, np.random.choice(eval_z.shape[1], size=variant['algo_params']['embedding_batch_size'], replace=False)], axis=1, keepdims=True) for _ in range(100)], axis=1)
    OfflineMetaRLAlgorithm.vis_task_embeddings(save_dir = str(log_dir/'figures'), fig_name='offline_mean_train_zs', zs=[train_sample_zs], subplot_title_lst=['offline_train_mean_zs'])
    OfflineMetaRLAlgorithm.vis_task_embeddings(save_dir = str(log_dir/'figures'), fig_name='offline_mean_eval_zs', zs=[eval_sample_zs], subplot_title_lst=['offline_eval_mean_zs'])

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
@click.option('--exp_name', default=None)
def main(config, gpu, seed, exp_name):
    variant = default_config
    if config:
        with open(os.path.join(config)) as f:
            exp_params = json.load(f)
        variant = deep_update_dict(exp_params, variant)
    variant['util_params']['gpu_id'] = gpu
    if not (exp_name == None):
        variant['util_params']['exp_name'] = exp_name
    show_tsne(variant, gpu, seed)
if __name__ == '__main__':
    main()