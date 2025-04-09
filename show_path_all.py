"""
Launcher for experiments with CSRO

"""
import os
import glob
from pathlib import Path
import numpy as np
import click
import json
import ast
import torch
import random
import multiprocessing as mp
from itertools import product
import torch.nn.functional as F
import sys
from tensorboardX import SummaryWriter

import numpy as np
np.int = int  # 动态修复 np.int 被废弃的问题


mujoco_version = '200'  # 默认版本
if '--mujoco_version' in sys.argv:
    idx = sys.argv.index('--mujoco_version')
    if idx + 1 < len(sys.argv):
        mujoco_version = sys.argv[idx + 1].strip()

# 设置 MuJoCo 环境变量
if mujoco_version == '131':
        os.environ['MUJOCO_PY_MJPRO_PATH'] = os.path.expanduser('~/.mujoco/mjpro131')
        os.environ['LD_LIBRARY_PATH'] = f"{os.environ.get('LD_LIBRARY_PATH', '')}:{os.path.expanduser('~/.mujoco/mjpro131/bin')}:/usr/lib/nvidia"
elif mujoco_version == '200':
    os.environ['MUJOCO_PY_MJPRO_PATH'] = '/home/autolab/.mujoco/mujoco200'
    os.environ['LD_LIBRARY_PATH'] = f"{os.environ.get('LD_LIBRARY_PATH', '')}:/home/autolab/.mujoco/mujoco200/bin:/usr/lib/nvidia"
else:
    raise ValueError(f"Unsupported MuJoCo version: {mujoco_version}. Supported versions: '131', '200'")

print(f"MuJoCo version {mujoco_version} set successfully!")


from rlkit.envs import ENVS
from rlkit.envs.wrappers import NormalizedBoxEnv
from rlkit.torch.sac.policies import TanhGaussianPolicy
from rlkit.torch.multi_task_dynamics import MultiTaskDynamics
from rlkit.torch.networks import FlattenMlp, MlpEncoder, RecurrentEncoder, MlpDecoder
from rlkit.torch.sac.sac import CERTAINSoftActorCritic
from rlkit.torch.sac.agent import PEARLAgent
from rlkit.launchers.launcher_util import setup_logger
import rlkit.torch.pytorch_util as ptu
from configs.default import default_config

def global_seed(seed=0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def Recon_loss(context, latent_z, obs_dim, action_dim, context_decoder, epsilon=1e-8):
    # b, _ = context.shape
    # latent_z = params.expand(b, -1) # [10, 1024, 20]
    # 10, 1024, 1
    epsilon = 1e-8
    r_ns = context[..., obs_dim+action_dim:]
    r_ns = context[..., obs_dim+action_dim:]
    s_z_a = torch.cat([context[..., : obs_dim], latent_z, context[..., obs_dim : obs_dim + action_dim]], dim=-1)
    pre_r_ns_param = context_decoder(s_z_a)
    split_size = int(context_decoder.output_size/2)
    pre_r_ns_mean = pre_r_ns_param[..., :split_size]
    pre_r_ns_var = F.softplus(pre_r_ns_param[..., split_size:])
    # 计算高斯
    probs = - (r_ns - pre_r_ns_mean)**2 / (pre_r_ns_var + epsilon) - torch.log(pre_r_ns_var**0.5 + epsilon)
    return - probs

def calculate(variant, gpu_id, seed):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    ptu.set_gpu_mode(True, gpu_id)
    # create multi-task environment and sample tasks, normalize obs if provided with 'normalizer.npz'
    if 'normalizer.npz' in os.listdir(variant['algo_params']['data_dir']):
        obs_absmax = np.load(os.path.join(variant['algo_params']['data_dir'], 'normalizer.npz'))['abs_max']
        env = NormalizedBoxEnv(ENVS[variant['env_name']](**variant['env_params']), obs_absmax=obs_absmax)
    else:
        env = NormalizedBoxEnv(ENVS[variant['env_name']](**variant['env_params']))

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

    context_decoder = MlpDecoder(
        hidden_sizes=[200, 200, 200],
        input_size=latent_dim+obs_dim+action_dim,
        output_size=2*(reward_dim+obs_dim) if variant['algo_params']['use_next_obs_in_context'] else 2*reward_dim,
        layer_norm=variant['algo_params']['layer_norm'] if 'layer_norm' in variant['algo_params'].keys() else False
    )

    classifier = MlpDecoder(
        hidden_sizes=[net_size],
        input_size=context_encoder_output_dim,
        output_size=variant['n_train_tasks'],
        layer_norm=variant['algo_params']['layer_norm'] if 'layer_norm' in variant['algo_params'].keys() else False
    )

    uncertainty_mlp = MlpDecoder(
        hidden_sizes=[net_size],
        input_size=latent_dim,
        output_size=1,
    )

    exp_name = variant['util_params']['exp_name']
    base_log_dir = variant['util_params']['base_log_dir']
    exp_prefix = variant['env_name']
    log_dir = Path(os.path.join(base_log_dir, exp_prefix.replace("_", "-"), exp_name, f"seed{seed}"))
    agent_path = log_dir/"agent.pth"
    if not agent_path.exists():
        exit(f"agent path {str(agent_path)} does not exist")
    agent_ckpt = torch.load(str(agent_path), weights_only=True)
    context_encoder.load_state_dict(agent_ckpt['context_encoder'])
    uncertainty_mlp.load_state_dict(agent_ckpt['uncertainty_mlp'])
    context_decoder.load_state_dict(agent_ckpt['context_decoder'])
    classifier.load_state_dict(agent_ckpt['classifier'])
    context_decoder.to(ptu.device)
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
            print(f'goal_idx{i}, step{j}')
            # if j % 400 == 0:
            #     continue
            for k in range(0, n_trj, 10):
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
    # eval_context = ptu.from_numpy(np.concatenate([np.array(obs_eval_lst), np.array(action_eval_lst), np.array(reward_eval_lst), np.array(next_obs_eval_lst)], axis=-1))
    train_z = context_encoder(train_context[..., :context_encoder_input_dim])
    # eval_z = context_encoder(eval_context[..., :context_encoder_input_dim])
    train_z_var = F.softplus(uncertainty_mlp(train_z)).detach().cpu().numpy()

    min_5 = np.percentile(train_z_var, 5)
    max_95 = np.percentile(train_z_var, 95)
    print(f'数据量5分位数: {min_5}')
    print(f'数据量95分位数: {max_95}')
    return min_5, max_95

def experiment(gpu_id, variant, seed=None, exp_names=None):
    min_5, max_95 = calculate(variant, gpu_id, seed)

    os.sched_setaffinity(0, [gpu_id*8+i for i in range(8)])
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

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

    if variant['algo_params']['club_use_sa']:
        club_input_dim = obs_dim + action_dim
    else:
        club_input_dim = obs_dim + action_dim + reward_dim if variant['algo_params']['use_next_obs_in_context'] else obs_dim + action_dim

    club_model = encoder_model(
        hidden_sizes=[200, 200, 200],
        input_size=club_input_dim,
        output_size=latent_dim * 2,
        output_activation=torch.tanh,
    )
    
    context_encoder = encoder_model(
        hidden_sizes=[200, 200, 200],
        input_size=context_encoder_input_dim,
        output_size=context_encoder_output_dim,
        output_activation=torch.tanh,
        layer_norm=variant['algo_params']['layer_norm'] if 'layer_norm' in variant['algo_params'].keys() else False
    )

    context_decoder = MlpDecoder(
        hidden_sizes=[200, 200, 200],
        input_size=latent_dim+obs_dim+action_dim,
        output_size=2*(reward_dim+obs_dim) if variant['algo_params']['use_next_obs_in_context'] else 2*reward_dim,
        layer_norm=variant['algo_params']['layer_norm'] if 'layer_norm' in variant['algo_params'].keys() else False
    )
    
    classifier = MlpDecoder(
        hidden_sizes=[net_size],
        input_size=context_encoder_output_dim,
        output_size=variant['n_train_tasks'],
        layer_norm=variant['algo_params']['layer_norm'] if 'layer_norm' in variant['algo_params'].keys() else False
    )

    uncertainty_mlp = MlpDecoder(
        hidden_sizes=[net_size],
        input_size=latent_dim,
        output_size=1,
    )

    reward_models = torch.nn.ModuleList()
    dynamic_models = torch.nn.ModuleList()
    for _ in range(variant['algo_params']['num_ensemble']):
        reward_models.append(
            FlattenMlp(hidden_sizes=[net_size, net_size, net_size],
                       input_size=latent_dim + obs_dim + action_dim,
                       output_size=1, )
        )
        dynamic_models.append(
            FlattenMlp(hidden_sizes=[net_size, net_size, net_size],
                       input_size=latent_dim + obs_dim + action_dim,
                       output_size=obs_dim, )
        )

    qf1 = FlattenMlp(
        hidden_sizes=[net_size, net_size, net_size],
        input_size=obs_dim + action_dim + latent_dim,
        output_size=1,
    )
    qf2 = FlattenMlp(
        hidden_sizes=[net_size, net_size, net_size],
        input_size=obs_dim + action_dim + latent_dim,
        output_size=1,
    )
    vf = FlattenMlp(
        hidden_sizes=[net_size, net_size, net_size],
        input_size=obs_dim + latent_dim,
        output_size=1,
    )

    c = FlattenMlp(
        hidden_sizes=[net_size, net_size, net_size],
        input_size=obs_dim + action_dim + latent_dim,
        output_size=1
    )

    policy = TanhGaussianPolicy(
        hidden_sizes=[net_size, net_size, net_size],
        obs_dim=obs_dim + latent_dim,
        latent_dim=latent_dim,
        action_dim=action_dim,
    )

    agent = PEARLAgent(
        latent_dim,
        context_encoder,
        uncertainty_mlp,
        policy,
        **variant['algo_params']
    )
    
    if variant['algo_params']['use_next_obs_in_context']:
        task_dynamics =  MultiTaskDynamics(num_tasks=len(tasks), 
            hidden_size=net_size, 
            num_hidden_layers=3, 
            action_dim=action_dim, 
            obs_dim=obs_dim,
            reward_dim=1,
            use_next_obs_in_context=variant['algo_params']['use_next_obs_in_context'],
            ensemble_size=variant['algo_params']['ensemble_size'],
            dynamics_weight_decay=[2.5e-5, 5e-5, 7.5e-5, 7.5e-5])
    else:
        task_dynamics = MultiTaskDynamics(num_tasks=len(tasks), 
            hidden_size=net_size, 
            num_hidden_layers=2, 
            action_dim=action_dim, 
            obs_dim=obs_dim,
            reward_dim=1,
            use_next_obs_in_context=variant['algo_params']['use_next_obs_in_context'],
            ensemble_size=variant['algo_params']['ensemble_size'],
            dynamics_weight_decay=[2.5e-5, 5e-5, 7.5e-5])

    # Setting up tasks
    if 'randomize_tasks' in variant.keys() and variant['randomize_tasks']:
        train_tasks = np.random.choice(len(tasks), size=variant['n_train_tasks'], replace=False)
    elif 'interpolation' in variant.keys() and variant['interpolation']:
        step = len(tasks)/variant['n_train_tasks']
        train_tasks = np.array([tasks[int(i*step)] for i in range(variant['n_train_tasks'])])
    eval_tasks = np.array(list(set(range(len(tasks))).difference(train_tasks)))
    goal_radius = variant['env_params']['goal_radius'] if 'goal_radius' in variant['env_params'] else 1
    
    # Choose algorithm
    algo_type = variant['algo_type']
    
    variant['util_params']['exp_name'] = exp_names[0]
    algorithm = CERTAINSoftActorCritic(
        env=env,
        train_tasks=train_tasks,
        eval_tasks=eval_tasks,
        nets=[agent, qf1, qf2, vf, c, club_model, context_decoder, classifier, reward_models, dynamic_models, task_dynamics],
        latent_dim=latent_dim,
        goal_radius=goal_radius,
        seed=seed,
        algo_type=algo_type,
        env_name = variant['env_name'],
        **variant['algo_params'],
    )
    variant['util_params']['exp_name'] = exp_names[1]
    algorithm_base = CERTAINSoftActorCritic(
        env=env,
        train_tasks=train_tasks,
        eval_tasks=eval_tasks,
        nets=[agent, qf1, qf2, vf, c, club_model, context_decoder, classifier, reward_models, dynamic_models, task_dynamics],
        latent_dim=latent_dim,
        goal_radius=goal_radius,
        seed=seed,
        algo_type=algo_type,
        env_name = variant['env_name'],
        **variant['algo_params'],
    )

    DEBUG = False
    os.environ['DEBUG'] = str(int(DEBUG))

    # directory
    base_log_dir = variant['util_params']['base_log_dir']
    exp_prefix = variant['env_name']
    log_dir_1 = Path(os.path.join(base_log_dir, exp_prefix.replace("_", "-"), exp_names[0], f"seed{seed}"))
    
    ##############################################
    file_list_1 = glob.glob(os.path.join(log_dir_1, 'agent_*.pth'))
    file_list_1.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))

    for i, file in enumerate(file_list_1):
        agent_ckpt_1 = torch.load(str(file))
        algorithm.agent.policy.load_state_dict(agent_ckpt_1['policy'])
        algorithm.agent.uncertainty_mlp.load_state_dict(agent_ckpt_1['uncertainty_mlp'])
        algorithm.agent.context_encoder.load_state_dict(agent_ckpt_1['context_encoder'])
        # algorithm.agent.context_decoder.load_state_dict(agent_ckpt_1['context_decoder'])
        if ptu.gpu_enabled():
            algorithm.to()
        algorithm.draw_path(5 * i, str(log_dir_1), min_5 = min_5, max_95 = max_95)
    
    ##############################################
    # agent_path_1 = log_dir_1/"agent.pth"
    # agent_ckpt_1 = torch.load(str(agent_path_1))
    # print("agent_path_1: ", agent_path_1)

    # algorithm.agent.policy.load_state_dict(agent_ckpt_1['policy'])
    # algorithm.agent.uncertainty_mlp.load_state_dict(agent_ckpt_1['uncertainty_mlp'])
    # algorithm.agent.context_encoder.load_state_dict(agent_ckpt_1['context_encoder'])
    # # algorithm.agent.context_decoder.load_state_dict(agent_ckpt_1['context_decoder'])
    
    # ptu.set_gpu_mode(variant['util_params']['use_gpu'], variant['util_params']['gpu_id'])
    # if ptu.gpu_enabled():
    #     algorithm.to()
    
    # if variant['env_name'] == 'point-robot':
    #     # algorithm.draw_path(variant['algo_params']['num_iterations'], str(log_dir_1), min_5 = min_5, max_95 = max_95)
    #     # algorithm.draw_neg_path(variant['algo_params']['num_iterations'], str(log_dir_1), min_5 = min_5, max_95 = max_95)
    #     # algorithm.draw_manual_path(variant['algo_params']['num_iterations'], str(log_dir_1))
    #     first_path = algorithm.draw_in_and_out_of_distribution_path(variant['algo_params']['num_iterations'], str(log_dir_1), min_5 = 0.9, max_95 = max_95)
    # # algorithm.draw_z(variant['algo_params']['num_iterations'], str(log_dir_1))


    log_dir_2 = Path(os.path.join(base_log_dir, exp_prefix.replace("_", "-"), exp_names[1], f"seed{seed}"))
    
    ##############################################
    file_list_2 = glob.glob(os.path.join(log_dir_2, 'agent_*.pth'))
    file_list_2.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))

    for i, file in enumerate(file_list_2):
        agent_ckpt_2 = torch.load(str(file))
        algorithm_base.agent.policy.load_state_dict(agent_ckpt_2['policy'])
        algorithm_base.agent.uncertainty_mlp.load_state_dict(agent_ckpt_1['uncertainty_mlp'])
        algorithm_base.agent.context_encoder.load_state_dict(agent_ckpt_2['context_encoder'])
        # algorithm.agent.context_decoder.load_state_dict(agent_ckpt_1['context_decoder'])
        algorithm_base.agent.z_strategy = 'mean'
        if ptu.gpu_enabled():
            algorithm_base.to()
        algorithm_base.draw_path(5 * i, str(log_dir_2), min_5 = min_5, max_95 = max_95)
    
    ##############################################
    # agent_path_2 = log_dir_2/"agent.pth"
    # agent_ckpt_2 = torch.load(str(agent_path_2))
    # print("agent_path_2: ", agent_path_2)

    # algorithm_base.agent.policy.load_state_dict(agent_ckpt_2['policy'])
    # algorithm_base.agent.uncertainty_mlp.load_state_dict(agent_ckpt_1['uncertainty_mlp'])
    # algorithm_base.agent.context_encoder.load_state_dict(agent_ckpt_2['context_encoder'])
    # algorithm_base.agent.z_strategy = 'mean'
    # # algorithm_base.agent.context_decoder.load_state_dict(agent_ckpt_2['context_decoder'])

    # ptu.set_gpu_mode(variant['util_params']['use_gpu'], variant['util_params']['gpu_id'])
    # if ptu.gpu_enabled():
    #     algorithm_base.to()
    
    # if variant['env_name'] == 'point-robot':
    #     # algorithm_base.draw_path(variant['algo_params']['num_iterations'], str(log_dir_2), min_5 = min_5, max_95 = max_95)
    #     # algorithm_base.draw_neg_path(variant['algo_params']['num_iterations'], str(log_dir_2), min_5 = min_5, max_95 = max_95)
    #     algorithm_base.draw_in_and_out_of_distribution_path(variant['algo_params']['num_iterations'], str(log_dir_2), min_5 = 0.9, max_95 = max_95, path = first_path)
    # # algorithm.draw_z(variant['algo_params']['num_iterations'], str(log_dir_2))
    
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
@click.option('--mujoco_version', type=click.Choice(['131', '200'], case_sensitive=False), default='200', help='MuJoCo version, default is --mujoco_version=200')
@click.option('--gpu', default="0,1,2,3", type=str, help="Comma-separated list of gpu.")
@click.option('--seed', default="0", type=str, help="Comma-separated list of seeds.")
@click.option('--algo_type', type=click.Choice(['FOCAL', 'CSRO', 'CORRO', 'UNICORN', 'CLASSIFIER', 'IDAQ'], case_sensitive=False), default=None)
@click.option('--train_z0_policy', type=click.Choice(['true', 'false'], case_sensitive=False), default=None)
@click.option('--use_hvar', type=click.Choice(['true', 'false'], case_sensitive=False), default=None)
@click.option('--z_strategy', type=click.Choice(['mean', 'min', 'weighted', 'quantile'], case_sensitive=False), default=None)
@click.option('--r_thres', default=None)
# python show_path_2.py configs/point-robot.json --gpu 0 --seed 0 --algo_type FOCAL --train_z0_policy true --use_hvar true --z_strategy weighted
def main(config, mujoco_version, gpu, seed, algo_type=None, train_z0_policy = None, use_hvar = None, z_strategy = None, r_thres=None):
    variant = default_config
    if config:
        with open(os.path.join(config)) as f:
            exp_params = json.load(f)
        variant = deep_update_dict(exp_params, variant)

    gpu = [int(g) for g in gpu.split(",")]
    print(f"Parsed gpus: {gpu}")
    variant['util_params']['gpu_id'] = gpu

    # exp_names = ['FOCAL0135/focal_mix_z0_hvar_p10_weighted', 'FOCAL0135/focal_mix_baseline']
    exp_names = ['CLASSIFIER0349/classifier_mix_z0_hvar_p10_weighted', 'CLASSIFIER0349/classifier_mix_baseline']
    # exp_names = ['UNICORN1237/unicorn_mix_z0_hvar_weighted', 'UNICORN1237/unicorn_mix_baseline']
    
    variant['util_params']['exp_name'] = exp_names[0]
    variant['algo_params']['pretrain'] = True
    if not (algo_type == None):
        variant['algo_type'] = algo_type.upper()
    if not (train_z0_policy == None):
        variant['algo_params']['train_z0_policy'] = train_z0_policy.lower() == 'true'
    if not (use_hvar == None):
        variant['algo_params']['use_hvar'] = use_hvar.lower() == 'true'
    if not (z_strategy == None):
        variant['algo_params']['z_strategy'] = z_strategy
    if not (r_thres == None):
        variant['algo_params']['r_thres'] = float(r_thres)

    seed = [int(s) for s in seed.split(",")]
    experiment(gpu_id=gpu[0], variant=variant, seed=seed[0], exp_names=exp_names)

if __name__ == "__main__":
    main()

