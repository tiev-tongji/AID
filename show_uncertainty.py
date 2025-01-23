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
import sys
import ast

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
from rlkit.torch.sac.sac import CSROSoftActorCritic
from rlkit.torch.sac.agent import PEARLAgent
from rlkit.launchers.launcher_util import setup_logger
import rlkit.torch.pytorch_util as ptu
from configs.default import default_config
from tqdm import tqdm

def global_seed(seed=0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

# def FOCAL_z_loss_per_sample(task_indices, task_z, sample_size_pos=5, sample_size_neg=5, epsilon=1e-3):
def FOCAL_z_loss_per_sample(task_indices, task_z, sample_size_pos=0, sample_size_neg=9, epsilon=1e-3):
    """
    task_indices: 每个样本的任务索引 (长度为样本数)
    task_z: 样本表示向量 (形状 [数量, 维度])
    sample_size: 每个样本计算时随机采样的其他样本数量
    epsilon: 防止数值问题的小值
    """
    num_samples = len(task_indices)
    sample_losses = torch.zeros(num_samples)  # 存储每个样本的 loss

    # 预计算任务索引对应的样本集合
    task_to_indices = {}
    for idx, task in enumerate(task_indices):
        if int(task) not in task_to_indices:
            task_to_indices[int(task)] = []
        task_to_indices[int(task)]+=[idx]

    print(f'task_to_indices: {num_samples}')
    
    for i in range(num_samples):
        pos_z_loss = 0.0
        neg_z_loss = 0.0
        pos_cnt = 0
        neg_cnt = 0
        
        # 当前样本的任务标签
        current_task = task_indices[i]
        
        # 正对采样：从当前任务的预计算索引集合中排除自己，并采样
        pos_candidates = task_to_indices[int(current_task)]
        pos_candidates = [j for j in pos_candidates if j != i]
        if len(pos_candidates) > 0:
            pos_samples = torch.tensor(pos_candidates)[torch.randint(0, len(pos_candidates), (min(len(pos_candidates), sample_size_pos),))]
        else:
            pos_samples = []

        # 负对采样：从非当前任务的所有预计算集合中采样
        neg_candidates = [j for task, indices in task_to_indices.items() if task != current_task for j in indices]
        if len(neg_candidates) > 0:
            neg_samples = torch.tensor(neg_candidates)[torch.randint(0, len(neg_candidates), (min(len(neg_candidates), sample_size_neg),))]
        else:
            neg_samples = []

        # 计算正对损失
        for j in pos_samples:
            pos_z_loss += torch.sqrt(torch.mean((task_z[i] - task_z[j]) ** 2) + epsilon)
            pos_cnt += 1

        # 计算负对损失
        for j in neg_samples:
            neg_z_loss += 1 / (torch.mean((task_z[i] - task_z[j]) ** 2) + epsilon * 100)
            neg_cnt += 1

        # 当前样本的 loss
        sample_loss = (pos_z_loss / (pos_cnt + epsilon)) + (neg_z_loss / (neg_cnt + epsilon))
        sample_losses[i] = sample_loss
        if(i % 300 == 0):
            print(f'i:{i}, pos_z_loss: {pos_z_loss}, neg_z_loss: {neg_z_loss}, sample_loss: {sample_loss}')
            # return sample_losses
    return sample_losses

def Recon_loss(pre_r_ns_param, split_size, r_ns, epsilon=1e-8):
    pre_r_ns_mean = pre_r_ns_param[..., :split_size]
    pre_r_ns_var = F.softplus(pre_r_ns_param[..., split_size:])
    # 计算高斯
    probs = -(r_ns-pre_r_ns_mean)**2/(pre_r_ns_var+epsilon) - torch.log(pre_r_ns_var**0.5+epsilon)
    return -torch.mean(probs).detach().cpu().numpy()

def show_uncertainty(variant, gpu_id, seed):
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
    agent_ckpt = torch.load(str(agent_path))
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
            if j % 400 == 0:
                continue
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
    eval_context = ptu.from_numpy(np.concatenate([np.array(obs_eval_lst), np.array(action_eval_lst), np.array(reward_eval_lst), np.array(next_obs_eval_lst)], axis=-1))
    train_z = context_encoder(train_context[..., :context_encoder_input_dim])
    eval_z = context_encoder(eval_context[..., :context_encoder_input_dim])
    train_z_var = F.softplus(uncertainty_mlp(train_z)).detach().cpu().numpy()
    eval_z_var = F.softplus(uncertainty_mlp(eval_z)).detach().cpu().numpy()
    print(f'10%分位数: {train_z_var.min() + 0.1*(train_z_var.max()-train_z_var.min())}')
    train_labels = torch.tensor([np.where(train_tasks == task_id)[0] for task_id in task_train_lst]).to(ptu.device).reshape(-1)

    if variant['algo_type'] == 'FOCAL':
        focal_loss = FOCAL_z_loss_per_sample(train_labels, train_z).detach().cpu().numpy()
        train_z_var[train_z_var > 2.0] = 2.0
        loss = focal_loss
        loss [loss > 2.0] = 2.0

    elif variant['algo_type'] == 'UNICORN':
        focal_loss = FOCAL_z_loss_per_sample(train_labels, train_z).detach().cpu().numpy()
        r_ns = train_context[..., obs_dim + action_dim:] # return nextstate
        s_z_a = torch.cat([train_context[..., :obs_dim], train_z, train_context[..., obs_dim: obs_dim + action_dim]], dim=-1) # state, z, action
        pre_r_ns_param = context_decoder(s_z_a)
        split_size = int(context_decoder.output_size/2)
        recon_loss = Recon_loss(pre_r_ns_param, split_size, r_ns)
        unicorn_focal_weight = variant['algo_params']['unicorn_focal_weight']
        unicorn_loss = recon_loss + unicorn_focal_weight * focal_loss
        loss = unicorn_loss
    
    elif variant['algo_type'] == 'CLASSIFIER':
        classifier_loss = F.cross_entropy(classifier(train_z), train_labels, reduction='none').detach().cpu().numpy()
        train_z_var[train_z_var > 0.01] = 0.01
        loss = classifier_loss
        loss[loss > 0.01] = 0.01

    print(f'train_context shape: {train_context.shape}')
    print(f'eval_context shape: {eval_context.shape}')
    print(f'train_z shape: {train_z.shape}')
    print(f'eval_z shape: {eval_z.shape}')
    print(f'train_z_var shape: {train_z_var.shape}')
    print(f'eval_z_var shape: {eval_z_var.shape}')
    print(f'loss shape: {loss.shape}')

    import matplotlib.pyplot as plt
    fig, ((ax1, ax2), (ax5, ax6)) = plt.subplots(2, 2, figsize=(16, 8), height_ratios=[2, 1])

    ax1.set_xlim(-1.2, 1.2)
    ax1.set_ylim(-0.2, 1.2)
    ax2.set_xlim(-1.2, 1.2)
    ax2.set_ylim(-0.2, 1.2)

    # 绘制训练数据的散点热力图
    sample_ids = np.random.choice(len(obs_train_lst), 1000, replace=False)
    # sample_ids = random.sample(range(5000), 1000)
    for i in tqdm(sample_ids, desc='Plotting train z variance'):
        z_var = train_z_var[i]
        loss_i = loss[i]
        # 根据 z_var 大小设置热力图颜色
        # z_var_color = plt.cm.coolwarm(z_var / train_z_var.max())
        # loss_color = plt.cm.coolwarm(loss_i / loss.max())
        z_var_color = plt.cm.coolwarm((z_var - train_z_var.min()) / (train_z_var.max() - train_z_var.min()))  # 应用颜色映射
        loss_color = plt.cm.coolwarm((loss_i - (loss.min())) / (loss.max() - (loss.min())))
        ax1.scatter(obs_train_lst[i][0], obs_train_lst[i][1], c=z_var_color, s=10)
        ax2.scatter(obs_train_lst[i][0], obs_train_lst[i][1], c=loss_color, s=10)
        # if z_var >= train_z_var.min() + 0.1 * (train_z_var.max() - train_z_var.min()):
        #     ax3.scatter(obs_train_lst[i][0], obs_train_lst[i][1], c=z_var_color, s=10)
        # if loss_i >= loss.min() + 0.1 * (loss.max() - (loss.min())):
        #     ax4.scatter(obs_train_lst[i][0], obs_train_lst[i][1], c=loss_color, s=10)

    # 为 ax1 和 ax2 设置热力图颜色条
    sm = plt.cm.ScalarMappable(cmap='coolwarm', norm=plt.Normalize(vmin=train_z_var.min(), vmax=train_z_var.max()))
    sm.set_array([])
    fig.colorbar(sm, ax=ax1, orientation='horizontal', label='Uncertainty')

    sm = plt.cm.ScalarMappable(cmap='coolwarm', norm=plt.Normalize(vmin=loss.min(), vmax=loss.max()))
    sm.set_array([])
    fig.colorbar(sm, ax=ax2, orientation='horizontal', label='Loss') 

    ax1.set_title('Train Uncertainty')
    ax2.set_title('Loss')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')

    # 画z_var的分布直方图
    # Create histograms with different colors for each bin and black edges
    train_hist, train_bins, _ = ax5.hist(train_z_var, bins=10, alpha=0.5, label='train', edgecolor='black')
    eval_hist, eval_bins, _ = ax6.hist(loss, bins=10, alpha=0.5, label='loss', edgecolor='black')
    print(f'eval_hist: {eval_hist}')
    print(f'eval_bins: {eval_bins}')

    plt.suptitle(f"{variant['util_params']['exp_name']}", fontsize=16)

    plt.tight_layout()
    # 保存图片
    plt.savefig('uncertainty.png')
    
def deep_update_dict(fr, to):
    ''' update dict of dicts with new values '''
    # assume dicts have same keys
    for k, v in fr.items():
        if type(v) is dict:
            deep_update_dict(v, to[k])
        else:
            to[k] = v
    return to

# python show_uncertainty.py configs/point-robot.json --gpu 0 --seed 5 --exp_name focal_mix_z0_hvar_p10_weighted --algo_type FOCAL --mujoco_version 200
@click.command()
@click.argument('config', default=None)
@click.option('--mujoco_version', type=click.Choice(['131', '200'], case_sensitive=False), default='200', help='MuJoCo version, default is --mujoco_version=200')
@click.option('--gpu', default=0)
@click.option('--seed', default=0)
@click.option('--exp_name', default=None)
@click.option('--algo_type', default=None)
def main(config, mujoco_version, gpu, seed, exp_name, algo_type):
    variant = default_config
    if config:
        with open(os.path.join(config)) as f:
            exp_params = json.load(f)
        variant = deep_update_dict(exp_params, variant)
    variant['util_params']['gpu_id'] = gpu
    if not (exp_name == None):
        variant['util_params']['exp_name'] = exp_name
    if not (algo_type == None):
        variant['algo_type'] = algo_type
    show_uncertainty(variant, gpu, seed)
if __name__ == '__main__':
    main()