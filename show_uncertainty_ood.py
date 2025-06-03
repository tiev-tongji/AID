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

mujoco_version = "200"  # 默认版本
if "--mujoco_version" in sys.argv:
    idx = sys.argv.index("--mujoco_version")
    if idx + 1 < len(sys.argv):
        mujoco_version = sys.argv[idx + 1].strip()

# 设置 MuJoCo 环境变量
if mujoco_version == "131":
    os.environ["MUJOCO_PY_MJPRO_PATH"] = os.path.expanduser("~/.mujoco/mjpro131")
    os.environ["LD_LIBRARY_PATH"] = (
        f"{os.environ.get('LD_LIBRARY_PATH', '')}:{os.path.expanduser('~/.mujoco/mjpro131/bin')}:/usr/lib/nvidia"
    )
elif mujoco_version == "200":
    os.environ["MUJOCO_PY_MJPRO_PATH"] = "/home/autolab/.mujoco/mujoco200"
    os.environ["LD_LIBRARY_PATH"] = (
        f"{os.environ.get('LD_LIBRARY_PATH', '')}:/home/autolab/.mujoco/mujoco200/bin:/usr/lib/nvidia"
    )
else:
    raise ValueError(
        f"Unsupported MuJoCo version: {mujoco_version}. Supported versions: '131', '200'"
    )

print(f"MuJoCo version {mujoco_version} set successfully!")

from rlkit.envs import ENVS
from rlkit.envs.wrappers import NormalizedBoxEnv
from rlkit.torch.sac.policies import TanhGaussianPolicy
from rlkit.torch.multi_task_dynamics import MultiTaskDynamics
from rlkit.torch.networks import FlattenMlp, MlpEncoder, RecurrentEncoder, MlpDecoder
from rlkit.torch.sac.certain import CERTAINSoftActorCritic
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
            # print(f"goal_idx{i}, step{j}")
            if j % 400 == 0:
                continue
            for k in range(0, n_trj, 10):
                train_trj_paths += [os.path.join(data_dir, f"goal_idx{i}", f"trj_evalsample{k}_step{j}.npy")]
                eval_trj_paths += [os.path.join(data_dir, f"goal_idx{i}", f"trj_evalsample{k}_step{j}.npy")]

    train_paths = [train_trj_path for train_trj_path in train_trj_paths if int(train_trj_path.split("/")[-2].split("goal_idx")[-1]) in train_tasks]
    train_task_idxs = [int(train_trj_path.split("/")[-2].split("goal_idx")[-1]) for train_trj_path in train_trj_paths if int(train_trj_path.split("/")[-2].split("goal_idx")[-1]) in train_tasks]

    obs_train_lst = []
    action_train_lst = []
    reward_train_lst = []
    reward_train_lst_neg_task = []
    next_obs_train_lst = []
    terminal_train_lst = []
    task_train_lst = []

    obs_train_lst_neg = []
    action_train_lst_neg = []
    reward_train_lst_neg = []
    next_obs_train_lst_neg = []

    radius = 1.0
    angles = np.linspace(0, np.pi, num=20)
    xs = radius * np.cos(angles)
    ys = radius * np.sin(angles)
    goals = np.stack([xs, ys], axis=1)
    goals = goals.tolist()

    goals_neg = np.stack([-xs, -ys], axis=1)
    goals_neg = goals_neg.tolist()

    for train_path, train_task_idx in zip(train_paths, train_task_idxs):
        trj_npy = np.load(train_path, allow_pickle=True)
        obs, action, reward, next_obs = np.array_split(trj_npy, [obs_dim, obs_dim+action_dim, -obs_dim], axis=-1)
        obs_train_lst += list(obs)
        action_train_lst += list(action)
        reward_train_lst += list(reward)

        goal = np.array(goals_neg[train_task_idx])  # shape (2,)
        obs_xy = obs[:, :2]  # shape (N, 2)
        dists = np.linalg.norm(obs_xy - goal, axis=1)  # shape (N,)
        reward_neg_task = -dists[:, None]  # shape (N, 1)，保持和原 reward 结构一致

        reward_train_lst_neg_task += list(reward_neg_task)
        next_obs_train_lst += list(next_obs)
        terminal = [0 for _ in range(trj_npy.shape[0])]
        terminal[-1] = 1
        terminal_train_lst += terminal
        task_train = [train_task_idx for _ in range(trj_npy.shape[0])]
        task_train_lst += task_train

    obs_train_lst = np.array(obs_train_lst)
    action_train_lst = np.array(action_train_lst)
    reward_train_lst_neg_task = np.array(reward_train_lst_neg_task)
    reward_train_lst = np.array(reward_train_lst)

    next_obs_train_lst = np.array(next_obs_train_lst)

    obs_train_lst_neg = np.copy(obs_train_lst)
    obs_train_lst_neg[:, 1] = obs_train_lst_neg[:, 1] * -1
    action_train_lst_neg = np.copy(action_train_lst)
    action_train_lst_neg[:, 1] = action_train_lst_neg[:, 1] * -1
    reward_train_lst_neg = np.copy(reward_train_lst_neg_task)
    next_obs_train_lst_neg = np.copy(next_obs_train_lst)
    next_obs_train_lst_neg[:, 1] = next_obs_train_lst_neg[:, 1] * -1

    train_context = ptu.from_numpy(np.concatenate([np.array(obs_train_lst), np.array(action_train_lst), np.array(reward_train_lst), np.array(next_obs_train_lst)], axis=-1))
    train_z = context_encoder(train_context[..., :context_encoder_input_dim])
    train_z_var = F.softplus(uncertainty_mlp(train_z)).detach().cpu().numpy()
    
    train_context_neg_task = ptu.from_numpy(np.concatenate([np.array(obs_train_lst), np.array(action_train_lst), np.array(reward_train_lst_neg_task), np.array(next_obs_train_lst)], axis=-1))
    train_z_neg_task = context_encoder(train_context_neg_task[..., :context_encoder_input_dim])
    train_z_var_neg_task = F.softplus(uncertainty_mlp(train_z_neg_task)).detach().cpu().numpy()
    
    train_context_neg = ptu.from_numpy(np.concatenate([np.array(obs_train_lst_neg), np.array(action_train_lst_neg), np.array(reward_train_lst_neg), np.array(next_obs_train_lst_neg)], axis=-1))
    train_z_neg = context_encoder(train_context_neg[..., :context_encoder_input_dim])
    train_z_var_neg = F.softplus(uncertainty_mlp(train_z_neg)).detach().cpu().numpy()

    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    import seaborn as sns
    from matplotlib import rcParams

    rcParams.update({"font.size": 16})
    sns.set_theme(style="white", font_scale=2.0)
    fig, (ax2, ax3) = plt.subplots(1, 2, figsize=(20, 12))

    ax2.set_xlim(-1.2, 1.2)
    ax2.set_ylim(-1.2, 1.2)
    ax2.set_frame_on(False)
    circle2 = patches.Arc((0, 0), 2, 2, angle=0, theta1=0, theta2=360, color=(180./255., 180./255., 180./255.), linewidth=2)
    ax2.add_artist(circle2)
    ax2.set_aspect('equal')
    ax2.set_xticks([])
    ax2.set_yticks([])
    
    ax3.set_xlim(-1.2, 1.2)
    ax3.set_ylim(-1.2, 1.2)
    ax3.set_frame_on(False)
    circle3 = patches.Arc((0, 0), 2, 2, angle=0, theta1=0, theta2=360, color=(180./255., 180./255., 180./255.), linewidth=2)
    ax3.add_artist(circle3)
    ax3.set_aspect('equal')
    ax3.set_xticks([])
    ax3.set_yticks([])

    # 绘制训练数据的散点热力图
    vmin1 = np.percentile(train_z_var, 5)
    vmax1 = np.percentile(train_z_var, 95)
    vmin2 = np.percentile(train_z_var_neg, 5)
    vmax2 = np.percentile(train_z_var_neg, 95)
    vmin3 = np.percentile(train_z_var_neg_task, 5)
    vmax3 = np.percentile(train_z_var_neg_task, 95)
    vmin = min(vmin1, vmin2, vmin3)
    vmax = max(vmax1, vmax2, vmax3)
    sample_ids = np.random.choice(len(obs_train_lst), 800, replace=False)
    lines_for_legend = []
    for i in tqdm(sample_ids, desc="Plotting train z variance"):
        z_var_neg = train_z_var_neg[i]
        normed_val_neg = np.clip((z_var_neg - vmin) / (vmax - vmin), 0, 1)
        z_var_color_neg = plt.cm.coolwarm(normed_val_neg)
        ax2.scatter(obs_train_lst_neg[i][0], obs_train_lst_neg[i][1], c=[z_var_color_neg], s=10)
        
        z_var_neg_task = train_z_var_neg_task[i]
        normed_val_neg_task = np.clip((z_var_neg_task - vmin) / (vmax - vmin), 0, 1)
        z_var_color_neg_task = plt.cm.coolwarm(normed_val_neg_task)
        ax3.scatter(obs_train_lst[i][0], obs_train_lst[i][1], c=[z_var_color_neg_task], s=10)

    # 绘制 goal
    neg_goal_color = (0.9, 0.2, 0.4)
    goal_color = (40 / 225, 75 / 225, 113 / 225)
    goal_size = 200
    goals_np = np.array(goals[::2])
    goal_point = ax2.scatter(goals_np[:, 0], goals_np[:, 1], color=goal_color, s=goal_size, marker='*', label='Goals(ID)')
    goals_neg_np = np.array(goals_neg[::2])
    goal_point_neg = ax3.scatter(goals_neg_np[:, 0], goals_neg_np[:, 1], color=neg_goal_color, s=goal_size, marker='*', label='Goals (OOD)')
    lines_for_legend.append(goal_point)
    lines_for_legend.append(goal_point_neg)
    
    ax2.set_title('Ood Context')
    ax3.set_title('Ood Task')

    plt.legend(handles=lines_for_legend, loc="upper right", fontsize=16)

    sm = plt.cm.ScalarMappable(cmap="coolwarm", norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    fig.subplots_adjust(bottom=0.15)
    cbar_ax = fig.add_axes([0.2, 0.05, 0.6, 0.02])  # [left, bottom, width, height]
    fig.colorbar(sm, cax=cbar_ax, orientation="horizontal")
    plt.tight_layout()
    plt.savefig("uncertainty.pdf")

def deep_update_dict(fr, to):
    """update dict of dicts with new values"""
    # assume dicts have same keys
    for k, v in fr.items():
        if type(v) is dict:
            deep_update_dict(v, to[k])
        else:
            to[k] = v
    return to


# python show_uncertainty_ood.py configs/point-robot.json --gpu 0 --seed 0 --exp_name CLASSIFIER0349/classifier_mix_z0_hvar_p10_weighted --algo_type CLASSIFIER
# python show_uncertainty_ood.py configs/point-robot.json --gpu 0 --seed 0 --exp_name FOCAL0135/focal_mix_z0_hvar_p10_weighted --algo_type FOCAL
# python show_uncertainty_ood.py configs/point-robot.json --gpu 0 --seed 1 --exp_name UNICORN1237/unicorn_mix_z0_hvar_p2_weighted --algo_type UNICORN
@click.command()
@click.argument("config", default=None)
@click.option(
    "--mujoco_version",
    type=click.Choice(["131", "200"], case_sensitive=False),
    default="200",
    help="MuJoCo version, default is --mujoco_version=200",
)
@click.option("--gpu", default=0)
@click.option("--seed", default=0)
@click.option("--exp_name", default=None)
@click.option("--algo_type", default=None)
def main(config, mujoco_version, gpu, seed, exp_name, algo_type):
    variant = default_config
    if config:
        with open(os.path.join(config)) as f:
            exp_params = json.load(f)
        variant = deep_update_dict(exp_params, variant)
    variant["util_params"]["gpu_id"] = gpu
    if not (exp_name == None):
        variant["util_params"]["exp_name"] = exp_name
    if not (algo_type == None):
        variant["algo_type"] = algo_type
    show_uncertainty(variant, gpu, seed)


if __name__ == "__main__":
    main()