import os
import click
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from configs.default import default_config
from rlkit.envs import ENVS

def deep_update_dict(fr, to):
    ''' update dict of dicts with new values '''
    # assume dicts have same keys
    for k, v in fr.items():
        if type(v) is dict:
            deep_update_dict(v, to[k])
        else:
            to[k] = v
    return to

def show_traj_in_dataset(variant):
    plt.figure(figsize=(10, 7))
    env = ENVS[variant['env_name']](**variant['env_params'])
    for x, y in env.goals:
        plt.plot(x, y, 'bo')
    obs_dim = np.prod(env.observation_space.shape)
    action_dim = np.prod(env.action_space.shape)
    """
    读取指定目录下的轨迹数据集。
    """
    data_dir = Path(variant['algo_params']['data_dir'])
    offline_data_quality = variant['algo_params']['offline_data_quality']
    tasks = list(range(variant['env_params']['n_tasks']))
    interpolation = variant['interpolation']
    if interpolation:
        gap = int(variant['n_train_tasks']/variant['n_eval_tasks']) + 1
        eval_tasks = np.arange(0, variant['n_train_tasks']+variant['n_eval_tasks'], gap) + int(gap/2)
        train_tasks = np.array(list(set(range(len(tasks))).difference(eval_tasks)))
    else:
        train_tasks = list(tasks[:variant['n_train_tasks']])
        eval_tasks = list(tasks[-variant['n_eval_tasks']:])
    train_trj_paths = []
    eval_trj_paths = []
    # trj entry format: [obs, action, reward, new_obs]
    for goal_dir in data_dir.glob('goal_idx*'):
        if not goal_dir.is_dir():
            continue
        goal_idx = int(goal_dir.stem.split('goal_idx')[-1])
        if goal_idx != 10:
            continue
        quality_steps = np.array(sorted(list(set([int(trj_path.stem.split('step')[-1]) for trj_path in goal_dir.rglob('trj_evalsample*_step*.npy')]))))
        low_quality_steps, mid_quality_steps, high_quality_steps = np.array_split(quality_steps, 3)
        if offline_data_quality == 'low':
            training_date_steps = low_quality_steps
        elif offline_data_quality == 'mid':
            training_date_steps = mid_quality_steps
        elif offline_data_quality == 'expert':
            training_date_steps = high_quality_steps[-1:]
        else:
            training_date_steps = quality_steps
        trj_idx = 0
        for trj_path in goal_dir.rglob('trj_evalsample*_step*.npy'):
            if int(trj_path.stem.split('step')[-1]) in training_date_steps:
                trj_idx += 1
                if goal_idx in train_tasks:
                    train_trj_paths.append(trj_path)
                else:
                    eval_trj_paths.append(trj_path)
                trj_data = np.load(str(trj_path), allow_pickle=True)
                obs, action, reward, next_obs = np.array_split(trj_data, [obs_dim, obs_dim+action_dim, -obs_dim], axis=-1)
                obs = obs.reshape(-1, 2)  # 假设 obs 中的每个观测值是一个 (x, y) 坐标对
                plt.plot(obs[:, 0], obs[:, 1])
    plt.title('Trajectories and Goals from offline_dataset/point-robot')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.savefig(f'traj_{variant["env_name"]}.png', dpi=400)
    plt.close()

@click.command()
@click.argument('config', default=None)
def main(config):
    variant = default_config
    if config:
        with open(os.path.join(config)) as f:
            exp_params = json.load(f)
        variant = deep_update_dict(exp_params, variant)
    show_traj_in_dataset(variant)

if __name__ == "__main__":
    main()

