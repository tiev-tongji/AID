"""
Training behavior policies for CSRO

"""

import click
import json
import os
from hydra import compose, initialize

import argparse
import multiprocessing as mp
from multiprocessing import Pool
from itertools import product

from rlkit.envs.wrappers import NormalizedBoxEnv
from rlkit.envs import ENVS
from configs.default import default_config
import pdb

import numpy as np
np.int = int  # 动态修复 np.int 被废弃的问题

def deep_update_dict(fr, to):
    ''' update dict of dicts with new values '''
    # assume dicts have same keys
    for k, v in fr.items():
        if type(v) is dict:
            deep_update_dict(v, to[k])
        else:
            to[k] = v
    return to

initialize(version_base='1.3', config_path="./rlkit/torch/sac/pytorch_sac/config/")

def experiment(gpu_id, variant, cfg, goal_idx=0, seed=0,  eval=False):
    env = NormalizedBoxEnv(ENVS[variant['env_name']](**variant['env_params']))
    if seed is not None:
        env.seed(seed) 
    env.reset_task(goal_idx)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    from rlkit.torch.sac.pytorch_sac.train import Workspace
    workspace = Workspace(cfg=cfg, env=env, env_name=variant['env_name'], goal_idx=goal_idx)
    if eval:
        print('evaluate:')
        workspace.run_evaluate()
    else:
        workspace.run()


@click.command()
@click.option("--config", default="./configs/hopper_rand_params.json")
@click.option("--gpu", default=0)
@click.option("--docker", is_flag=True, default=False)
@click.option("--debug", is_flag=True, default=False)
@click.option("--eval", is_flag=True, default=False)
def main(config, gpu, docker, debug, eval, goal_idx=0, seed=0):
    variant = default_config
    cwd = os.getcwd()
    files = os.listdir(cwd)
    if config:
        with open(os.path.join(config)) as f:
            exp_params = json.load(f)
        variant = deep_update_dict(exp_params, variant)

    if variant["env_name"] == "point-robot":
        cfg = compose(config_name="train_point")
    else:
        cfg = compose(config_name="train")

    print('cfg.agent', cfg.agent)
    args_list = []
    gpu_count = 2
    for task_id in range(variant['env_params']['n_tasks']):
        gpu_id = task_id % gpu_count + gpu
        args_list.append([gpu_id, variant, cfg, task_id])
    # multi-processing
    p = mp.Pool(10)
    if len(list(args_list)) > 1:
        # p.starmap(experiment, product([variant], [cfg], list(task_idx_list)))
        p.starmap(experiment, args_list)
    else:
        experiment(gpu, variant=variant, cfg=cfg, goal_idx=goal_idx)


if __name__ == '__main__':
    #add a change 
    main()
