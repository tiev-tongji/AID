"""
Launcher for experiments with CSRO

"""
import os
from pathlib import Path
import numpy as np
import click
import json
import ast
import torch
import random
import multiprocessing as mp
from itertools import product
import sys
from tensorboardX import SummaryWriter

mujoco_version = '200'  # 默认版本
if '--mujoco_version' in sys.argv:
    idx = sys.argv.index('--mujoco_version')
    if idx + 1 < len(sys.argv):
        mujoco_version = sys.argv[idx + 1].strip()

# 设置 MuJoCo 环境变量
if mujoco_version == '131':
    os.environ['MUJOCO_PY_MJPRO_PATH'] = '/home/autolab/.mujoco/mujoco131'
    os.environ['LD_LIBRARY_PATH'] = f"{os.environ.get('LD_LIBRARY_PATH', '')}:/home/autolab/.mujoco/mujoco131/bin:/usr/lib/nvidia"
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

def global_seed(seed=0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def experiment(gpu_id, variant, seed=None):
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
    
    if variant['algo_params']['separate_train'] == True and variant['algo_params']['pretrain'] == True:
        variant['algo_params']['z_strategy'] = 'mean'


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
        # output_activation_half=True
    )
    
    context_encoder = encoder_model(
        hidden_sizes=[200, 200, 200],
        input_size=context_encoder_input_dim,
        output_size=context_encoder_output_dim,
        output_activation=torch.tanh,
        layer_norm=variant['algo_params']['layer_norm'] if 'layer_norm' in variant['algo_params'].keys() else False
    )

    # behavior_encoder = encoder_model(
    #     hidden_sizes=[200, 200, 200],
    #     input_size=obs_dim + action_dim,
    #     output_size=2*latent_dim,
    #     output_activation=torch.tanh,
    #     # output_activation_half=True,
    #     layer_norm=variant['algo_params']['layer_norm'] if 'layer_norm' in variant['algo_params'].keys() else False
    # )

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
    algorithm = CSROSoftActorCritic(
        env=env,
        train_tasks=train_tasks,
        eval_tasks=eval_tasks,
        nets=[agent, qf1, qf2, vf, c, club_model, context_decoder, classifier, reward_models, dynamic_models],
        latent_dim=latent_dim,
        goal_radius=goal_radius,
        seed=seed,
        algo_type=algo_type,
        **variant['algo_params'],
    )
    # focal nets=[agent, qf1, qf2, vf, c]
    # focal_loss

    # CSRO nets=[agent, qf1, qf2, vf, c, club_model]
    # focal_loss + W * club_loss
    
    # CORRO nets=[agent, qf1, qf2, vf, c]
    # infoNCE_loss

    # UNICORN nets=[agent, qf1, qf2, vf, c, context_decoder]
    # focal_loss + W * recon_loss

    # CLASSIFIER nets=[agent, qf1, qf2, vf, c, classifier]
    # classifier_loss

    # CROO nets=[agent, qf1, qf2, vf, c, behavior_encoder, context_decoder]

    # optionally load pre-trained weights
    if variant['path_to_weights'] is not None:
        path = variant['path_to_weights']
        epoch = variant['epoch_to_weights']
        agent_ckpt = torch.load(os.path.join(path, "seed"+str(seed), f'agent_itr_{epoch}.pth'))
        club_model.load_state_dict(agent_ckpt['club_model'])
        context_encoder.load_state_dict(agent_ckpt['context_encoder'])
        qf1.load_state_dict(agent_ckpt['qf1'])
        qf2.load_state_dict(agent_ckpt['qf2'])
        vf.load_state_dict(agent_ckpt['vf'])
        algorithm.networks[-3].load_state_dict(agent_ckpt['target_vf'])
        policy.load_state_dict(agent_ckpt['policy'])
        c.load_state_dict(agent_ckpt['c'])
        context_decoder.load_state_dict(agent_ckpt['context_decoder'])
        # behavior_encoder.load_state_dict(agent_ckpt['behavior_encoder'])

    # optional GPU mode
    ptu.set_gpu_mode(variant['util_params']['use_gpu'], variant['util_params']['gpu_id'])
    if ptu.gpu_enabled():
        algorithm.to()

    # debugging triggers a lot of printing and logs to a debug directory
    DEBUG = variant['util_params']['debug']
    os.environ['DEBUG'] = str(int(DEBUG))

    # create logging directory
    # TODO support Docker
    exp_id = 'debug' if DEBUG else variant['util_params']['exp_name']
    experiment_log_dir = setup_logger(
        variant['env_name'],
        variant=variant,
        exp_id=exp_id,
        base_log_dir=variant['util_params']['base_log_dir'],
        seed=seed,
        snapshot_mode="gap_and_last",
        snapshot_gap=5
    )

    # optionally save eval trajectories as pkl files
    if variant['algo_params']['dump_eval_paths']:
        pickle_dir = experiment_log_dir + '/eval_trajectories'
        Path(pickle_dir).mkdir(parents=True, exist_ok=True)

    tb_writer = SummaryWriter(log_dir=experiment_log_dir)
    # run the algorithm
    algorithm.train(tb_writer)

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
@click.option('--exp_name', default=None)
@click.option('--pretrain', type=click.Choice(['true', 'false'], case_sensitive=False), default=None)
@click.option('--algo_type', type=click.Choice(['FOCAL', 'CSRO', 'CORRO', 'UNICORN', 'CLASSIFIER', 'IDAQ'], case_sensitive=False), default=None)
@click.option('--train_z0_policy', type=click.Choice(['true', 'false'], case_sensitive=False), default=None)
@click.option('--use_hvar', type=click.Choice(['true', 'false'], case_sensitive=False), default=None)
@click.option('--z_strategy', type=click.Choice(['mean', 'min', 'weighted', 'quantile'], case_sensitive=False), default=None)
@click.option('--r_thres', default=None)
def main(config, mujoco_version, gpu, seed, exp_name=None, pretrain=None, algo_type=None, train_z0_policy = None, use_hvar = None, z_strategy = None, r_thres=None):
    variant = default_config
    if config:
        with open(os.path.join(config)) as f:
            exp_params = json.load(f)
        variant = deep_update_dict(exp_params, variant)

    gpu = [int(g) for g in gpu.split(",")]
    print(f"Parsed gpus: {gpu}")
    variant['util_params']['gpu_id'] = gpu
    
    if not (exp_name == None):
        variant['util_params']['exp_name'] = exp_name
    if not (pretrain == None):
        variant['algo_params']['pretrain'] = pretrain.lower() == 'true'
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

    # multi-processing
    seed = [int(s) for s in seed.split(",")]
    print(f"Parsed seeds: {seed}")
    if len(seed) > 1:
        p = mp.Pool(2*len(gpu))
        args = []
        for i, s in enumerate(seed):
            gpu_id = gpu[i % len(gpu)]
            args.append((gpu_id, variant, s))
        p.starmap(experiment, args)
    else:
        experiment(gpu_id=gpu[0], variant=variant, seed=seed[0])

if __name__ == "__main__":
    main()

