import numpy as np
import logging
import torch

def offline_rollout(env, agent, buffer, max_path_length=np.inf, accum_context=True, animated=False, save_frames=False):
    # perform online rollout as in rollout()
    # If accum_context=True, aggregate offline context
    batch_dict = buffer.task_buffers[env._goal_idx].random_batch(max_path_length)
    # observations = batch_dict['observations']
    # actions = batch_dict['actions']
    # rewards = batch_dict['rewards']
    # terminals = batch_dict['terminals']
    # next_observations = batch_dict['next_observations']
    observations = []
    actions = []
    rewards = []
    terminals = []
    next_observations = []
    agent_infos = [{} for _ in actions]
    env_infos = [{} for _ in actions]
    path_length = 0

    if callable(getattr(env, "sparsify_rewards", None)):
        env_infos = [{'sparse_reward': env.sparsify_rewards(r)} for r in rewards]
    # batch_dict = dict(
    #     observations=self._observations[indices],
    #     actions=self._actions[indices],
    #     rewards=self._rewards[indices],
    #     terminals=self._terminals[indices],
    #     next_observations=self._next_obs[indices],
    #     sparse_rewards=self._sparse_rewards[indices],
    # )

    if accum_context:
        # update the agent's current context
        # while idx < max_path_length:
        #     o = batch_dict['observations'][idx]
        #     a = batch_dict['actions'][idx]
        #     r = batch_dict['rewards'][idx]
        #     t = batch_dict['terminals'][idx]
        #     next_o = batch_dict['next_observations'][idx]
        #     if callable(getattr(env, "sparsify_rewards", None)):
        #         env_info = {'sparse_reward': env.sparsify_rewards(r)}
        #     else:
        #         env_info = {}
        #     print(o, a, r, o.shape, a.shape, r.shape)
        #     agent.update_context([o, a, r, next_o, t, env_info])
        #     idx += 1
        agent.update_context_dict(batch_dict=batch_dict, env=env)

    agent.infer_posterior(agent.context, task_indices=env._goal_idx)

    o = env.reset()
    # perform online evaluation
    while path_length < max_path_length:
        logging.info('task_indices:')
        logging.info(agent.task_indices)
        logging.info('z')
        logging.info(agent.z)
        logging.info('o:')
        logging.info(o)
        a, agent_info = agent.get_action(o)
        next_o, r, d, env_info = env.step(a)
        logging.info('next_o:')
        logging.info(next_o)
        logging.info('a')
        logging.info(a)
        logging.info('r')
        logging.info(r)
        observations.append(o)
        next_observations.append(next_o)
        rewards.append(r)
        terminals.append(d)
        actions.append(a)
        agent_infos.append(agent_info)
        path_length += 1
        o = next_o
        if animated:
            env.render()
        if save_frames:
            from PIL import Image
            image = Image.fromarray(np.flipud(env.get_image()))
            env_info['frame'] = image
        env_infos.append(env_info)
        if d:
            break

    actions = np.array(actions)
    if len(actions.shape) == 1:
        actions = np.expand_dims(actions, 1)
    observations = np.array(observations)
    if len(observations.shape) == 1:
        observations = np.expand_dims(observations, 1)
    next_observations = np.array(next_observations)
    if len(next_observations.shape) == 1:
        next_observations = np.expand_dims(next_observations, 1)

    return dict(
        observations=observations,
        actions=actions,
        rewards=np.array(rewards).reshape(-1, 1),
        next_observations=next_observations,
        terminals=np.array(terminals).reshape(-1, 1),
        agent_infos=agent_infos,
        env_infos=env_infos,
    )


def offline_sample(env, agent, buffer, max_path_length=np.inf, accum_context=True):
    """
       The following value for the following keys will be a 2D array, with the
       first dimension corresponding to the time dimension.
        - observations
        - actions
        - rewards
        - next_observations
        - terminals

       The next two elements will be lists of dictionaries, with the index into
       the list being the index into the time
        - agent_infos
        - env_infos

       :param env:
       :param agent:
       :param max_path_length:
       :param accum_context: if True, accumulate the collected context
       :param animated:
       :param save_frames: if True, save video of rollout
       :return:
       """
    # observations = []
    # actions = []
    # rewards = []
    # terminals = []

    # o = env.reset()
    # goal_idx = env.goal_idx
    # next_o = None
    # path_length = 0

    batch_dict = buffer.task_buffers[env._goal_idx].random_batch(max_path_length)
    observations = batch_dict['observations']
    actions = batch_dict['actions']
    rewards = batch_dict['rewards']
    terminals = batch_dict['terminals']
    next_observations = batch_dict['next_observations']
    agent_infos = [{} for _ in actions]
    env_infos = [{} for _ in actions]

    if callable(getattr(env, "sparsify_rewards", None)):
        env_infos = [{'sparse_reward': env.sparsify_rewards(r)} for r in rewards]
    # batch_dict = dict(
    #     observations=self._observations[indices],
    #     actions=self._actions[indices],
    #     rewards=self._rewards[indices],
    #     terminals=self._terminals[indices],
    #     next_observations=self._next_obs[indices],
    #     sparse_rewards=self._sparse_rewards[indices],
    # )

    if accum_context:
        # while idx < max_path_length:
        #     o = batch_dict['observations'][idx]
        #     a = batch_dict['actions'][idx]
        #     r = batch_dict['rewards'][idx]
        #     t = batch_dict['terminals'][idx]
        #     next_o = batch_dict['next_observations'][idx]
        #     if callable(getattr(env, "sparsify_rewards", None)):
        #         env_info = {'sparse_reward': env.sparsify_rewards(r)}
        #     print(o, a, r, o.shape, a.shape, r.shape)
        #     agent.update_context([o, a, r, next_o, t, env_info])
        #     idx += 1
        agent.update_context_dict(batch_dict=batch_dict, env=env)
        # agent.infer_posterior(agent.context, task_indices=np.array([env._goal_idx]))

    actions = np.array(actions)
    if len(actions.shape) == 1:
        actions = np.expand_dims(actions, 1)
    observations = np.array(observations)
    if len(observations.shape) == 1:
        observations = np.expand_dims(observations, 1)
    next_observations = np.array(next_observations)
    if len(next_observations.shape) == 1:
        next_observations = np.expand_dims(next_observations, 1)


    return dict(
        observations=observations,
        actions=actions,
        rewards=np.array(rewards).reshape(-1, 1),
        next_observations=next_observations,
        terminals=np.array(terminals).reshape(-1, 1),
        agent_infos=agent_infos,
        env_infos=env_infos,
    )


def manual_rollout(env, agent, max_path_length=np.inf, accum_context=True, 
                   update_z_per_step=False, animated=False, save_frames=False):
    observations = []
    actions = []
    rewards = []
    terminals = []
    agent_infos = []
    env_infos = []

    # 使用 env.reset() 初始化环境，但覆盖其返回的 observation 为随机下半圆采样
    o = env.reset()
    next_o = None
    path_length = 0

    if animated:
        env.render()
    while path_length < max_path_length:
        # 生成随机 action：位于下半圆的单位向量
        theta = np.random.uniform(np.pi, 2 * np.pi)
        a = np.array([np.cos(theta), np.sin(theta)])
        agent_info = {}
        next_o, r, d, env_info = env.step(a)

        # 如果环境支持稀疏奖励，则计算并记录
        if callable(getattr(env, "sparsify_rewards", None)):
            env_info = {'sparse_reward': env.sparsify_rewards(r)}
        # 更新 agent 的上下文
        if accum_context:
            agent.update_context([o, a, r, next_o, d, env_info])
        if update_z_per_step:
            agent.infer_posterior(agent.context)
        observations.append(o)
        rewards.append(r)
        terminals.append(d)
        actions.append(a)
        agent_infos.append(agent_info)
        path_length += 1
        o = next_o  # 使用环境的下一状态
        if animated:
            env.render()
        if save_frames:
            image = Image.fromarray(np.flipud(env.get_image()))
            env_info['frame'] = image
        env_infos.append(env_info)
        if d:
            break

    actions = np.array(actions)
    if len(actions.shape) == 1:
        actions = np.expand_dims(actions, 1)
    observations = np.array(observations)
    if len(observations.shape) == 1:
        observations = np.expand_dims(observations, 1)
        next_o = np.array([next_o])
    next_observations = np.vstack(
        (
            observations[1:, :],
            np.expand_dims(next_o, 0)
        )
    )
    return dict(
        observations=observations,
        actions=actions,
        rewards=np.array(rewards).reshape(-1, 1),
        next_observations=next_observations,
        terminals=np.array(terminals).reshape(-1, 1),
        agent_infos=agent_infos,
        env_infos=env_infos,
    )


def rollout(env, agent, max_path_length=np.inf, accum_context=True, update_z_per_step=False, animated=False, save_frames=False):
    """
    The following value for the following keys will be a 2D array, with the
    first dimension corresponding to the time dimension.
     - observations
     - actions
     - rewards
     - next_observations
     - terminals

    The next two elements will be lists of dictionaries, with the index into
    the list being the index into the time
     - agent_infos
     - env_infos

    :param env:
    :param agent:
    :param max_path_length:
    :param accum_context: if True, accumulate the collected context
    :param animated:
    :param save_frames: if True, save video of rollout
    :return:
    """
    observations = []
    actions = []
    rewards = []
    terminals = []
    agent_infos = []
    env_infos = []
    o = env.reset()
    next_o = None
    path_length = 0


    if animated:
        env.render()
    while path_length < max_path_length:
        a, agent_info = agent.get_action(o)
        next_o, r, d, env_info = env.step(a)

        if callable(getattr(env, "sparsify_rewards", None)):
            env_info = {'sparse_reward': env.sparsify_rewards(r)}
        # update the agent's current context
        if accum_context:
            agent.update_context([o, a, r, next_o, d, env_info])
        if update_z_per_step:
            agent.infer_posterior(agent.context)
        observations.append(o)
        rewards.append(r)
        terminals.append(d)
        actions.append(a)
        agent_infos.append(agent_info)
        path_length += 1
        o = next_o
        if animated:
            env.render()
        if save_frames:
            from PIL import Image
            image = Image.fromarray(np.flipud(env.get_image()))
            env_info['frame'] = image
        env_infos.append(env_info)
        if d:
            break

    actions = np.array(actions)
    if len(actions.shape) == 1:
        actions = np.expand_dims(actions, 1)
    observations = np.array(observations)
    if len(observations.shape) == 1:
        observations = np.expand_dims(observations, 1)
        next_o = np.array([next_o])
    next_observations = np.vstack(
        (
            observations[1:, :],
            np.expand_dims(next_o, 0)
        )
    )
    return dict(
        observations=observations,
        actions=actions,
        rewards=np.array(rewards).reshape(-1, 1),
        next_observations=next_observations,
        terminals=np.array(terminals).reshape(-1, 1),
        agent_infos=agent_infos,
        env_infos=env_infos,
    )

def np_online_rollout(env, agent, max_path_length=np.inf, accum_context=True, update_z_per_step=False, animated=False, save_frames=False, use_np_online_decay=False, 
                    init_num=0, decay_function=None):
    """
    The following value for the following keys will be a 2D array, with the
    first dimension corresponding to the time dimension.
     - observations
     - actions
     - rewards
     - next_observations
     - terminals

    The next two elements will be lists of dictionaries, with the index into
    the list being the index into the time
     - agent_infos
     - env_infos

    :param env:
    :param agent:
    :param max_path_length:
    :param accum_context: if True, accumulate the collected context
    :param animated:
    :param save_frames: if True, save video of rollout
    :return:
    """
    observations = []
    actions = []
    rewards = []
    terminals = []
    agent_infos = []
    env_infos = []
    o = env.reset()
    next_o = None
    path_length = 0
    action_size = env.action_space.shape[0]

    num = init_num

    if animated:
        env.render()
    while path_length < max_path_length:
        a_noise = np.random.uniform(-1,1,action_size)
        if use_np_online_decay:
            a, agent_info = agent.get_action(o)
            alpha = decay_function(num)
            p = np.random.uniform(0,1)
            if p <= alpha:
                a = a_noise
        else:
            a = a_noise
            agent_info = {}
        next_o, r, d, env_info = env.step(a)
        num += 1

        if callable(getattr(env, "sparsify_rewards", None)):
            env_info = {'sparse_reward': env.sparsify_rewards(r)}
        # update the agent's current context
        if accum_context:
            agent.update_context([o, a, r, next_o, d, env_info])
        if update_z_per_step:
            agent.infer_posterior(agent.context)
        observations.append(o)
        rewards.append(r)
        terminals.append(d)
        actions.append(a)
        agent_infos.append(agent_info)
        path_length += 1
        o = next_o

        if animated:
            env.render()
        if save_frames:
            from PIL import Image
            image = Image.fromarray(np.flipud(env.get_image()))
            env_info['frame'] = image
        env_infos.append(env_info)
        if d:
            break

    actions = np.array(actions)
    if len(actions.shape) == 1:
        actions = np.expand_dims(actions, 1)
    observations = np.array(observations)
    if len(observations.shape) == 1:
        observations = np.expand_dims(observations, 1)
        next_o = np.array([next_o])
    next_observations = np.vstack(
        (
            observations[1:, :],
            np.expand_dims(next_o, 0)
        )
    )
    return dict(
        observations=observations,
        actions=actions,
        rewards=np.array(rewards).reshape(-1, 1),
        next_observations=next_observations,
        terminals=np.array(terminals).reshape(-1, 1),
        agent_infos=agent_infos,
        env_infos=env_infos,
    )

def ensemble_rollout(env, agent, max_path_length=np.inf, accum_context=True, is_select=False, animated=False,
            save_frames=False, r_thres=0., is_onlineadapt_max=False, is_sparse_reward=False,reward_models=None,dynamic_models=None,update_score=True,use_std=False):
    """
    The following value for the following keys will be a 2D array, with the
    first dimension corresponding to the time dimension.
     - observations
     - actions
     - rewards
     - next_observations
     - terminals

    The next two elements will be lists of dictionaries, with the index into
    the list being the index into the time
     - agent_infos
     - env_infos

    :param env:
    :param agent:
    :param max_path_length:
    :param accum_context: if True, accumulate the collected context
    :param animated:
    :param save_frames: if True, save video of rollout
    :return:
    """
    observations = []
    actions = []
    rewards = []
    terminals = []
    agent_infos = []
    env_infos = []
    o = env.reset()
    next_o = None
    path_length = 0

    context = []
    scores = []

    if animated:
        env.render()
    while path_length < max_path_length:
        a, agent_info = agent.get_action(o)
        next_o, r, d, env_info = env.step(a)

        context.append([o, a, r, next_o, d, env_info])
        if is_select or is_onlineadapt_max:
            if is_sparse_reward:
                scores.append(env_info['sparse_reward'])
            else:
                scores.append(r)

        '''
        # update the agent's current context
        if accum_context:
            if not is_select:
                agent.update_context([o, a, r, next_o, d, env_info])
            elif env_info['sparse_reward'] > 0.:
                agent.update_context([o, a, r, next_o, d, env_info])
        '''

        observations.append(o)
        rewards.append(r)
        terminals.append(d)
        actions.append(a)
        agent_infos.append(agent_info)
        path_length += 1
        o = next_o
        if animated:
            env.render()
        if save_frames:
            from PIL import Image
            image = Image.fromarray(np.flipud(env.get_image()))
            env_info['frame'] = image
        env_infos.append(env_info)
        if d:
            break
    observation_batch = torch.from_numpy(np.array(observations)).to(agent.z.device)
    action_batch = torch.from_numpy(np.array(actions)).to(agent.z.device)

    n_observations = np.array(observations)
    if len(n_observations.shape) == 1:
        n_observations = np.expand_dims(n_observations, 1)
        n_next_o = np.array([next_o])
    else:
        n_next_o = next_o
    n_next_observations = np.vstack(
        (
            n_observations[1:, :],
            np.expand_dims(n_next_o, 0)
        )
    )

    # num_ensemble = len(reward_models)
    # reward_predictions = []
    # prediction_errors = []
    # dynamics_predictions = []
    # pes = []
    # for i in range(num_ensemble):
    #     reward_prediction = reward_models[i].forward(0, 0, agent.z.detach().float().repeat(observation_batch.shape[0],1), observation_batch.float(), action_batch.float())
    #     dynamic_prediction = dynamic_models[i].forward(0, 0, agent.z.detach().float().repeat(observation_batch.shape[0], 1),
    #                                           observation_batch.float(), action_batch.float())
    #     reward_predictions.append(reward_prediction)
    #     dynamics_predictions.append(dynamic_prediction)
    #     pe = ((reward_prediction-torch.from_numpy(np.array(rewards).reshape(-1, 1)).to(agent.z.device).float())**2).mean()+((dynamic_prediction-torch.from_numpy(n_next_observations).to(agent.z.device).float())**2).mean()
    #     pes.append(pe)
    #     prediction_errors.append(((reward_prediction-torch.from_numpy(np.array(rewards).reshape(-1, 1)).to(agent.z.device).float())**2).mean().item()+((dynamic_prediction-torch.from_numpy(n_next_observations).to(agent.z.device).float())**2).mean().item())
    # reward_predictions = torch.stack(reward_predictions)
    # dynamics_predictions = torch.stack(dynamics_predictions)
    # pes = torch.stack(pes)


    # uncentainty = pes.mean()
    # if use_std:
    #     uncentainty = torch.std(reward_predictions, dim=1).mean() + torch.std(dynamics_predictions, dim=1).mean()
    # update the agent's current context
    if accum_context:
        if is_onlineadapt_max:
            if update_score:
                agent.update_onlineadapt_max(-1 * uncentainty, context)
            else:
                agent.fix_update_onlineadapt_max(-1 * uncentainty, context)
            if update_score:
                print(use_std,uncentainty.item(),',',np.sum(scores),'!!!')
        elif not is_select or np.sum(scores) > r_thres: # return-based
            # print('A!')
            for c in context:
                agent.update_context(c)

    actions = np.array(actions)
    if len(actions.shape) == 1:
        actions = np.expand_dims(actions, 1)
    observations = np.array(observations)
    if len(observations.shape) == 1:
        observations = np.expand_dims(observations, 1)
        next_o = np.array([next_o])
    next_observations = np.vstack(
        (
            observations[1:, :],
            np.expand_dims(next_o, 0)
        )
    )
    return dict(
        observations=observations,
        actions=actions,
        rewards=np.array(rewards).reshape(-1, 1),
        next_observations=next_observations,
        terminals=np.array(terminals).reshape(-1, 1),
        agent_infos=agent_infos,
        env_infos=env_infos,
        # uncertanties=-1 * uncentainty.item(),
        # prediction_errors=prediction_errors
    )

def z_random_switch_rollout(env, agent, max_path_length=np.inf, accum_context=True, update_z_per_step=False, animated=False, save_frames=False):
    observations = []
    actions = []
    rewards = []
    terminals = []
    agent_infos = []
    env_infos = []
    o = env.reset()
    next_o = None
    path_length = 0
    action_size = env.action_space.shape[0]

    if animated:
        env.render()
    while path_length < max_path_length:
        a_noise = np.random.uniform(-1,1,action_size)
        if path_length % 2 == 1:
            a, agent_info = agent.get_action(o)
        else:
            a = a_noise
            agent_info = {}
        next_o, r, d, env_info = env.step(a)

        if callable(getattr(env, "sparsify_rewards", None)):
            env_info = {'sparse_reward': env.sparsify_rewards(r)}
        # update the agent's current context
        if accum_context:
            agent.update_context([o, a, r, next_o, d, env_info])
        if update_z_per_step:
            agent.infer_posterior(agent.context)
        observations.append(o)
        rewards.append(r)
        terminals.append(d)
        actions.append(a)
        agent_infos.append(agent_info)
        path_length += 1
        o = next_o
        if animated:
            env.render()
        if save_frames:
            from PIL import Image
            image = Image.fromarray(np.flipud(env.get_image()))
            env_info['frame'] = image
        env_infos.append(env_info)
        if d:
            break

    actions = np.array(actions)
    if len(actions.shape) == 1:
        actions = np.expand_dims(actions, 1)
    observations = np.array(observations)
    if len(observations.shape) == 1:
        observations = np.expand_dims(observations, 1)
        next_o = np.array([next_o])
    next_observations = np.vstack(
        (
            observations[1:, :],
            np.expand_dims(next_o, 0)
        )
    )
    return dict(
        observations=observations,
        actions=actions,
        rewards=np.array(rewards).reshape(-1, 1),
        next_observations=next_observations,
        terminals=np.array(terminals).reshape(-1, 1),
        agent_infos=agent_infos,
        env_infos=env_infos,
    )

def split_paths(paths):
    """
    Stack multiples obs/actions/etc. from different paths
    :param paths: List of paths, where one path is something returned from
    the rollout functino above.
    :return: Tuple. Every element will have shape batch_size X DIM, including
    the rewards and terminal flags.
    """
    rewards = [path["rewards"].reshape(-1, 1) for path in paths]
    terminals = [path["terminals"].reshape(-1, 1) for path in paths]
    actions = [path["actions"] for path in paths]
    obs = [path["observations"] for path in paths]
    next_obs = [path["next_observations"] for path in paths]
    rewards = np.vstack(rewards)
    terminals = np.vstack(terminals)
    obs = np.vstack(obs)
    actions = np.vstack(actions)
    next_obs = np.vstack(next_obs)
    assert len(rewards.shape) == 2
    assert len(terminals.shape) == 2
    assert len(obs.shape) == 2
    assert len(actions.shape) == 2
    assert len(next_obs.shape) == 2
    return rewards, terminals, obs, actions, next_obs

def split_paths_to_dict(paths):
    rewards, terminals, obs, actions, next_obs = split_paths(paths)
    return dict(
        rewards=rewards,
        terminals=terminals,
        observations=obs,
        actions=actions,
        next_observations=next_obs,
    )


def get_stat_in_paths(paths, dict_name, scalar_name):
    if len(paths) == 0:
        return np.array([[]])

    if type(paths[0][dict_name]) == dict:
        # Support rllab interface
        return [path[dict_name][scalar_name] for path in paths]

    return [
        [info[scalar_name] for info in path[dict_name]]
        for path in paths
    ]
