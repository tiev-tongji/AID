# default CSRO experiment settings
# all experiments should modify these settings only as needed
default_config = dict(
    env_name='cheetah-dir',
    n_train_tasks=2,
    n_eval_tasks=2,
    latent_size=20, # dimension of the latent context vector
    net_size=256, # number of units per FC layer in each network
    path_to_weights=None, # path to pre-trained weights to load into networks
    seed_list=[0, 1, 2], # list of random seeds
    env_params=dict(
        n_tasks=2, # number of distinct tasks in this domain, should equal sum of train and eval tasks
        randomize_tasks=True, # shuffle the tasks after creating them
        max_episode_steps=200, # built-in max episode length for this environment
    ),
    algo_params=dict(
        separate_train=True, # whether to train separate models for each task
        pretrain=True, # whether to pretrain the context encoder
        train_z0_policy=True, # whether to train a policy conditioned on z0
        use_hvar=True, # whether to use heteroscedasticity loss
        hvar_punish_w=10, # weight on heteroscedasticity
        recurrent=False, # recurrent or permutation-invariant encoder
        use_relabel = False, # whether to use relabeling in the context encoder
        use_z_min = True, # whether to use z_min in the context encoder
        eval_deterministic=False, # whether to evaluate the deterministic policy
        policy_update_strategy='BRAC', # choose between 'BRAC' and 'TD3BC'

        # 基本训练参数
        soft_target_tau=0.005, # for SAC target network update
        allow_backward_z=False, # whether to allow gradients to flow back through z

        # BRAC参数
        use_brac=True, # whether to use BRAC regularization (compare with batch PEARL)
        train_alpha=True, # whether to train alpha (BRAC)
        alpha_max=2000., # Maximum value for alpha
        alpha_init=500., # Initialized value for alpha (BRAC)
        target_divergence=0.05, # For training alpha adaptively. As in BEAR, if train_alpha=True, increase alpha when div > target_divergence, lower alpha when div < target_divergence (BRAC)
        c_iter=3, # number of dual critic steps per iteration
        use_value_penalty=False, # whether to use value penalty in BRAC, only effective if use_brac=True
        divergence_name='kl', # divergence type in BRAC algo, offline method (CSRO) only
        kl_lambda=.1, # weight on KL divergence term in encoder loss
        policy_mean_reg_weight=1e-3, #
        policy_std_reg_weight=1e-3,
        policy_pre_activation_weight=0.,

        # TD3BC参数
        step_to_update_policy=2, # how often to update the policy
        delay_frequency=2, # how often to update the Q function
        bc_weight=2.5, # weight on behavior cloning loss in TD3BC
        gamma=0.99, # discount factor

        # 奖励设置
        sparse_rewards=False, # whether to sparsify rewards as determined in env
        max_entropy=True, # whether to include max-entropy term (as in SAC and PEARL) in value function

        # 更新步长
        policy_lr=3e-4, # learning rate for policy
        qf_lr=3e-4, # learning rate for q network
        vf_lr=3e-4, # learning rate for v network
        context_lr=3e-4, # learning rate for context encoder
        c_lr=1e-4, # dual critic learning rate (BRAC dual)
        alpha_lr=1, # alpha learning rate (BRAC)

        # loss
        recon_loss_weight = 1.0, # weight on reconstruction loss
        focal_loss_weight = 1.0, # weight on focal loss
        club_loss_weight = 1.0, # weight on club loss
        croo_loss_weight = 1.0, # weight on CROO loss
        infoNCE_loss_weight = 1.0, # weight on infoNCE loss
        infoNCE_temp = 0.1, # temperature for infoNCE loss
        unicorn_focal_weight = 0.1, # unicorn focal loss weight

        # 基本配置
        club_use_sa=False, # whether to use state-action pairs in CLUB
        use_next_obs_in_context=False, # use next obs if it is useful in distinguishing tasks
        offline_data_quality = 'mix', # quality of the offline data for training, one of [mix, low, mid, expert]
        ensemble_size = 7, # number of models in the ensemble
        meta_batch=16, # number of tasks to average the gradient across
        batch_size=256, # number of transitions in the RL batch
        num_iterations=500, # number of data sampling / training iterates
        num_tasks_sample=5, # number of randomly sampled tasks to collect data for each iteration
        num_train_steps_per_itr=2000, # number of meta-gradient steps taken per iteration
        num_evals=2, # number of independent evals
        num_steps_per_eval=600,  # number of transitions to eval on        
        embedding_batch_size=64, # number of transitions in the context batch
        embedding_mini_batch_size=64, # number of context transitions to backprop through (should equal the arg above except in the recurrent encoder case)
        max_path_length=200, # max path length for this environment
        discount=0.99, # RL discount factor
        replay_buffer_size=1000000,
        save_replay_buffer=False,
        save_algorithm=False,
        save_environment=False,
        reward_scale=5., # scale rewards before constructing Bellman update, effectively controls weight on the entropy of the policy
        use_information_bottleneck=False, # False makes latent context deterministic
        update_post_train=1, # how often to resample the context when collecting data during training (in trajectories)
        num_exp_traj_eval=1, # how many exploration trajs to collect before beginning posterior sampling at test time
        sample=1, # whether to train with stochastic (noise-sampled) trajectories, for offline method (CSRO) only
        train_epoch=6e5, # corresponding epoch of the model used to generate meta-training trajectories, offline method (CSRO) only
        eval_epoch=6e5, # corresponding epoch of the model used to generate meta-testing trajectories, offline method (CSRO) only
        z_loss_weight=10, # z_loss weight
        allow_eval=True, # if it is True, enable evaluation
        mb_replace=False, # meta batch sampling, replace or not
        dropout=0.1, # dropout for context encoder
        # data_dir="./data/walker_randparam_new_norm", # default data directory
        data_dir="./data", # default data directory
        FOCAL = dict(
            use_recon_loss = False,
            use_focal_loss = True,
            use_club_loss = False,
            use_croo_loss = False,
            use_croo_max = False,
            use_classify_loss = False,
            use_infoNCE_loss = False,
            recon_loss_weight = 1.0,
            focal_loss_weight = 1.0,
            club_loss_weight = 25.0,
            club_model_loss_weight = 10,
            croo_loss_weight = 1.0,
            classify_loss_weight = 1.0,
            infoNCE_loss_weight = 1.0,
            infoNCE_temp = 0.1,
        ),
        CSRO = dict(
            use_recon_loss = False,
            use_focal_loss = True,
            use_club_loss = True,
            use_croo_loss = False,
            use_croo_max = False,
            use_classify_loss = False,
            use_infoNCE_loss = False,
            recon_loss_weight = 1.0,
            focal_loss_weight = 1.0,
            club_loss_weight = 25.0,
            club_model_loss_weight = 10,
            croo_loss_weight = 1.0,
            classify_loss_weight = 1.0,
            infoNCE_loss_weight = 1.0,
            infoNCE_temp = 0.1,
        ),
        CORRO = dict(
            use_recon_loss = False,
            use_focal_loss = False,
            use_club_loss = False,
            use_croo_loss = False,
            use_croo_max = False,
            use_classify_loss = False,
            use_infoNCE_loss = True,
            recon_loss_weight = 1.0,
            focal_loss_weight = 1.0,
            club_loss_weight = 25.0,
            club_model_loss_weight = 10,
            croo_loss_weight = 1.0,
            classify_loss_weight = 1.0,
            infoNCE_loss_weight = 1.0,
            infoNCE_temp = 0.1,
        ),
        UNICORN = dict(
            use_recon_loss = True,
            use_focal_loss = True,
            use_club_loss = False,
            use_croo_loss = False,
            use_croo_max = False,
            use_classify_loss = False,
            use_infoNCE_loss = False,
            recon_loss_weight = 1.0,
            focal_loss_weight = 0.1,
            club_loss_weight = 25.0,
            club_model_loss_weight = 10,
            croo_loss_weight = 1.0,
            classify_loss_weight = 1.0,
            infoNCE_loss_weight = 1.0,
            infoNCE_temp = 0.1,
        ),
        CLASSIFIER = dict(
            use_recon_loss = False,
            use_focal_loss = False,
            use_club_loss = False,
            use_croo_loss = False,
            use_croo_max = False,
            use_classify_loss = True,
            use_infoNCE_loss = False,
            recon_loss_weight = 1.0,
            focal_loss_weight = 1.0,
            club_loss_weight = 25.0,
            club_model_loss_weight = 10,
            croo_loss_weight = 1.0,
            classify_loss_weight = 1.0,
            infoNCE_loss_weight = 1.0,
            infoNCE_temp = 0.1,
        ),
        CROO = dict(
            use_recon_loss = False,
            use_focal_loss = False,
            use_club_loss = False,
            use_croo_loss = True,
            use_croo_max = False,
            use_classify_loss = False,
            use_infoNCE_loss = False,
            recon_loss_weight = 1.0,
            focal_loss_weight = 1.0,
            club_loss_weight = 25.0,
            club_model_loss_weight = 10,
            croo_loss_weight = 1.0,
            classify_loss_weight = 1.0,
            infoNCE_loss_weight = 1.0,
            infoNCE_temp = 0.1,
        ),
    ),
    util_params=dict(
        base_log_dir='./logs',
        use_gpu=True,
        gpu_id=0,
        debug=True, # debugging triggers printing and writes logs to debug directory
        docker=False, # TODO docker is not yet supported
        #machine='non_mujoco' #non_mujoco or mujoco, when non_mujoco is chosen, can train offline in non-mujoco environments
    ),
    algo_type='CSRO'
)



