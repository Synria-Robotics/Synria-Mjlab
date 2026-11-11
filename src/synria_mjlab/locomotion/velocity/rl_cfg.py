"""RL config for Corina velocity."""

from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg


def corina_velocity_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  return RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      # Keep the transferred get-up controller in its raw observation
      # coordinate system. Updating normalization statistics during PPO would
      # otherwise invalidate the transferred first-layer mapping.
      obs_normalization=False,
      distribution_cfg={
        "class_name": "GaussianDistribution",
        # Parameterize noise in log space so it cannot cross zero.  A 0.05
        # policy-space std becomes about 0.0125 rad at the leg joints: enough
        # for local gait exploration without destroying the balance transfer.
        "init_std": 0.05,
        "std_range": (0.01, 0.10),
        "std_type": "log",
        "learn_std": True,
      },
    ),
    critic=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
    ),
    algorithm=RslRlPpoAlgorithmCfg(
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      # Large exploratory actions immediately destabilize this light biped.
      # Start from the transferred balance controller and explore locally.
      entropy_coef=0.0,
      num_learning_epochs=3,
      num_mini_batches=4,
      learning_rate=1.0e-5,
      schedule="fixed",
      gamma=0.99,
      lam=0.95,
      desired_kl=0.01,
      max_grad_norm=1.0,
    ),
    experiment_name="corina_velocity",
    save_interval=50,
    num_steps_per_env=24,
    max_iterations=10_000,
  )
