"""Register Corina velocity tasks."""

from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from .env_cfgs import corina_flat_env_cfg
from .rl_cfg import corina_velocity_ppo_runner_cfg

register_mjlab_task(
  task_id="Velocity-Flat-Corina",
  env_cfg=corina_flat_env_cfg(),
  play_env_cfg=corina_flat_env_cfg(play=True),
  rl_cfg=corina_velocity_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)
