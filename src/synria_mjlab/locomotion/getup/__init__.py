"""Register Corina getup tasks."""

from mjlab.rl import MjlabOnPolicyRunner
from mjlab.tasks.registry import register_mjlab_task

from .env_cfgs import corina_getup_env_cfg
from .rl_cfg import corina_getup_ppo_runner_cfg

register_mjlab_task(
  task_id="Getup-Flat-Corina",
  env_cfg=corina_getup_env_cfg(),
  play_env_cfg=corina_getup_env_cfg(play=True),
  rl_cfg=corina_getup_ppo_runner_cfg(),
  runner_cls=MjlabOnPolicyRunner,
)
