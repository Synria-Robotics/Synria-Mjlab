"""Register Synria Reach tasks."""

from mjlab.rl import MjlabOnPolicyRunner
from mjlab.tasks.registry import register_mjlab_task

from synria_mjlab.manipulation.reach.env_cfgs import reach_env_cfg
from synria_mjlab.manipulation.rl_cfg import manip_ppo_runner_cfg
from synria_mjlab.robots.alicia_d import get_alicia_d_robot_cfg
from synria_mjlab.robots.alicia_m import get_alicia_m_robot_cfg

_ROBOTS = {
  "Alicia-D": get_alicia_d_robot_cfg,
  "Alicia-M": get_alicia_m_robot_cfg,
}

for name, robot_fn in _ROBOTS.items():
  register_mjlab_task(
    task_id=f"Reach-{name}",
    env_cfg=reach_env_cfg(robot_cfg_fn=robot_fn),
    play_env_cfg=reach_env_cfg(play=True, robot_cfg_fn=robot_fn),
    rl_cfg=manip_ppo_runner_cfg(experiment_name=f"reach_{name.lower().replace('-', '_')}"),
    runner_cls=MjlabOnPolicyRunner,
  )
