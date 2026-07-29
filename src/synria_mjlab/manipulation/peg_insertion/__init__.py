"""Register Synria PegInsertion tasks."""

from mjlab.rl import MjlabOnPolicyRunner
from mjlab.tasks.registry import register_mjlab_task

from synria_mjlab.manipulation.peg_insertion.env_cfgs import peg_insertion_env_cfg
from synria_mjlab.manipulation.rl_cfg import manip_ppo_runner_cfg
from synria_mjlab.robots.bessica_d import get_bessica_d_robot_cfg
from synria_mjlab.robots.bessica_m import get_bessica_m_robot_cfg

_ROBOTS = {
  "Bessica-D": get_bessica_d_robot_cfg,
  "Bessica-M": get_bessica_m_robot_cfg,
}

for name, robot_fn in _ROBOTS.items():
  register_mjlab_task(
    task_id=f"PegInsertion-{name}",
    env_cfg=peg_insertion_env_cfg(robot_cfg_fn=robot_fn),
    play_env_cfg=peg_insertion_env_cfg(play=True, robot_cfg_fn=robot_fn),
    rl_cfg=manip_ppo_runner_cfg(
      experiment_name=f"peg_{name.lower().replace('-', '_')}"
    ),
    runner_cls=MjlabOnPolicyRunner,
  )
