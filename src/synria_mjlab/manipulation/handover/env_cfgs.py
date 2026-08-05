"""Dual-arm handover task skeleton for Bessica (Aloha-inspired)."""

from __future__ import annotations

from collections.abc import Callable

from mjlab.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions.actions import RelativeJointPositionActionCfg
from mjlab.managers import ObservationTermCfg, RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.velocity import mdp

from synria_mjlab.manipulation.lift.env_cfgs import get_cube_spec, lift_env_cfg
from synria_mjlab.robots import RobotCfg
from synria_mjlab.robots.bessica_d import get_bessica_d_robot_cfg


def handover_env_cfg(
  play: bool = False,
  robot_cfg_fn: Callable[[], RobotCfg] = get_bessica_d_robot_cfg,
) -> ManagerBasedRlEnvCfg:
  """Dual-arm joint-position handover over a shared cube.

  Skeleton MDP: both arms actuate jointly; rewards encourage either EE near the
  object (proxy for a full Aloha-style handover curriculum).
  """
  robot = robot_cfg_fn()
  cfg = lift_env_cfg(play=play, robot_cfg_fn=robot_cfg_fn)
  cfg.scene.entities["cube"] = EntityCfg(
    spec_fn=lambda: get_cube_spec(cube_size=0.025, mass=0.08),
    init_state=EntityCfg.InitialStateCfg(pos=(0.35, 0.0, 0.55)),
  )

  cfg.actions.clear()
  cfg.actions["left_arm"] = RelativeJointPositionActionCfg(
    entity_name="robot",
    actuator_names=(robot.left_arm_joint_pattern,),
    scale=0.05,
  )
  cfg.actions["right_arm"] = RelativeJointPositionActionCfg(
    entity_name="robot",
    actuator_names=(robot.right_arm_joint_pattern,),
    scale=0.05,
  )
  cfg.actions["grippers"] = RelativeJointPositionActionCfg(
    entity_name="robot",
    actuator_names=(robot.gripper_actuator_pattern,),
    scale=0.02,
  )

  cfg.viewer.body_name = robot.viewer_body
  cfg.viewer.lookat = (0.35, 0.0, 0.6)
  cfg.viewer.distance = 2.0

  left_ee = SceneEntityCfg("robot", site_names=(robot.left_ee_site,))
  right_ee = SceneEntityCfg("robot", site_names=(robot.right_ee_site,))
  cfg.rewards.clear()
  cfg.rewards["left_reach_cube"] = RewardTermCfg(
    func=_ee_to_cube_distance,
    weight=2.0,
    params={"std": 0.3, "asset_cfg": left_ee},
  )
  cfg.rewards["right_reach_cube"] = RewardTermCfg(
    func=_ee_to_cube_distance,
    weight=2.0,
    params={"std": 0.3, "asset_cfg": right_ee},
  )
  cfg.rewards["action_rate_l2"] = RewardTermCfg(func=mdp.action_rate_l2, weight=-0.01)

  for group in ("actor", "critic"):
    terms = cfg.observations[group].terms
    terms.pop("ee_to_cube", None)
    terms["left_ee_to_cube"] = ObservationTermCfg(
      func=_ee_to_cube_vector,
      params={"asset_cfg": left_ee},
    )
    terms["right_ee_to_cube"] = ObservationTermCfg(
      func=_ee_to_cube_vector,
      params={"asset_cfg": right_ee},
    )

  if play:
    cfg.observations["actor"].enable_corruption = False
    cfg.curriculum = {}

  return cfg


def _ee_to_cube_vector(env, asset_cfg: SceneEntityCfg):
  import torch
  from mjlab.entity import Entity

  robot: Entity = env.scene[asset_cfg.name]
  cube: Entity = env.scene["cube"]
  ee = robot.data.site_pos_w[:, asset_cfg.site_ids].squeeze(1)
  return cube.data.root_link_pos_w - ee


def _ee_to_cube_distance(env, std: float, asset_cfg: SceneEntityCfg):
  import torch

  delta = _ee_to_cube_vector(env, asset_cfg)
  return 1.0 - torch.tanh(torch.norm(delta, dim=-1) / std)
