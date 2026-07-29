"""Dual-arm peg insertion task skeleton for Bessica (Aloha-inspired)."""

from __future__ import annotations

from collections.abc import Callable

import mujoco
from mjlab.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions.actions import RelativeJointPositionActionCfg
from mjlab.managers import ObservationTermCfg, RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.velocity import mdp

from synria_mjlab.manipulation.lift.env_cfgs import lift_env_cfg
from synria_mjlab.robots import RobotCfg
from synria_mjlab.robots.bessica_d import get_bessica_d_robot_cfg


def get_peg_spec() -> mujoco.MjSpec:
  spec = mujoco.MjSpec()
  body = spec.worldbody.add_body(name="peg")
  body.add_freejoint(name="peg_joint")
  body.add_geom(
    name="peg_geom",
    type=mujoco.mjtGeom.mjGEOM_CYLINDER,
    size=(0.015, 0.08),
    mass=0.05,
    rgba=(0.2, 0.6, 0.9, 1.0),
  )
  return spec


def get_hole_spec() -> mujoco.MjSpec:
  spec = mujoco.MjSpec()
  body = spec.worldbody.add_body(name="hole", pos=(0.4, 0.0, 0.5))
  body.add_geom(
    name="hole_geom",
    type=mujoco.mjtGeom.mjGEOM_BOX,
    size=(0.05, 0.05, 0.02),
    rgba=(0.3, 0.3, 0.3, 1.0),
  )
  return spec


def peg_insertion_env_cfg(
  play: bool = False,
  robot_cfg_fn: Callable[[], RobotCfg] = get_bessica_d_robot_cfg,
) -> ManagerBasedRlEnvCfg:
  """Skeleton peg-insertion MDP with dual-arm joint control."""
  robot = robot_cfg_fn()
  cfg = lift_env_cfg(play=play, robot_cfg_fn=robot_cfg_fn)
  cfg.scene.entities = {
    "robot": robot.entity_cfg,
    "peg": EntityCfg(
      spec_fn=get_peg_spec,
      init_state=EntityCfg.InitialStateCfg(pos=(0.3, 0.1, 0.55)),
    ),
    "hole": EntityCfg(spec_fn=get_hole_spec),
  }

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
    actuator_names=(robot.gripper_joint_pattern,),
    scale=0.02,
  )

  # Peg uses free object; drop cube-specific lift command.
  cfg.commands.clear()
  cfg.rewards.clear()
  right_ee = SceneEntityCfg("robot", site_names=(robot.right_ee_site,))
  cfg.rewards["peg_to_hole"] = RewardTermCfg(
    func=_peg_to_hole_distance,
    weight=5.0,
    params={"std": 0.2},
  )
  cfg.rewards["right_ee_to_peg"] = RewardTermCfg(
    func=_ee_to_peg_distance,
    weight=2.0,
    params={"std": 0.2, "asset_cfg": right_ee},
  )
  cfg.rewards["action_rate_l2"] = RewardTermCfg(func=mdp.action_rate_l2, weight=-0.01)

  for group in ("actor", "critic"):
    terms = cfg.observations[group].terms
    for name in ("ee_to_cube", "cube_to_goal"):
      terms.pop(name, None)
    terms["peg_to_hole"] = ObservationTermCfg(func=_peg_to_hole_vector)
    terms["right_ee_to_peg"] = ObservationTermCfg(
      func=_ee_to_peg_vector,
      params={"asset_cfg": right_ee},
    )

  cfg.viewer.body_name = robot.viewer_body
  cfg.viewer.lookat = (0.35, 0.0, 0.55)
  cfg.viewer.distance = 2.0

  if play:
    cfg.observations["actor"].enable_corruption = False
    cfg.curriculum = {}

  return cfg


def _peg_to_hole_vector(env):
  peg = env.scene["peg"]
  hole = env.scene["hole"]
  return hole.data.root_link_pos_w - peg.data.root_link_pos_w


def _peg_to_hole_distance(env, std: float):
  import torch

  return 1.0 - torch.tanh(torch.norm(_peg_to_hole_vector(env), dim=-1) / std)


def _ee_to_peg_vector(env, asset_cfg: SceneEntityCfg):
  robot = env.scene[asset_cfg.name]
  peg = env.scene["peg"]
  ee = robot.data.site_pos_w[:, asset_cfg.site_ids].squeeze(1)
  return peg.data.root_link_pos_w - ee


def _ee_to_peg_distance(env, std: float, asset_cfg: SceneEntityCfg):
  import torch

  return 1.0 - torch.tanh(torch.norm(_ee_to_peg_vector(env, asset_cfg), dim=-1) / std)
