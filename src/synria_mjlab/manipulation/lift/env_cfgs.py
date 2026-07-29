"""Lift-cube environment configs for Synria single-arm robots."""

from __future__ import annotations

from collections.abc import Callable

import mujoco
from mjlab.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensorCfg
from mjlab.tasks.manipulation.lift_cube_env_cfg import make_lift_cube_env_cfg

from synria_mjlab.robots import RobotCfg
from synria_mjlab.robots.alicia_d import get_alicia_d_robot_cfg


def get_cube_spec(
  cube_size: float = 0.025,
  mass: float = 0.1,
  rgba: tuple[float, float, float, float] = (0.8, 0.2, 0.2, 1.0),
) -> mujoco.MjSpec:
  spec = mujoco.MjSpec()
  body = spec.worldbody.add_body(name="cube")
  body.add_freejoint(name="cube_joint")
  body.add_geom(
    name="cube_geom",
    type=mujoco.mjtGeom.mjGEOM_BOX,
    size=(cube_size,) * 3,
    mass=mass,
    rgba=rgba,
  )
  return spec


def lift_env_cfg(
  play: bool = False,
  robot_cfg_fn: Callable[[], RobotCfg] = get_alicia_d_robot_cfg,
) -> ManagerBasedRlEnvCfg:
  """State-based cube lift for Synria arms (YAM-style, baked-in gripper)."""
  robot = robot_cfg_fn()
  cfg = make_lift_cube_env_cfg()

  cfg.scene.entities = {
    "robot": robot.entity_cfg,
    "cube": EntityCfg(
      spec_fn=get_cube_spec,
      init_state=EntityCfg.InitialStateCfg(pos=(0.35, 0.0, 0.05)),
    ),
  }

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.actuator_names = (
    f"({robot.arm_joint_pattern}|{robot.gripper_joint_pattern})",
  )
  joint_pos_action.scale = 0.5

  cfg.observations["actor"].terms["ee_to_cube"].params["asset_cfg"] = SceneEntityCfg(
    "robot", site_names=(robot.ee_site,)
  )
  cfg.observations["critic"].terms["ee_to_cube"].params["asset_cfg"] = SceneEntityCfg(
    "robot", site_names=(robot.ee_site,)
  )
  cfg.rewards["lift"].params["asset_cfg"] = SceneEntityCfg(
    "robot", site_names=(robot.ee_site,)
  )

  for key in ("fingertip_friction_slide", "fingertip_friction_spin", "fingertip_friction_roll"):
    cfg.events[key].params["asset_cfg"].geom_names = robot.fingertip_geom_pattern

  assert cfg.scene.sensors is not None
  for sensor in cfg.scene.sensors:
    if sensor.name == "ee_ground_collision" and isinstance(sensor, ContactSensorCfg):
      sensor.primary.pattern = robot.collision_link_pattern

  cfg.viewer.body_name = robot.viewer_body
  cfg.viewer.lookat = (0.35, 0.0, 0.2)
  cfg.viewer.distance = 1.5

  lift_cmd = cfg.commands["lift_height"]
  if hasattr(lift_cmd, "object_pose_range") and lift_cmd.object_pose_range is not None:
    lift_cmd.object_pose_range.x = (0.25, 0.45)
    lift_cmd.object_pose_range.y = (-0.15, 0.15)
    lift_cmd.object_pose_range.z = (0.02, 0.05)

  if play:
    cfg.observations["actor"].enable_corruption = False
    cfg.curriculum = {}

  return cfg
