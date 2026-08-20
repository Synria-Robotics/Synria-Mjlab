"""Corina biped robot config (synriard MJCF) for locomotion tasks."""

from __future__ import annotations

from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.spec_config import CollisionCfg

from synria_mjlab.robots import RobotCfg
from synria_mjlab.robots.tools import (
  ensure_imu,
  load_synriard_spec,
  make_floating_base,
  name_body_geoms,
)

_FOOT_BODIES = (
  "right_leg_link6",
  "left_leg_link6",
)

LEG_ACTUATORS = (
  BuiltinPositionActuatorCfg(
    target_names_expr=(r".*_leg_joint.*",),
    stiffness=400.0,
    damping=20.0,
    effort_limit=80.0,
    armature=0.01,
  ),
)

ARM_ACTUATORS = (
  BuiltinPositionActuatorCfg(
    target_names_expr=(r".*_hand_joint.*",),
    stiffness=200.0,
    damping=10.0,
    effort_limit=40.0,
    armature=0.01,
  ),
)

HOME = EntityCfg.InitialStateCfg(
  pos=(0.0, 0.0, 0.302526),
  joint_pos={
    # Legs: G1-style semi-crouch (hip pitch / knee / ankle mapped to Corina chain).
    "right_leg_joint1": -0.691,
    "right_leg_joint2": 0.0,
    "right_leg_joint3": 0.0,
    "right_leg_joint4": 1.21,
    "right_leg_joint5": -0.5124,
    "left_leg_joint1": -0.691,
    "left_leg_joint2": 0.0,
    "left_leg_joint3": 0.0,
    "left_leg_joint4": 1.21,
    "left_leg_joint5": 0.5124,
    # Arms: forward and slightly inward (Corina shoulder chain != G1, tuned in sim).
    "left_hand_joint1": 0.0,
    "left_hand_joint2": -1.57,
    "left_hand_joint3": 0.0,
    "left_hand_joint4": -1.05,
    "right_hand_joint1": 0.0,
    "right_hand_joint2": -1.57,
    "right_hand_joint3": 0.0,
    "right_hand_joint4": -1.05,
    ".*": 0.0,
  },
  joint_vel={".*": 0.0},
)

COLLISION = CollisionCfg(
  geom_names_expr=(".*",),
  contype={
    r"(left|right)_leg_link6_collision.*": 1,
    ".*": 0,
  },
  conaffinity={
    r"(left|right)_leg_link6_collision.*": 1,
    ".*": 0,
  },
  condim={r"(left|right)_leg_link6_collision.*": 3, ".*": 1},
  friction={r"(left|right)_leg_link6_collision.*": (0.8,), ".*": (0.6,)},
  priority={r"(left|right)_leg_link6_collision.*": 1},
)


def get_spec():
  spec = load_synriard_spec("Corina", "v1_2", None, strip_world=True)
  make_floating_base(spec, "base_link")
  ensure_imu(spec, "base_link", "imu")
  name_body_geoms(spec, _FOOT_BODIES)
  return spec


def get_corina_robot_cfg() -> RobotCfg:
  return RobotCfg(
    entity_cfg=EntityCfg(
      init_state=HOME,
      spec_fn=get_spec,
      articulation=EntityArticulationInfoCfg(
        actuators=LEG_ACTUATORS + ARM_ACTUATORS,
        soft_joint_pos_limit_factor=0.9,
      ),
      collisions=(COLLISION,),
    ),
    arm_joint_pattern=r".*_joint.*",
    gripper_joint_pattern=r"$a",  # no gripper
    ee_site="imu",
    viewer_body="base_link",
    collision_link_pattern=r"(left|right)_leg_link6",
    fingertip_geom_pattern=r"(left|right)_leg_link6_collision.*",
  )


def get_corina_entity_cfg() -> EntityCfg:
  """Return the raw EntityCfg used by locomotion task factories."""
  return get_corina_robot_cfg().entity_cfg
