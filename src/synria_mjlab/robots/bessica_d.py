"""Bessica_D dual-arm robot config (synriard MJCF)."""

from __future__ import annotations

from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.spec_config import CollisionCfg

from synria_mjlab.robots import RobotCfg
from synria_mjlab.robots.tools import load_synriard_spec

_FINGER_BODIES = (
  "right_arm_gripper_left_finger",
  "right_arm_gripper_right_finger",
  "left_arm_gripper_left_finger",
  "left_arm_gripper_right_finger",
)

ARM_ACTUATORS = (
  BuiltinPositionActuatorCfg(
    target_names_expr=(r"(left|right)_arm_joint[1-7]",),
    stiffness=80.0,
    damping=4.0,
    effort_limit=5.0,
    armature=0.01,
  ),
  BuiltinPositionActuatorCfg(
    target_names_expr=(r"(left|right)_arm_gripper_joint.*",),
    stiffness=200.0,
    damping=10.0,
    effort_limit=5.0,
    armature=0.001,
  ),
)

HOME = EntityCfg.InitialStateCfg(
  pos=(0.0, 0.0, 0.0),
  joint_pos={
    "right_arm_joint2": 0.8,
    "right_arm_joint4": 1.2,
    "left_arm_joint2": 0.8,
    "left_arm_joint4": 1.2,
    ".*": 0.0,
  },
  joint_vel={".*": 0.0},
)

COLLISION = CollisionCfg(
  geom_names_expr=(".*",),
  contype={
    r".*gripper.*_collision.*": 1,
    r".*arm_link7.*": 1,
    ".*": 0,
  },
  conaffinity={
    r".*gripper.*_collision.*": 1,
    r".*arm_link7.*": 1,
    ".*": 0,
  },
  condim={r".*gripper.*_collision.*": 6, ".*": 3},
  friction={r".*gripper.*_collision.*": (1.0, 5e-3, 5e-4), ".*": (0.6,)},
  priority={r".*gripper.*_collision.*": 1},
)


def get_spec():
  return load_synriard_spec(
    "Bessica_D",
    "v1_1",
    "covered",
    finger_bodies=_FINGER_BODIES,
  )


def get_bessica_d_robot_cfg() -> RobotCfg:
  return RobotCfg(
    entity_cfg=EntityCfg(
      init_state=HOME,
      spec_fn=get_spec,
      articulation=EntityArticulationInfoCfg(
        actuators=ARM_ACTUATORS,
        soft_joint_pos_limit_factor=0.95,
      ),
      collisions=(COLLISION,),
    ),
    arm_joint_pattern=r"(left|right)_arm_joint[1-7]",
    gripper_joint_pattern=r"(left|right)_arm_gripper_joint.*",
    ee_site="right_tool0_site",
    viewer_body="base_link",
    collision_link_pattern=r".*_arm_link7",
    fingertip_geom_pattern=r".*gripper.*_collision.*",
    left_arm_joint_pattern=r"left_arm_joint[1-7]",
    right_arm_joint_pattern=r"right_arm_joint[1-7]",
    left_ee_site="left_tool0_site",
    right_ee_site="right_tool0_site",
    left_gripper_joint_pattern=r"left_arm_gripper_joint.*",
    right_gripper_joint_pattern=r"right_arm_gripper_joint.*",
  )
