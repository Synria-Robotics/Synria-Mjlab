"""Alicia_D robot config (synriard MJCF, gripper baked in)."""

from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.spec_config import CollisionCfg

from synria_mjlab.robots import RobotCfg
from synria_mjlab.robots.tools import load_synriard_spec

_FINGER_BODIES = ("left_gripper", "right_gripper")

ARM_ACTUATORS = (
  BuiltinPositionActuatorCfg(
    target_names_expr=(r"Joint[1-6]",),
    stiffness=80.0,
    damping=4.0,
    effort_limit=5.0,
    armature=0.01,
  ),
  BuiltinPositionActuatorCfg(
    target_names_expr=(r"(left|right)_finger",),
    stiffness=200.0,
    damping=10.0,
    effort_limit=5.0,
    armature=0.001,
  ),
)

HOME = EntityCfg.InitialStateCfg(
  pos=(0.0, 0.0, 0.0),
  joint_pos={
    "Joint1": 0.0,
    "Joint2": 0.4,
    "Joint3": 0.8,
    "Joint4": 0.0,
    "Joint5": 0.6,
    "Joint6": 0.0,
    "left_finger": 0.0,
    "right_finger": 0.0,
  },
  joint_vel={".*": 0.0},
)

COLLISION = CollisionCfg(
  geom_names_expr=(".*",),
  contype={
    r"(left|right)_gripper_collision.*": 1,
    r"link6.*": 1,
    ".*": 0,
  },
  conaffinity={
    r"(left|right)_gripper_collision.*": 1,
    r"link6.*": 1,
    ".*": 0,
  },
  condim={r"(left|right)_gripper_collision.*": 6, ".*": 3},
  friction={r"(left|right)_gripper_collision.*": (1.0, 5e-3, 5e-4), ".*": (0.6,)},
  solref={r"(left|right)_gripper_collision.*": (0.01, 1)},
  priority={r"(left|right)_gripper_collision.*": 1},
)


def get_spec():
  return load_synriard_spec(
    "Alicia_D",
    "v5_6",
    "gripper_50mm",
    finger_bodies=_FINGER_BODIES,
  )


def get_alicia_d_robot_cfg() -> RobotCfg:
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
    arm_joint_pattern=r"Joint[1-6]",
    gripper_joint_pattern=r"(left|right)_finger",
    ee_site="tool0_site",
    viewer_body="base_link",
    collision_link_pattern=r"link6",
    fingertip_geom_pattern=r"(left|right)_gripper_collision.*",
  )
