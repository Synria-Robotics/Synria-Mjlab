"""Corina biped robot config (synriard MJCF) for locomotion tasks."""

from __future__ import annotations

import math

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

# High-gain controller used by the trained recovery and velocity policies.
_NATURAL_FREQ = 10 * 2.0 * math.pi
_DAMPING_RATIO = 2.0
ARMATURE_HIP = 0.010177520 * 0.35 * 8.0
ARMATURE_KNEE = 0.025101925 * 0.35 * 8.0
ARMATURE_ANKLE = 0.007219450 * 0.35 * 8.0
ARMATURE_ARM = 0.00425

def _pd_gains(armature: float) -> tuple[float, float]:
  return armature * _NATURAL_FREQ**2, 2.0 * _DAMPING_RATIO * armature * _NATURAL_FREQ

_STIFFNESS_HIP, _DAMPING_HIP = _pd_gains(ARMATURE_HIP)
_STIFFNESS_KNEE, _DAMPING_KNEE = _pd_gains(ARMATURE_KNEE)
_STIFFNESS_ANKLE, _DAMPING_ANKLE = _pd_gains(ARMATURE_ANKLE)
_STIFFNESS_ARM, _DAMPING_ARM = _pd_gains(ARMATURE_ARM)

HIP_ACTUATOR = BuiltinPositionActuatorCfg(
  # MJCF: right hip is right_leg_joint1; left hip is left_leg_joint (no "1" suffix).
  target_names_expr=(r"right_leg_joint1", r"left_leg_joint$", r".*_leg_joint3"),
  stiffness=_STIFFNESS_HIP,
  damping=_DAMPING_HIP,
  effort_limit=80.0,
  armature=ARMATURE_HIP,
)

KNEE_ACTUATOR = BuiltinPositionActuatorCfg(
  target_names_expr=(r".*_leg_joint2", r".*_leg_joint4"),
  stiffness=_STIFFNESS_KNEE,
  damping=_DAMPING_KNEE,
  effort_limit=80.0,
  armature=ARMATURE_KNEE,
)

ANKLE_ACTUATOR = BuiltinPositionActuatorCfg(
  target_names_expr=(r".*_leg_joint5", r".*_leg_joint6"),
  stiffness=_STIFFNESS_ANKLE,
  damping=_DAMPING_ANKLE,
  effort_limit=80.0,
  armature=ARMATURE_ANKLE,
)

ARM_ACTUATOR = BuiltinPositionActuatorCfg(
  target_names_expr=(r".*_hand_joint.*",),
  stiffness=_STIFFNESS_ARM,
  damping=_DAMPING_ARM,
  effort_limit=5.0,
  armature=ARMATURE_ARM,
)

CORINA_ACTUATORS = (HIP_ACTUATOR, KNEE_ACTUATOR, ANKLE_ACTUATOR, ARM_ACTUATOR)

HOME = EntityCfg.InitialStateCfg(
  # Corina's straight-leg sole height is 0.3555 m.  Mini-Pi also trains from
  # an all-zero straight pose; starting from the old crouch left both feet
  # about 4.5 cm above the plane and injected an impact into every reset.
  pos=(0.0, 0.0, 0.3555),
  joint_pos={
    # Legs: straight nominal pose, matching the Mini-Pi reference convention.
    "right_leg_joint1": 0.0,
    "right_leg_joint2": 0.0,
    "right_leg_joint3": 0.0,
    "right_leg_joint4": 0.0,
    "right_leg_joint5": 0.0,
    "left_leg_joint$": 0.0,
    "left_leg_joint2": 0.0,
    "left_leg_joint3": 0.0,
    "left_leg_joint4": 0.0,
    "left_leg_joint5": 0.0,
    # Arms: forward and slightly inward (Corina shoulder chain != G1, tuned in sim).
    # Use $ on base joints — MJCF names are left/right_hand_joint (no "1" suffix).
    "left_hand_joint2": -1.57,
    "left_hand_joint3": 0.0,
    "left_hand_joint4": -1.05,
    "left_hand_joint$": 0.0,
    "right_hand_joint2": -1.57,
    "right_hand_joint3": 0.0,
    "right_hand_joint4": -1.05,
    "right_hand_joint$": 0.0,
    ".*": 0.0,
  },
  joint_vel={".*": 0.0},
)

COLLISION = CollisionCfg(
  geom_names_expr=(".*",),
  contype={
    r"(left|right)_leg_link6_collision_1": 1,
    ".*": 0,
  },
  conaffinity={
    r"(left|right)_leg_link6_collision_1": 1,
    ".*": 0,
  },
  condim={r"(left|right)_leg_link6_collision_1": 3, ".*": 1},
  friction={r"(left|right)_leg_link6_collision_1": (0.8,), ".*": (0.6,)},
  priority={r"(left|right)_leg_link6_collision_1": 1},
)


def get_spec():
  spec = load_synriard_spec("Corina", "v1_2", None, strip_world=True)
  make_floating_base(spec, "base_link")
  ensure_imu(spec, "base_link", "imu")
  name_body_geoms(spec, _FOOT_BODIES)
  # Dedicated sole sites are required for terrain-height and foot-velocity
  # rewards; the old fallback IMU site measured the torso instead of the feet.
  for body_name, site_name, sole_y in zip(
    _FOOT_BODIES, ("right_foot", "left_foot"), (0.035, -0.035)
  ):
    body = spec.body(body_name)
    if spec.site(site_name) is None:
      # The mirrored foot bodies use opposite local Y axes toward the sole.
      # Mesh bounds place the actual contact surface 35 mm from each origin.
      body.add_site(name=site_name, pos=(0.0, sole_y, 0.0), size=(0.012, 0.012, 0.012), group=5)
  return spec


def get_corina_robot_cfg() -> RobotCfg:
  return RobotCfg(
    entity_cfg=EntityCfg(
      init_state=HOME,
      spec_fn=get_spec,
      articulation=EntityArticulationInfoCfg(
        actuators=CORINA_ACTUATORS,
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
