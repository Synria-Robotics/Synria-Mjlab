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

# PD gains derived from the G1 motor model (Kp = J·ω_n², Kd = 2ζJω_n) with
# ω_n = 10 Hz and ζ = 2.0. Leg armatures are scaled ~0.35× for Corina's lighter
# mass (~8 kg vs G1 ~35 kg). Arms use the G1 4010 tier (MJCF actuatorfrcrange ±5 N·m).
_NATURAL_FREQ = 10 * 2.0 * math.pi
_DAMPING_RATIO = 2.0

# G1 reflected inertias (Unitree motor tiers, from mjlab g1_constants).
_G1_ARMATURE_7520_14 = 0.010177520
_G1_ARMATURE_7520_22 = 0.025101925
_G1_ARMATURE_5020 = 0.007219450
_G1_ARMATURE_4010 = 0.00425

# Scale leg motor inertia for Corina (~8 kg vs G1 ~35 kg). An extra sim gain
# multiplier improves free-base default-pose hold until motor specs are available.
_MASS_SCALE = 0.35
_SIM_GAIN_MULTIPLIER = 8.0
_LEG_ARMATURE_SCALE = _MASS_SCALE * _SIM_GAIN_MULTIPLIER

ARMATURE_HIP = _G1_ARMATURE_7520_14 * _LEG_ARMATURE_SCALE
ARMATURE_KNEE = _G1_ARMATURE_7520_22 * _LEG_ARMATURE_SCALE
ARMATURE_ANKLE = _G1_ARMATURE_5020 * _LEG_ARMATURE_SCALE
ARMATURE_ARM = _G1_ARMATURE_4010


def _pd_gains(armature: float) -> tuple[float, float]:
  stiffness = armature * _NATURAL_FREQ**2
  damping = 2.0 * _DAMPING_RATIO * armature * _NATURAL_FREQ
  return stiffness, damping


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
  effort_limit=40.0,
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
  pos=(0.0, 0.0, 0.302526),
  joint_pos={
    # Legs: G1-style semi-crouch (hip pitch / knee / ankle mapped to Corina chain).
    "right_leg_joint1": -0.691,
    "right_leg_joint2": 0.0,
    "right_leg_joint3": 0.0,
    "right_leg_joint4": 1.21,
    "right_leg_joint5": -0.5124,
    "left_leg_joint$": -0.691,
    "left_leg_joint2": 0.0,
    "left_leg_joint3": 0.0,
    "left_leg_joint4": 1.21,
    "left_leg_joint5": 0.5124,
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
