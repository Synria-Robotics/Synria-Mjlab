"""Synria robot configuration helpers for mjlab tasks."""

from __future__ import annotations

from dataclasses import dataclass

from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityCfg
from mjlab.envs.mdp.actions import (
  DifferentialIKActionCfg,
  RelativeJointPositionActionCfg,
)


@dataclass
class RobotCfg:
  """Robot description used by Synria manipulation / locomotion tasks.

  Synria MJCFs already include grippers (unlike mjlab_manipulation bare arms).
  """

  entity_cfg: EntityCfg
  arm_joint_pattern: str = r"Joint[1-6]|joint[1-6]"
  gripper_joint_pattern: str = r"(left|right)_finger"
  gripper_actuator_pattern: str = "left_finger"
  ee_site: str = "tool0_site"
  viewer_body: str = "base_link"
  collision_link_pattern: str = r"link6|link_6"
  fingertip_geom_pattern: str = r".*gripper.*_collision"
  left_arm_joint_pattern: str = r"left_arm_joint[1-7]"
  right_arm_joint_pattern: str = r"right_arm_joint[1-7]"
  left_ee_site: str = "left_tool0_site"
  right_ee_site: str = "right_tool0_site"
  left_gripper_joint_pattern: str = r"left_arm_gripper_joint.*"
  right_gripper_joint_pattern: str = r"right_arm_gripper_joint.*"

  def arm_joint_action_cfg(self, scale: float = 0.05) -> RelativeJointPositionActionCfg:
    return RelativeJointPositionActionCfg(
      entity_name="robot",
      actuator_names=(self.arm_joint_pattern,),
      scale=scale,
    )

  def gripper_joint_action_cfg(self, scale: float = 0.02) -> RelativeJointPositionActionCfg:
    return RelativeJointPositionActionCfg(
      entity_name="robot",
      actuator_names=(self.gripper_actuator_pattern,),
      scale=scale,
    )

  def lift_joint_action_scales(
    self,
    arm_scale: float = 0.5,
    gripper_scale_factor: float = 1.0,
  ) -> dict[str, float]:
    """Per-actuator scales for Lift-style absolute joint-position actions."""
    scales: dict[str, float] = {}
    actuators = self.entity_cfg.articulation.actuators
    assert actuators is not None
    for actuator in actuators:
      assert isinstance(actuator, BuiltinPositionActuatorCfg)
      effort = actuator.effort_limit
      stiffness = actuator.stiffness
      assert effort is not None and stiffness is not None
      for pattern in actuator.target_names_expr:
        if "finger" in pattern:
          scales[pattern] = gripper_scale_factor * effort / stiffness
        else:
          scales[pattern] = arm_scale
    return scales

  def arm_ik_action_cfg(self) -> DifferentialIKActionCfg:
    return DifferentialIKActionCfg(
      entity_name="robot",
      actuator_names=(self.arm_joint_pattern,),
      frame_name=self.ee_site,
      frame_type="site",
      use_relative_mode=True,
      delta_pos_scale=0.05,
      delta_ori_scale=0.25,
      position_weight=1.0,
      orientation_weight=0.0,
      damping=0.1,
      max_dq=0.5,
      joint_limit_weight=1.0,
    )
