"""Synria robot configuration helpers for mjlab tasks."""

from __future__ import annotations

from dataclasses import dataclass

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
      actuator_names=(self.gripper_joint_pattern,),
      scale=scale,
    )

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
