"""Getup MDP helpers (adapted from mjlab_playground)."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import sample_uniform

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")
_UP_VEC = torch.tensor([0.0, 0.0, -1.0])


def orientation_reward(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  gravity = asset.data.projected_gravity_b
  up = _UP_VEC.to(gravity.device)
  error = torch.sum(torch.square(up - gravity), dim=-1)
  return torch.exp(-2.0 * error)


def height_reward(
  env: ManagerBasedRlEnv,
  desired_height: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  height = asset.data.body_link_pos_w[:, asset_cfg.body_ids, 2].squeeze(-1)
  clamped = torch.clamp(height, max=desired_height)
  return (torch.exp(clamped) - 1.0) / (math.exp(desired_height) - 1.0)


def reset_fallen_or_standing(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  fall_probability: float = 0.6,
  fall_height: float = 0.4,
  velocity_range: float = 0.5,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> None:
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)

  n = len(env_ids)
  asset: Entity = env.scene[asset_cfg.name]

  default_root_state = asset.data.default_root_state
  assert default_root_state is not None
  default_joint_pos = asset.data.default_joint_pos
  assert default_joint_pos is not None
  default_joint_vel = asset.data.default_joint_vel
  assert default_joint_vel is not None
  soft_joint_pos_limits = asset.data.soft_joint_pos_limits
  assert soft_joint_pos_limits is not None

  fall_mask = torch.rand(n, device=env.device) < fall_probability
  root_states = default_root_state[env_ids].clone()

  random_quat = F.normalize(torch.randn(n, 4, device=env.device), dim=-1)
  fallen_positions = env.scene.env_origins[env_ids].clone()
  fallen_positions[:, 2] += fall_height
  fallen_velocities = sample_uniform(-velocity_range, velocity_range, (n, 6), env.device)

  standing_positions = root_states[:, 0:3] + env.scene.env_origins[env_ids]
  standing_positions[:, 2] += 0.05

  root_states[:, 0:3] = torch.where(
    fall_mask.unsqueeze(-1), fallen_positions, standing_positions
  )
  root_states[:, 3:7] = torch.where(
    fall_mask.unsqueeze(-1), random_quat, root_states[:, 3:7]
  )
  root_states[:, 7:13] = torch.where(
    fall_mask.unsqueeze(-1), fallen_velocities, torch.zeros_like(fallen_velocities)
  )

  joint_pos = default_joint_pos[env_ids].clone()
  joint_vel = default_joint_vel[env_ids].clone()
  if fall_mask.any():
    lo = soft_joint_pos_limits[env_ids][:, :, 0]
    hi = soft_joint_pos_limits[env_ids][:, :, 1]
    rand_pos = sample_uniform(lo, hi, joint_pos.shape, env.device)
    joint_pos = torch.where(fall_mask.unsqueeze(-1), rand_pos, joint_pos)
    joint_vel = torch.where(
      fall_mask.unsqueeze(-1),
      sample_uniform(-0.5, 0.5, joint_vel.shape, env.device),
      joint_vel,
    )

  asset.write_root_state_to_sim(root_states, env_ids=env_ids)
  asset.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
