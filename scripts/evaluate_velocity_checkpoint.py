"""Deterministically evaluate Corina gait geometry without opening a viewer."""

from __future__ import annotations

import argparse
from dataclasses import asdict

import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.lab_api.math import quat_apply


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("checkpoint", nargs="?")
  parser.add_argument("--zero", action="store_true")
  parser.add_argument("--steps", type=int, default=240)
  parser.add_argument("--device", default="cuda:0")
  parser.add_argument("--command-x", type=float)
  parser.add_argument("--cycle-time", type=float)
  parser.add_argument("--hip-amplitude", type=float)
  parser.add_argument("--knee-amplitude", type=float)
  parser.add_argument("--ankle-amplitude", type=float)
  parser.add_argument("--lateral-shift-amplitude", type=float)
  parser.add_argument("--lateral-phase-lead", type=float)
  args = parser.parse_args()

  task_id = "Velocity-Flat-Corina"
  env_cfg = load_env_cfg(task_id, play=True)
  joint_action = env_cfg.actions["joint_pos"]
  for name in (
    "cycle_time",
    "hip_amplitude",
    "knee_amplitude",
    "ankle_amplitude",
    "lateral_shift_amplitude",
    "lateral_phase_lead",
  ):
    value = getattr(args, name)
    if value is not None:
      setattr(joint_action, name, value)
  if args.command_x is not None:
    env_cfg.commands["twist"].ranges.lin_vel_x = (args.command_x, args.command_x)
  env_cfg.scene.num_envs = 1
  agent_cfg = load_rl_cfg(task_id)
  env = RslRlVecEnvWrapper(
    ManagerBasedRlEnv(cfg=env_cfg, device=args.device),
    clip_actions=agent_cfg.clip_actions,
  )
  if args.zero:
    policy = lambda obs: torch.zeros((1, env.num_actions), device=args.device)
  else:
    if args.checkpoint is None:
      parser.error("checkpoint is required unless --zero is used")
    runner_cls = load_runner_cls(task_id) or MjlabOnPolicyRunner
    runner = runner_cls(env, asdict(agent_cfg), device=args.device)
    runner.load(
      args.checkpoint,
      load_cfg={"actor": True},
      strict=True,
      map_location=args.device,
    )
    policy = runner.get_inference_policy(device=args.device)
  obs = env.get_observations()

  robot = env.unwrapped.scene["robot"]
  foot_ids = robot.find_bodies(
    ("right_leg_link6", "left_leg_link6"), preserve_order=True
  )[0]
  knee_ids = robot.find_joints(
    ("right_leg_joint4", "left_leg_joint4"), preserve_order=True
  )[0]
  foot_site_ids = robot.find_sites(("right_foot", "left_foot"), preserve_order=True)[0]
  contact_sensor = env.unwrapped.scene["feet_ground_contact"]
  local_normals = torch.tensor(
    ((0.0, 1.0, 0.0), (0.0, -1.0, 0.0)), device=args.device
  )
  angles: list[list[float]] = [[], []]
  knees: list[list[float]] = [[], []]
  swing_heights: list[list[float]] = [[], []]
  swing_contacts: list[list[float]] = [[], []]
  velocities: list[float] = []
  tilts: list[float] = []
  done_count = 0

  with torch.inference_mode():
    for _ in range(args.steps):
      actions = policy(obs)
      obs, _, dones, _ = env.step(actions)
      done_count += int(dones.sum().item())
      quats = robot.data.body_link_quat_w[:, foot_ids, :]
      normals = torch.stack(
        (quat_apply(quats[:, 0], local_normals[0].expand(1, -1)),
         quat_apply(quats[:, 1], local_normals[1].expand(1, -1))),
        dim=1,
      )
      foot_tilt = torch.rad2deg(torch.acos(torch.clamp(-normals[0, :, 2], -1.0, 1.0)))
      phase = torch.sin(
        2.0
        * torch.pi
        * env.episode_length_buf[0]
        * env.unwrapped.step_dt
        / joint_action.cycle_time
      )
      stance = (phase < 0.0, phase >= 0.0)
      contacts = (contact_sensor.data.found[0] > 0).float()
      for index in range(2):
        if bool(stance[index]):
          angles[index].append(float(foot_tilt[index]))
        else:
          swing_heights[index].append(float(robot.data.site_pos_w[0, foot_site_ids[index], 2]))
          swing_contacts[index].append(float(contacts[index]))
        knees[index].append(float(robot.data.joint_pos[0, knee_ids[index]]))
      velocities.append(float(robot.data.root_link_lin_vel_b[0, 0]))
      gravity = robot.data.projected_gravity_b[0]
      tilts.append(float(torch.acos(torch.clamp(-gravity[2], -1.0, 1.0))))

  print(f"steps={args.steps} dones={done_count}")
  print(
    "stance_sole_tilt_deg="
    f"right:{sum(angles[0]) / len(angles[0]):.2f} "
    f"left:{sum(angles[1]) / len(angles[1]):.2f}"
  )
  print(
    "knee_rad="
    f"right_max:{max(knees[0]):.3f} left_max:{max(knees[1]):.3f}"
  )
  print(
    "swing_foot_height_m="
    f"right_max:{max(swing_heights[0]):.3f} left_max:{max(swing_heights[1]):.3f}"
  )
  print(
    "swing_contact_fraction="
    f"right:{sum(swing_contacts[0]) / len(swing_contacts[0]):.2f} "
    f"left:{sum(swing_contacts[1]) / len(swing_contacts[1]):.2f}"
  )
  print(f"forward_velocity_mps={sum(velocities) / len(velocities):.3f}")
  print(
    f"base_tilt_rad=mean:{sum(tilts) / len(tilts):.3f} max:{max(tilts):.3f}"
  )
  env.close()


if __name__ == "__main__":
  main()
