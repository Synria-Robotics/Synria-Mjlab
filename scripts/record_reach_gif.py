"""Record a short Reach play GIF under media/ (headless).

Examples:
  python scripts/record_reach_gif.py
  python scripts/record_reach_gif.py --task Reach-Alicia-M --seconds 3
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
from dataclasses import asdict
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
import torch

import mjlab.tasks  # noqa: F401
import synria_mjlab  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.viewer import ViewerConfig

ROOT = Path(__file__).resolve().parents[1]
DEVICE = "cuda:0"
NUM_ENVS = 16
STEP_DT = 0.02
FRAME_STRIDE = 5
GIF_FPS = 10


def _task_slug(task: str) -> str:
  return task.lower().replace("-", "_")


def _latest_checkpoint(experiment: str) -> Path:
  root = ROOT / "logs" / "rsl_rl" / experiment
  runs = sorted(root.glob("*/model_*.pt"))
  if not runs:
    raise FileNotFoundError(f"no checkpoints under {root}")
  # Prefer highest iteration in the newest run directory.
  newest_run = sorted([p.parent for p in runs], key=lambda p: p.name)[-1]
  ckpts = sorted(
    newest_run.glob("model_*.pt"),
    key=lambda p: int(p.stem.split("_")[1]),
  )
  return ckpts[-1]


def _zoom_camera(env: ManagerBasedRlEnv) -> None:
  """Free camera close on env-0; peer envs still drawn and may appear at edges."""
  renderer = env._offline_renderer
  assert renderer is not None
  origins = env.scene.env_origins.cpu().numpy()
  o0 = origins[0]
  cam = renderer._cam
  cam.type = mujoco.mjtCamera.mjCAMERA_FREE.value
  cam.trackbodyid = -1
  cam.fixedcamid = -1
  cam.lookat[:] = (float(o0[0]) + 0.3, float(o0[1]), float(o0[2]) + 0.22)
  cam.distance = 0.55
  cam.elevation = -10.0
  cam.azimuth = 135.0
  renderer._cfg.max_extra_envs = max(0, env.num_envs - 1)
  renderer._extra_env_ids = None
  renderer._model.vis.global_.fovy = 32.0
  print(
    f"[record] lookat={list(cam.lookat)} distance={cam.distance} "
    f"elev={cam.elevation} azim={cam.azimuth}"
  )


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--task", default="Reach-Alicia-D")
  parser.add_argument("--checkpoint", type=Path, default=None)
  parser.add_argument("--seconds", type=float, default=3.0)
  parser.add_argument("--num-envs", type=int, default=NUM_ENVS)
  args = parser.parse_args()

  task = args.task
  slug = _task_slug(task)
  ckpt = args.checkpoint or _latest_checkpoint(slug)
  out_gif = ROOT / "media" / f"{slug}.gif"
  out_mp4 = ROOT / "media" / f"{slug}.mp4"
  num_steps = max(1, int(round(args.seconds / STEP_DT)))

  assert ckpt.is_file(), f"missing checkpoint: {ckpt}"
  assert task in list_tasks(), f"{task} not registered"
  print(f"[record] task={task} ckpt={ckpt} steps={num_steps} (~{args.seconds}s)")

  env_cfg = load_env_cfg(task, play=True)
  env_cfg.scene.num_envs = args.num_envs
  env_cfg.viewer.height = 720
  env_cfg.viewer.width = 1280
  env_cfg.viewer.max_extra_envs = args.num_envs - 1
  env_cfg.viewer.origin_type = ViewerConfig.OriginType.WORLD
  env_cfg.viewer.distance = 0.55
  env_cfg.viewer.elevation = -10.0
  env_cfg.viewer.azimuth = 135.0
  env_cfg.viewer.fovy = 32.0
  agent_cfg = load_rl_cfg(task)

  env = ManagerBasedRlEnv(cfg=env_cfg, device=DEVICE, render_mode="rgb_array")
  _zoom_camera(env)
  env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

  runner_cls = load_runner_cls(task) or MjlabOnPolicyRunner
  runner = runner_cls(env, asdict(agent_cfg), device=DEVICE)
  runner.load(str(ckpt), load_cfg={"actor": True}, strict=True, map_location=DEVICE)
  policy = runner.get_inference_policy(device=DEVICE)

  obs = env.get_observations()
  frames: list[np.ndarray] = []
  with torch.inference_mode():
    for i in range(num_steps):
      actions = policy(obs)
      obs, _, _, _ = env.step(actions)
      frame = env.unwrapped.render()
      if frame is not None and i % FRAME_STRIDE == 0:
        frames.append(np.asarray(frame))

  env.close()
  print(f"captured {len(frames)} frames")

  out_gif.parent.mkdir(parents=True, exist_ok=True)
  imageio.mimsave(out_mp4, frames, fps=GIF_FPS)
  if shutil.which("ffmpeg"):
    palette = out_gif.with_suffix(".palette.png")
    subprocess.run(
      [
        "ffmpeg", "-y", "-i", str(out_mp4),
        "-vf", "fps=10,scale=960:-1:flags=lanczos,palettegen=stats_mode=diff",
        str(palette),
      ],
      check=True,
      capture_output=True,
    )
    subprocess.run(
      [
        "ffmpeg", "-y", "-i", str(out_mp4), "-i", str(palette),
        "-lavfi",
        "fps=10,scale=960:-1:flags=lanczos[x];[x][1:v]paletteuse=dither=bayer:bayer_scale=5",
        str(out_gif),
      ],
      check=True,
      capture_output=True,
    )
    palette.unlink(missing_ok=True)
  else:
    imageio.mimsave(out_gif, frames, fps=GIF_FPS, loop=0)
  print(f"wrote {out_gif} ({out_gif.stat().st_size / 1024:.1f} KiB)")
  print(f"wrote {out_mp4} ({out_mp4.stat().st_size / 1024:.1f} KiB)")


if __name__ == "__main__":
  main()
