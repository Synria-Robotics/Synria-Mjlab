"""Record ~5s Reach-Alicia-D play to media/reach_alicia_d.gif (headless).

Close-up framing: robots fill the frame; some of the 16 envs may be cropped.
"""

from __future__ import annotations

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
TASK = "Reach-Alicia-D"
CKPT = ROOT / "logs/rsl_rl/reach_alicia_d/2026-07-31_14-41-14/model_1499.pt"
OUT_GIF = ROOT / "media/reach_alicia_d.gif"
OUT_MP4 = ROOT / "media/reach_alicia_d.mp4"
NUM_ENVS = 16
NUM_STEPS = 250  # step_dt=0.02 -> ~5s sim
DEVICE = "cuda:0"


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
  # Tight framing — a few large arms; remaining envs may be cropped.
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
  assert CKPT.is_file(), f"missing checkpoint: {CKPT}"
  assert TASK in list_tasks(), f"{TASK} not registered"

  env_cfg = load_env_cfg(TASK, play=True)
  env_cfg.scene.num_envs = NUM_ENVS
  env_cfg.viewer.height = 720
  env_cfg.viewer.width = 1280
  env_cfg.viewer.max_extra_envs = NUM_ENVS - 1
  env_cfg.viewer.origin_type = ViewerConfig.OriginType.WORLD
  env_cfg.viewer.distance = 0.55
  env_cfg.viewer.elevation = -10.0
  env_cfg.viewer.azimuth = 135.0
  env_cfg.viewer.fovy = 32.0
  agent_cfg = load_rl_cfg(TASK)

  env = ManagerBasedRlEnv(cfg=env_cfg, device=DEVICE, render_mode="rgb_array")
  _zoom_camera(env)
  env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

  runner_cls = load_runner_cls(TASK) or MjlabOnPolicyRunner
  runner = runner_cls(env, asdict(agent_cfg), device=DEVICE)
  runner.load(str(CKPT), load_cfg={"actor": True}, strict=True, map_location=DEVICE)
  policy = runner.get_inference_policy(device=DEVICE)

  obs = env.get_observations()
  frames: list[np.ndarray] = []
  with torch.inference_mode():
    for i in range(NUM_STEPS):
      actions = policy(obs)
      obs, _, _, _ = env.step(actions)
      frame = env.unwrapped.render()
      if frame is not None and i % 5 == 0:
        frames.append(np.asarray(frame))

  env.close()
  print(f"captured {len(frames)} frames")

  OUT_GIF.parent.mkdir(parents=True, exist_ok=True)
  imageio.mimsave(OUT_MP4, frames, fps=10)
  if shutil.which("ffmpeg"):
    palette = OUT_GIF.with_suffix(".palette.png")
    subprocess.run(
      [
        "ffmpeg", "-y", "-i", str(OUT_MP4),
        "-vf", "fps=10,scale=960:-1:flags=lanczos,palettegen=stats_mode=diff",
        str(palette),
      ],
      check=True,
      capture_output=True,
    )
    subprocess.run(
      [
        "ffmpeg", "-y", "-i", str(OUT_MP4), "-i", str(palette),
        "-lavfi",
        "fps=10,scale=960:-1:flags=lanczos[x];[x][1:v]paletteuse=dither=bayer:bayer_scale=5",
        str(OUT_GIF),
      ],
      check=True,
      capture_output=True,
    )
    palette.unlink(missing_ok=True)
  else:
    imageio.mimsave(OUT_GIF, frames, fps=10, loop=0)
  print(f"wrote {OUT_GIF} ({OUT_GIF.stat().st_size / 1024:.1f} KiB)")
  print(f"wrote {OUT_MP4} ({OUT_MP4.stat().st_size / 1024:.1f} KiB)")


if __name__ == "__main__":
  main()
