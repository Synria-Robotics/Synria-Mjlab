"""Move a robot to its home pose and visualize it in MuJoCo.

Home poses for Synria robots are copied from ``synria_mjlab.robots.*``. Open-RD
robots use the MJCF keyframe when present, otherwise a neutral zero pose.

Examples:
  python scripts/home_pose.py --robot corina
  python scripts/home_pose.py --library synriard --robot alicia-d
  python scripts/home_pose.py --library openrd --robot unitree_g1
  python scripts/home_pose.py --library openrd --robot franka_panda --variant panda_hand
  python scripts/home_pose.py --list
"""

from __future__ import annotations

import argparse
import re
import time
from collections.abc import Callable
from dataclasses import replace
from typing import Literal

import mujoco
import mujoco.viewer
import numpy as np
from mjlab.entity import Entity, EntityCfg

from synria_mjlab.robots.alicia_d import get_alicia_d_robot_cfg
from synria_mjlab.robots.alicia_m import get_alicia_m_robot_cfg
from synria_mjlab.robots.bessica_d import get_bessica_d_robot_cfg
from synria_mjlab.robots.bessica_m import get_bessica_m_robot_cfg
from synria_mjlab.robots.corina import HOME as CORINA_HOME
from synria_mjlab.robots.corina import get_corina_robot_cfg

Library = Literal["synriard", "openrd"]

# ---------------------------------------------------------------------------
# Home poses (copied from synria_mjlab/robots/*.py)
# ---------------------------------------------------------------------------

ALICIA_D_HOME = EntityCfg.InitialStateCfg(
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

ALICIA_M_HOME = EntityCfg.InitialStateCfg(
  pos=(0.0, 0.0, 0.0),
  joint_pos={
    "joint1": 0.0,
    "joint2": -0.5,
    "joint3": -0.8,
    "joint4": 0.0,
    "joint5": 0.5,
    "joint6": 0.0,
    "left_finger": 0.0,
    "right_finger": 0.0,
  },
  joint_vel={".*": 0.0},
)

BESSICA_D_HOME = EntityCfg.InitialStateCfg(
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

BESSICA_M_HOME = EntityCfg.InitialStateCfg(
  pos=(0.0, 0.0, 0.0),
  joint_pos={
    "right_joint2": 0.8,
    "right_joint4": 1.0,
    "left_joint2": 0.8,
    "left_joint4": -1.0,
    ".*": 0.0,
  },
  joint_vel={".*": 0.0},
)

UNITREE_G1_HOME = EntityCfg.InitialStateCfg(
  pos=(0.0, 0.0, 0.8),
  rot=(1.0, 0.0, 0.0, 0.0),
  lin_vel=(0.0, 0.0, 0.0),
  ang_vel=(0.0, 0.0, 0.0),
  joint_pos={
    "left_hip_pitch_joint": -0.312,
    "left_hip_roll_joint": 0.0,
    "left_hip_yaw_joint": 0.0,
    "left_knee_joint": 0.669,
    "left_ankle_pitch_joint": -0.363,
    "left_ankle_roll_joint": 0.0,
    "right_hip_pitch_joint": -0.312,
    "right_hip_roll_joint": 0.0,
    "right_hip_yaw_joint": 0.0,
    "right_knee_joint": 0.669,
    "right_ankle_pitch_joint": -0.363,
    "right_ankle_roll_joint": 0.0,
    "waist_yaw_joint": 0.0,
    "waist_roll_joint": 0.0,
    "waist_pitch_joint": 0.0,
    "left_shoulder_pitch_joint": 0.2,
    "left_shoulder_roll_joint": 0.2,
    "left_shoulder_yaw_joint": 0.0,
    "left_elbow_joint": 0.6,
    "left_wrist_roll_joint": 0.0,
    "left_wrist_pitch_joint": 0.0,
    "left_wrist_yaw_joint": 0.0,
    "right_shoulder_pitch_joint": 0.2,
    "right_shoulder_roll_joint": -0.2,
    "right_shoulder_yaw_joint": 0.0,
    "right_elbow_joint": 0.6,
    "right_wrist_roll_joint": 0.0,
    "right_wrist_pitch_joint": 0.0,
    "right_wrist_yaw_joint": 0.0,
    ".*": 0.0,
  },
  joint_vel={".*": 0.0},
)

SYNRIARD_ROBOTS: dict[str, tuple[Callable[[], EntityCfg], EntityCfg.InitialStateCfg]] = {
  "corina": (lambda: get_corina_robot_cfg().entity_cfg, CORINA_HOME),
  "alicia-d": (lambda: get_alicia_d_robot_cfg().entity_cfg, ALICIA_D_HOME),
  "alicia-m": (lambda: get_alicia_m_robot_cfg().entity_cfg, ALICIA_M_HOME),
  "bessica-d": (lambda: get_bessica_d_robot_cfg().entity_cfg, BESSICA_D_HOME),
  "bessica-m": (lambda: get_bessica_m_robot_cfg().entity_cfg, BESSICA_M_HOME),
}

OPENRD_HOMES: dict[str, EntityCfg.InitialStateCfg] = {
  "unitree-g1": UNITREE_G1_HOME,
}


def _normalize_robot(name: str) -> str:
  return re.sub(r"[_\s]+", "-", name.strip().lower())


def _load_openrd_spec(
  name: str,
  *,
  variant: str | None,
  version: str | None,
) -> mujoco.MjSpec:
  try:
    from openrd import get_model_path
  except ImportError as exc:
    raise SystemExit(
      "openrd is not installed. Install Open-Robot-Descriptions, e.g.\n"
      "  pip install -e ../../Open-Robot-Descriptions"
    ) from exc

  path = get_model_path(name, version=version, variant=variant, model_format="mjcf")
  return mujoco.MjSpec.from_file(str(path))


def _openrd_name(robot: str) -> str:
  return robot.replace("-", "_")


def _openrd_robots() -> list[str]:
  try:
    from openrd import list_available_models
  except ImportError:
    return []

  robots: set[str] = set()
  table = list_available_models(model_format="mjcf")
  for line in table.splitlines():
    line = line.strip()
    if not line or line.startswith(("+", "-", "|")):
      continue
    if "|" not in line:
      continue
    parts = [p.strip() for p in line.strip("|").split("|")]
    name = parts[0] if parts else ""
    if not name or name.lower() in {"robot name", "name"}:
      continue
    robots.add(name)
  return sorted(robots)


def _find_floor_geom(spec: mujoco.MjSpec) -> mujoco.MjsGeom | None:
  for geom in spec.geoms:
    if geom.type == mujoco.mjtGeom.mjGEOM_PLANE:
      return geom
  return None


def _ensure_checker_floor_assets(spec: mujoco.MjSpec) -> str:
  """Return the checker material name, creating assets if needed."""
  material_name = "home_pose_grid"
  texture_name = "home_pose_grid_tex"
  if spec.texture(texture_name) is None:
    spec.add_texture(
      name=texture_name,
      type=mujoco.mjtTexture.mjTEXTURE_2D,
      builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
      rgb1=(0.16, 0.22, 0.28),
      rgb2=(0.26, 0.32, 0.38),
      width=512,
      height=512,
    )
  if spec.material(material_name) is None:
    material = spec.add_material(
      name=material_name,
      texrepeat=(10.0, 10.0),
      reflectance=0.25,
    )
    material.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = texture_name
  return material_name


def _add_scene_environment(spec: mujoco.MjSpec) -> None:
  """Add a checker ground plane and lighting (synriard MJCFs strip both)."""
  material_name = _ensure_checker_floor_assets(spec)
  floor = _find_floor_geom(spec)
  if floor is None:
    spec.worldbody.add_geom(
      name="home_pose_floor",
      type=mujoco.mjtGeom.mjGEOM_PLANE,
      size=(20.0, 20.0, 0.05),
      pos=(0.0, 0.0, 0.0),
      material=material_name,
      contype=1,
      conaffinity=1,
      friction=(0.8, 0.005, 0.0001),
    )
  elif not floor.material:
    floor.material = material_name

  if not spec.lights:
    spec.worldbody.add_light(
      name="key_light",
      pos=(3.0, 2.0, 4.0),
      dir=(-0.35, -0.25, -1.0),
      type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
      castshadow=True,
    )
    spec.worldbody.add_light(
      name="fill_light",
      pos=(-2.5, -2.0, 3.0),
      dir=(0.25, 0.15, -1.0),
      type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
      castshadow=False,
      diffuse=(0.4, 0.4, 0.4),
    )


def _build_synriard_entity_cfg(robot: str) -> EntityCfg:
  cfg_fn, home = SYNRIARD_ROBOTS[robot]
  entity_cfg = cfg_fn()
  return replace(entity_cfg, init_state=home)


def _build_openrd_entity_cfg(
  robot: str,
  variant: str | None,
  version: str | None,
) -> EntityCfg:
  home = OPENRD_HOMES.get(robot)
  if home is not None:
    init_state = home
  else:
    probe = _load_openrd_spec(_openrd_name(robot), variant=variant, version=version)
    if probe.keys:
      init_state = EntityCfg.InitialStateCfg(joint_pos=None)
    else:
      init_state = EntityCfg.InitialStateCfg()

  def spec_fn() -> mujoco.MjSpec:
    return _load_openrd_spec(_openrd_name(robot), variant=variant, version=version)

  return EntityCfg(init_state=init_state, spec_fn=spec_fn)


def build_entity_cfg(
  library: Library,
  robot: str,
  *,
  variant: str | None = None,
  version: str | None = None,
) -> EntityCfg:
  robot = _normalize_robot(robot)
  if library == "synriard":
    if robot not in SYNRIARD_ROBOTS:
      known = ", ".join(sorted(SYNRIARD_ROBOTS))
      raise SystemExit(f"Unknown synriard robot {robot!r}. Choose from: {known}")
    entity_cfg = _build_synriard_entity_cfg(robot)
    return entity_cfg

  entity_cfg = _build_openrd_entity_cfg(robot, variant, version)
  return entity_cfg


def _init_state_key_id(model: mujoco.MjModel) -> int:
  key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "init_state")
  if key_id >= 0:
    return key_id
  if model.nkey > 0:
    return 0
  return -1


def _geom_bottom_z(model: mujoco.MjModel, data: mujoco.MjData, gi: int) -> float:
  """World-frame lowest z for a geom (mesh vertices when available)."""
  geom_type = model.geom_type[gi]
  xpos = data.geom_xpos[gi]
  xmat = data.geom_xmat[gi].reshape(3, 3)

  if geom_type == mujoco.mjtGeom.mjGEOM_MESH:
    mesh_id = model.geom_dataid[gi]
    vert_adr = model.mesh_vertadr[mesh_id]
    vert_num = model.mesh_vertnum[mesh_id]
    verts = model.mesh_vert[vert_adr : vert_adr + vert_num].reshape(-1, 3)
    world_z = (verts @ xmat.T + xpos)[:, 2]
    return float(world_z.min())

  # Fallback for primitive geoms: center minus half-height along local z.
  half_z = model.geom_size[gi][2] if geom_type != mujoco.mjtGeom.mjGEOM_SPHERE else model.geom_size[gi][0]
  bottom = xpos - xmat[:, 2] * half_z
  return float(bottom[2])


def _foot_bottom_z(
  model: mujoco.MjModel,
  data: mujoco.MjData,
  foot_bodies: tuple[str, ...],
) -> float:
  """Lowest world-frame z among mesh geoms on the foot bodies."""
  mujoco.mj_forward(model, data)
  foot_body_ids = {
    mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name) for name in foot_bodies
  }
  foot_z = float("inf")
  for gi in range(model.ngeom):
    if model.geom_bodyid[gi] not in foot_body_ids:
      continue
    if model.geom_type[gi] != mujoco.mjtGeom.mjGEOM_MESH:
      continue
    foot_z = min(foot_z, _geom_bottom_z(model, data, gi))
  if foot_z == float("inf"):
    foot_z = min(
      data.xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)][2]
      for name in foot_bodies
    )
  return foot_z


def _snap_feet_to_floor(
  model: mujoco.MjModel,
  data: mujoco.MjData,
  *,
  foot_bodies: tuple[str, ...] = ("right_leg_link6", "left_leg_link6"),
  clearance: float = 0.0,
) -> None:
  """Shift the free root so feet sit on the floor (Corina spawn height is nominal)."""
  if model.nq < 7:
    return
  foot_z = _foot_bottom_z(model, data, foot_bodies)
  data.qpos[2] -= foot_z - clearance
  data.qvel[:] = 0.0
  mujoco.mj_forward(model, data)


def apply_home_pose(
  model: mujoco.MjModel,
  data: mujoco.MjData,
  *,
  snap_feet: bool = False,
) -> None:
  key_id = _init_state_key_id(model)
  if key_id >= 0:
    mujoco.mj_resetDataKeyframe(model, data, key_id)
    key = model.key(key_id)
    if model.nu > 0 and key.ctrl.size == model.nu:
      data.ctrl[:] = key.ctrl
  else:
    mujoco.mj_resetData(model, data)
  if snap_feet:
    _snap_feet_to_floor(model, data)
  else:
    mujoco.mj_forward(model, data)


def run_viewer(
  model: mujoco.MjModel,
  data: mujoco.MjData,
  *,
  hold: bool,
  kinematic: bool,
  home_qpos: np.ndarray | None = None,
  home_ctrl: np.ndarray | None = None,
) -> None:
  if home_qpos is None:
    home_qpos = data.qpos.copy()
  if home_ctrl is None and model.nu > 0:
    key_id = _init_state_key_id(model)
    if key_id >= 0:
      key = model.key(key_id)
      if key.ctrl.size == model.nu:
        home_ctrl = key.ctrl.copy()

  with mujoco.viewer.launch_passive(model, data) as viewer:
    viewer.cam.lookat[:] = (0.0, 0.0, 0.45)
    viewer.cam.distance = 2.8
    viewer.cam.elevation = -15.0
    viewer.cam.azimuth = 135.0

    while viewer.is_running():
      if hold and kinematic:
        data.qpos[:] = home_qpos
        data.qvel[:] = 0.0
        if home_ctrl is not None:
          data.ctrl[:] = home_ctrl
        mujoco.mj_forward(model, data)
      elif hold and home_ctrl is not None and model.nu > 0:
        data.ctrl[:] = home_ctrl
        mujoco.mj_step(model, data)
        # Floating-base robots: keep root at home while joints track under PD.
        if model.nq > 7:
          data.qpos[:7] = home_qpos[:7]
          data.qvel[:6] = 0.0
          data.ctrl[:] = home_ctrl
          mujoco.mj_forward(model, data)
      else:
        mujoco.mj_forward(model, data)
      viewer.sync()
      time.sleep(model.opt.timestep)


def _print_robot_lists() -> None:
  print("Synriard robots (home pose defined in this script):")
  for name in sorted(SYNRIARD_ROBOTS):
    print(f"  {name}")
  openrd = _openrd_robots()
  print("\nOpen-RD robots (custom home pose in this script, else MJCF keyframe):")
  if openrd:
    for name in openrd:
      key = _normalize_robot(name)
      tag = " [home defined]" if key in OPENRD_HOMES else ""
      print(f"  {name}{tag}")
  else:
    print("  (install openrd to list models)")


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
    "--library",
    choices=("synriard", "openrd"),
    default="synriard",
    help="Robot description package (default: synriard)",
  )
  parser.add_argument(
    "--robot",
    default="corina",
    help="Robot name, e.g. corina, alicia-d, unitree_g1",
  )
  parser.add_argument(
    "--variant",
    default=None,
    help="Open-RD variant (optional, e.g. franka_panda for franka_panda robot)",
  )
  parser.add_argument(
    "--version",
    default=None,
    help="Open-RD version suffix when the module name includes one",
  )
  parser.add_argument(
    "--no-hold",
    action="store_true",
    help="Do not apply position-control targets (free physics / manual dragging)",
  )
  parser.add_argument(
    "--kinematic",
    action="store_true",
    help="Lock qpos each frame instead of physics simulation (debug only)",
  )
  parser.add_argument(
    "--list",
    action="store_true",
    help="List supported robots and exit",
  )
  args = parser.parse_args()

  if args.list:
    _print_robot_lists()
    return

  robot = _normalize_robot(args.robot)
  entity_cfg = build_entity_cfg(
    args.library,
    robot,
    variant=args.variant,
    version=args.version,
  )
  entity = Entity(entity_cfg)
  _add_scene_environment(entity.spec)
  model = entity.spec.compile()
  data = mujoco.MjData(model)
  snap_feet = robot == "corina"
  apply_home_pose(model, data, snap_feet=snap_feet)
  home_qpos = data.qpos.copy()
  home_ctrl = data.ctrl.copy() if model.nu > 0 else None

  print(f"[home_pose] library={args.library} robot={robot} nu={model.nu} nq={model.nq}")
  run_viewer(
    model,
    data,
    hold=not args.no_hold,
    kinematic=args.kinematic,
    home_qpos=home_qpos,
    home_ctrl=home_ctrl,
  )


if __name__ == "__main__":
  main()
