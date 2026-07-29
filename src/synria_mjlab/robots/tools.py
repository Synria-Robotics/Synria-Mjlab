"""Helpers for loading Synria MJCF assets into mjlab Entity specs."""

from __future__ import annotations

from pathlib import Path

import mujoco
from synriard import get_model_path


def resolve_mjcf(name: str, version: str, variant: str | None = None) -> Path:
  """Resolve an absolute MJCF path from synriard."""
  if variant:
    return Path(get_model_path(name, version, variant, model_format="mjcf"))

  # Models without a discrete variant (Bessica_M, Corina) expose xml on the
  # version module or on a `{name}_{version}` namespace object.
  try:
    return Path(get_model_path(name, version, None, model_format="mjcf"))
  except ValueError:
    from synriard import mjcf as mjcf_pkg

    mod = getattr(mjcf_pkg, f"{name}_{version}")
    if hasattr(mod, "xml"):
      return Path(mod.xml)
    obj = getattr(mod, f"{name}_{version}")
    return Path(obj.xml)


def strip_world_extras(spec: mujoco.MjSpec) -> None:
  """Remove embedded floor / lights so mjlab Scene owns the world."""
  for geom in list(spec.geoms):
    name = geom.name or ""
    if name in {"floor", "ground", "terrain"} or (
      geom.type == mujoco.mjtGeom.mjGEOM_PLANE and geom.parent == spec.worldbody
    ):
      spec.delete(geom)
  for light in list(spec.lights):
    if light.parent == spec.worldbody:
      spec.delete(light)


def name_body_geoms(spec: mujoco.MjSpec, body_names: tuple[str, ...], suffix: str = "_collision") -> None:
  """Ensure geoms under selected bodies have stable names for sensors / DR."""
  for body_name in body_names:
    try:
      body = spec.body(body_name)
    except Exception:
      continue
    for i, geom in enumerate(body.geoms):
      if not geom.name:
        geom.name = f"{body_name}{suffix}" if i == 0 else f"{body_name}{suffix}_{i}"


def load_synriard_spec(
  name: str,
  version: str,
  variant: str | None = None,
  *,
  finger_bodies: tuple[str, ...] = (),
  strip_world: bool = True,
) -> mujoco.MjSpec:
  """Load a synriard MJCF and prepare it for mjlab scenes."""
  path = resolve_mjcf(name, version, variant)
  spec = mujoco.MjSpec.from_file(str(path))
  if strip_world:
    strip_world_extras(spec)
  if finger_bodies:
    name_body_geoms(spec, finger_bodies)
  return spec


def make_floating_base(spec: mujoco.MjSpec, base_body: str = "base_link") -> None:
  """Add a freejoint to a fixed-base humanoid/biped if missing."""
  has_free = any(j.type == mujoco.mjtJoint.mjJNT_FREE for j in spec.joints)
  if has_free:
    return
  body = spec.body(base_body)
  body.add_freejoint(name="floating_base_joint")


def ensure_imu(spec: mujoco.MjSpec, body_name: str = "base_link", site_name: str = "imu") -> None:
  """Attach a minimal IMU site + sensors used by mjlab locomotion tasks."""
  try:
    spec.site(site_name)
  except Exception:
    body = spec.body(body_name)
    body.add_site(name=site_name, pos=(0.0, 0.0, 0.0), size=(0.01, 0.01, 0.01), group=5)

  existing = {s.name for s in spec.sensors}
  for name, stype in (
    ("imu_ang_vel", mujoco.mjtSensor.mjSENS_GYRO),
    ("imu_lin_vel", mujoco.mjtSensor.mjSENS_VELOCIMETER),
    ("imu_lin_acc", mujoco.mjtSensor.mjSENS_ACCELEROMETER),
  ):
    if name in existing:
      continue
    spec.add_sensor(
      name=name,
      type=stype,
      objtype=mujoco.mjtObj.mjOBJ_SITE,
      objname=site_name,
    )
