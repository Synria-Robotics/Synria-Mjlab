"""Corina flat-terrain velocity tracking."""

from __future__ import annotations

import math

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers import TerminationTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg, ObjRef, RingPatternCfg, TerrainHeightSensorCfg
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg

from synria_mjlab.robots.corina import get_corina_entity_cfg


def corina_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create Corina flat-terrain velocity configuration."""
  cfg = make_velocity_env_cfg()
  robot = get_corina_entity_cfg()
  cfg.scene.entities = {"robot": robot}
  # Corina mesh collisions exceed the velocity template default (35).
  cfg.sim.nconmax = 96

  # Flat plane.
  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_type = "plane"
  cfg.scene.terrain.terrain_generator = None

  foot_sites = ("imu",)  # fallback site if foot sites missing
  foot_geoms = (
    "right_leg_link6_collision",
    "left_leg_link6_collision",
  )

  # Wire terrain scan to base.
  for sensor in cfg.scene.sensors or ():
    if sensor.name == "terrain_scan":
      sensor.frame = ObjRef(type="body", name="base_link", entity="robot")
    if sensor.name == "foot_height_scan":
      assert isinstance(sensor, TerrainHeightSensorCfg)
      sensor.frame = tuple(
        ObjRef(type="site", name="imu", entity="robot") for _ in foot_sites
      )
      sensor.pattern = RingPatternCfg.single_ring(radius=0.05, num_samples=4)

  feet_ground_cfg = ContactSensorCfg(
    name="feet_ground_contact",
    primary=ContactMatch(mode="geom", pattern=foot_geoms, entity="robot"),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
    track_air_time=True,
  )
  # Drop rough-terrain scans; keep feet contact.
  cfg.scene.sensors = (feet_ground_cfg,)
  cfg.observations["actor"].terms.pop("height_scan", None)
  cfg.observations["critic"].terms.pop("height_scan", None)
  cfg.observations["actor"].terms.pop("foot_height", None)
  cfg.observations["critic"].terms.pop("foot_height", None)

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = 0.25

  cfg.viewer.body_name = "base_link"
  cfg.viewer.distance = 2.0

  if "foot_friction" in cfg.events:
    cfg.events["foot_friction"].params["asset_cfg"].geom_names = foot_geoms
  if "base_com" in cfg.events:
    cfg.events["base_com"].params["asset_cfg"].body_names = ("base_link",)

  # Soften pose / contact rewards for biped without Go1-specific sites.
  if "pose" in cfg.rewards:
    cfg.rewards["pose"].params["std_standing"] = {r".*": 0.3}
    cfg.rewards["pose"].params["std_walking"] = {r".*": 0.5}
    cfg.rewards["pose"].params["std_running"] = {r".*": 0.5}
  if "upright" in cfg.rewards:
    cfg.rewards["upright"].params["asset_cfg"].body_names = ("base_link",)
    cfg.rewards["upright"].params.pop("terrain_sensor_names", None)
  if "body_ang_vel" in cfg.rewards:
    cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("base_link",)
  for reward_name in ("foot_clearance", "foot_slip", "air_time", "foot_swing_height"):
    cfg.rewards.pop(reward_name, None)

  cfg.terminations.pop("illegal_contact", None)
  cfg.terminations.pop("out_of_terrain_bounds", None)
  cfg.terminations["fell_over"] = TerminationTermCfg(
    func=mdp.bad_orientation,
    params={"limit_angle": math.radians(70.0)},
  )

  if play:
    cfg.episode_length_s = int(1e9)
    cfg.observations["actor"].enable_corruption = False
    cfg.events.pop("push_robot", None)
    cfg.curriculum = {}

  if "reset_base" in cfg.events:
    # HOME.pos z already places feet on the ground; only randomize xy/yaw.
    cfg.events["reset_base"].params["pose_range"]["z"] = (0.0, 0.0)

  return cfg
