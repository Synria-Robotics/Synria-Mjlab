"""Corina getup (fall recovery) environment config."""

from __future__ import annotations

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.tasks.velocity import mdp as vel_mdp
from mjlab.terrains import TerrainEntityCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig

from synria_mjlab.locomotion.getup import mdp as getup_mdp
from synria_mjlab.robots.corina import get_corina_entity_cfg

_TORSO_HEIGHT = 0.45


def corina_getup_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  robot = get_corina_entity_cfg()

  actor_terms = {
    "base_ang_vel": ObservationTermCfg(
      func=vel_mdp.builtin_sensor,
      params={"sensor_name": "robot/imu_ang_vel"},
      noise=Unoise(n_min=-0.2, n_max=0.2),
    ),
    "projected_gravity": ObservationTermCfg(
      func=vel_mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05)
    ),
    "joint_pos": ObservationTermCfg(
      func=vel_mdp.joint_pos_rel,
      noise=Unoise(n_min=-0.03, n_max=0.03),
    ),
    "joint_vel": ObservationTermCfg(
      func=vel_mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5)
    ),
    "actions": ObservationTermCfg(func=vel_mdp.last_action),
  }
  observations = {
    "actor": ObservationGroupCfg(
      terms=actor_terms, concatenate_terms=True, enable_corruption=True
    ),
    "critic": ObservationGroupCfg(
      terms=dict(actor_terms), concatenate_terms=True, enable_corruption=False
    ),
  }

  actions = {
    "joint_pos": JointPositionActionCfg(
      entity_name="robot",
      actuator_names=(".*",),
      scale=0.25,
      use_default_offset=True,
    )
  }

  events = {
    "reset_fallen_or_standing": EventTermCfg(
      func=getup_mdp.reset_fallen_or_standing,
      mode="reset",
      params={
        "fall_probability": 0.6,
        "fall_height": 0.4,
        "velocity_range": 0.5,
      },
    ),
  }

  rewards = {
    "orientation": RewardTermCfg(func=getup_mdp.orientation_reward, weight=1.0),
    "torso_height": RewardTermCfg(
      func=getup_mdp.height_reward,
      weight=1.0,
      params={
        "desired_height": _TORSO_HEIGHT,
        "asset_cfg": SceneEntityCfg("robot", body_names=("base_link",)),
      },
    ),
    "action_rate_l2": RewardTermCfg(func=vel_mdp.action_rate_l2, weight=-0.01),
    "dof_pos_limits": RewardTermCfg(func=vel_mdp.joint_pos_limits, weight=-1.0),
  }

  terminations = {
    "time_out": TerminationTermCfg(func=vel_mdp.time_out, time_out=True),
  }

  cfg = ManagerBasedRlEnvCfg(
    scene=SceneCfg(
      terrain=TerrainEntityCfg(terrain_type="plane"),
      num_envs=1,
      extent=2.0,
      entities={"robot": robot},
    ),
    observations=observations,
    actions=actions,
    commands={},
    events=events,
    rewards=rewards,
    terminations=terminations,
    curriculum={},
    viewer=ViewerConfig(
      origin_type=ViewerConfig.OriginType.ASSET_BODY,
      entity_name="robot",
      body_name="base_link",
      distance=2.0,
      elevation=-10.0,
      azimuth=90.0,
    ),
    sim=SimulationCfg(
      njmax=300,
      mujoco=MujocoCfg(
        timestep=0.005,
        iterations=10,
        ls_iterations=20,
        impratio=10,
        cone="elliptic",
      ),
    ),
    decimation=4,
    episode_length_s=6.0,
  )

  if play:
    cfg.observations["actor"].enable_corruption = False

  return cfg
