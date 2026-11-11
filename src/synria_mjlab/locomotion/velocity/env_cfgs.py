"""Corina flat-terrain velocity tracking."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp.actions import JointPositionAction, JointPositionActionCfg
from mjlab.managers import TerminationTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg, ObjRef, RingPatternCfg, TerrainHeightSensorCfg
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg
from mjlab.utils.lab_api.math import quat_apply, quat_apply_inverse

from synria_mjlab.robots.corina import get_corina_entity_cfg


@dataclass(kw_only=True)
class PhaseJointPositionActionCfg(JointPositionActionCfg):
  """Joint targets with a deterministic swing-knee feed-forward component."""

  cycle_time: float = 0.6
  hip_amplitude: float = 0.36
  knee_amplitude: float = 0.56
  ankle_amplitude: float = 0.20
  stance_extension_amplitude: float = 0.0
  lateral_shift_amplitude: float = 0.0
  lateral_phase_lead: float = math.pi / 4.0

  def build(self, env):
    return PhaseJointPositionAction(self, env)


class PhaseJointPositionAction(JointPositionAction):
  """Add knee flexion to the policy residual instead of relying on reward alone."""

  cfg: PhaseJointPositionActionCfg

  def __init__(self, cfg: PhaseJointPositionActionCfg, env):
    super().__init__(cfg, env)
    self._right_hip_idx = self._target_names.index("right_leg_joint1")
    self._left_hip_idx = self._target_names.index("left_leg_joint")
    self._right_knee_idx = self._target_names.index("right_leg_joint4")
    self._left_knee_idx = self._target_names.index("left_leg_joint4")
    self._right_ankle_idx = self._target_names.index("right_leg_joint5")
    self._left_ankle_idx = self._target_names.index("left_leg_joint5")
    self._right_hip_roll_idx = self._target_names.index("right_leg_joint2")
    self._left_hip_roll_idx = self._target_names.index("left_leg_joint2")
    self._right_ankle_roll_idx = self._target_names.index("right_leg_joint6")
    self._left_ankle_roll_idx = self._target_names.index("left_leg_joint6")

  def apply_actions(self) -> None:
    phase = self._env.episode_length_buf * self._env.step_dt / self.cfg.cycle_time
    sin_phase = torch.sin(2.0 * torch.pi * phase)
    target = self._processed_actions.clone()
    right_swing = torch.clamp(sin_phase, min=0.0)
    left_swing = torch.clamp(-sin_phase, min=0.0)
    # Mini-Pi's coordinated reference: hip + knee + ankle pitch sums to zero,
    # so the knee flexes while the swing sole stays parallel to the ground.
    target[:, self._right_hip_idx] -= self.cfg.hip_amplitude * right_swing
    target[:, self._right_knee_idx] += self.cfg.knee_amplitude * right_swing
    target[:, self._right_ankle_idx] -= self.cfg.ankle_amplitude * right_swing
    target[:, self._left_hip_idx] -= self.cfg.hip_amplitude * left_swing
    target[:, self._left_knee_idx] += self.cfg.knee_amplitude * left_swing
    target[:, self._left_ankle_idx] += self.cfg.ankle_amplitude * left_swing
    # A flat-foot hip/ankle pair moves the support foot rearward relative to
    # the torso, supplying forward propulsion instead of swing-leg recoil.
    target[:, self._right_hip_idx] += self.cfg.stance_extension_amplitude * left_swing
    target[:, self._right_ankle_idx] -= self.cfg.stance_extension_amplitude * left_swing
    target[:, self._left_hip_idx] += self.cfg.stance_extension_amplitude * right_swing
    target[:, self._left_ankle_idx] += self.cfg.stance_extension_amplitude * right_swing
    # Shift the pelvis over the stance foot before unloading the swing leg.
    # Corina's mirrored ankle-roll axes require opposite numeric compensation.
    lateral = self.cfg.lateral_shift_amplitude * torch.sin(
      2.0 * torch.pi * phase + self.cfg.lateral_phase_lead
    )
    target[:, self._right_hip_roll_idx] += lateral
    target[:, self._left_hip_roll_idx] += lateral
    target[:, self._right_ankle_roll_idx] -= 0.77 * lateral
    target[:, self._left_ankle_roll_idx] += 0.77 * lateral
    encoder_bias = self._entity.data.encoder_bias[:, self._target_ids]
    self._entity.set_joint_position_target(
      target - encoder_bias, joint_ids=self._target_ids
    )


def _straight_leg_lateral_cost(env, site_names: tuple[str, str]) -> torch.Tensor:
  """Penalize lateral foot excursion; preserve a straight sagittal gait."""
  asset = env.scene["robot"]
  site_ids = asset.find_sites(site_names, preserve_order=True)[0]
  p = asset.data.site_pos_w[:, site_ids, :]
  rel_w = p - asset.data.root_link_pos_w[:, None, :]
  root_quat = asset.data.root_link_quat_w[:, None, :].expand(-1, len(site_ids), -1)
  rel_b = quat_apply_inverse(root_quat, rel_w)
  # Corina's right/left feet are at negative/positive body Y respectively.
  target_y = torch.tensor([-0.10, 0.10], device=p.device, dtype=p.dtype)[None, :]
  return torch.sum(torch.square(rel_b[..., 1] - target_y), dim=1)


def _single_support_reward(env, sensor_name: str) -> torch.Tensor:
  """Prefer a clean one-foot support phase for marching gait."""
  sensor = env.scene[sensor_name]
  assert sensor.data.found is not None
  contacts = (sensor.data.found > 0).sum(dim=1)
  return (contacts == 1).float()


def _gait_clock(env, cycle_time: float) -> torch.Tensor:
  """Mini-Pi style periodic clock (sin, cos) for a deterministic gait cycle."""
  phase = env.episode_length_buf * env.step_dt / cycle_time
  angle = 2.0 * torch.pi * phase
  return torch.stack((torch.sin(angle), torch.cos(angle)), dim=-1)


def _expected_stance(env, cycle_time: float, double_support_width: float) -> torch.Tensor:
  """Expected right/left stance mask for the Mini-Pi alternating gait."""
  sin_phase = _gait_clock(env, cycle_time)[:, 0]
  stance = torch.stack((sin_phase < 0.0, sin_phase >= 0.0), dim=-1)
  double_support = torch.abs(sin_phase) < double_support_width
  return torch.where(double_support[:, None], torch.ones_like(stance), stance)


def _phase_contact_reward(
  env,
  sensor_name: str,
  cycle_time: float,
  double_support_width: float,
) -> torch.Tensor:
  """Reward foot contacts that agree with the commanded gait phase."""
  sensor = env.scene[sensor_name]
  assert sensor.data.found is not None
  contact = sensor.data.found > 0
  expected = _expected_stance(env, cycle_time, double_support_width)
  return torch.where(contact == expected, 1.0, -0.3).float().mean(dim=-1)


def _swing_contact_cost(
  env, sensor_name: str, cycle_time: float, double_support_width: float
) -> torch.Tensor:
  """Non-saturating cost for a foot that remains on the floor during swing."""
  sensor = env.scene[sensor_name]
  assert sensor.data.found is not None
  contact = sensor.data.found > 0
  swing = ~_expected_stance(env, cycle_time, double_support_width)
  return torch.sum((contact & swing).float(), dim=-1)


def _phase_foot_height_reward(
  env,
  height_sensor_name: str,
  cycle_time: float,
  target_height: float,
  std: float,
) -> torch.Tensor:
  """Track a low, alternating swing-foot arc like the Mini-Pi reference."""
  height_sensor = env.scene[height_sensor_name]
  foot_height = height_sensor.data.heights
  sin_phase = _gait_clock(env, cycle_time)[:, 0]
  # Site order is right, left.  Positive phase swings the right foot; negative
  # phase swings the left foot.  Stance foot target remains on the floor.
  target = target_height * torch.stack(
    (torch.clamp(sin_phase, min=0.0), torch.clamp(-sin_phase, min=0.0)), dim=-1
  )
  error = torch.mean(torch.square((foot_height - target) / std), dim=-1)
  return torch.exp(-error)


def _swing_foot_height_l1_cost(
  env, height_sensor_name: str, cycle_time: float, target_height: float
) -> torch.Tensor:
  """Keep an informative gradient until each swing foot clears the floor."""
  foot_height = env.scene[height_sensor_name].data.heights
  sin_phase = _gait_clock(env, cycle_time)[:, 0]
  swing_amount = torch.stack(
    (torch.clamp(sin_phase, min=0.0), torch.clamp(-sin_phase, min=0.0)), dim=-1
  )
  target = target_height * swing_amount
  swing = swing_amount > 0.0
  return torch.sum(torch.abs(foot_height - target) * swing, dim=-1)


def _base_height_reward(env, target_height: float, std: float) -> torch.Tensor:
  """Keep the torso at the Mini-Pi reference height instead of leaning to step."""
  height = env.scene["robot"].data.root_link_pos_w[:, 2]
  return torch.exp(-torch.square((height - target_height) / std))


def _reference_leg_pose_reward(
  env,
  cycle_time: float,
  hip_scale: float,
  knee_scale: float,
  ankle_scale: float,
  stance_scale: float,
) -> torch.Tensor:
  """Return an unsaturated cost for the Mini-Pi swing-leg reference."""
  asset = env.scene["robot"]
  joint_names = (
    "right_leg_joint1",
    "right_leg_joint4",
    "right_leg_joint5",
    "left_leg_joint",
    "left_leg_joint4",
    "left_leg_joint5",
  )
  joint_ids = asset.find_joints(joint_names, preserve_order=True)[0]
  phase = _gait_clock(env, cycle_time)[:, 0]
  right_swing = torch.clamp(phase, min=0.0)
  left_swing = torch.clamp(-phase, min=0.0)
  # Corina's mirrored ankle axes need opposite numeric signs for the same
  # physical pitch. Knee flexion peaks mid-swing and returns to zero at landing.
  target = torch.stack(
    (
      -hip_scale * right_swing + stance_scale * left_swing,
      knee_scale * right_swing,
      -ankle_scale * right_swing - stance_scale * left_swing,
      -hip_scale * left_swing + stance_scale * right_swing,
      knee_scale * left_swing,
      ankle_scale * left_swing + stance_scale * right_swing,
    ),
    dim=-1,
  )
  # L1 stays informative when a transferred policy starts far from the target;
  # the previous clipped term became flat and allowed both knees to lock at 0.
  return torch.mean(torch.abs(asset.data.joint_pos[:, joint_ids] - target), dim=-1)


def _phase_lateral_joint_reward(
  env,
  cycle_time: float,
  amplitude: float,
  phase_lead: float,
  std: float,
) -> torch.Tensor:
  """Track the small phase-led pelvis shift while keeping hip yaw at HOME."""
  asset = env.scene["robot"]
  joint_ids = asset.find_joints(
    (
      "right_leg_joint2",
      "right_leg_joint3",
      "left_leg_joint2",
      "left_leg_joint3",
    ),
    preserve_order=True,
  )[0]
  phase = env.episode_length_buf * env.step_dt / cycle_time
  lateral = amplitude * torch.sin(2.0 * torch.pi * phase + phase_lead)
  target = asset.data.default_joint_pos[:, joint_ids].clone()
  target[:, 0] += lateral
  target[:, 2] += lateral
  error = torch.linalg.vector_norm(
    asset.data.joint_pos[:, joint_ids] - target, dim=-1
  )
  return torch.exp(-error / std)


def _stance_foot_flat_cost(
  env,
  body_names: tuple[str, str],
  cycle_time: float,
  double_support_width: float,
) -> torch.Tensor:
  """Penalize pitched/rolled support feet using each sole's world normal."""
  asset = env.scene["robot"]
  body_ids = asset.find_bodies(body_names, preserve_order=True)[0]
  quat_w = asset.data.body_link_quat_w[:, body_ids, :]
  batch = quat_w.shape[0]
  right_normal = quat_apply(
    quat_w[:, 0],
    torch.tensor([0.0, 1.0, 0.0], device=quat_w.device).expand(batch, -1),
  )
  left_normal = quat_apply(
    quat_w[:, 1],
    torch.tensor([0.0, -1.0, 0.0], device=quat_w.device).expand(batch, -1),
  )
  sole_normal_w = torch.stack((right_normal, left_normal), dim=1)
  # A flat sole normal is world -Z.  Its XY magnitude is sin(tilt), providing
  # useful gradients even for the old 20-37 degree toe/heel-only contacts.
  tilt_cost = torch.sum(torch.square(sole_normal_w[..., :2]), dim=-1)
  stance = _expected_stance(env, cycle_time, double_support_width).float()
  return torch.sum(tilt_cost * stance, dim=-1) / torch.clamp(
    torch.sum(stance, dim=-1), min=1.0
  )


def _forward_velocity_l1_cost(env, command_name: str) -> torch.Tensor:
  """Non-saturating forward-speed error for converting stepping into walking."""
  command = env.command_manager.get_command(command_name)
  assert command is not None
  actual = env.scene["robot"].data.root_link_lin_vel_b
  return torch.abs(command[:, 0] - actual[:, 0]) + 0.5 * torch.abs(actual[:, 1])


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
  # The template enables terrain-level curriculum for generated rough terrain.
  # A plane has no terrain generator or levels, so keeping the term makes the
  # first environment reset fail in terrain_levels_vel().
  cfg.curriculum.pop("terrain_levels", None)

  foot_sites = ("right_foot", "left_foot")
  foot_geoms = (
    "right_leg_link6_collision_1",
    "left_leg_link6_collision_1",
  )

  # Wire terrain scan to base.
  for sensor in cfg.scene.sensors or ():
    if sensor.name == "terrain_scan":
      sensor.frame = ObjRef(type="body", name="base_link", entity="robot")
    if sensor.name == "foot_height_scan":
      assert isinstance(sensor, TerrainHeightSensorCfg)
      sensor.frame = tuple(ObjRef(type="site", name=name, entity="robot") for name in foot_sites)
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
  # Keep the terrain height sensor for foot-clearance rewards; only the rough
  # terrain scan is removed from observations.
  height_sensors = tuple(
    sensor for sensor in (cfg.scene.sensors or ()) if sensor.name == "foot_height_scan"
  )
  cfg.scene.sensors = height_sensors + (feet_ground_cfg,)
  cfg.observations["actor"].terms.pop("height_scan", None)
  cfg.observations["critic"].terms.pop("height_scan", None)
  cfg.observations["actor"].terms.pop("foot_height", None)
  cfg.observations["critic"].terms.pop("foot_height", None)

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action = PhaseJointPositionActionCfg(
    entity_name=joint_pos_action.entity_name,
    actuator_names=joint_pos_action.actuator_names,
    scale=joint_pos_action.scale,
    offset=joint_pos_action.offset,
    clip=joint_pos_action.clip,
    preserve_order=joint_pos_action.preserve_order,
    use_default_offset=joint_pos_action.use_default_offset,
    cycle_time=0.6,
    hip_amplitude=0.36,
    knee_amplitude=0.56,
    ankle_amplitude=0.20,
    stance_extension_amplitude=0.0,
    lateral_shift_amplitude=0.0,
    lateral_phase_lead=math.pi / 4.0,
  )
  cfg.actions["joint_pos"] = joint_pos_action
  joint_pos_action.clip = {
    r".*_leg_joint5": (-0.26, 0.26),
    r".*_leg_joint6": (-0.26, 0.26),
  }
  # Corina is a light biped; the quadruped template's 0.25 rad action range
  # produces large transients before the policy has learned to balance.
  # Mini-Pi has no upper-body actions.  Freeze Corina's arms at HOME so the
  # policy must balance with a repeatable leg gait instead of arm flailing.
  joint_pos_action.scale = {
    r".*_leg_joint.*": 0.25,
    r".*_hand_joint.*": 0.0,
  }

  # Mini-Pi exposes sin/cos gait phase to both actor and critic.  Without this
  # clock, a memoryless policy cannot associate a foot with stance or swing.
  gait_clock = ObservationTermCfg(func=_gait_clock, params={"cycle_time": 0.6})
  cfg.observations["actor"].terms["gait_clock"] = gait_clock
  cfg.observations["critic"].terms["gait_clock"] = gait_clock

  # Start with balance and low-speed sagittal motion.  The stock template asks
  # for ±1 m/s lateral motion immediately, while also kicking the robot every
  # 1–3 seconds in all six base-velocity axes.  That combination made every
  # rollout terminate before PPO received a useful locomotion trajectory.
  twist = cfg.commands["twist"]
  twist.rel_standing_envs = 0.0
  twist.rel_forward_envs = 1.0
  # Corina's mesh faces local -X; use that as visual/semantic forward.
  twist.ranges.lin_vel_x = (-0.5, -0.3)
  # Viser's command slider requires each positive range endpoint to be >= 0.1.
  twist.ranges.lin_vel_y = (-0.1, 0.1)
  twist.ranges.ang_vel_z = (-0.2, 0.2)
  cfg.events.pop("push_robot", None)
  if "encoder_bias" in cfg.events:
    cfg.events["encoder_bias"].params["bias_range"] = (-0.005, 0.005)
  if "base_com" in cfg.events:
    cfg.events["base_com"].params["ranges"] = {
      0: (-0.01, 0.01),
      1: (-0.01, 0.01),
      2: (-0.01, 0.01),
    }
  if "foot_friction" in cfg.events:
    cfg.events["foot_friction"].params["ranges"] = (0.6, 1.0)

  command_curriculum = cfg.curriculum.get("command_vel")
  if command_curriculum is not None:
    command_curriculum.params["velocity_stages"] = [
      {
        "step": 0,
        "lin_vel_x": (-0.5, -0.3),
        "lin_vel_y": (-0.1, 0.1),
        "ang_vel_z": (-0.2, 0.2),
      },
      {
        "step": 3_000 * 24,
        "lin_vel_x": (-0.6, 1.0),
        "lin_vel_y": (-0.1, 0.1),
        "ang_vel_z": (-0.4, 0.4),
      },
      {
        "step": 6_000 * 24,
        "lin_vel_x": (-1.0, 1.5),
        "lin_vel_y": (-0.2, 0.2),
        "ang_vel_z": (-0.6, 0.6),
      },
    ]

  cfg.viewer.body_name = "base_link"
  cfg.viewer.distance = 2.0

  if "foot_friction" in cfg.events:
    cfg.events["foot_friction"].params["asset_cfg"].geom_names = foot_geoms
  if "base_com" in cfg.events:
    cfg.events["base_com"].params["asset_cfg"].body_names = ("base_link",)
  # Bootstrap nominal locomotion before introducing dynamics randomization.
  # The transferred recovery controller was trained without these perturbations.
  for event_name in ("foot_friction", "base_com", "encoder_bias"):
    cfg.events.pop(event_name, None)
  cfg.observations["actor"].enable_corruption = False

  # Soften pose / contact rewards for biped without Go1-specific sites.
  if "pose" in cfg.rewards:
    cfg.rewards["pose"].params["std_standing"] = {
      r".*_leg_joint.*": 0.25,
      r".*_hand_joint.*": 0.06,
    }
    cfg.rewards["pose"].params["std_walking"] = {
      r".*_leg_joint.*": 0.50,
      r".*_hand_joint.*": 0.08,
    }
    cfg.rewards["pose"].params["std_running"] = {
      r".*_leg_joint.*": 0.60,
      r".*_hand_joint.*": 0.10,
    }
  if "upright" in cfg.rewards:
    cfg.rewards["upright"].params["asset_cfg"].body_names = ("base_link",)
    cfg.rewards["upright"].params.pop("terrain_sensor_names", None)
    # Keep upright recovery dominant over velocity reward so rolling across the
    # floor cannot become a locally optimal way to match the command.
    cfg.rewards["upright"].params["std"] = 0.12
    cfg.rewards["upright"].weight = 10.0
  if "pose" in cfg.rewards:
    cfg.rewards["pose"].weight = 1.0
  if "track_linear_velocity" in cfg.rewards:
    # Keep a non-vanishing gradient while converting stable in-place marching
    # into forward travel.  At a 0.5 m/s error, std=0.15 is effectively zero.
    cfg.rewards["track_linear_velocity"].weight = 10.0
    cfg.rewards["track_linear_velocity"].params["std"] = 0.40
  if "track_angular_velocity" in cfg.rewards:
    cfg.rewards["track_angular_velocity"].weight = 3.0
    cfg.rewards["track_angular_velocity"].params["std"] = 0.20
  if "body_ang_vel" in cfg.rewards:
    cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("base_link",)
    cfg.rewards["body_ang_vel"].weight = -1.0
  if "angular_momentum" in cfg.rewards:
    # Corina's MJCF does not expose the template's root_angmom sensor.
    cfg.rewards["angular_momentum"].weight = 0.0
  cfg.rewards["air_time"] = RewardTermCfg(
    func=mdp.feet_air_time,
    weight=0.05,
    params={"sensor_name": "feet_ground_contact", "threshold_min": 0.08,
            "threshold_max": 0.45, "command_name": "twist", "command_threshold": 0.05},
  )
  cfg.rewards["foot_slip"] = RewardTermCfg(
    func=mdp.feet_slip,
    weight=-0.20,
    params={"sensor_name": "feet_ground_contact", "command_name": "twist",
            "command_threshold": 0.05,
            "asset_cfg": SceneEntityCfg("robot", site_names=foot_sites)},
  )
  cfg.rewards["foot_clearance"] = RewardTermCfg(
    func=mdp.feet_clearance,
    weight=-0.75,
    params={"target_height": 0.02, "height_sensor_name": "foot_height_scan",
            "command_name": "twist", "command_threshold": 0.05,
            "asset_cfg": SceneEntityCfg("robot", site_names=foot_sites)},
  )
  cfg.rewards["foot_swing_height"] = RewardTermCfg(
    func=mdp.feet_swing_height,
    weight=-0.50,
    params={"sensor_name": "feet_ground_contact", "height_sensor_name": "foot_height_scan",
            "target_height": 0.02, "command_name": "twist", "command_threshold": 0.05},
  )
  cfg.rewards["straight_leg"] = RewardTermCfg(
    func=_straight_leg_lateral_cost,
    weight=-80.0,
    params={"site_names": foot_sites},
  )
  cfg.rewards["single_support"] = RewardTermCfg(
    func=_single_support_reward,
    weight=0.20,
    params={"sensor_name": "feet_ground_contact"},
  )
  cfg.rewards["phase_contact"] = RewardTermCfg(
    func=_phase_contact_reward,
    weight=5.0,
    params={
      "sensor_name": "feet_ground_contact",
      "cycle_time": 0.6,
      "double_support_width": 0.1,
    },
  )
  cfg.rewards["swing_contact"] = RewardTermCfg(
    func=_swing_contact_cost,
    weight=-2.0,
    params={
      "sensor_name": "feet_ground_contact",
      "cycle_time": 0.6,
      "double_support_width": 0.1,
    },
  )
  cfg.rewards["phase_foot_height"] = RewardTermCfg(
    func=_phase_foot_height_reward,
    weight=5.0,
    params={
      "height_sensor_name": "foot_height_scan",
      "cycle_time": 0.6,
      "target_height": 0.04,
      "std": 0.02,
    },
  )
  cfg.rewards["swing_foot_height_l1"] = RewardTermCfg(
    func=_swing_foot_height_l1_cost,
    weight=-5.0,
    params={
      "height_sensor_name": "foot_height_scan",
      "cycle_time": 0.6,
      "target_height": 0.04,
    },
  )
  cfg.rewards["reference_leg_pose"] = RewardTermCfg(
    func=_reference_leg_pose_reward,
    weight=-12.0,
    params={
      "cycle_time": 0.6,
      "hip_scale": 0.36,
      "knee_scale": 0.56,
      "ankle_scale": 0.20,
      "stance_scale": 0.0,
    },
  )
  cfg.rewards["default_lateral_joint_pose"] = RewardTermCfg(
    func=_phase_lateral_joint_reward,
    weight=4.0,
    params={
      "cycle_time": 0.6,
      "amplitude": 0.0,
      "phase_lead": math.pi / 4.0,
      "std": 0.10,
    },
  )
  cfg.rewards["stance_foot_flat"] = RewardTermCfg(
    func=_stance_foot_flat_cost,
    weight=-12.0,
    params={
      "body_names": ("right_leg_link6", "left_leg_link6"),
      "cycle_time": 0.6,
      "double_support_width": 0.1,
    },
  )
  cfg.rewards["forward_velocity_l1"] = RewardTermCfg(
    func=_forward_velocity_l1_cost,
    weight=-6.0,
    params={"command_name": "twist"},
  )
  cfg.rewards["base_height"] = RewardTermCfg(
    func=_base_height_reward,
    weight=5.0,
    params={"target_height": 0.3555, "std": 0.025},
  )
  cfg.rewards["alive"] = RewardTermCfg(func=envs_mdp.is_alive, weight=3.0)
  cfg.rewards["action_rate_l2"].weight = -0.20

  cfg.terminations.pop("illegal_contact", None)
  cfg.terminations.pop("out_of_terrain_bounds", None)
  cfg.terminations["fell_over"] = TerminationTermCfg(
    func=mdp.bad_orientation,
    params={"limit_angle": math.radians(45.0)},
  )
  cfg.terminations["base_too_low"] = TerminationTermCfg(
    func=envs_mdp.root_height_below_minimum,
    params={"minimum_height": 0.24},
  )
  cfg.rewards["termination"] = RewardTermCfg(
    func=envs_mdp.is_terminated,
    weight=-200.0,
  )

  if play:
    cfg.seed = 42
    cfg.episode_length_s = int(1e9)
    cfg.observations["actor"].enable_corruption = False
    cfg.curriculum = {}
    # Match the Mini-Pi reference evaluation: fixed forward command.
    twist.rel_standing_envs = 0.0
    twist.rel_forward_envs = 1.0
    # Viser requires a positive upper slider bound; fixed-command evaluation
    # is provided by scripts/evaluate_velocity_checkpoint.py.
    twist.ranges.lin_vel_x = (-0.51, 0.51)
    twist.ranges.lin_vel_y = (-0.1, 0.1)
    twist.ranges.ang_vel_z = (-0.1, 0.1)

  if "reset_base" in cfg.events:
    # HOME.pos z already places feet on the ground; only randomize xy/yaw.
    cfg.events["reset_base"].params["pose_range"]["z"] = (0.0, 0.0)

  return cfg
