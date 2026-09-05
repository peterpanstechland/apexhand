"""Shared coin-roll scene / MDP configs. Stage-specific overrides live in config/."""

from __future__ import annotations

from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg
from isaaclab_physx.physics import PhysxCfg

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim.simulation_cfg import SimulationCfg
from isaaclab.sim.spawners.materials import RigidBodyMaterialCfg
from isaaclab.utils.configclass import configclass
from isaaclab.utils.noise import GaussianNoiseCfg as Gnoise
from isaaclab_tasks.utils import PresetCfg

import isaaclab_tasks.manager_based.manipulation.inhand.mdp as inhand_mdp

from pan_dexterous_lab.assets.apex_cfg import RobotPresetCfg
from pan_dexterous_lab.assets.cameras import (
    OVERHEAD_RGB128,
    SIDE_RGB128,
    WRIST_RGB128,
    OverheadCameraPresetCfg,
    SideCameraPresetCfg,
    WristCameraPresetCfg,
)
from pan_dexterous_lab.assets.joints import (
    ACTUATED_JOINT_NAMES,
    COUPLED_JOINT_NAMES,
    COUPLED_SOURCE_NAMES,
)
from pan_dexterous_lab.assets.objects import Object2PresetCfg, ObjectPresetCfg
from pan_dexterous_lab.tasks.coin_roll import mdp as coin_mdp
from pan_dexterous_lab.tasks.coin_roll.hand_side import apply_hand_side


@configclass
class PhysicsBackendCfg(PresetCfg):
    """Typed selector: ``physics=physx`` or ``physics=newton_mjwarp``."""

    physx = PhysxCfg(
        bounce_threshold_velocity=0.2,
        gpu_max_rigid_contact_count=2**20,
        gpu_max_rigid_patch_count=2**23,
    )
    newton_mjwarp = NewtonCfg(
        solver_cfg=MJWarpSolverCfg(
            solver="newton",
            integrator="implicitfast",
            njmax=200,
            nconmax=70,
            impratio=10.0,
            cone="elliptic",
            update_data_interval=2,
            iterations=100,
        ),
        num_substeps=2,
        debug_mode=False,
    )
    default = physx


@configclass
class CoinRollSceneCfg(InteractiveSceneCfg):
    robot: RobotPresetCfg = RobotPresetCfg()
    object: ObjectPresetCfg = ObjectPresetCfg()
    object2: Object2PresetCfg = Object2PresetCfg()
    wrist_camera: WristCameraPresetCfg = WristCameraPresetCfg()
    overhead_camera: OverheadCameraPresetCfg = OverheadCameraPresetCfg()
    side_camera: SideCameraPresetCfg = SideCameraPresetCfg()
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(color=(0.95, 0.95, 0.92), intensity=1200.0),
    )
    dome_light = AssetBaseCfg(
        prim_path="/World/domeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.78, 0.80, 0.86), intensity=700.0),
    )


@configclass
class ActionsCfg:
    joint_pos = coin_mdp.ApexCoupledEMAActionCfg(
        asset_name="robot",
        joint_names=ACTUATED_JOINT_NAMES,
        coupled_joint_names=COUPLED_JOINT_NAMES,
        coupled_source_names=COUPLED_SOURCE_NAMES,
        alpha=0.6,
        # Per-joint delta in radians about the default pose, not a normalized
        # fraction of the joint range (see ApexCoupledEMAAction).
        #
        # Finger abduction (j0) is deliberately tiny. Self-collision is off, so
        # with the URDF's full +/-25 deg of abduction the policy learned to splay
        # adjacent fingers *through* each other to clamp the coin: measured
        # lateral gaps of -23 mm, i.e. fingers swapping sides. A knuckle roll
        # needs the fingers parallel and moves the coin by flexion, so removing
        # abduction authority costs nothing and makes crossing geometrically
        # impossible. The thumb keeps full authority to catch the coin.
        scale={
            ".*_thumb_j[0-3]": 0.5,
            ".*_(index|middle|ring|pinky)_j0": 0.04,
            ".*_(index|middle|ring|pinky)_j[12]": 0.5,
        },
        rescale_to_limits=False,
        preserve_order=True,
    )


def _actuated_asset_cfg() -> SceneEntityCfg:
    return SceneEntityCfg("robot", joint_names=ACTUATED_JOINT_NAMES, preserve_order=True)


def joint_pos_obs_term() -> ObsTerm:
    """Normalized actuated joint angles. Shared by every stage's policy group."""
    return ObsTerm(
        func=mdp.joint_pos_limit_normalized,
        noise=Gnoise(std=0.005),
        params={"asset_cfg": _actuated_asset_cfg()},
    )


def joint_vel_obs_term() -> ObsTerm:
    """Actuated joint velocities. Shared by every stage's policy group."""
    return ObsTerm(
        func=mdp.joint_vel_rel,
        scale=0.2,
        noise=Gnoise(std=0.01),
        params={"asset_cfg": _actuated_asset_cfg()},
    )


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos = joint_pos_obs_term()
        joint_vel = joint_vel_obs_term()
        object_pos = ObsTerm(
            func=mdp.root_pos_w, noise=Gnoise(std=0.002), params={"asset_cfg": SceneEntityCfg("object")}
        )
        object_quat = ObsTerm(
            func=mdp.root_quat_w, params={"asset_cfg": SceneEntityCfg("object"), "make_quat_unique": False}
        )
        object_lin_vel = ObsTerm(
            func=mdp.root_lin_vel_w, noise=Gnoise(std=0.002), params={"asset_cfg": SceneEntityCfg("object")}
        )
        object_ang_vel = ObsTerm(
            func=mdp.root_ang_vel_w,
            scale=0.2,
            noise=Gnoise(std=0.002),
            params={"asset_cfg": SceneEntityCfg("object")},
        )
        fingertip_pos = ObsTerm(func=coin_mdp.fingertip_pos_w, noise=Gnoise(std=0.002))
        coin_to_knuckle = ObsTerm(func=coin_mdp.coin_to_knuckle_rel_pos, noise=Gnoise(std=0.002))
        last_action = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()

    @configclass
    class WristImageCfg(ObsGroup):
        rgb = ObsTerm(
            func=mdp.image,
            params={"sensor_cfg": SceneEntityCfg("wrist_camera"), "data_type": "rgb"},
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class OverheadImageCfg(ObsGroup):
        rgb = ObsTerm(
            func=mdp.image,
            params={"sensor_cfg": SceneEntityCfg("overhead_camera"), "data_type": "rgb"},
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class SideImageCfg(ObsGroup):
        rgb = ObsTerm(
            func=mdp.image,
            params={"sensor_cfg": SceneEntityCfg("side_camera"), "data_type": "rgb"},
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    # Populated only by vision env cfgs. rsl_rl's MLP actor reads ``policy`` only.
    wrist_image: WristImageCfg | None = None
    overhead_image: OverheadImageCfg | None = None
    side_image: SideImageCfg | None = None


@configclass
class EventCfg:
    """Domain randomization. Stage A narrows the gain ranges in __post_init__."""

    robot_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.7, 1.3),
            "dynamic_friction_range": (0.7, 1.3),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 250,
        },
    )
    robot_scale_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "mass_distribution_params": (0.95, 1.05),
            "operation": "scale",
        },
    )
    robot_joint_stiffness_and_damping = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stiffness_distribution_params": (0.8, 1.25),
            "damping_distribution_params": (0.8, 1.25),
            "operation": "scale",
            "distribution": "log_uniform",
        },
    )
    object_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("object", body_names=".*"),
            "static_friction_range": (0.55, 0.90),
            "dynamic_friction_range": (0.50, 0.85),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 250,
        },
    )
    object_scale_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("object"),
            "mass_distribution_params": (0.85, 1.15),
            "operation": "scale",
        },
    )
    reset_robot_joints = EventTerm(
        func=inhand_mdp.reset_joints_within_limits_range,
        mode="reset",
        params={
            # Narrow: the coin is placed relative to knuckle frames that are still
            # one step stale at reset, so a wide joint jitter spawns it inside a finger.
            "position_range": {".*": [0.05, 0.05]},
            "velocity_range": {".*": [0.0, 0.0]},
            "use_default_offset": True,
            "operation": "scale",
        },
    )
    reset_object = EventTerm(
        func=coin_mdp.reset_coin_on_knuckles,
        mode="reset",
        params={
            "height": 0.004,
            "jitter": 0.002,
            "finger_indices": (0, 1),
        },
    )
    randomize_lighting = EventTerm(
        func=coin_mdp.randomize_lighting,
        mode="reset",
        params={
            "intensity_range": (400.0, 1400.0),
            "color_lo": (0.62, 0.64, 0.70),
            "color_hi": (1.0, 0.98, 0.92),
        },
    )
    rand_overhead_cam = None
    rand_wrist_cam = None
    rand_side_cam = None


@configclass
class HoldRewardsCfg:
    coin_knuckle_distance = RewTerm(func=coin_mdp.coin_knuckle_distance, weight=-8.0)
    coin_seat_offset = RewTerm(func=coin_mdp.coin_seat_offset, weight=-4.0)
    desired_contact = RewTerm(func=coin_mdp.desired_contact, weight=2.0)
    joint_vel_l2 = RewTerm(func=mdp.joint_vel_l2, weight=-2.5e-5)
    action_l2 = RewTerm(func=mdp.action_l2, weight=-0.0001)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    drop = RewTerm(func=coin_mdp.drop_penalty, weight=-10.0)
    # Same predicate as the hold_success termination, so it doubles as the metric.
    hold_ok = RewTerm(func=coin_mdp.hold_bonus, weight=1.0)
    finger_crossing = RewTerm(func=coin_mdp.finger_crossing, weight=-20.0)
    user_term = RewTerm(func=coin_mdp.user_reward, weight=0.0)


@configclass
class TransferRewardsCfg(HoldRewardsCfg):
    # Stage A hold_ok / index-middle contact anchored the policy to "stay put".
    # Retarget seating & contact toward the transfer bridge; kill hold_ok.
    hold_ok = RewTerm(func=coin_mdp.hold_bonus, weight=0.0)
    desired_contact = RewTerm(
        func=coin_mdp.desired_contact,
        weight=2.0,
        params={"finger_indices": (0, 1, 2), "threshold": 0.024},
    )
    coin_knuckle_distance = RewTerm(func=coin_mdp.coin_bridge_distance, weight=-6.0)
    coin_seat_offset = RewTerm(func=coin_mdp.coin_bridge_seat_offset, weight=-3.0)
    progress = RewTerm(func=coin_mdp.progress, weight=8.0)
    roll_rotation = RewTerm(func=coin_mdp.roll_rotation, weight=0.5)
    slip = RewTerm(func=coin_mdp.slip_penalty, weight=-2.0)
    success = RewTerm(func=coin_mdp.success_bonus, weight=50.0)


@configclass
class HoldTerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    drop = DoneTerm(func=coin_mdp.coin_dropped)
    object_out_of_reach = DoneTerm(func=inhand_mdp.object_away_from_robot, params={"threshold": 0.45})
    success = DoneTerm(
        func=coin_mdp.hold_success,
        params={"hold_steps": 150, "stable_steps": 30, "max_speed": 0.45, "contact_threshold": 0.040},
    )


@configclass
class TransferTerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    drop = DoneTerm(func=coin_mdp.coin_dropped)
    object_out_of_reach = DoneTerm(func=inhand_mdp.object_away_from_robot, params={"threshold": 0.45})


@configclass
class CoinHoldEnvCfg(ManagerBasedRLEnvCfg):
    scene: CoinRollSceneCfg = CoinRollSceneCfg(num_envs=512, env_spacing=0.6)
    sim: SimulationCfg = SimulationCfg(
        physics_material=RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=1.0),
        physics=PhysicsBackendCfg(),
    )
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands = None
    rewards: HoldRewardsCfg = HoldRewardsCfg()
    terminations: HoldTerminationsCfg = HoldTerminationsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self):
        self.decimation = 4
        self.episode_length_s = 3.0
        self.sim.dt = 1.0 / 240.0
        self.sim.render_interval = self.decimation
        # Palm-down: the knuckle bridge sits near (-0.157, 0.0, 0.49), fingers along -X.
        # Look down on the backs of the fingers or the coin hides behind a knuckle.
        self.viewer.eye = (-0.09, -0.10, 0.61)
        self.viewer.lookat = (-0.17, 0.0, 0.488)
        self.viewer.origin_type = "env"
        self.scene.clone_in_fabric = True


@configclass
class CoinTransferEnvCfg(CoinHoldEnvCfg):
    rewards: TransferRewardsCfg = TransferRewardsCfg()
    terminations: TransferTerminationsCfg = TransferTerminationsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = 5.0
        # Widen domain randomization for Stage B.
        self.events.robot_joint_stiffness_and_damping.params["stiffness_distribution_params"] = (0.3, 3.0)
        self.events.robot_joint_stiffness_and_damping.params["damping_distribution_params"] = (0.75, 1.5)
        self.events.object_scale_mass.params["mass_distribution_params"] = (0.4, 1.6)


@configclass
class CoinHoldEnvCfg_PLAY(CoinHoldEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.observations.policy.enable_corruption = False
        self.terminations.time_out = None


@configclass
class CoinTransferEnvCfg_PLAY(CoinTransferEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.observations.policy.enable_corruption = False
        self.terminations.time_out = None


@configclass
class CoinHoldLeftEnvCfg(CoinHoldEnvCfg):
    """Left-hand asset sanity / later training. Not the primary policy."""

    def __post_init__(self):
        super().__post_init__()
        apply_hand_side(self, "left", "palm_down_knuckle")


@configclass
class CoinHoldVisionEnvCfg(CoinHoldEnvCfg):
    """Hold + wrist / overhead / side cameras. MLP policy still reads state only."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 128
        self.scene.wrist_camera = WRIST_RGB128
        self.scene.overhead_camera = OVERHEAD_RGB128
        self.scene.side_camera = SIDE_RGB128
        self.observations.wrist_image = ObservationsCfg.WristImageCfg()
        self.observations.overhead_image = ObservationsCfg.OverheadImageCfg()
        self.observations.side_image = ObservationsCfg.SideImageCfg()
        self.events.rand_overhead_cam = EventTerm(
            func=coin_mdp.randomize_camera_offset,
            mode="reset",
            params={"sensor_name": "overhead_camera", "pos_jitter_m": 0.005, "rot_jitter_deg": 3.0},
        )
        self.events.rand_wrist_cam = EventTerm(
            func=coin_mdp.randomize_camera_offset,
            mode="reset",
            params={"sensor_name": "wrist_camera", "pos_jitter_m": 0.004, "rot_jitter_deg": 3.0},
        )
        self.events.rand_side_cam = EventTerm(
            func=coin_mdp.randomize_camera_offset,
            mode="reset",
            params={"sensor_name": "side_camera", "pos_jitter_m": 0.005, "rot_jitter_deg": 3.0},
        )


@configclass
class CoinHoldVisionEnvCfg_PLAY(CoinHoldVisionEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 4
        self.observations.policy.enable_corruption = False
        self.terminations.time_out = None
