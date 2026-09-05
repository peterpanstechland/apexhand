"""Baoding rotation: two 30 mm wooden balls circulating in a palm-up cradle.

Split out of ``coin_roll_env_cfg`` because almost nothing about the task matches
the coin stages any more -- different object, different hand pose, different
actuation, and in particular a different observation contract.

That contract is the important part. The coin policy was trained on privileged
world state (coin pose, fingertip positions) that the real hand cannot measure,
so deployment had to feed it stub zeros and the policy saturated. Here the actor
and critic are split:

* ``policy``     -- proprioception, plus only the ball features a camera can
                    actually recover. This is what ships to the robot.
* ``privileged`` -- ground-truth ball and fingertip state, critic only, never
                    exported.

See ``BaodingPPORunnerCfg.obs_groups`` for the wiring.
"""

from __future__ import annotations

import isaaclab.envs.mdp as mdp
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils.configclass import configclass
from isaaclab.utils.noise import GaussianNoiseCfg as Gnoise

import isaaclab_tasks.manager_based.manipulation.inhand.mdp as inhand_mdp

from pan_dexterous_lab.assets.apex_cfg import CradleRobotPresetCfg
from pan_dexterous_lab.assets.objects import BaodingObject2PresetCfg, BaodingObjectPresetCfg
from pan_dexterous_lab.tasks.coin_roll import mdp as coin_mdp
from pan_dexterous_lab.tasks.coin_roll.coin_roll_env_cfg import (
    CoinHoldEnvCfg,
    CoinRollSceneCfg,
    joint_pos_obs_term,
    joint_vel_obs_term,
)
from pan_dexterous_lab.tasks.coin_roll.hand_side import apply_hand_side

# Per-joint action delta in radians about the cradle pose.
#
# The knuckle-roll stages pinned finger abduction to 0.04 rad because
# self-collision is disabled and the policy had learned to splay fingers
# *through* each other to clamp the coin. Baoding inverts the priority: sweeping
# the fingers sideways is the main way the balls get driven around the palm, so
# abduction needs real authority. 0.30 rad is roughly two thirds of the URDF's
# +/-25 deg travel, enough to circulate the pair while still short of the limit;
# the finger_crossing penalty stays in the reward as the guard.
BAODING_ACTION_SCALE = {
    ".*_thumb_j[0-3]": 0.5,
    ".*_(index|middle|ring|pinky)_j0": 0.30,
    ".*_(index|middle|ring|pinky)_j[12]": 0.5,
}


@configclass
class BaodingObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        """Everything here must be reproducible on the real hand.

        Joint state comes from the SDK, ``pair`` from ``real/ball_tracker.py``,
        and the command is whatever direction we ask for at run time. Terms are
        declared explicitly rather than inherited-and-nulled so the
        concatenation order stays readable -- the real runner has to rebuild
        this vector in exactly this order.
        """

        joint_pos = joint_pos_obs_term()
        joint_vel = joint_vel_obs_term()
        pair = ObsTerm(func=coin_mdp.baoding_pair_obs, noise=Gnoise(std=0.002))
        spin_command = ObsTerm(func=coin_mdp.baoding_spin_command)
        last_action = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class PrivilegedCfg(ObsGroup):
        """Critic-only ground truth. Reduces value variance without ever being
        asked of the hardware. Uncorrupted -- noising the critic's input buys
        nothing."""

        object_pos = ObsTerm(func=mdp.root_pos_w, params={"asset_cfg": SceneEntityCfg("object")})
        object_lin_vel = ObsTerm(
            func=mdp.root_lin_vel_w, params={"asset_cfg": SceneEntityCfg("object")}
        )
        object_ang_vel = ObsTerm(
            func=mdp.root_ang_vel_w, scale=0.2, params={"asset_cfg": SceneEntityCfg("object")}
        )
        object2_pos = ObsTerm(func=coin_mdp.object2_pos_w)
        object2_lin_vel = ObsTerm(func=coin_mdp.object2_lin_vel_w)
        fingertip_pos = ObsTerm(func=coin_mdp.fingertip_pos_w)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    privileged: PrivilegedCfg = PrivilegedCfg()


@configclass
class BaodingRewardsCfg:
    # Rotation is the task; everything else exists to stop a degenerate way of
    # collecting it. Without ball_gap the policy parks one ball and orbits the
    # other, and without pair_centering it walks the pair off the palm edge.
    spin = RewTerm(func=coin_mdp.spin, weight=8.0)
    ball_gap = RewTerm(func=coin_mdp.ball_gap, weight=-4.0)
    pair_centering = RewTerm(func=coin_mdp.pair_centering, weight=-3.0)
    hold_pair = RewTerm(func=coin_mdp.hold_pair, weight=1.0)
    drop = RewTerm(func=coin_mdp.baoding_drop_penalty, weight=-12.0)
    joint_vel_l2 = RewTerm(func=mdp.joint_vel_l2, weight=-2.5e-5)
    action_l2 = RewTerm(func=mdp.action_l2, weight=-0.0001)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    # Self-collision is off, so this is the only thing stopping the now-freed
    # abduction from driving fingers through one another.
    finger_crossing = RewTerm(func=coin_mdp.finger_crossing, weight=-20.0)
    user_term = RewTerm(func=coin_mdp.user_reward, weight=0.0)


@configclass
class BaodingTerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    drop = DoneTerm(func=coin_mdp.balls_dropped)
    object_out_of_reach = DoneTerm(
        func=inhand_mdp.object_away_from_robot, params={"threshold": 0.45}
    )


@configclass
class BaodingSceneCfg(CoinRollSceneCfg):
    robot: CradleRobotPresetCfg = CradleRobotPresetCfg()
    object: BaodingObjectPresetCfg = BaodingObjectPresetCfg()
    object2: BaodingObject2PresetCfg = BaodingObject2PresetCfg()


@configclass
class BaodingRotateEnvCfg(CoinHoldEnvCfg):
    scene: BaodingSceneCfg = BaodingSceneCfg(num_envs=512, env_spacing=0.6)
    observations: BaodingObservationsCfg = BaodingObservationsCfg()
    rewards: BaodingRewardsCfg = BaodingRewardsCfg()
    terminations: BaodingTerminationsCfg = BaodingTerminationsCfg()

    def __post_init__(self):
        super().__post_init__()
        # Circulating the pair takes several seconds of build-up, so episodes
        # have to be long enough to contain more than one hand-off.
        self.episode_length_s = 10.0
        self.actions.joint_pos.scale = dict(BAODING_ACTION_SCALE)

        # Palm-up: look down into the cradle instead of at the knuckle backs.
        self.viewer.eye = (-0.05, -0.18, 0.66)
        self.viewer.lookat = (-0.04, 0.0, 0.545)

        self.events.reset_object = EventTerm(
            func=coin_mdp.reset_objects_in_palm,
            mode="reset",
            # Height is above the metacarpal-head centroid, and must clear the
            # metacarpal head bodies themselves (~10 mm radius, centres 7 mm
            # above the palm) or PhysX depenetrates the pair straight up at
            # 1 m/s on the first step. The balls then settle by gravity.
            params={"height": 0.028, "pair_gap": 0.003, "jitter": 0.002},
        )
        self.events.spin_direction = EventTerm(
            func=coin_mdp.resample_spin_direction,
            mode="reset",
            params={"p_reverse": 0.0},
        )
        # Wooden balls on rubber pads grip far better than the metal coin did.
        self.events.object_physics_material.params["static_friction_range"] = (0.60, 0.95)
        self.events.object_physics_material.params["dynamic_friction_range"] = (0.50, 0.85)
        # The pair was weighed at 9.55 g, so mass is not the uncertain quantity
        # here -- friction and joint gains are. Keep the spread narrow.
        self.events.object_scale_mass.params["mass_distribution_params"] = (0.92, 1.08)


@configclass
class BaodingRotateEnvCfg_PLAY(BaodingRotateEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.observations.policy.enable_corruption = False
        self.terminations.time_out = None


@configclass
class BaodingRotateLeftEnvCfg(BaodingRotateEnvCfg):
    """The deployment target: the physical hand on the bench is a left hand."""

    def __post_init__(self):
        super().__post_init__()
        apply_hand_side(self, "left", "palm_up_cradle")


@configclass
class BaodingRotateLeftEnvCfg_PLAY(BaodingRotateLeftEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.observations.policy.enable_corruption = False
        self.terminations.time_out = None
