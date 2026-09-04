"""Coin-roll MDP terms."""

from .actions import ApexCoupledEMAAction, ApexCoupledEMAActionCfg
from .events import randomize_camera_offset, randomize_lighting, reset_coin_on_knuckles, reset_objects_in_palm
from .observations import coin_to_knuckle_rel_pos, fingertip_pos_w, object2_lin_vel_w, object2_pos_w
from .rewards import (
    coin_bridge_distance,
    coin_bridge_seat_offset,
    coin_knuckle_distance,
    coin_seat_offset,
    desired_contact,
    drop_penalty,
    finger_crossing,
    hold_bonus,
    progress,
    roll_rotation,
    slip_penalty,
    success_bonus,
)
from .rewards_baoding import ball_gap, balls_dropped, hold_pair, pair_centering, spin
from .rewards_baoding import drop_penalty as baoding_drop_penalty
from .terminations import coin_dropped, hold_ok, hold_success
from .user_rewards import user_reward

__all__ = [
    "ApexCoupledEMAAction",
    "ApexCoupledEMAActionCfg",
    "ball_gap",
    "baoding_drop_penalty",
    "balls_dropped",
    "coin_bridge_distance",
    "coin_bridge_seat_offset",
    "coin_dropped",
    "coin_knuckle_distance",
    "coin_seat_offset",
    "coin_to_knuckle_rel_pos",
    "desired_contact",
    "drop_penalty",
    "finger_crossing",
    "fingertip_pos_w",
    "hold_bonus",
    "hold_ok",
    "hold_pair",
    "hold_success",
    "object2_lin_vel_w",
    "object2_pos_w",
    "pair_centering",
    "progress",
    "randomize_camera_offset",
    "randomize_lighting",
    "reset_coin_on_knuckles",
    "reset_objects_in_palm",
    "roll_rotation",
    "slip_penalty",
    "spin",
    "success_bonus",
    "user_reward",
]
