"""User-authored reward terms. The Web UI writes new functions here after review.

``user_reward`` is what ``HoldRewardsCfg.user_term`` / ``TransferRewardsCfg.user_term``
call. Keep the weight at 0 until you have inspected a Reward Probe run.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def user_reward(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """No-op placeholder. Replace via the Web UI dialog or edit this file."""
    return torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)
