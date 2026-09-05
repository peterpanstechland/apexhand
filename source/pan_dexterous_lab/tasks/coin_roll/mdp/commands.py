"""The baoding rotation-direction command.

Its own module because it is shared state: the reward scores against it, the
observation feeds it to the policy, and a reset event samples it. Sampling has
to be the event's job -- rewards run before resets and observations after, so a
term that resampled lazily would hand the policy one direction and score it
against another.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv

_DIRECTION = "_pan_baoding_direction"


def spin_direction(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """(N,) commanded rotation direction, +1 or -1. Read-only."""
    direction = getattr(env, _DIRECTION, None)
    if direction is None:
        # Before the first reset event: default to the positive direction so
        # the term is well defined during manager start-up dimension probing.
        direction = torch.ones(env.num_envs, device=env.device)
        setattr(env, _DIRECTION, direction)
    return direction


def resample_spin_direction(
    env: "ManagerBasedEnv",
    env_ids: torch.Tensor,
    p_reverse: float = 0.0,
) -> None:
    """Reset event: draw a direction per episode.

    ``p_reverse = 0`` keeps every episode turning the same way, which is what
    the first training run wants. The command channel exists regardless, so
    enabling bidirectional training later does not change the policy's input
    layout and therefore does not invalidate an exported ONNX.
    """
    if len(env_ids) == 0:
        return
    direction = spin_direction(env)
    if p_reverse <= 0.0:
        direction[env_ids] = 1.0
        return
    flip = torch.rand(len(env_ids), device=direction.device) < p_reverse
    direction[env_ids] = torch.where(flip, -1.0, 1.0)
