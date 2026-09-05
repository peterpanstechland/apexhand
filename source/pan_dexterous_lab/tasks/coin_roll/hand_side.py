"""Retarget an env cfg between the left and right Apex Hand.

Three things have to move together or the env breaks in confusing ways:

1. the robot USD,
2. every joint-name list (action term, joint observations),
3. the ``side`` argument the MDP terms use to resolve palm / knuckle / pad /
   fingertip **body** names.

Only the first two used to be handled, by string-replacing ``right_`` with
``left_``. Point 3 was missed entirely: the MDP terms fall back to
``DEFAULT_SIDE = "right"``, so a left-hand cfg looked fine until a reward asked
a left USD for ``right_palm_link``. Terms do not spell ``side`` out in their
params, so it has to be injected wherever the term function accepts it.
"""

from __future__ import annotations

import inspect
from typing import Any

from isaaclab.managers import SceneEntityCfg

from pan_dexterous_lab.assets.apex_cfg import make_hand_cfg
from pan_dexterous_lab.assets.joints import (
    Side,
    actuated_joint_names,
    coupled_joint_names,
    coupled_source_names,
)


def _fields(cfg: Any) -> dict[str, Any]:
    """Attribute dict of a configclass instance, or empty for anything else."""
    return dict(vars(cfg)) if hasattr(cfg, "__dict__") else {}


def _iter_terms(manager_cfg: Any):
    """Yield term cfgs, descending one level so observation groups are covered."""
    for value in _fields(manager_cfg).values():
        if value is None:
            continue
        if hasattr(value, "func"):
            yield value
        else:
            for inner in _fields(value).values():
                if inner is not None and hasattr(inner, "func"):
                    yield inner


def _retarget_term(term: Any, side: Side, joints: list[str]) -> None:
    params = getattr(term, "params", None)
    if params is None:
        return
    # Body names come from the ``side`` argument; inject it only where the term
    # function actually takes one, so unrelated terms keep their signature.
    try:
        accepts_side = "side" in inspect.signature(term.func).parameters
    except (TypeError, ValueError):
        accepts_side = False
    if accepts_side:
        params["side"] = side
    for key, value in params.items():
        if isinstance(value, SceneEntityCfg) and value.joint_names is not None:
            params[key] = SceneEntityCfg(
                value.name, joint_names=joints, preserve_order=value.preserve_order
            )


def detect_hand_side(robot: Any) -> Side:
    """Which hand a live articulation actually is, read off its body names.

    Evaluation and export scripts otherwise fall back to ``DEFAULT_SIDE`` and
    silently measure the wrong bodies when the task is the left-hand variant.
    """
    names = set(robot.body_names)
    for side in ("left", "right"):
        if f"{side}_palm_link" in names:
            return side  # type: ignore[return-value]
    raise RuntimeError(f"no left_/right_palm_link among {sorted(names)[:8]}...")


def apply_hand_side(cfg: Any, side: Side, pose: str) -> None:
    """Point a coin-roll / baoding env cfg at ``side``, in place.

    ``pose`` is the hand's spawn pose (``palm_down_knuckle`` / ``palm_up_cradle``)
    and must match what the task expects; it is not inferable from the cfg
    without reverse-engineering the spawn quaternion.
    """
    joints = actuated_joint_names(side)

    cfg.scene.robot = make_hand_cfg(side, pose).replace(prim_path="{ENV_REGEX_NS}/Robot")

    cfg.actions.joint_pos.joint_names = joints
    cfg.actions.joint_pos.coupled_joint_names = coupled_joint_names(side)
    cfg.actions.joint_pos.coupled_source_names = coupled_source_names(side)

    for manager in (cfg.observations, cfg.rewards, cfg.terminations, cfg.events):
        if manager is None:
            continue
        for term in _iter_terms(manager):
            _retarget_term(term, side, joints)
