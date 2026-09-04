"""16-DoF policy actions with 1:1 write-through of the 5 passive Apex joints.

Isaac Sim 6 writes URDF ``<mimic>`` as ``newton:mimic*`` attributes. PhysX
ignores those, so DIP/IP joints would be unactuated free joints unless we
copy the source joint targets here.
"""

from __future__ import annotations

from dataclasses import MISSING

import torch

from isaaclab.envs.mdp.actions.actions_cfg import EMAJointPositionToLimitsActionCfg
from isaaclab.envs.mdp.actions.joint_actions_to_limits import EMAJointPositionToLimitsAction
from isaaclab.utils.configclass import configclass

if False:  # TYPE_CHECKING without circular import at runtime
    from isaaclab.envs import ManagerBasedEnv


class ApexCoupledEMAAction(EMAJointPositionToLimitsAction):
    """Map a 16-dim action onto 21 joints; passive joints 1:1 follow sources.

    Targets are a bounded delta about the *default* joint pose rather than the
    base class's rescale-to-limits mapping. With rescale-to-limits, action 0 is
    forced to be the mid-point of every joint's range, which fixes the neutral
    pose to a tight curl -- the knuckle bridge comes out sloped and the coin
    rolls off. Offsetting from the default pose lets the calibrated palm-down
    pose be the neutral action, so the policy's initial zero-mean Gaussian holds
    the coin instead of flicking it away.
    """

    cfg: "ApexCoupledEMAActionCfg"

    def __init__(self, cfg: "ApexCoupledEMAActionCfg", env: "ManagerBasedEnv"):
        super().__init__(cfg, env)
        if isinstance(self._joint_ids, slice):
            raise RuntimeError(
                "ApexCoupledEMAAction expects an explicit 16-joint id list. "
                "Set joint_names to ACTUATED_JOINT_NAMES and preserve_order=True."
            )
        self._cpl_ids, _ = self._asset.find_joints(cfg.coupled_joint_names, preserve_order=True)
        src_ids, _ = self._asset.find_joints(cfg.coupled_source_names, preserve_order=True)
        src_slot = []
        for src_id in src_ids:
            try:
                src_slot.append(self._joint_ids.index(src_id))
            except ValueError as exc:
                raise RuntimeError(
                    f"Coupled source joint id {src_id} is not in the 16 actuated joints {self._joint_ids}."
                ) from exc
        self._src_slot = torch.tensor(src_slot, device=self.device, dtype=torch.long)
        self._src_ids_t = torch.tensor(src_ids, device=self.device, dtype=torch.long)
        self._cpl_ids_t = torch.tensor(self._cpl_ids, device=self.device, dtype=torch.int32)

    def process_actions(self, actions: torch.Tensor):
        # Deliberately does not call super(): the base chain would either rescale
        # to limits (losing the default-pose origin) or drop the offset entirely.
        self._raw_actions[:] = actions
        default = self._asset.data.default_joint_pos.torch[:, self._joint_ids]
        target = default + self._raw_actions * self._scale
        limits = self._asset.data.soft_joint_pos_limits.torch[:, self._joint_ids]
        ema = self._alpha * target + (1.0 - self._alpha) * self._prev_applied_actions
        self._processed_actions[:] = torch.clamp(ema, limits[..., 0], limits[..., 1])
        self._prev_applied_actions[:] = self._processed_actions[:]

    def apply_actions(self):
        super().apply_actions()
        # Command the same PD target as the source joint...
        coupled_targets = self.processed_actions[:, self._src_slot]
        self._asset.set_joint_position_target_index(target=coupled_targets, joint_ids=self._cpl_ids_t)
        # ...and snap the passive DoF. PhysX ignores newton:mimic*, so without this
        # the DIP/IP joints are free and lag or collapse.
        src_q = self._asset.data.joint_pos.torch[:, self._src_ids_t]
        self._asset.write_joint_position_to_sim_index(position=src_q, joint_ids=self._cpl_ids_t)


@configclass
class ApexCoupledEMAActionCfg(EMAJointPositionToLimitsActionCfg):
    """Config for :class:`ApexCoupledEMAAction`."""

    class_type: type = ApexCoupledEMAAction
    coupled_joint_names: list[str] = MISSING
    coupled_source_names: list[str] = MISSING
    preserve_order: bool = True
