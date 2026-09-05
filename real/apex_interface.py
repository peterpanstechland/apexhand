"""Rysen Apex Hand SDK wrapper.

Official: Ethernet TCP 5856/5857, Ubuntu 22.04 / Python 3.10.
https://docs.rysenbot.com/apex-hand/get-started

Actuated order is always ``ACTUATED_LOGICAL``. Coupled joints are written as
1:1 copies of their source — never commanded independently.
"""

from __future__ import annotations

import os
from typing import Sequence

from real.joint_table import ACTUATED_LOGICAL, COUPLED_LOGICAL, COUPLED_SOURCE_LOGICAL
from real.safety import clamp_joint

_FINGER_SDK = {
    "thumb": "THUMB",
    "index": "INDEX",
    "middle": "MIDDLE",
    "ring": "RING",
    "pinky": "PINKY",
}

DEFAULT_HAND_IP = os.environ.get("APEX_HAND_IP", "192.168.88.200")


def logical_to_joint_id(name: str):
    from rysen_apexhand_sdk import JointId

    finger, joint = name.split("_", 1)
    attr = f"JOINT_ID_{_FINGER_SDK[finger]}_{joint.upper()}"
    return getattr(JointId, attr)


class ApexInterface:
    def __init__(self, ip: str | None = None) -> None:
        self.ip = ip or DEFAULT_HAND_IP
        self._sdk = None
        self._side: str | None = None

    @property
    def side(self) -> str:
        if self._side is None:
            raise RuntimeError("connect() first")
        return self._side

    def connect(self) -> None:
        from rysen_apexhand_sdk import ConnectionType, ErrorCode, HandDir, Rysen

        sdk = Rysen()
        ret = sdk.connect(self.ip, ConnectionType.CONNECTION_TYPE_ETHERNET)
        if ret != ErrorCode.ERROR_CODE_OK:
            raise RuntimeError(f"SDK connect({self.ip}) failed: {ret}")
        direction = sdk.get_hand_dir()
        self._side = "left" if direction == HandDir.LEFT else "right"
        self._sdk = sdk

    def configure_motion(self, *, max_speed: float, max_accel: float, finger_torque_pct: float) -> None:
        from rysen_apexhand_sdk import FingerId, JointId, MaxFingerTorque, MaxJointAccel, MaxJointSpeed

        sdk = self._require()
        speeds, accels = [], []
        for i in range(21):
            s, a = MaxJointSpeed(), MaxJointAccel()
            s.joint_id = a.joint_id = JointId(i)
            s.speed = float(max_speed)
            a.accel = float(max_accel)
            speeds.append(s)
            accels.append(a)
        torques = []
        for i in range(5):
            t = MaxFingerTorque()
            t.finger_id = FingerId(i)
            t.torque = float(finger_torque_pct)
            torques.append(t)
        sdk.set_max_joint_speed(speeds)
        sdk.set_max_joint_accel(accels)
        sdk.set_max_finger_torque(torques)

    def enable(self) -> None:
        from rysen_apexhand_sdk import ErrorCode

        ret = self._require().set_all_fingers_enabled()
        if ret != ErrorCode.ERROR_CODE_OK:
            raise RuntimeError(f"enable fingers failed: {ret}")

    def disable(self) -> None:
        if self._sdk is None:
            return
        self._sdk.set_all_fingers_disabled()

    def get_joint_positions(self) -> list[float]:
        q, _ = self.get_actuated_pv()
        return q

    def get_actuated_pv(self) -> tuple[list[float], list[float]]:
        """Actuated joint positions and velocities, ``ACTUATED_LOGICAL`` order."""
        states = self._require().get_joint_states()

        def _as_int(jid) -> int:
            value = getattr(jid, "value", jid)
            return int(value)

        pos = {_as_int(j.joint_id): float(j.position) for j in states.joint_states}
        vel = {_as_int(j.joint_id): float(j.velocity) for j in states.joint_states}
        ids = [_as_int(logical_to_joint_id(name)) for name in ACTUATED_LOGICAL]
        return [pos[i] for i in ids], [vel[i] for i in ids]

    def get_finger_currents(self) -> dict[str, float]:
        """Max |motor current| (A) per finger: thumb / index / middle / ring / pinky."""
        states = self._require().get_motor_states()
        out = {name: 0.0 for name in _FINGER_SDK}
        for motor in states.motors:
            label = str(motor.motor_id).upper()
            finger = "pinky" if "LITTLE" in label or "PINKY" in label else None
            if finger is None:
                for name in _FINGER_SDK:
                    if name.upper() in label:
                        finger = name
                        break
            if finger is None:
                continue
            raw = abs(float(motor.current))
            # Hand firmware reports milliamps. 449–453 is a stale full-scale rail.
            if 440.0 <= raw <= 460.0:
                continue
            amp = raw * 0.001
            if amp > out[finger]:
                out[finger] = amp
        return out

    def set_joint_positions(self, q: Sequence[float], *, torque_nmm: float = 200.0) -> None:
        from rysen_apexhand_sdk import ErrorCode, create_move_j_position_follow_param

        if len(q) != len(ACTUATED_LOGICAL):
            raise ValueError(f"expected {len(ACTUATED_LOGICAL)} actuated joints, got {len(q)}")
        by_name = {name: clamp_joint(name, float(v)) for name, v in zip(ACTUATED_LOGICAL, q)}
        for src, dst in zip(COUPLED_SOURCE_LOGICAL, COUPLED_LOGICAL):
            by_name[dst] = clamp_joint(dst, by_name[src])
        params = [
            create_move_j_position_follow_param(logical_to_joint_id(name), pos, torque_nmm)
            for name, pos in by_name.items()
        ]
        ret = self._require().move_j_position_follow(params)
        if ret == ErrorCode.ERROR_CODE_OUT_OF_RANGE:
            print(f"move_j_position_follow clamped/out of range, skipped tick", flush=True)
            return
        if ret != ErrorCode.ERROR_CODE_OK:
            raise RuntimeError(f"move_j_position_follow failed: {ret}")

    def close(self) -> None:
        if self._sdk is None:
            return
        try:
            self.disable()
        finally:
            self._sdk.disconnect()
            self._sdk = None
            self._side = None

    def _require(self):
        if self._sdk is None:
            raise RuntimeError("connect() first")
        return self._sdk
