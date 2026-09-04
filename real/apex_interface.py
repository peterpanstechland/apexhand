"""Rysen Apex Hand SDK wrapper.

Not implemented: no physical hand is attached on this machine.
Official link: Ethernet, TCP 5856 / 5857, Ubuntu 22.04 / Python 3.10.
https://docs.rysenbot.com/apex-hand/get-started
"""

from __future__ import annotations


class ApexInterface:
    """Placeholder for `rysen-sdk` joint get/set.

    When hardware arrives:
    1. Map ``ACTUATED_JOINT_NAMES`` through ``logs/.../exported/joint_map.json``.
    2. Send position targets at 60 Hz over the wired Ethernet link.
    3. Never send the 5 coupled joints independently — the firmware owns those.
    """

    def connect(self) -> None:
        raise NotImplementedError("No Apex Hand on this host. See class docstring.")

    def get_joint_positions(self) -> list[float]:
        raise NotImplementedError

    def set_joint_positions(self, q: list[float]) -> None:
        raise NotImplementedError
