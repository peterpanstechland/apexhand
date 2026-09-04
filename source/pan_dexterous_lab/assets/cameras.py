"""Multi-view cameras for the Apex in-hand tasks.

Default Hold/Transfer policies stay state-based. These sensors exist so we can
(1) record more than the viewer camera, (2) randomize viewpoint/lighting, and
(3) later swap in a CNN encoder without redoing spawn math.
"""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.sensors import CameraCfg
from isaaclab.utils.configclass import configclass
from isaaclab_tasks.utils import PresetCfg

from .joints import DEFAULT_SIDE, palm_body_name


def _pinhole() -> sim_utils.PinholeCameraCfg:
    return sim_utils.PinholeCameraCfg(focal_length=24.0, horizontal_aperture=20.955, clipping_range=(0.02, 1.2))


def make_wrist_camera(
    width: int = 128,
    height: int = 128,
    data_types: list[str] | None = None,
    side: str = DEFAULT_SIDE,
) -> CameraCfg:
    return CameraCfg(
        prim_path=f"{{ENV_REGEX_NS}}/Robot/{palm_body_name(side)}/wrist_camera",
        offset=CameraCfg.OffsetCfg(
            pos=(-0.02, 0.0, 0.07),
            rot=(0.653, 0.271, 0.271, 0.653),
            convention="opengl",
        ),
        data_types=data_types or ["rgb"],
        spawn=_pinhole(),
        width=width,
        height=height,
    )


def make_overhead_camera(width: int = 128, height: int = 128, data_types: list[str] | None = None) -> CameraCfg:
    return CameraCfg(
        prim_path="{ENV_REGEX_NS}/overhead_camera",
        offset=CameraCfg.OffsetCfg(
            pos=(-0.09, -0.10, 0.61),
            rot=(0.020, 0.852, 0.522, 0.033),
            convention="world",
        ),
        data_types=data_types or ["rgb"],
        spawn=_pinhole(),
        width=width,
        height=height,
    )


def make_side_camera(width: int = 128, height: int = 128, data_types: list[str] | None = None) -> CameraCfg:
    return CameraCfg(
        prim_path="{ENV_REGEX_NS}/side_camera",
        offset=CameraCfg.OffsetCfg(
            pos=(-0.16, -0.22, 0.54),
            rot=(0.653, 0.271, 0.271, 0.653),
            convention="world",
        ),
        data_types=data_types or ["rgb"],
        spawn=_pinhole(),
        width=width,
        height=height,
    )


WRIST_RGB64 = make_wrist_camera(64, 64)
WRIST_RGB128 = make_wrist_camera(128, 128)
WRIST_RGB256 = make_wrist_camera(256, 256)
OVERHEAD_RGB64 = make_overhead_camera(64, 64)
OVERHEAD_RGB128 = make_overhead_camera(128, 128)
OVERHEAD_RGB256 = make_overhead_camera(256, 256)
SIDE_RGB64 = make_side_camera(64, 64)
SIDE_RGB128 = make_side_camera(128, 128)
SIDE_RGB256 = make_side_camera(256, 256)


@configclass
class WristCameraPresetCfg(PresetCfg):
    off = None
    rgb64: CameraCfg = WRIST_RGB64
    rgb128: CameraCfg = WRIST_RGB128
    rgb256: CameraCfg = WRIST_RGB256
    default = off


@configclass
class OverheadCameraPresetCfg(PresetCfg):
    off = None
    rgb64: CameraCfg = OVERHEAD_RGB64
    rgb128: CameraCfg = OVERHEAD_RGB128
    rgb256: CameraCfg = OVERHEAD_RGB256
    default = off


@configclass
class SideCameraPresetCfg(PresetCfg):
    off = None
    rgb64: CameraCfg = SIDE_RGB64
    rgb128: CameraCfg = SIDE_RGB128
    rgb256: CameraCfg = SIDE_RGB256
    default = off
