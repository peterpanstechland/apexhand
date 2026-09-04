"""ONNX policy loop for the real Apex Hand. Unimplemented without hardware."""

from __future__ import annotations


def main() -> None:
    raise NotImplementedError(
        "Real-hand loop: SDK feedback → normalize → ONNX → Δq → SafetyFilter → SDK. "
        "Requires joint_map.json from scripts/export_onnx.py."
    )


if __name__ == "__main__":
    main()
