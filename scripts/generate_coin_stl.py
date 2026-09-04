#!/usr/bin/env python3
"""Generate a 32 mm x 4 mm PΛN curriculum token STL (visual only).

Collision in Isaac Lab is a plain cylinder — do not feed this STL to the solver.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import trimesh

OUT = Path(__file__).resolve().parents[1] / "assets" / "token" / "PAN_coin.stl"
RADIUS = 0.016
HEIGHT = 0.004
EMBOSS = 0.0007


def _letter_boxes() -> list[trimesh.Trimesh]:
    """Very coarse raised P Λ N blocks on the +Z face."""
    z0 = HEIGHT / 2.0
    specs = [
        # P
        (-0.009, 0.000, 0.0025, 0.010),
        (-0.0065, 0.004, 0.004, 0.0025),
        (-0.0065, 0.001, 0.004, 0.0020),
        # Λ (two slashes as boxes)
        (-0.0015, 0.000, 0.0020, 0.010),
        (0.0015, 0.000, 0.0020, 0.010),
        # N
        (0.006, 0.000, 0.0020, 0.010),
        (0.010, 0.000, 0.0020, 0.010),
        (0.008, 0.002, 0.0040, 0.0025),
    ]
    meshes = []
    for x, y, sx, sy in specs:
        box = trimesh.creation.box(extents=(sx, sy, EMBOSS))
        box.apply_translation((x, y, z0 + EMBOSS / 2.0))
        meshes.append(box)
    return meshes


def main() -> None:
    disk = trimesh.creation.cylinder(radius=RADIUS, height=HEIGHT, sections=64)
    token = trimesh.util.concatenate([disk] + _letter_boxes())
    OUT.parent.mkdir(parents=True, exist_ok=True)
    token.export(OUT)
    print(f"Wrote {OUT}  verts={len(token.vertices)}  faces={len(token.faces)}")
    print("Collision must stay a cylinder; this mesh is visual-only.")


if __name__ == "__main__":
    main()
