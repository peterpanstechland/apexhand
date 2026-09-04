#!/usr/bin/env bash
# Convert official Apex URDFs to USD. Do NOT pass --merge-joints (pads/tips must stay).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck disable=SC1091
source "$ROOT/env.sh"
cd "$ISAACLAB_PATH"

for H in right left; do
  out_dir="$ROOT/assets/apex/usd/$H"
  mkdir -p "$out_dir"
  python scripts/tools/convert_urdf.py \
    "$ROOT/assets/apex-hand-urdf/apex_hand_$H/apex_hand_$H.urdf" \
    "$out_dir" \
    --fix-base --joint-stiffness 3.0 --joint-damping 0.1 --headless
  echo "Converted $H -> $out_dir"
done
