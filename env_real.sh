#!/usr/bin/env bash
# Apex Hand real-SDK env (separate from Isaac Sim env.sh)
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$ROOT/.venv-apex-real/bin/activate"
export LD_LIBRARY_PATH="$ROOT/third_party/Rysen_SDK/rysen_sdk/lib/x86_64:${LD_LIBRARY_PATH:-}"
export APEX_HAND_IP="${APEX_HAND_IP:-192.168.88.200}"
echo "[env_real] Python=$(python -V 2>&1)"
echo "[env_real] APEX_HAND_IP=$APEX_HAND_IP"
echo "[env_real] LD_LIBRARY_PATH includes: $ROOT/third_party/Rysen_SDK/rysen_sdk/lib/x86_64"
