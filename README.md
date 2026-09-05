# PΛN Dexterous Lab

Apex Hand coin-roll training on this host's existing **Isaac Sim 6.0.1** + **Isaac Lab `v3.0.0-beta2.patch1`**.

**换机续做先看进度：** [PROGRESS.md](PROGRESS.md) · 门闸/数字表：[RESULTS.md](RESULTS.md) · 真机 SDK：[docs/REAL_SDK.zh.md](docs/REAL_SDK.zh.md)

**完整使用教程（环境搭建 / 训练 / Sim2Real）：** [docs/USAGE_GUIDE.zh.md](docs/USAGE_GUIDE.zh.md)

**可视化调参控制台：** [docs/WEBUI.zh.md](docs/WEBUI.zh.md) · 参数百科：[docs/PARAMS.zh.md](docs/PARAMS.zh.md) · RL 术语：[docs/RL_GLOSSARY.zh.md](docs/RL_GLOSSARY.zh.md)

## Machine facts (do not ignore)

- RTX 4080 Laptop **12 GB**, not 16 GB. Start at 512 envs.
- Do not reinstall Isaac Sim. Use `~/isaacsim-env`.
- Every new terminal: `source ~/Documents/apexhand/env.sh`
- Never pass `--merge-joints` to the URDF converter (pads/tips disappear).
- PhysX ignores URDF mimic joints. Coupling is done in `ApexCoupledEMAAction`.

## One-day command sequence

```bash
source env.sh
pip install -e . --no-deps

# Gate 1
python IsaacLab/scripts/environments/list_envs.py | grep -iE "cartpole|allegro|repose"
python IsaacLab/scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Cartpole-v0 --headless --max_iterations 20
python IsaacLab/scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Repose-Cube-Allegro-v0 --headless --num_envs 1024 --max_iterations 20

# Assets
bash scripts/convert_apex_urdf.sh
python scripts/inspect_apex.py --headless
python scripts/generate_coin_stl.py

# Gate 3
python scripts/check_action.py --headless

# Stage A then B
python scripts/train.py --task PAN-CoinHold-Apex-v0 --headless --num_envs 512
python scripts/train.py --task PAN-CoinTransfer-Apex-v0 --headless --num_envs 512
python scripts/play.py --task PAN-CoinHold-Apex-Play-v0 --checkpoint latest --headless --video
python scripts/export_onnx.py --task PAN-CoinHold-Apex-v0 --headless
```

Gymnasium ids: `PAN-CoinHold-Apex-v0`, `PAN-CoinTransfer-Apex-v0`, `PAN-CoinHold-Apex-Left-v0`, `PAN-BaodingRotate-Apex-v0`, `PAN-CoinHold-Apex-Vision-v0`.

This repository is source only: training logs, checkpoints, ONNX exports, and mp4 replays stay on the machine (`logs/`, `*.pt`, `*.onnx`). Clone [Isaac Lab](https://github.com/isaac-sim/IsaacLab) next to the repo (or point `ISAACLAB_PATH` at your checkout); it is not vendored here. Put the official Apex URDF checkout at `assets/apex-hand-urdf/` (also not vendored).

## Web console

```bash
source env.sh
python -m webui    # http://127.0.0.1:8090
```
