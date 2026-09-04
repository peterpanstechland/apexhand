# Apex Hand Coin Knuckle Roll — 进度交接

> 换机 / 换环境时先读这份。最后更新：2026-09-04（本机 RTX 4080 Laptop 12GB）。

## 一句话现状

- **任务定义（已纠正）**：掌心朝下，硬币平躺在**指背（knuckles）**上做 knuckle roll，不是掌心朝上指腹托币。
- **Stage A（Hold）**：完成，离线 eval success **100%**（干净 ckpt `model_100`）。
- **Stage B（Index→Middle）**：完成。离线 **`eval_transfer.py`：61.5% (315/512)**，phase@success≈0.54；drop 0.4%。目标 >50% **已过**。
- **交付**：最终 15s 视频已有；ONNX/`joint_map.json` 见 `.../rebalance_v2_fin/exported/`（若目录尚空则导出脚本刚修好，重跑即可）。
- **残留风险**：eval 时指缝 min gap 可到 ~0.4 mm、偶发 squeeze；视频里手指动作大，肉眼确认滚动质量仍建议看最终 mp4。

## 环境（新机器必做）

```bash
# Isaac Sim 已装在 ~/isaacsim-env；不要重装
source ~/Documents/apexhand/env.sh
cd ~/Documents/apexhand
# 每个新终端都要 source env.sh
```

| 项 | 值 |
| --- | --- |
| Isaac Sim | 6.0.1（`~/isaacsim-env`） |
| Isaac Lab | `v3.0.0-beta2.patch1`（仓库内 `IsaacLab/`） |
| GPU 本机 | RTX 4080 Laptop **12 GB** |
| 常用并行 | `num_envs=2048` ≈ 4.5 GB，~22–25k steps/s |

更完整的教程见 [docs/USAGE_GUIDE.zh.md](docs/USAGE_GUIDE.zh.md)。门闸数字表见 [RESULTS.md](RESULTS.md)。

## 正确任务几何（勿回退）

- 掌心朝下 quat：`(0.7071068, 0.0, -0.7071068, 0.0)`（`source/pan_dexterous_lab/assets/apex_cfg.py`）
- 指背座位：knuckle `link1`/`link2` 中点 + **世界 +Z** 偏移 `KNUCKLE_SEAT_HEIGHT=0.013`
- `enabled_self_collisions=False`（触觉壳重叠，开自碰会 PhysX 发散）
- 动作：`ApexCoupledEMAAction`，`target = default_joint_pos + action * scale`，`rescale_to_limits=False`
- 外展 `*_j0` scale 钳到 **0.04** + `finger_crossing` 惩罚 **-20**（防穿指 exploit）

参考动作：[Instructables Coin knuckle roll](https://www.instructables.com/Coin-knuckle-roll/)

## 有效 Checkpoint（请只用这些）

### Stage A — Coin Hold

| 路径 | 用途 |
| --- | --- |
| `logs/rsl_rl/pan_coin_hold/2026-09-04_04-20-40/model_100.pt` | **推荐**：Stage B 热启动；eval gap≈20.6 mm，abd 小，无 squeeze |
| `.../model_1499.pt` | 训满末帧；success 仍高，但 gap 收到 ~10 mm、abd 变大，**不如 100 干净** |

```bash
python scripts/eval_hold.py --num_envs 64 --episodes 2 \
  --checkpoint logs/rsl_rl/pan_coin_hold/2026-09-04_04-20-40/model_100.pt
```

Gate 4：Hold success > 80% 已满足（eval 100%）。

### Stage B — Index→Middle Transfer

| 路径 | 用途 |
| --- | --- |
| `logs/rsl_rl/pan_coin_transfer/2026-09-04_11-26-02_rebalance_v2_fin/model_2099.pt` | **当前最终权**（从 hold`model_100` → rebalance_v2 → cont → fin） |
| `logs/rsl_rl/pan_coin_transfer/2026-09-04_10-02-11_rebalance_v2/` | 中段 run（曾在 ~612 停去录像）；视频在此 |
| `logs/rsl_rl/pan_coin_transfer/2026-09-04_10-45-19_rebalance_v2_cont/` | 续训中间产物（到 ~1600） |

**不要用：**

| 路径 | 原因 |
| --- | --- |
| `pan_coin_hold/2026-09-04_01-29-58` | 掌心朝上指腹托币（错误任务） |
| `pan_coin_hold/2026-09-04_04-02-17` | knuckle 位姿但对，手指交叉穿透 exploit |
| `pan_coin_hold/2026-09-04_04-17-33_crossfix` | 交叉问题对照 |
| `pan_coin_transfer/2026-09-04_08-46-28` | Stage B **第一轮**（未重平衡）：刷 `roll_rotation`，`success≈0` |

## 训练命令备忘

```bash
source ~/Documents/apexhand/env.sh && cd ~/Documents/apexhand

# Stage A
python scripts/train.py --task PAN-CoinHold-Apex-v0 --headless \
  --num_envs 2048 --max_iterations 1500 --seed 42

# Stage B（从 Hold 干净 ckpt 续；绝对路径 + --resume）
python scripts/train.py --task PAN-CoinTransfer-Apex-v0 --headless \
  --num_envs 2048 --max_iterations 2000 --seed 42 --resume \
  --checkpoint /ABS/PATH/to/pan_coin_hold/.../model_100.pt

# Play 录像（5s≈300 step，15s≈900 step；PLAY 任务 time_out=None）
python scripts/play.py --task PAN-CoinTransfer-Apex-Play-v0 --num_envs 1 \
  --checkpoint /ABS/PATH/to/model_2099.pt --video --video_length 900 --headless
```

Gym id：

- `PAN-CoinHold-Apex-v0` / `PAN-CoinHold-Apex-Play-v0`
- `PAN-CoinTransfer-Apex-v0` / `PAN-CoinTransfer-Apex-Play-v0`

## 已有视频

| 文件 | 说明 |
| --- | --- |
| `logs/rsl_rl/pan_coin_transfer/2026-09-04_10-02-11_rebalance_v2/videos/play/rl-video-15s.mp4` | model_600，15s，用户看过觉得效果可以但偏短→已加长 |
| `.../rl-video-step-0.mp4`（同目录） | 同内容 / 较短版 |
| `logs/rsl_rl/pan_coin_transfer/2026-09-04_11-26-02_rebalance_v2_fin/videos/play/rl-video-15s-final.mp4` | **最终权 `model_2099`，15s @ 1280×720** |

## Stage B 奖励教训（换机后勿改回旧权重）

第一轮失败原因：`hold_ok≈0.95` + 未封顶的 `roll_rotation≈6` 压死 `progress`，策略原地转币。

当前 `TransferRewardsCfg`（`coin_roll_env_cfg.py` + `mdp/rewards.py`）：

- `hold_ok` 权重 **0**
- `progress`：前向 Δφ + 朝 target phase≈0.55 的稠密项，权重 **8**
- `roll_rotation`：ω 封顶且需横向前进才计分，权重 **0.5**
- `slip` **-2**，`success` **50**
- 座位距离用 `coin_bridge_distance`（index-middle / middle-ring 两桥取 min）

末段日志（fin，~2099）：reward≈39–41，`Episode_Reward/success≈0.16`（≈ 50/300，多数满长 episode 拿一次 bonus），drop≈2–3%。

## 关键修复清单（踩过的坑）

1. Hold success 曾恒为 0 → 改为「撑满 ≥150 step + 末 30 稳定」；评估用 `scripts/eval_hold.py`
2. action=0 曾映射到关节限位中点 → 改为相对 `default_joint_pos`
3. 手指交叉 + self-collision 关 → abduction 钳制 + `finger_crossing`
4. 硬币 depene / 出生穿模 → `max_depenetration_velocity=1.0`，`reset_coin_on_knuckles`，关节 jitter 0.05
5. 座位偏移用 body-local 会偏到指缝 → 改世界 +Z
6. 续训必须用**绝对** `--checkpoint` + `--resume`；rsl_rl 从 ckpt 的 iteration 再加 `max_iterations`
7. 12GB 上不要并行两个 `train.py`（曾误开双进程）
8. `DISPLAY` 常空 → 用 `--video` headless，别依赖本机 Kit 窗

## 未完成 / 可选后续

- [x] 最终 `model_2099` 15s 视频
- [x] 离线 transfer success **61.5%**（>50%）— `scripts/eval_transfer.py`
- [x] ONNX + `joint_map.json` → `logs/rsl_rl/pan_coin_transfer/2026-09-04_11-26-02_rebalance_v2_fin/exported/`
- [ ] 人工再看最终视频确认滚动观感（手指动作偏大、指缝偶发收紧）
- [ ] Hold 也可另导一份 ONNX（若交付要双策略）
- [ ] `real/` 仅有骨架，硬件未接

```bash
python scripts/eval_transfer.py --num_envs 128 --episodes 4 --headless \
  --checkpoint logs/rsl_rl/pan_coin_transfer/2026-09-04_11-26-02_rebalance_v2_fin/model_2099.pt
python scripts/export_onnx.py --task PAN-CoinTransfer-Apex-v0 --headless \
  --checkpoint logs/rsl_rl/pan_coin_transfer/2026-09-04_11-26-02_rebalance_v2_fin/model_2099.pt
```

## 关键代码入口

| 模块 | 路径 |
| --- | --- |
| 手/币资产 | `source/pan_dexterous_lab/assets/apex_cfg.py`, `joints.py` |
| 环境与奖励权重 | `source/pan_dexterous_lab/tasks/coin_roll/coin_roll_env_cfg.py` |
| MDP | `source/pan_dexterous_lab/tasks/coin_roll/mdp/` |
| PPO cfg | `.../config/apex_hand/agents/rsl_rl_ppo_cfg.py` |
| 训练/播放 | `scripts/train.py`, `scripts/play.py` |
| Hold 评估 | `scripts/eval_hold.py` |

## 对话上下文（可选）

本机 Cursor agent 长对话：`~/.cursor/projects/home-peter-laptop-Documents-apexhand/agent-transcripts/91fdf184-5fbc-43ac-8fa9-9e99c9d99fcd/`
