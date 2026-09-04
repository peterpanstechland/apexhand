# Results log — Apex Hand Coin Knuckle Roll

Host: RTX 4080 Laptop **12 GB**, Ubuntu 22.04, Isaac Sim **6.0.1**, Isaac Lab **v3.0.0-beta2.patch1**.

**换机续做请先读 [PROGRESS.md](PROGRESS.md)**（任务定义、有效 ckpt、坑、下一步）。

## Gate 1 — Isaac Lab smoke

| Check | Result | Notes |
| --- | --- | --- |
| Cartpole 20 iter | PASS | 早期冒烟 |
| Allegro Repose 1024 env | PASS | 定标基线；本机 Hold@2048 ≈ **4.5 GB / ~22–25k steps/s** |

## Gate 2 — Apex USD

| Side | joints | pads | tips | Result |
| --- | --- | --- | --- | --- |
| right | 21 actuated (+coupled) | ok | ok | PASS（勿 `--merge-joints`） |
| left | 同结构 | ok | ok | 资产可用；主策略用右手 |

## Gate 3 — Coupled actions

| Case | Result |
| --- | --- |
| 0 / +1 / -1 | PASS（`scripts/check_action.py`；action 原点=default pose） |

## Gate 4 — Stage A Coin Hold（knuckle，掌心朝下）

| Run | num_envs | iters | VRAM | steps/s | eval success | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| `pan_coin_hold/2026-09-04_04-20-40` | 2048 | 1500 | ~4.5 GB | ~25k | **100%** @`model_100`/`1499` | 有效 run；推荐热启动 **`model_100`** |
| `.../01-29-58` | 2048 | 1500 | — | — | — | **作废**：掌心朝上指腹 |
| `.../04-02-17` | 2048 | — | — | — | 高但作弊 | **作废**：穿指 |

Hold 判据：撑满 ≥150 control steps + 末 30 稳定；评估 `scripts/eval_hold.py`。

## Stage B — Index→Middle transfer

| Run | 说明 | 结局 |
| --- | --- | --- |
| `pan_coin_transfer/2026-09-04_08-46-28` | 旧奖励 | **失败**：刷 roll，`success≈0` |
| `.../10-02-11_rebalance_v2` | 重平衡后从 hold`model_100` | 中断于 ~612（录像）；有 15s 视频 |
| `.../10-45-19_rebalance_v2_cont` | 从 `model_600` 续 | 到 ~1600+ |
| `.../11-26-02_rebalance_v2_fin` | 从 `model_1600` 再续 | **完成 `model_2099.pt`** |

末段（~2099）：mean reward ≈ 39–41，`Episode_Reward/success` ≈ 0.16，drop ≈ 2–3%。

**离线评估（`scripts/eval_transfer.py`，128 env × 4 ep = 512）：**

| Metric | Value |
| --- | --- |
| transfer_success | **61.5%** (315/512) — Gate >50% **PASS** |
| phase_at_success | mean 0.544 (p10 0.521 / p90 0.572) |
| drop | 0.4% (2/512) |
| finger min gap | 0.0004 m（有挤压风险；squeezed steps 2669/153600） |
| max abduction | 0.44 rad |

## Deliverables

| Item | Status |
| --- | --- |
| Hold play / eval | Gate 4 PASS（100% @`model_100`） |
| Transfer play 15s（`model_600`） | `.../rebalance_v2/videos/play/rl-video-15s.mp4` |
| Transfer play（`model_2099`） | `.../rebalance_v2_fin/videos/play/rl-video-15s-final.mp4` |
| Transfer eval | **61.5%** PASS |
| ONNX + `joint_map.json` | `.../rebalance_v2_fin/exported/` |
| `real/` 骨架 | stubs only |
| [PROGRESS.md](PROGRESS.md) | 换机交接 |

## Known issues

- Isaac Lab 3.0 beta：`packaging` pin 冲突时保留 isaacsim 侧 26.0。
- URDF 动力学未标定；**不能假设**策略直接上真机。
- `enabled_self_collisions` 必须关；靠 abduction clamp + `finger_crossing`。
- Stage B 第一轮奖励可被原地旋转 hack；已改，勿回退。
- Hold 后期（iter>500）action std 上升、指缝收紧 → 热启动用早中期 ckpt。
- 12GB 勿并行两个 Isaac/train。
- AppLauncher 须在 Hydra env import 之前（现有 `train.py`/`play.py` 已按此）。
