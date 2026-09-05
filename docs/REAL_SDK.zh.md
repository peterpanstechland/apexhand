# 真机 SDK 联调（本机）

> 与 Isaac 训练环境**完全分离**。最后更新：2026-09-05。

## 已在本机做好的

| 项 | 路径 / 状态 |
| --- | --- |
| 官方 SDK 源码 | `third_party/Rysen_SDK/`（GitHub `RysenRobotics/Rysen_SDK`） |
| Python 3.10 venv | `.venv-apex-real/` |
| 激活脚本 | `source env_real.sh`（**不要** `source env.sh`） |
| 空载冒烟 | `python scripts/real_sdk_smoke.py`（IP 默认取 `APEX_HAND_IP`） |
| 系统库 | spdlog / fmt / boost 已在系统里，**import 已通过** |
| 实机 | **已连通**：左手 `192.168.88.200`，手固件 3.2.5，SDK 1.5.2，控制 100 Hz |

官方文档：[Apex Hand 入门](https://docs.rysenbot.com/apex-hand/get-started) · 仓库说明：`third_party/Rysen_SDK/README_CN.md`

## IP 与网段

本机这只手已改到本地网段，**开箱默认值不是这个**：

| | 地址 | TCP |
| --- | --- | --- |
| **本机实机（左手）** | `192.168.88.200` | **5856 / 5857** |
| 出厂默认 · 右手 | `192.168.0.103` | 同上 |
| 出厂默认 · 左手 | `192.168.0.102` | 同上 |

`env_real.sh` 会把 `APEX_HAND_IP` 设成 `192.168.88.200`，所有脚本默认读它，`--ip` 可覆盖。

```bash
ping -c 2 "$APEX_HAND_IP"
```

若换了一只出厂状态的手，电脑有线口和手要在同一网段。直连时给网卡临时加个别名即可：

```bash
sudo ip addr add 192.168.0.50/24 dev enp109s0
ping -c 2 192.168.0.103
```

## 环境用法

```bash
cd ~/Documents/apexhand
source env_real.sh                 # Python 3.10 + rysen .so 路径

# 仅连通 + 读关节（默认不动电机）
python scripts/real_sdk_smoke.py

# 摄像头遥操作（Space 才使能电机，Esc 退出）
python scripts/landmark_teleop.py --side left

# ONNX 策略空载回放（会动！清空周围再跑，gain 从小起）
python -m real.policy_runner --gain 0.30 --seconds 12

# 官方完整示例（会动！空载、周围清障后再跑）
cd third_party/Rysen_SDK/python
python example.py --ip "$APEX_HAND_IP"
```

## 还没做 / 不要混

- `sudo ./third_party/Rysen_SDK/install_rysen_deps.sh`：本机已有运行库，**暂未跑**（sudo 要密码）。若之后 `import` 报缺库再装。
- **不要**在 `isaacsim-env` / `source env.sh` 里装 rysen-sdk。
- 冒烟脚本直接调官方 API；`real/apex_interface.py` 是给策略/遥操作用的封装（固定 `ACTUATED_LOGICAL` 顺序，耦合关节由它 1:1 复制，不单独下发）。
- 仿真策略要的观测里有相机/特权量，不是「插上就能滚币」；`real/obs_assembler.py` 会在启动时检查每一项能不能供上，供不上直接报错。

## 验收清单

- [x] 网线插好，`ping` 通手 IP
- [x] `tcp 5856/5857` 探测 OK（冒烟脚本会打）
- [x] `CONNECT_OK` + 能打印关节状态
- [x] 摄像头遥操作驱动实机（已连续跑 19 分钟）
- [x] ONNX 策略空载回放（`real.policy_runner`，60 Hz，安全层在环）
- [ ] 掌面标定 + 装球闭环（`--camera --calib`）
