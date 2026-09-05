# 真机 SDK 空载联调（本机）

> 与 Isaac 训练环境**完全分离**。最后更新：2026-09-04。

## 已在本机做好的

| 项 | 路径 / 状态 |
| --- | --- |
| 官方 SDK 源码 | `third_party/Rysen_SDK/`（GitHub `RysenRobotics/Rysen_SDK`） |
| Python 3.10 venv | `.venv-apex-real/` |
| 激活脚本 | `source env_real.sh`（**不要** `source env.sh`） |
| 空载冒烟 | `python scripts/real_sdk_smoke.py --ip 192.168.0.103` |
| 系统库 | spdlog / fmt / boost 已在系统里，**import 已通过** |

官方文档：[Apex Hand 入门](https://docs.rysenbot.com/apex-hand/get-started) · 仓库说明：`third_party/Rysen_SDK/README_CN.md`

## 出厂 IP（必看）

| 手 | 默认 IP | TCP |
| --- | --- | --- |
| **右手** | `192.168.0.103` | **5856 / 5857** |
| **左手** | `192.168.0.102` | 同上 |

## ⚠ 本机网段冲突（现在卡在这里）

查过：有线网卡 `enp109s0` 是 **`192.168.88.102/24`**，和灵巧手默认的 **`192.168.0.x`** **不在同一网段**。  
Wi‑Fi `wlo1` 也是 `192.168.88.101`。  
**不改网络，SDK 连不上。**

### 推荐做法（空载联调）

1. 网线：**电脑有线口 ↔ 灵巧手**（直连或同交换机，别走 Wi‑Fi）。
2. 把有线 IPv4 改成手动，例如：
   - 地址：`192.168.0.50`
   - 掩码：`255.255.255.0`
   - 网关：可留空（直连不需要）
3. 应用后：

```bash
ping -c 2 192.168.0.103          # 右手
# 或
ping -c 2 192.168.0.102          # 左手
```

临时加别名（不改 GUI，重启可能丢）：

```bash
sudo ip addr add 192.168.0.50/24 dev enp109s0
ping -c 2 192.168.0.103
```

（需要你本机输入 sudo 密码。）

可选：若手支持改 IP 到 `192.168.88.x`，也可保持现有网段，改手侧 IP。

## 环境用法

```bash
cd ~/Documents/apexhand
source env_real.sh                 # Python 3.10 + rysen .so 路径

# 仅连通 + 读关节（默认不动电机）
python scripts/real_sdk_smoke.py --ip 192.168.0.103

# 官方完整示例（会动！空载、周围清障后再跑）
cd third_party/Rysen_SDK/python
python example.py --ip 192.168.0.103
```

## 还没做 / 不要混

- `sudo ./third_party/Rysen_SDK/install_rysen_deps.sh`：本机已有运行库，**暂未跑**（sudo 要密码）。若之后 `import` 报缺库再装。
- **不要**在 `isaacsim-env` / `source env.sh` 里装 rysen-sdk。
- 冒烟脚本直接调官方 API；`real/apex_interface.py` 是给策略/遥操作用的封装（固定 `ACTUATED_LOGICAL` 顺序，耦合关节由它 1:1 复制，不单独下发）。
- 空载通过后再谈 ONNX / 放币；仿真策略有特权观测，不能直接当「插上就能滚币」。

## 验收清单

- [ ] 网线插好，`ping` 通手 IP  
- [ ] `tcp 5856/5857` 探测 OK（冒烟脚本会打）  
- [ ] `CONNECT_OK` + 能打印关节状态  
- [ ] （可选）官方 `example.py` 小动作，急停可达  
