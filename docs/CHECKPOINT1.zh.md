# AIx Origin 深圳场 · Checkpoint 1 进展说明

| 项 | 内容 |
| --- | --- |
| 项目 | PΛN Dexterous Lab — Apex Hand 灵巧手在手内操作（in-hand manipulation） |
| 仓库 | https://github.com/peterpanstechland/apexhand （public，无需协作者权限） |
| 硬件 | Rysen Apex Hand（**左手**，固件 3.2.5，SDK 1.5.2，Ethernet 100 Hz） |
| 仿真 | Isaac Sim 6.0.1 + Isaac Lab v3.0.0-beta2.patch1，RTX 4080 Laptop 12 GB |
| 日期 | 2026-09-05 |

目标是让这只 16 自由度的灵巧手在**手内**完成两类精细操作：指背滚币（coin knuckle roll）和掌心保健球对转（baoding rotate），全部用 PPO 在仿真里训练，再迁到实物手上。

---

## 一、已完成的部分

### 1. 仿真训练栈（可复现）

Apex Hand 的左右手 USD 都已跑通：21 个关节，其中 5 个 DIP 是固件耦合的，不能独立下发。动作层 `ApexCoupledEMAAction` 统一处理耦合复制和 EMA 平滑，动作原点是 `default_joint_pos` 而非关节限位中点 —— 这样 action=0 对应一个物理上合理的姿态。

任务、物体、相机、奖励权重都做成了 preset + Hydra override，配方存在 `configs/recipes/`，另有一个 WebUI（`webui/`）把 200 多个参数连同「调大/调小会怎样」的说明暴露给非专业用户。

### 2. 指背滚币（Coin Knuckle Roll）—— 已达标

掌心朝下，32×4 mm 硬币平躺在食指/中指指背上滚动。分两阶段训练：

| 阶段 | 指标 | 结果 |
| --- | --- | --- |
| Stage A · Hold（托住不掉） | 离线 eval success | **100%** |
| Stage B · Index→Middle 传递 | 离线 eval success（512 episode） | **61.5%**（315/512），门闸 >50% 通过 |
| Stage B | drop rate | 0.4%（2/512） |

交付物：15 秒演示视频、ONNX 权重 + `joint_map.json`（含完整观测布局）。评估脚本 `scripts/eval_transfer.py`，数字表见 [RESULTS.md](../RESULTS.md)。

### 3. 真机 SDK 全链路打通 —— 已在实物手上跑通

不是「能连上」，是策略已经在真手上动过：

- `real/apex_interface.py`：连接、使能、读关节位置/速度/电机电流、按 `ACTUATED_LOGICAL` 固定顺序下发位置目标，耦合关节由这一层 1:1 复制。
- `real/safety.py`：关节限位 + 每 tick Δq 上限 + 过流保护（超限的手指冻结并把屈曲缓缓放开）。
- **空载策略回放已成功**：`python -m real.policy_runner --gain 0.30 --seconds 12`，ONNX 策略以 60 Hz 驱动实物手，安全层全程在环。
- **MediaPipe 遥操作已连续运行 19 分钟**：`scripts/landmark_teleop.py`，摄像头 → 21 landmarks → 16 个驱动关节 → 实物手，Space 键才使能电机。

真机环境与 Isaac 环境**完全隔离**（`env_real.sh` / Python 3.10 独立 venv），但共用同一份关节表 —— `real/joint_table.py` 按路径加载 `assets/joints.py`，避免出现第二份关节顺序。

### 4. 保健球对转（Baoding）—— 训练进行中

这是 Checkpoint 1 的主线任务：掌心朝上托住两颗球，让它们绕掌心法向互相转圈。

物体按**实物**建模：桌上那对是 30 mm 车制木球，实测 9.55 g/颗（反推密度 676 kg/m³，正好落在山毛榉/桦木区间）。仿真里的目标间距和复位位置都从这一个半径推导，重新量球不需要改奖励项。

手侧也改成了**左手** —— 实物就是左手，训左手可以省掉部署时镜像策略这一步。

当前长训（`PAN-BaodingRotate-Apex-Left-v0`，4096 并行环境，计划 3000 iteration）：

| 指标 | iteration ~1600 时 | 含义 |
| --- | --- | --- |
| Mean episode length | 600 / 600 步 | 满 10 秒，基本不掉球 |
| Termination/drop | 1.9% | 掉球率 |
| Episode_Reward/spin | 0.55，仍在上升 | 转起来了，且没到平台 |
| Episode_Reward/hold_pair | 0.99 | 稳定托住 |
| Mean reward | 13.4 | — |

吞吐约 37–43k steps/s，训完 3000 iteration 约 1 小时 50 分。曲线逐条解读见 [BAODING_TRAINING.zh.md](BAODING_TRAINING.zh.md)。

### 5. Sim→Real 观测对齐机制

这是这个 checkpoint 里最关键的一块基础设施。`scripts/export_onnx.py` 把 actor 的观测布局**逐项**写进 `joint_map.json`（每个 term 的名字和维度、驱动关节顺序、动作缩放、EMA α、控制频率、左右手）。真机侧 `real/obs_assembler.py` 按这份 spec 逐项填充，任何一项没有 provider 就在启动时报错。

配套的 `real/ball_tracker.py` 用一颗固定相机产出与仿真 `baoding_pair_obs` **完全同序同单位**的六个数；`scripts/calibrate_palm.py` 点四个掌指关节做一次性相似变换标定（合成数据回归残差 < 0.01 mm）。

---

## 二、剩余待完成的部分及计划

| # | 待完成 | 计划 |
| --- | --- | --- |
| 1 | 保健球训练跑满 3000 iteration | 已在跑，约 1 小时内完成 |
| 2 | 保健球离线评估 | `scripts/eval_baoding.py` 已写好，指标是**每 episode 转多少圈**（而不是只看奖励），同时统计掉球、飞出掌心、穿指三种作弊 |
| 3 | 导出保健球 ONNX | `scripts/export_onnx.py --task PAN-BaodingRotate-Apex-Left-v0`，会一并写出观测 spec |
| 4 | **真机装球闭环** | 目前真机只做过空载。下一步：固定相机 → `calibrate_palm.py` 标定 → `policy_runner --camera --calib` 带球跑，从 `--gain 0.3` 起步逐步放开 |
| 5 | Sim2Real 增益整定 | URDF 的动力学参数没标定过，预计需要在真机上调 EMA α、Δq 上限、力矩上限；若差距大则回仿真加域随机化 |
| 6 | 演示视频 | 仿真 play 视频 + 真机视频各一段 |

**风险与备选**：若带球闭环在 checkpoint 2 之前无法稳定，回退方案是用已经跑通的 MediaPipe 遥操作演示手内操作能力，同时给出仿真里的圈数指标 —— 两条链路是独立的，不会一起失败。

---

## 三、遇到的技术难点

### 1. 观测静默错位（最严重的一个）

第一次上真机时，观测向量是在真机脚本里手写的 88 维布局，尾部还拿 `np.zeros(15)` 补齐。策略换了之后观测形状变了，**没有任何报错**，向量整体错位，手只是轻微抽动。这类 bug 不会崩溃，只会让人误以为是「sim2real gap 太大」而去调错方向。

解法是把观测布局变成训练侧的导出产物，真机侧按 spec 重建，缺项直接启动失败。现在观测不匹配是一个开机错误，不是一次失败的实验。

### 2. 左右手不是字符串替换

原来把 env 改成左手的做法是把关节名里的 `right_` 替换成 `left_`。但 MDP 的奖励/观测项还要用 `side` 参数去解析 **body** 名（掌心、指背、指腹、指尖），这些项默认 `side="right"`，于是左手 USD 被问「right_palm_link 在哪」，报错点离原因很远。现在 `hand_side.apply_hand_side()` 把 USD、关节列表、注入给各 term 的 `side` 三者一起搬，导出时也把解析出的 side 记进 `joint_map.json`，部署侧不可能和训练侧不一致。

### 3. 固件的两个坑

- 关节上限写 100°，但 `deg2rad(100)` 会**超过**固件内部的 1.7453 rad，直接触发越界报错。限位统一留 1 mrad 余量。
- 电机电流上报会卡在 449–453 mA 的假满量程值。不过滤的话，过流保护会认为手指永远处于碰撞状态。

### 4. 过流保护和主动捏取无法区分

安全层靠电流突增判断碰撞，然后把该手指放开。但拇指-食指主动捏东西时的电流特征和撞到障碍物一模一样。解法是让上层把「当前意图捏取的程度」前馈给安全层，捏取时豁免这两根手指 —— 纯靠电流本身是分不出来的。

### 5. 自碰撞必须关闭，于是要防穿指

Apex Hand 的触觉外壳在 URDF 里互相重叠，打开自碰撞 PhysX 会直接发散。关掉之后策略立刻学会**把手指互相穿过去**来完成任务。对策是把外展关节 `*_j0` 的动作缩放钳到 0.04，再加一项 `finger_crossing` 惩罚（权重 -20）。

### 6. 两颗球长得一模一样

同一批木球，相机分不出谁是谁。所以观测里不能有「球 1 / 球 2」的身份 —— 一旦有，策略会学到一个真机上不存在的信息。改成只报**这一对**的中点、间距、以及**倍角**后的轴向（倍角对两个 blob 的检出顺序不变）。仿真侧和真机侧用的是同一个定义。

### 7. 单目相机测不出深度

球离掌心多高这一维，一颗固定相机恢复不了。当前策略靠本体感受和平面内的对位来补，但这是保健球任务上真机最大的不确定项。后续可能加第二个视角或用手上的触觉阵列。

### 8. 奖励被 hack

指背传递第一轮训练里，`hold_ok≈0.95` 加上没有封顶的 `roll_rotation` 把 `progress` 项压死了，策略学会**原地转币**刷分，传递成功率≈0。重平衡后（`hold_ok` 权重归零、`progress` 提到 8、`roll_rotation` 封顶且要求横向前进才计分）才拿到 61.5%。

### 9. 12 GB 显存

单卡 4080 Laptop 只有 12 GB，2048 环境约占 4.5 GB，不能同时跑两个训练进程。保健球任务开到 4096 环境是这台机器的实际上限。

---

## 四、代码入口速查

| 模块 | 路径 |
| --- | --- |
| 手 / 物体资产 | `source/pan_dexterous_lab/assets/` |
| 保健球任务 | `source/pan_dexterous_lab/tasks/coin_roll/baoding_env_cfg.py` |
| 滚币任务 | `source/pan_dexterous_lab/tasks/coin_roll/coin_roll_env_cfg.py` |
| 奖励 / 观测 / 事件 | `source/pan_dexterous_lab/tasks/coin_roll/mdp/` |
| 左右手改造 | `source/pan_dexterous_lab/tasks/coin_roll/hand_side.py` |
| 真机 SDK 封装 | `real/apex_interface.py`, `real/safety.py` |
| 真机策略回放 | `real/policy_runner.py`, `real/obs_assembler.py` |
| 相机球追踪 | `real/ball_tracker.py`, `scripts/calibrate_palm.py` |
| 遥操作 | `tracking/`, `scripts/landmark_teleop.py` |
| 训练 / 播放 / 评估 | `scripts/train.py`, `play.py`, `eval_baoding.py`, `eval_transfer.py` |
| WebUI | `webui/` |

完整文档：[README](../README.md) · [使用教程](USAGE_GUIDE.zh.md) · [真机 SDK](REAL_SDK.zh.md) · [训练曲线解读](BAODING_TRAINING.zh.md) · [进度交接](../PROGRESS.md) · [数字表](../RESULTS.md)
