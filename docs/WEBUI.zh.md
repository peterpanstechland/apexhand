# 可视化训练控制台

本地网页，用来调参、开训、看曲线/日志/视频，以及用自然语言生成奖励函数。

## 启动

```bash
source ~/Documents/apexhand/env.sh
cd ~/Documents/apexhand
python -m webui
```

浏览器打开 [http://127.0.0.1:8090](http://127.0.0.1:8090)（本机 8080 已被 Open WebUI 占用）。

训练进程会再走一遍 `source env.sh`，所以 **Isaac 的 libstdc++ 手术不会被网页进程带偏**。网页本身不启动 Kit；只有你点「开始训练 / 预览 / 玩手 / 探针」才拉起仿真。

## 页面怎么用

1. 左边选任务、物理引擎、手部姿态、物体、相机。
2. 每个参数右边有「解释」：作用、调大/调小会发生什么、新手建议。完整百科见 [PARAMS.zh.md](PARAMS.zh.md)。
3. 点 **参数体检**。红色是硬错误（比如小批次不能整除），黄灯是「你可能没意识到」。
4. **开始训练**。右侧「运行」看日志；「曲线」读 TensorBoard 事件；「视频」在回放出 mp4 之后出现。
5. **场景预览** 只 spawn 几步，写 `logs/webui/preview/scene.png`，改完球的尺寸先看摆位。
6. **玩手** 开 1 个环境的交互仿真。拖滑条拨关节；在画面上左键拖动环绕、右键平移、滚轮缩放。硬币/球仍受重力。Kit 冷启动大约 30–90 秒。一张 12GB 卡不要和训练同时开。
7. **Reward 探针** 随机动作 100 步，列出每项均值/方差/是否死项。
8. **回放录像** 对当前配方跑 `play.py --video`。
9. Stage 串联：`configs/recipes/hold_default.yaml` 里的 `chain` 会在 Hold 正常结束后自动 `--resume` 开 Transfer。

## 物体与姿态

| 预设 | 建议姿态 | 说明 |
| --- | --- | --- |
| `pan_coin_32mm` | 掌心朝下 · 指背沟 | 现在的主任务 |
| `baoding_38/45/50mm` | 掌心朝上 · 托举 | 单球；双球请用 Baoding 任务（自带第二颗） |
| `cube_60mm` / `rod_80mm` / `egg_45mm` | 视任务而定 | 几何换了，奖励还是硬币那套的话会没有意义 |

重量按千克写在 hydra 路径 `env.scene.object.spawn.mass_props.mass`（4 g = 0.004）。界面「物体物理」分组就是这些覆盖。

## 物理引擎

- **PhysX**：默认，当前主训练。
- **Newton / MuJoCo-Warp**：同一任务换求解器。控制台会自动加 `env.scene.clone_in_fabric=false`。
- **导出 MJCF**：工具页按钮，产物 `assets/mjcf/apex_hand.xml`。
- **跨引擎一致性**：同一 checkpoint 两边各跑一段，对比成功率。

## 相机与显存

三视角挂在 `PAN-CoinHold-Apex-Vision-v0`。当前 PPO 仍是 MLP，**图像默认不进策略**，只用于预览、录制和以后换视觉编码器。RTX 4080 Laptop 是 12 GB：开相机请把并行环境降到 128。页面上的显存数字是粗估。

圆顶光已经从几乎全黑改成冷白；视觉任务还会在重置时随机强度和色温。

## 自然语言奖励

1. 在「Reward 对话」填 OpenAI 兼容的 `base_url` / `model` / `api_key`（DeepSeek、通义、Kimi、智谱、本地 vLLM 都行）。Key 存在 `webui/.llm.json`，已进 `.gitignore`。
2. 用中文描述行为，点生成。背后会把现有 `rewards.py` 签名和坐标系事实塞进 system prompt。
3. 先看 diff 和静态检查（语法、禁止的 import、必须有函数）。
4. 点「审阅通过」才写入 `source/pan_dexterous_lab/tasks/coin_roll/mdp/user_rewards.py`，并拉起 1 环境 headless 自检（shape / NaN）。
5. 默认权重是 0。自检通过后，把「用户自定义奖励权重」从 0 调到 0.5～1 再训。

## 配方文件

保存在 `configs/recipes/*.yaml`。字段：`task`、`physics`、`object`、`hand_pose`、`cameras`、`cli`、`overrides`（hydra 路径）、可选 `chain`。
