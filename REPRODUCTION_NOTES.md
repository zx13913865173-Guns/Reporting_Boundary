# 与稿件的差异核对（上传前必读）

本文件回答一个问题：**这套代码相对当前稿件，哪些地方是按稿件改的、哪些地方是稿件没说、由本实现定的。**

- 第 1 节：以当前稿件为准修正的条目（旧仓库与稿件不一致的地方）。
- 第 2 节：**稿件未指定**、由本实现给出默认值的条目 —— 需作者确认后再对外发布。
- 第 3 节：稿件报告数值与实现定义的逐项对照。
- 第 4 节：验证方式与结果。

---

## 1. 按当前稿件修正的条目（相对早期仓库）

| # | 条目 | 早期仓库 | 当前稿件 | 代码位置 |
|---|---|---|---|---|
| 1 | 正弦边界振幅 | 256 px | **128 px** | `config.SINE_AMPLITUDE` |
| 2 | 边框任务厚度 | 81 px | **20 px** | `config.FRAME_THICKNESS` |
| 3 | horizontal / vertical 两个任务 | 切分方向与实现互相对调 | horizontal 用 y = 512；vertical 用 x = 512 | `masks.build_mask` |
| 4 | `TMA-IoU_C` 的 A 方向 | 取 `P(cyberpunk) ≥ 0.5` 去和水墨（255）掩码求 IoU | 取 `p_i^A` 对风格 A 的掩码 `m_i^A` 求 IoU | `classifier.predict_prob_map_A`、`metrics/tma_iou.py` |
| 5 | Frame 任务是否进主聚合 | 与其他任务一起聚合 | **排除**，仅作敏感性检查 | `config.FRAME_EXCLUDED_FROM_MAIN` |
| 6 | 统计口径 | 混合效应模型、ANOVA、FDR 家族 | **不报** SRCC、p 值、置信区间、混合效应 | `statistics.py` 的 guard 函数 |
| 7 | 依赖清单位置 | 写在名为 `Contact` 的文件里 | 仓库根目录 `requirements.txt` | `requirements.txt` |

第 4 条影响最大：方向反了会整体改变所有 IoU 结论，程序照样跑完、产出看着正常但数值全错。

---

## 2. 稿件未指定、本实现给出默认值的条目（**需作者确认**）

| # | 条目 | 稿件原文口径 | 本实现采用的默认值 | 代码位置 |
|---|---|---|---|---|
| 1 | 掩码→patch 网格映射 | 稿件明说未指定用「patch 中心采样」还是「patch 面积覆盖」，规则应取自发布实现 | 主口径 `patch_center`，另出 `patch_area` 敏感性变体 | `config.MASK_TO_GRID_PRIMARY`、`masks.mask_to_grid` |
| 2 | G 与 D 的构造差异 | 只说 G/D 同属硬掩码子组，D1.0 属「D dual-LoRA 子组」，未说明 G 与 D 的 pass 差别 | G = 单次生成不切换 LoRA；D = 每区域各一次、各带本区域 LoRA | `generation.generate_unit`、`config.CONDITION_SETUP` |
| 3 | 条件 C 的定义 | 与 A、B 并列为「仅描述性参考」，但只定义了 A 与 B | 暴露为可插拔工厂 `condition_C`，**未实现**，调用即抛错 | `generation.condition_C` |
| 4 | base contribution 的精确定义 | 只给数值（G1.0 = 0.000，H = 0.009，I = 0.010） | 边界带内「合成图与 base 图在每通道容差 2 灰度级内相等」的像素比例 | `metrics/mask_defined.base_contribution` |
| 5 | `CMMD` 的 RBF 带宽 | 只说用高斯 RBF 核 | 中位数启发式（合并样本的成对距离中位数） | `metrics/common.mmd_rbf` |
| 6 | 概率校准方法 | 只说「calibrated probabilities」 | 单参数温度缩放，在留出集上最小化 NLL | `classifier.calibrate_temperature` |
| 7 | CLIP 骨干 | 只说用 CLIP embeddings | ViT-B/32 | `metrics/common.load_clip` |
| 8 | 过渡宽度的平滑核 | 只说「segmenter-smoothed upper bound」 | 高斯平滑 σ = 3 px 后测 0.1→0.9 距离 | `metrics/mask_defined.transition_width_median` |
| 9 | 负控制中扰动的聚合方式 | 只说报告平均下降量 | `compositing_mask_perturbation` 取 ±8/±16 px 平移与 8 px 膨胀/腐蚀共 6 个扰动的**均值** | `negative_controls.run_compositing_mask_perturbation` |
| 10 | 人类 1–5 量表的具体种子子集 | 只说「10-seed subset」 | 取种子列表前 10 个（42、100–108） | `config.HUMAN_RATING_SEEDS` |
| 11 | 配对比较的左右呈现顺序 | 未说明 | 每个配对随机化左右位置，随机种子 42 | `human_study.build_paired_schedule` |
| 12 | 智能体间的 G/D 单次/两次生成是否有中间产物 | 未说明 | 两次 pass 的区域输出都保存（`regionA.png`、`regionB.png`），供 `PRSC` 使用 | `generation.generate_unit` |

---

## 3. 稿件报告数值与实现定义的对照

`paper_values.py` 是稿件数值的唯一存放处，`figures.py` 默认从它取数，
因此无 GPU 也能重建全部五张图。

| 指标 | 稿件值（六条件） | 实现定义 |
|---|---|---|
| `TMA-IoU_C` | 0.69 / 0.68 / 0.67 / 0.72 / 0.71 / 0.70 | `metrics.tma_iou.tma_iou_c`（对称平均） |
| `PRSC_A` | 0.84 / 0.80 / 0.72 / 0.65 / 0.86 / 0.86 | `metrics.mask_defined.prsc` |
| `CRSC_A interior` | 0.06 / 0.24 / 0.03 / 0.18 / 0.04 / 0.05 | `metrics.mask_defined.crsc_interior` |
| `CMMD_A` | 0.12 / 0.14 / 0.11 / 0.18 / 0.13 / 0.10 | `metrics.mask_defined.masked_cmmd` |
| 掩码风格纯度 A | 0.80 / 0.82 / 0.75 / 0.84 / 0.86 / 0.85 | `metrics.mask_defined.masked_purity` |
| `PB-IoU` | 0.62 / 0.60 / 0.59 / 0.65 / 0.67 / 0.66 | `metrics.non_task_mask.pb_iou` |

负控制（平均 `TMA-IoU_C` 下降量，稿件 Table 1）：

| 控制 | G0.5 | D0.5 | G1.0 | D1.0 | H | I |
|---|---|---|---|---|---|---|
| mask_swap | 0.31 | 0.30 | 0.29 | 0.33 | 0.28 | 0.27 |
| label_swap | 0.27 | 0.26 | 0.24 | 0.29 | 0.23 | 0.22 |
| compositing_mask_perturbation | 0.12 | 0.11 | 0.10 | 0.14 | 0.09 | 0.08 |

其余数值（过渡宽度、base 贡献、泛化剖面、分类器 FPR、设计敏感性）见
`paper_values.py`。

---

## 4. 验证方式与结果

`python selftest.py` 在无 GPU、无模型权重条件下检查 14 项，全部通过：

```
PASS  mask geometry (manuscript Section 3.1)
PASS  patch grid and mask->grid mapping
PASS  soft probability IoU
PASS  TMA-IoU_C directions and variants
PASS  signed distance and overlap masks (H/I)
PASS  PRSC / CRSC interior / noise floor
PASS  transition width and base contribution
PASS  non-task-mask references
PASS  core negative controls
PASS  human-study schedules
PASS  reporting boundary and design sensitivity
PASS  classifier validation helpers
PASS  paper_values internal consistency
PASS  figures 1-5 (low dpi)

passed 14   failed 0
```

其中由自检直接确认的稿件约定：

- horizontal_split 必须是 y = 512、vertical_split 必须是 x = 512；
- irregular_sine 振幅必须是 128 px（在 x = 256 处边界落在 y = 640）；
- frame_border 厚度必须是 20 px；
- center_circle 半径必须是 256 px；
- patch 网格必须是 125 × 125；
- 全不确定网格在 uncertain-excluded 变体下必须返回空样本；
- 掩码交换与标签交换必须显著降低 `TMA-IoU_C`，且合成掩码微扰的降幅必须小于它；
- 设计敏感性在 p₁ = 0.65 时 power > 0.99、在 p₁ = 0.60 时约 0.97；
- `statistics.py` 必须拒绝计算 SRCC / p 值 / 置信区间 / 混合效应。

`python run_all.py --dry-run --stages masks metrics human controls stats`
可进一步验证阶段连通性（区域 A 面积、六条件计数、描述性表格均与稿件一致）。

---

## 5. 上传仓库前的待办

1. 确认第 2 节 12 条默认值，尤其是**条目 1（网格映射规则）**与**条目 2（G/D 构造差异）**——
   这两条会直接改变 `TMA-IoU_C` 与 `PRSC` 的绝对数值。
2. 在仓库首页补写版本说明，并列出 `DATA_AVAILABILITY.md` 第 2 节的三处数值差异。
3. 按稿件的数据可用性口径，把「生成图与标注需向通讯作者索取」写进仓库说明，
   不要把未分发的数据放进 Git 历史。
