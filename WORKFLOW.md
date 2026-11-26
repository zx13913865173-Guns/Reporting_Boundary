# 复现流程（7 个阶段）

流程严格按稿件顺序组织：先固定构造（任务掩码、条件矩阵、生成），
再在三个互相不替代的证据族上取数。每一阶段都可以单独运行。

---

## 阶段 1 — 任务与掩码（对应 3.1）

目标：参数化生成 8 个合成边界任务的目标掩码，并固定掩码→patch 网格映射规则。

- 输入：无（几何写死在 `masks.build_mask`）
- 输出：`outputs/target_masks/{task}.png`（255 = 区域 A）
- 参数：1024×1024 画布；正弦振幅 128 px；边框 20 px；中心圆半径 256 px
- 命令：`python run_all.py --stages masks`

任务清单（区域 A = 水墨 ink-wash，区域 B = 赛博 cyberpunk）：

| # | 任务 | 几何 | 区域 A 面积占比 |
|---|---|---|---|
| 1 | horizontal_split | 边界 y = 512，A 为 y < 512 | 50.00% |
| 2 | vertical_split | 边界 x = 512，A 为 x < 512 | 50.00% |
| 3 | diagonal_45 | 边界 y = x | 49.95% |
| 4 | diagonal_135 | 边界 y = 1024 − x | 49.95% |
| 5 | center_circle | 圆心 (512,512)、半径 256 | 19.63% |
| 6 | corner_quadrant | 左上象限 | 25.00% |
| 7 | irregular_sine | y = 512 + 128 sin(2πx/1024) | 50.05% |
| 8 | frame_border | 20 px 外边框 | 7.66% |

Frame 任务因 20 px 边框小于 32×32 分类器 patch，排除在主聚合之外。

---

## 阶段 2 — LoRA 训练（对应 3.5）

目标：每个风格训练一个 LoRA。

- 训练图：每风格 50 张，两位领域专家按 3 点量表独立评分，
  仅保留**两位都评 3 分**的图；评分一致性用 Cohen's κ 报告（κ = 0.88）
- 超参：rank 32、alpha 1.0、3000 步、学习率 1e-4
- 命令：`python run_all.py --stages lora`（依赖 kohya-ss / sd-scripts）

---

## 阶段 3 — 风格分类器（对应 3.6.1、4.12）

目标：得到 patch 级的**校准**风格 A 概率 `p_i^A`，并给出方向特定误报率。

- 骨干：ResNet-50；patch 32×32、stride 8 → 125×125 网格
- 训练集：200 张水墨 + 200 张赛博，与 LoRA 训练集相互独立
- 校准：温度缩放（`classifier.calibrate_temperature`）
- 不确定带：0.4 ≤ p < 0.6
- 方向特定 FPR：`FPR_A on B = 0.03`、`FPR_B on A = 0.05`
- 校验指标：accuracy、F1、AUC、ECE、mean uncertain、FNR、混淆矩阵
- 命令：`python run_all.py --stages classifier`

分类器准确率**仅用于**描述分类器对 `TMA-IoU_C` 的残余贡献，
不作为 `TMA-IoU_C` 是有效质量度量的证据。

---

## 阶段 4 — 区域生成与合成（对应 3.3、3.4、3.5）

目标：产出每个（任务，条件，种子）单元的**区域输出**与**合成图**。

- RLIMC 两阶段：
  1. 两次区域局部 inpainting，各自加载本区域风格的 LoRA；
     区域 A 的输出在任何跨区域合成**之前**保存（`PRSC` 的评估对象）
  2. 按掩码规格在像素空间合成
- 合成规格：硬掩码（G/D）；重叠带 alpha 混合（H: δ = 8，I: δ = 16）
- 生成参数：30 步、CFG 7.5、DPM++ 2M Karras、inpainting strength 0.80
- 规模：8 任务 × 6 条件 × 20 种子 = **960** 个单元
- 命令：`python run_all.py --stages generate`

---

## 阶段 5 — 指标（对应 3.6）

目标：在三类证据族上取数。全部指标实现见 `metrics/`。

| 指标 | 评估域 | 方向 | 不确定带处理 |
|---|---|---|---|
| `TMA-IoU_C`（对称平均） | 任务掩码 patch 网格 | 越高越好 | 主口径全部保留，另出排除变体 |
| `PRSC_A` / `PRSC_B` | erode(M_A, 32 px) / erode(M_B, 32 px) | 越高越好 | **排除** uncertain |
| `CRSC interior` | erode(M_B, 32 px) / erode(M_A, 32 px) | 越低越好 | **保留** uncertain |
| 边界带泄漏 | 距边界 < 32 px 的 patch | 越低越好 | 不并入主 CRSC |
| 掩码风格纯度 | 掩码区域（中性填充） | 越高越好 | — |
| `CMMD` | 同上，与 200 张参考图 | 越低越好 | — |
| 过渡宽度 | 平滑后概率图的 0.1→0.9 距离 | — | 平滑上界，非几何宽度 |
| base 贡献 | 边界带内 | — | 描述性，无置信区间 |
| `PB-IoU` | 分类器恢复区域 vs 共识感知划分 | 越高越好 | 非任务掩码参考 |

- 命令：`python run_all.py --stages metrics`
- `PRSC` 排除 uncertain、`CRSC` 保留 uncertain —— 这一不对称是稿件明示的局限。
- `CRSC` 值接近或低于方向特定 FPR 时，表述为
  「not distinguishable from classifier error」，不得说成零泄漏或机制性抑制。

---

## 阶段 6 — 核心负控制（对应 3.6.5）

三个构造敏感性测试，**生成区域固定不动**，只扰动其周围的构造：

| 控制 | 扰动对象 |
|---|---|
| mask_swap | 用错误的合成掩码，仍对正确任务掩码评估 |
| label_swap | 交换风格标签，任务掩码不变 |
| compositing_mask_perturbation | 边界法向 ±8/±16 px 平移；8 px 膨胀/腐蚀 |

- 设计：3 控制 × 4 个 G/D 条件 × 10 prompt = **120**；
  3 控制 × 2 个 H/I 条件 × 10 prompt = **60**；合计 **180** 个合成
- 报告量：相对该条件未扰动合成的 `TMA-IoU_C` **平均下降量**
- 命令：`python run_all.py --stages controls`
- 结论边界：控制说明 `TMA-IoU_C` 对构造扰动有响应，
  但**本身不证明**它作为质量度量无效。

---

## 阶段 7 — 人类数据与描述性分析（对应 3.6.6、3.6.7、3.7、3.8）

- 配对偏好：304 个配对，每对三个**独立**的二元问题
  （无掩码 / 总体偏好 / 强制合规）→ 912 条二元响应；比例仅描述性
- 1–5 量表：自然度、边界伪影两个**分开**的量表，仅在 10 种子子集上评；
  Mask-only 未评自然度
- `PB-IoU`：40 图 × 2 标注者 = 80 条边界标注；仅标注边界笔画，
  参考划分取两标注者边界派生区域的**交集**；标注者间 boundary F1 = 0.86、MAD = 4.1 px
- 统计：只出描述性表格与设计敏感性；**不计算** SRCC、p 值、置信区间、
  Fisher 变换区间与混合效应模型（`statistics.py` 主动抛错拒绝）
- 图：Fig. 1–Fig. 5
- 命令：`python run_all.py --stages human stats figures`

---

## 自检

`python selftest.py` 会在无 GPU、无模型权重的条件下检查 14 项：
掩码几何、patch 网格、soft IoU、`TMA-IoU_C` 方向与变体、带符号距离与重叠掩码、
`PRSC`/`CRSC`/噪声底线、过渡宽度与 base 贡献、非任务掩码参考、核心负控制、
人类数据表格、报告边界与设计敏感性、分类器校验函数、`paper_values` 一致性、
五张图生成。
