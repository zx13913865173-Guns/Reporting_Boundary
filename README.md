# 复现材料：面向掩码指定区域风格生成的报告边界分析

本仓库是下述稿件的复现材料（代码 + 流程）：

> **Construction-Dependent Agreement Is Not, by Itself, Independent Quality
> Evidence: A Reporting-Boundary Analysis for Mask-Specified Regional Style
> Generation**

本仓库对应的是**当前「报告边界」版本**的稿件，而不是早期「方法导向」草稿。
两者的差异集中在三处条件级数值上（见 `DATA_AVAILABILITY.md`）。

---

## 1. 这篇稿件到底在算什么

稿件的核心主张是：对于硬掩码 G/D 条件，`TMA-IoU_C` 是一个**构造依赖的诊断量**，
其定义与合成构造共享同一个任务掩码，因此它的取值不能独立于该构造来解释。
为此，稿件报告三类互相不替代的证据：

| 证据族 | 内容 | 是否以任务掩码为评估域 |
|---|---|---|
| (i) 构造一致性诊断 | `TMA-IoU_C`（对称平均 + 分方向） | 是 |
| (ii) 掩码定义的描述性指标 | `PRSC`、`CRSC interior`、边界带泄漏、掩码风格纯度、`CMMD`、过渡宽度、base 贡献 | 是 |
| (iii) 非任务掩码参考 | 无掩码人类偏好（304 配对）、`PB-IoU`（40 图 × 2 标注者） | 否 |

`TMA-IoU_C` 只用于同一流水线内的描述性诊断与任务版式合规检查，
**不用于**把构造定义的条件与非构造条件相互排序。

---

## 2. 目录结构

```
Reporting_Boundary_Reproduction/
├── README.md                      # 本文件
├── requirements.txt               # 依赖清单
├── DATA_AVAILABILITY.md           # 数据可用性声明 + 三处数值差异
├── config.py                      # 全局配置：任务、条件矩阵、超参、指标参数
├── paper_values.py                # 稿件报告的全部数值（供离线画图/核对）
├── masks.py                       # 8 个任务掩码、带符号距离、掩码→patch 网格映射
├── lora_train.py                  # 风格 LoRA 训练 + 训练图双专家筛选
├── classifier.py                  # 32×32 patch 风格分类器、温度校准、方向特定 FPR
├── generation.py                  # RLIMC 两阶段生成 + 六条件矩阵 + H/I 重叠合成
├── negative_controls.py           # 三类核心负控制（180 个合成）
├── human_study.py                 # 配对偏好、1–5 量表、PB-IoU 标注
├── statistics.py                  # 描述性表格 + 设计敏感性；显式拒绝越界统计
├── figures.py                     # Fig. 1–Fig. 5
├── run_all.py                     # 端到端驱动（9 个阶段）
├── selftest.py                    # 无 GPU 冒烟自检（14 项）
├── metrics/
│   ├── __init__.py
│   ├── common.py                  # patch 网格、soft IoU、CLIP 嵌入、CMMD 核
│   ├── tma_iou.py                 # TMA-IoU_C：分方向、对称平均、不确定带变体
│   ├── mask_defined.py            # PRSC、CRSC interior、边界带泄漏、纯度、CMMD…
│   └── non_task_mask.py           # PB-IoU、共识划分、偏好比例
├── docs/
│   ├── WORKFLOW.md                # 7 阶段复现流程
│   ├── CONDITION_MATRIX.md        # 六条件 × 八任务 × 20 种子设计表
│   └── REPRODUCTION_NOTES.md      # 与稿件的逐条差异核对（必读）
├── data/                          # 训练图、生成图、标注（未随仓库分发）
└── outputs/                       # 掩码、生成结果、指标 CSV、图
```

---

## 3. 快速开始

```bash
# 0) 依赖
pip install -r requirements.txt

# 1) 无 GPU 的自检：几何约定、指标链路、负控制、统计边界、五张图
python selftest.py

# 2) 分阶段运行（可任选阶段）
python run_all.py --stages masks
python run_all.py --stages lora
python run_all.py --stages classifier
python run_all.py --stages generate
python run_all.py --stages metrics controls
python run_all.py --stages human stats figures

# 3) 无 GPU / 无权重时验证管线连通性
python run_all.py --dry-run --stages masks metrics figures --dpi 150
```

阶段说明见 `docs/WORKFLOW.md`。

---

## 4. 关键约定（改代码前必看）

- **方向约定**：`classifier.predict_prob_map_A` 返回的是**风格 A（ink-wash）**的
  校准概率 `p_i^A`；`TMA-IoU_C` 的 A 方向必须用 `p_i^A` 对风格 A 的任务掩码
  `m_i^A` 求交并。旧版本取 `P(cyberpunk) ≥ 0.5` 去和水墨（255）掩码求 IoU，
  方向相反，会整体改变 IoU 结论。
- **掩码→网格规则**：`config.MASK_TO_GRID_PRIMARY` 默认 `patch_center`，
  另一规则 `patch_area` 作为敏感性变体。
- **不确定带**：`0.4 ≤ p < 0.6` 为 uncertain。`PRSC` 排除 uncertain patch，
  `CRSC interior` 保留，二者的不对称是稿件明示的局限。
- **Frame 任务**：20 px 边框小于 32×32 patch，按稿件排除在主聚合之外，只作敏感性检查。
- **报告边界**：`statistics.py` 会**主动抛错**拒绝计算 SRCC、p 值、置信区间与
  混合效应模型，而不是静默产出。

---

## 5. 与稿件的差异

见 `docs/REPRODUCTION_NOTES.md`。该文件逐条列出「按稿件修正」与
「稿件未指定、本实现给出默认值」两类条目，其中后者需要作者确认。

---

## 6. 硬件与运行环境

- 硬件：单卡 NVIDIA RTX 4070 Super（12 GB）
- 软件：CUDA 12.1、PyTorch 2.1、diffusers 0.25
- LoRA 训练：kohya-ss / sd-scripts（需另行克隆，见 `lora_train.py`）
- 生成参数：1024×1024、30 步、CFG 7.5、DPM++ 2M Karras、inpainting strength 0.80

---

## 7. 联系

见 `DATA_AVAILABILITY.md`。
