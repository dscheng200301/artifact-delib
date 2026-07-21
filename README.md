# ArtifactDelib

**Candidate-Disagreement-Driven Dynamic Multi-Expert Deliberation for Ancient Artifact Identification**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.12+](https://img.shields.io/badge/Python-3.12+-blue.svg)](pyproject.toml)

基于候选分歧驱动动态多专家协作的古代文物细粒度识别系统。

---

## 一、项目核心流程

主方法 `ArtifactDelib-Rule` 由以下组件构成：

```
Input Artifact Image
→ VisualPerception
→ Five Specialized Experts
→ ReportSummarizer
→ Top-K CandidateGenerator
→ DisagreementAnalyzer
→ Dynamic Router
    ├── FAST
    ├── Targeted Expert Recheck
    └── Controlled Deliberation
→ Deferred ArtifactJudge
→ Final Identification
```

### 五个专家

1. **Shape Expert** — 器形分析
2. **Style Expert** — 纹饰与艺术风格分析
3. **Glyph Expert** — 铭文款识分析
4. **Material & Craft Expert** — 材质与工艺分析
5. **Local Detail Expert** — 局部细节分析

### 测试阶段约束

测试阶段输入只能是图片。禁止向模型提供：测试标签、文物标题、馆藏说明、博物馆描述、年代元数据、文件名中的类别、数据集说明文本、对应对象 ID、任何能够直接泄露答案的信息。真实标签只能用于离线评价。

---

## 二、实验体系

项目实验体系分为两个部分：

1. **外部基线实验（External Baselines）** — 与已有方法的对比
2. **核心消融实验（Core Ablations）** — ArtifactDelib 自身模块的消融分析

### 外部基线

| ID | Method | Description |
|----|--------|-------------|
| B1 | CLIP Zero-Shot | CLIP 零样本图像分类 |
| B2 | DINOv2 k-NN | 冻结 DINOv2 特征的 k-NN 检索 |
| B3 | BLIP-2 Zero-Shot | BLIP-2 零样本图文识别 |
| B4 | Direct Single-VLM | 单次 VLM 调用 |
| B5 | Self-Consistency VLM | N=5 次独立 VLM 采样 + 多数投票 |
| B6 | Multi-Agent Debate | 4 智能体 2 轮自由辩论 |

加上本文方法：

| O1 | **ArtifactDelib-Rule** | 完整流水线 + RuleRouter |
| O2 | ArtifactDelib-MLP | 完整流水线 + MLPRouter（仅当 MLP 可运行时） |

### 核心消融

| ID | Variant | What's removed / changed |
|----|---------|--------------------------|
| A1 | w/o Expert Specialization | 五个专家使用相同通用 Prompt |
| A2 | w/o Disagreement Analysis | 仅使用置信度 margin 路由 |
| A3 | w/o Dynamic Routing | 所有样本执行固定完整路径 |
| A4 | Random Recheck | 随机专家取代定向重审（3 种子） |
| A5 | w/o Controlled Deliberation | 困难样本跳过协商 |
| A6 | Free Debate | 自由讨论取代受控假设协商 |
| A7 | w/o Critic | 仅 Hypothesis A+B，无 CriticAgent |
| A8 | Early Judge | Judge 在重审前即作出裁决 |

### 暂缓基线

```text
AutoGen and CAMEL are deferred to a future experimental phase.
Deferred baselines: AutoGen and CAMEL are intentionally not
implemented in the current experimental phase.
```

**不安装 `autogen` 或 `camel-ai`，不加入实验矩阵，不创建伪实现。**

### 外部基线实现状态

| Baseline | 状态 | 依赖 |
|----------|------|------|
| direct_single_vlm | **完整实现** | 无额外依赖 |
| self_consistency_vlm | **完整实现** | 无额外依赖 |
| multi_agent_debate | **完整实现** | 无额外依赖 |
| clip_zero_shot | **完整实现** | `torch`, `transformers` |
| dinov2_knn | **完整实现** | `torch`, `transformers`, `scikit-learn` |
| blip2_zero_shot | **完整实现** | `torch`, `transformers` |

所有视觉基线默认 `allow_model_download=False`，不自动下载模型权重。

### 旧实验重新归类

以下方法已从外部基线表移除，重新归类为内部诊断 / 遗留消融：

| 旧名称 | 新归类 | 说明 |
|--------|--------|------|
| FixedMultiExpert | Legacy / Internal Diagnostic | 不路由的固定 5 专家 |
| FixedFull | Legacy / Internal Diagnostic | 始终执行全部步骤 |
| GenericMAD | Legacy / Internal Diagnostic | 通用多 Agent 辩论（内部） |
| NoRouter | Legacy Ablation | 固定路径 |
| FixedAllRecheck | Legacy Ablation | 始终执行全部重审 |
| RandomRouter / ConfidenceRouter | Legacy Ablation | 备选路由策略（已弃用） |
| EntropyRouter / MarginRouter | Legacy Ablation | 备选路由策略（已弃用） |
| OracleRouter | Oracle Upper Bound | 仅开发调试，不可部署 |

这些类保留兼容性，但在论文主表中不出现。

---

## 三、项目结构

```
artifact-delib/
├── src/artifact_delib/
│   ├── agents/                  # 专家、协商、Judge 模块
│   │   ├── experts/             # 5 个专项专家 Agent
│   │   ├── deliberation/        # HypothesisAgent, CriticAgent, Manager
│   │   └── structured_report.py # 专家混合输出格式解析
│   ├── baselines/               # 外部基线实现
│   │   ├── direct_vlm.py        # Direct Single-VLM
│   │   ├── self_consistency.py  # Self-Consistency VLM
│   │   ├── multi_agent_debate.py # Multi-Agent Debate
│   │   ├── clip_zero_shot.py    # CLIP Zero-Shot（需可选依赖 torch）
│   │   ├── dinov2_knn.py        # DINOv2 k-NN（需可选依赖 torch）
│   │   ├── blip2_zero_shot.py   # BLIP-2 Zero-Shot（需可选依赖 torch）
│   │   ├── legacy.py            # 旧基线（FixedMultiExpert 等）
│   │   ├── base.py              # BaselineProtocol
│   │   └── registry.py          # 方法注册表
│   ├── ablations/               # 核心消融实验
│   │   ├── no_expert_specialization.py
│   │   ├── no_disagreement_analysis.py
│   │   ├── no_dynamic_routing.py
│   │   ├── random_recheck.py
│   │   ├── no_controlled_deliberation.py
│   │   ├── free_debate.py
│   │   ├── no_critic.py
│   │   └── early_judge.py
│   ├── ablations.py             # 向后兼容（旧 A1-A12）
│   ├── baselines.py             # 向后兼容（旧 B1-B4）
│   ├── api/                     # 模型客户端、Mock、缓存、预算、重试
│   ├── data/                    # 导入、划分、验证、下载、BatchRunner
│   ├── evaluation/              # 预测解析、指标计算、实验日志
│   ├── pipeline/                # 主流水线编排
│   ├── router/                  # RuleRouter, OracleRouteBuilder, MLPRouter
│   └── schemas.py               # 核心数据结构
├── tests/                       # 单元测试
├── scripts/                     # 数据下载、Oracle 构建、表生成
├── configs/                     # 实验配置文件
│   ├── baselines/
│   ├── ablations/
│   └── experiments/
├── results/                     # 实验输出目录
│   ├── external_baselines/
│   ├── core_ablations/
│   ├── tables/
│   └── figures/
└── README.md
```

---

## 四、安装

**Requirements:** Python 3.12+, Conda environment `histo-delib`

```bash
git clone https://github.com/dscheng200301/artifact-delib.git
cd artifact-delib

# 核心安装
conda run -n histo-delib pip install -e .

# 可选：视觉基线依赖
conda run -n histo-delib pip install -e ".[vision-baselines]"
```

### 依赖分组

- **Core:** pydantic, httpx, Pillow, PyYAML, Jinja2, filelock
- **Vision Baselines (可选):** torch, transformers, timm, scikit-learn

普通安装不会被迫安装大型视觉依赖。

---

## 五、运行实验

### Mock 模式（不调用付费 API）

```bash
# 运行所有单元测试
PYTHONPATH=src conda run -n histo-delib python -m pytest tests/ -v

# 生成结果表骨架（无实验数据）
conda run -n histo-delib python scripts/generate_tables.py
```

### 真实实验（需要授权和 API Key）

```bash
# 注意：以下命令会调用付费 API，请先确认配置
# 修改 configs/experiments/external_baselines.yaml 中的 allow_remote_calls: true
# 然后运行 BatchRunner
```

**在运行真实基线前，必须确认：**
1. API Key 已配置且不硬编码在代码中
2. 数据集已下载到 `data/artifact/`
3. 数据划分清单已生成
4. 已理解实验会消耗 Token 和产生费用
5. 模型权重下载（CLIP/DINOv2/BLIP-2）需单独授权

### 未授权时禁止自动下载或远程调用

项目默认 `allow_remote_calls: false` 和 `allow_model_download: false`。
所有单元测试使用 MockClient，不调用付费 API，也不下载真实模型。

---

## 六、数据防护

### 泄漏防护措施

1. 文件 SHA-256 完全重复检测
2. 感知哈希近重复检测
3. 相同 Met object ID 分组
4. 同一文物不同拍摄角度分组
5. 相同馆藏编号分组
6. 文件名标签泄漏检查
7. 图片损坏与不可读取检查
8. 类别样本数量统计
9. train/validation/test 标签覆盖统计

### 正式划分

使用 **Object-disjoint stratified split**：
- 同一对象的所有图片只能进入一个 split
- 验证集用于选择 k、阈值和 Prompt
- 测试集不得用于调参
- split 清单落盘，后续运行复用
- 不得每次运行重新随机划分

### 当前数据状态

读取实际 manifest 获取真实数量。README 不包含虚构数字。

---

## 七、评价指标

### 识别指标

- Category Accuracy
- Artifact-Type Accuracy
- Dynasty/Period Accuracy
- Material Accuracy
- Top-1 Joint Accuracy
- Top-3 Accuracy
- Macro-F1
- Micro-F1
- Parse Failure Rate

### 效率指标

- Average API Calls
- Average Input/Output/Total Tokens
- Average Latency / P50 / P95
- Estimated Cost

### 协商相关指标（仅完整方法与相关消融）

- Correction Rate
- Harm Rate
- No-Change Rate
- Deliberation Trigger Count
- Average Deliberation Rounds

---

## 八、专家输出格式

专家报告采用**自然语言 + 少量结构化控制字段**的混合格式：

```json
{
  "expert_type": "shape",
  "report": "完整自然语言视觉分析报告",
  "top_candidates": [
    {"name": "宋代青瓷碗", "confidence": 0.67},
    {"name": "元代早期瓷碗", "confidence": 0.23}
  ],
  "uncertainty_focus": ["底足", "口沿"],
  "recommended_expert": "style"
}
```

- `report` 必须保留完整自然语言视觉理解
- `top_candidates` 限制为 1–3 个
- `confidence` 仅作为路由特征，不是真实概率
- `uncertainty_focus` 用于分歧分析和定向重审
- `recommended_expert` 仅为专家建议，Router 不直接无条件采用
- JSON 解析失败时保留原始自然语言输出

---

## 九、结果

### 重要声明

**本 README 不包含虚构实验结果。** 所有实验数据必须在真实实验运行后才能填入。

当前结果表（`results/tables/`）只包含表头和行名。数据单元格为空。

### 统计方法

- 确定性方法：至少运行一次，Bootstrap 95% CI
- 随机方法：至少 3 个种子（42, 123, 2026），报告 Mean ± Std
- 主方法与主要外部基线间支持 McNemar 检验
- 不得只保留最好的一次运行

---

## 十、Pilot 实验状态

**当前正式实验尚未运行。** 所有结果表仅含表头和方法名，无实验数据。

Pilot 执行命令（需要先修改 `allow_remote_calls: true`）：

```bash
PYTHONPATH=src python -m artifact_delib.cli run configs/experiments/pilot.yaml
```

Pilot 成功条件：
- 完成率 ≥ 99%
- Parse failure rate ≤ 5%
- FAST / RECHECK / DELIBERATION 三类路由均出现
- 抽查 Token Accounting 一致

---

## 十一、代码质量

- Python 3.12+，类型注解完整
- 新增类和公共函数有 docstring
- 不硬编码 API Key
- 不提交 `.env`
- 普通 `pip install` 不强制安装 torch
- 模型和特征使用 lazy loading
- 不自动下载模型
- 不自动执行付费实验
- 不伪造论文结果

---

## 十二、引用

If you use this work in your research, please cite:

```bibtex
@software{artifactdelib2026,
  title = {ArtifactDelib: Candidate-Disagreement-Driven Dynamic Multi-Expert Deliberation
           for Ancient Artifact Identification},
  author = {Cheng, D.},
  year = {2026},
  url = {https://github.com/dscheng200301/artifact-delib}
}
```

## License

[MIT](LICENSE)
