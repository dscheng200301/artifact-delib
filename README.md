# ArtifactDelib

**Candidate-Disagreement-Driven Dynamic Multi-Expert Deliberation for Ancient Artifact Identification**

基于候选分歧驱动动态多专家协作的古代文物细粒度识别系统。

## 三大核心创新

1. **多专家自然语言视觉理解** — 5 个专业专家（器形/纹饰/铭文/材质/局部细节）从不同角度分析图像
2. **Top-K 候选分歧驱动动态路由** — 基于候选差异方向，精准调用最相关的定向专家重审
3. **受控假设级协商 + 延迟裁决** — 难样本触发 Top-1 vs Top-2 的受控协商，最后由 Judge 裁决

## 完整流水线

```
Image → VisualPerception → 5 Experts → Summarizer → Top-K Candidates
    → DisagreementAnalyzer → RuleRouter
        ├── FAST → ArtifactJudge
        ├── Targeted Expert Recheck → Re-summarize → Re-route
        └── DELIBERATION → HypothesisAgent A/B + Critic → ArtifactJudge
    → Final NL Identification
```

## 项目结构

```
src/artifact_delib/
├── agents/              # 所有专家、协商、裁决模块
├── api/                 # 模型客户端（OpenAI 兼容 / Mock / 缓存 / 预算 / 重试）
├── data/                # 数据导入、划分、验证、批量运行、Met 下载器
├── evaluation/          # NL 解析器、指标、实验日志
├── models/              # 模型客户端封装
├── pipeline/            # 流水线编排
├── router/              # RuleRouter、OracleRouteBuilder、MLPRouter
├── baselines.py         # B1-B4 基线
├── ablations.py         # A1-A12 消融变体
├── schemas.py           # 数据结构
└── constants.py         # 常量
```

## 安装

```bash
conda run -n histo-delib pip install -e .
```

## 测试

```bash
# 运行全部 116 个测试
PYTHONPATH=src conda run -n histo-delib python -m pytest tests/
```

## 环境

- Python 3.12
- Conda 环境：`histo-delib`
- 依赖：pydantic, httpx, Pillow, PyYAML, Jinja2, filelock

## 数据集

支持 [Met Museum Open Access](https://github.com/metmuseum/openaccess) (CC0 许可)：

```python
from artifact_delib.data import MetDownloader, ArtifactDatasetSplitter
import asyncio

# 下载
downloader = MetDownloader(Path("data/artifact"), max_objects=500, concurrency=20)
samples = asyncio.run(downloader.run())

# 划分 70/10/20
splitter = ArtifactDatasetSplitter(train_ratio=0.7, validation_ratio=0.1)
splits = splitter.split(samples)
```

## 基线 & 消融

| 基线 | 方法 | 用途 |
|---|---|---|
| B1 DirectVLM | 单次 VLM 调用 | 证明多专家有效性 |
| B2 FixedMultiExpert | 固定多专家，无路由 | 证明动态路由有效性 |
| B3 GenericMAD | 自由多智能体辩论 | 证明受控协商有效性 |
| B4 FixedFull | 全量执行 | 证明动态路由降本 |
| B5 ArtifactDelib-Rule | 完整系统 + RuleRouter | 主系统 |

消融涵盖三大创新的各个维度（详见 `src/artifact_delib/ablations.py`）。
