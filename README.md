# ArtifactDelib

**Candidate-Disagreement-Driven Dynamic Multi-Expert Deliberation for Ancient Artifact Identification**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.12+](https://img.shields.io/badge/Python-3.12+-blue.svg)](pyproject.toml)
[![Tests](https://img.shields.io/badge/Tests-121%20passing-brightgreen.svg)](tests/)

基于候选分歧驱动动态多专家协作的古代文物细粒度识别系统。

---

## Overview

ArtifactDelib is a **dynamic multi-expert deliberation framework** for fine-grained visual recognition of ancient Chinese artifacts. Instead of relying on a single VLM call or a fixed set of expert analyses, it adaptively allocates computational resources based on the **disagreement pattern among top-K candidates**.

The system is designed around three core innovations:

1. **Multi-Expert Visual Understanding** — Five specialized experts (Shape, Style, Glyph, Material, Local Detail) analyze the artifact image from complementary perspectives, producing natural-language reports.

2. **Disagreement-Driven Dynamic Routing** — After expert reports are summarized into top-K candidate predictions, a **DisagreementAnalyzer** identifies the type of disagreement (e.g., shape confusion vs. style confusion), and a **RuleRouter** decides the optimal action:
   - **FAST** → Direct judge if confident
   - **Targeted Recheck** → Route to the single most relevant expert for re-examination
   - **DELIBERATION** → Trigger controlled hypothesis-level debate when candidates are closely contested

3. **Controlled Deliberation + Deferred Judgment** — For hard samples, a HypothesisAgent argues for Top-1 vs. Top-2 candidates while a CriticAgent refutes, followed by a final judge that renders the decision with full context.

## Pipeline

```
                        ┌──────────────────────┐
                        │    Input Image        │
                        └──────────┬───────────┘
                                   ▼
                        ┌──────────────────────┐
                        │  Visual Perception    │
                        └──────────┬───────────┘
                                   ▼
            ┌───────────────────────────────────┐
            │   Five Specialized Experts       │
            │  ┌─────┐ ┌────┐ ┌─────┐ ┌─────┐┌──────┐│
            │  │Shape│ │Style││Glyph│ │Mat. ││Detail││
            │  └─────┘ └────┘ └─────┘ └─────┘└──────┘│
            └──────────────────┬────────────────────┘
                               ▼
            ┌──────────────────────────────────────┐
            │         Report Summarizer             │
            └──────────────────┬───────────────────┘
                               ▼
            ┌──────────────────────────────────────┐
            │     Top-K Candidate Generation        │
            └──────────────────┬───────────────────┘
                               ▼
            ┌──────────────────────────────────────┐
            │     Disagreement Analyzer             │
            │     + RuleRouter                      │
            └────────────┬─────────────┬───────────┘
                         │             │
              ┌──────────┘             └──────────┐
              ▼                                    ▼
    ┌──────────────────┐              ┌──────────────────────┐
    │   FAST           │              │ Needs Recheck        │
    │ → ArtifactJudge  │              │ → Targeted Expert    │
    └──────────────────┘              │ → Re-summarizer      │
                                      │ → Re-route           │
                                      └──────────┬───────────┘
                                                 │
                                       ┌─────────┴─────────┐
                                       ▼                   ▼
                            ┌──────────────────┐  ┌──────────────────┐
                            │  DELIBERATION    │  │  FAST (re-route) │
                            │ Hypothesis A/B   │  │ → ArtifactJudge  │
                            │ + Critic Debate  │  └──────────────────┘
                            │ → ArtifactJudge  │
                            └──────────────────┘
                                      │
                                      ▼
                        ┌──────────────────────┐
                        │  Final Identification │
                        └──────────────────────┘
```

## Baselines & Ablations

The framework is evaluated against a comprehensive suite of baselines and ablations.

### Baselines (B1–B5)

| ID | Method | Description | Purpose |
|----|--------|-------------|---------|
| B1 | **DirectVLM** | Single VLM call, no experts | Baseline lower bound |
| B2 | **FixedMultiExpert** | All 5 experts always called, no routing | Prove dynamic routing value |
| B3 | **GenericMAD** | N-agent free-form debate with rounds | Prove controlled deliberation value |
| B4 | **FixedFull** | All experts + all rechecks + deliberation | Upper bound on cost |
| B5 | **ArtifactDelib-Rule** | Full system with RuleRouter | **Proposed method** |

### Ablations (A1–A12)

| ID | Variant | What's removed / changed |
|----|---------|--------------------------|
| A1 | NoMultiExpert | VP → Judge only (no experts) |
| A2 | SingleExpert | One generic expert instead of 5 specialized |
| A3 | NoRouter | Always all experts (like B2) |
| A4 | NoRecheck | Router can't request re-examination |
| A5 | NoDeliberation | Router can't trigger debate; goes straight to judge |
| A6 | NoDeferredJudge | Judge decides immediately, no deliberation context |
| A7 | FixedAllRecheck | Always recheck all 5 experts (like B4) |
| A8 | FixedDeliberation | Always run deliberation |
| A9 | FreeDebate | Unstructured debate vs. controlled hypothesis-critic |
| A10–A12 | *(additional routing/deliberation variants)* | |

Ablations are implemented in [`src/artifact_delib/ablations.py`](src/artifact_delib/ablations.py).

## Learned Router

Building on the RuleRouter, the framework also includes a **Learned Router** pipeline:

1. **OracleRouteBuilder** — Runs the full pipeline on training samples, extracts route features (top-K confidences, margin, disagreement type, candidate count), and scores every possible route action against ground truth to determine the optimal (oracle) route.
2. **MLPRouter** — A lightweight multi-layer perceptron trained on the oracle dataset to predict the optimal route from features alone, enabling data-driven routing decisions at inference time.

Implementation in [`src/artifact_delib/router/`](src/artifact_delib/router/).

## Project Structure

```
artifact-delib/
├── src/artifact_delib/
│   ├── agents/              # Experts, deliberation, judge modules
│   │   ├── experts/         # 5 specialized expert agents
│   │   └── deliberation/    # HypothesisAgent, CriticAgent, Manager
│   ├── api/                 # Model clients (OpenAI-compatible, mock, cache, budget, retry)
│   ├── data/                # Importer, splitter, validator, Met downloader, batch runner
│   ├── evaluation/          # Prediction parser, metrics, experiment logger
│   ├── models/              # Mock client for testing
│   ├── pipeline/            # Main pipeline orchestrator
│   ├── router/              # RuleRouter, OracleRouteBuilder, MLPRouter
│   ├── baselines.py         # B1–B4 baselines
│   ├── ablations.py         # A1–A12 ablation variants
│   └── schemas.py           # Core data structures
├── tests/                   # 121 tests across phases 2–8
├── scripts/                 # Data download, Oracle build, router training
├── prompts/                 # Prompt templates for agents
└── docs/                    # Documentation
```

## Installation

**Requirements:** Python 3.12+, Conda environment `histo-delib`

```bash
# Clone
git clone https://github.com/dscheng200301/artifact-delib.git
cd artifact-delib

# Install package
conda run -n histo-delib pip install -e .

# Run tests (121 tests)
PYTHONPATH=src conda run -n histo-delib python -m pytest tests/ -v
```

### Dependencies
- **Core:** pydantic, httpx, Pillow
- **Utilities:** PyYAML, Jinja2, filelock

## Dataset

The system is designed for [Met Museum Open Access](https://github.com/metmuseum/openaccess) (CC0 licensed) Chinese artifacts, covering 10+ categories:

| Category | Target |
|----------|--------|
| Ceramics (瓷器) | 1,500 |
| Jade (玉器) | 1,200 |
| Paintings (绘画) | 800 |
| Metalwork (金属器) | 800 |
| Textiles (纺织品) | 800 |
| Snuff Bottles (鼻烟壶) | 400 |
| Sculpture (雕塑) | 300 |
| Tomb Pottery (陶俑) | 250 |
| Lacquer (漆器) | 200 |
| Calligraphy (书法) | 130 |
| Enamels (珐琅/景泰蓝) | 20 |
| **Total** | **~6,400** |

```python
from pathlib import Path
from artifact_delib.data import MetDownloader, ArtifactDatasetSplitter
import asyncio

# Download up to 500 Met artifacts
downloader = MetDownloader(Path("data/artifact"), max_objects=500, concurrency=20)
samples = asyncio.run(downloader.run())

# Split 70/10/20
splitter = ArtifactDatasetSplitter(train_ratio=0.7, validation_ratio=0.1)
splits = splitter.split(samples)
```

To run the full download (6,400 images, ~75 min with rate limiting):
```bash
PYTHONPATH=src python scripts/download_met_images.py
```

## Running Experiments

```python
from artifact_delib.data.batch_runner import BatchRunner
from artifact_delib.pipeline.artifact_delib_pipeline import ArtifactDelibPipeline
from artifact_delib.data.importer import ArtifactDatasetImporter

# Load dataset
importer = ArtifactDatasetImporter(image_root=Path("data/artifact/images"))
samples = importer.import_manifest(Path("data/artifact/met_artifact_manifest.csv"))

# Load test split
with open("data/artifact/splits/test.txt") as f:
    test_ids = set(line.strip() for line in f if line.strip())
test_samples = [s for s in samples if s.sample_id in test_ids]

# Run B5 (full pipeline)
runner = BatchRunner(
    method=ArtifactDelibPipeline(),
    output_root=Path("output/b5"),
    experiment_id="b5_full_pipeline",
    method_name="ArtifactDelib-Rule"
)
results = runner.run(test_samples)
metrics = runner.evaluate(results, test_samples)
print(metrics)
```

## Evaluation

The evaluation framework computes:

- **Top-1 / Top-5 accuracy** — Fine-grained type, category, period, and joint
- **Macro F1** — Per-class F1 weighted equally across categories
- **Cost metrics** — Average API calls and tokens per sample
- **Correction / Harm rates** — How often recheck or deliberation corrects (or worsens) the prediction

## Citation

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
