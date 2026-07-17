# HistoDelib

**Disagreement-Triggered Bimodal Deliberation with Deferred Cross-Modal Adjudication for Historical Image-Caption Verification**.

This repository is an API-only Python 3.12 engineering framework. It does not download or deploy local model weights, search/download formal datasets, or run formal experiments. Fixture runs are explicitly marked `SYNTHETIC_FIXTURE` and `NOT_FOR_RESEARCH_RESULTS`; formal data is `NOT_SELECTED` and formal experiments are `NOT_RUN`.

## Setup

The existing Conda environment must be named `histo-delib` and use Python 3.12.

```powershell
conda run -n histo-delib python -m pip install -r requirements/base.txt -r requirements/api.txt -r requirements/dev.txt
conda run -n histo-delib python -m pip install -e .
```

Copy `.env.example` to `.env` only when configuring a remote API. Paid calls are disabled by default.

## Verification

```powershell
conda run -n histo-delib python -m pytest
conda run -n histo-delib python -m histodelib.cli doctor
```

See `STATUS.md` for current execution state and `docs/architecture.md` for the evolving system description.
