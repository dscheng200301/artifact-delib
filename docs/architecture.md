# Architecture

HistoDelib separates fixture/data validation, normalized model requests, guarded API infrastructure, modality-isolated methods, structured predictions, metrics, and CLI commands. `GuardedModelClient` wraps either the deterministic mock or an OpenAI-compatible provider with a deterministic cache key, conservative pre-call budget reservation, transient retry, and redacted JSONL audit records. Mock API clients implement the same normalized request/response schema as an OpenAI-compatible client. Paid calls are disabled by default, and generated fixture artifacts are not research results.

Local YAML files under `configs/` can be passed to the CLI; the resolved mapping is persisted in each run directory alongside resumable predictions. The fixture builder currently creates twelve labelled images (four per class), all marked `SYNTHETIC_FIXTURE` and `NOT_FOR_RESEARCH_RESULTS`.
