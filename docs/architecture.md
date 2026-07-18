# Architecture

HistoDelib separates fixture/data validation, normalized model requests, guarded API infrastructure, modality-isolated methods, structured predictions, metrics, and CLI commands. `GuardedModelClient` wraps either the deterministic mock or an OpenAI-compatible provider with a deterministic cache key, conservative pre-call budget reservation, transient retry, and redacted JSONL audit records. Mock API clients implement the same normalized request/response schema as an OpenAI-compatible client. Paid calls are disabled by default, and generated fixture artifacts are not research results.

Local YAML files under `configs/` can be passed to the CLI; the resolved mapping is persisted in each run directory alongside resumable predictions. The fixture builder currently creates twelve labelled images (four per class), all marked `SYNTHETIC_FIXTURE` and `NOT_FOR_RESEARCH_RESULTS`.

The core deliberation path now keeps modality-specific `TextEvidence` and `ImageEvidence` separate from the final Judge label. `ClaimFactPair` records explicit alignment and conflict types for routing. API routing validates a bounded `RouteDecision` and records rule/API disagreement or parse fallback. Targeted reinspection creates deterministic Pillow views with source/view hashes; missing coordinates fall back to a recorded full view. Cross-examination stores a bounded stateful transcript without hidden chain-of-thought.

Run artifacts include a deterministic fingerprint over samples and resolved runtime configuration. Resume is rejected when the fingerprint changes. Metrics pair predictions and labels by `sample_id`; API audit records distinguish cache hits, logical requests, retry attempts, and provider responses, with pricing provenance and thread-safe budget reservations.
