# Changelog

## Unreleased - 2026-07-18

- Added typed text/image evidence schemas, prompt provenance, safer relation probing, validated API-router fallback, deterministic image views, stateful bounded cross-examination, and explicit baseline protocol metadata.
- Added run fingerprints that reject unsafe resume, sample-ID-safe metric pairing, image readability/hash/duplicate checks, group-aware statistics helpers, thread-safe budget reservation, retry-attempt audit records, and pricing provenance fields.
- Formal datasets and formal experiments remain `NOT_SELECTED` / `NOT_RUN`; no paid API calls or research results were generated.

## 0.1.0 - 2026-07-17

- Added `scripts/run_qwen_smoke.sh` for an explicit, guarded Qwen remote synthetic smoke run; it validates `.env` and the paid-call gate without printing credentials.
- Added the equivalent native PowerShell entrypoint `scripts/run_qwen_smoke.ps1` for Windows environments without Bash/WSL.
- Added fail-closed smoke artifact validation, explicit API run metadata, runtime YAML contract validation, bounded provider concurrency, and a no-call experiment-matrix dry-run.

- Added environment audit, approved engineering design, and implementation plan.
- Added API-only project metadata, typed schemas/settings, synthetic data validation, safe mock API infrastructure, disagreement routing, deferred judging, metrics, and CLI fixture smoke path.
- Added deferred paper-writing PDF prompt and research-boundary documentation; no formal datasets, experiments, or real API calls.
- Added versioned prompt loading, audited token/latency/cost/error JSONL records, resumable baseline runs, and the four-page Chinese paper-writing specification PDF.
- Verification evidence: 53 tests passed; Ruff, mypy, pip check, fixture validation, and mock smoke workflow passed. Formal data and experiments remain NOT_SELECTED/NOT_RUN.
- Added guarded client integration for pre-call budget reservation, deterministic response caching, retry handling, resolved YAML run configs, twelve-sample synthetic fixtures, and per-sample API-router call accounting.
- Set the default API-only model to `qwen3.5-flash` across all modalities and baseline request paths.
