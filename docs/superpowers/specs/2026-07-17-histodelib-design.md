# HistoDelib Engineering Design

## Scope and constraints

HistoDelib is an API-only Python 3.12 research-engineering package for historical image--caption verification. It must never download, train, or serve a local LLM/VLM model; it must not search for or download formal datasets; and it must not present fixture output as research results. The existing Conda environment is `histo-delib`; it is not recreated by this project.

The first delivery is a runnable, testable package with OpenAI-compatible and mock clients, a three-class data schema (`TRUE`, `MISCAPTIONED`, `OUT_OF_CONTEXT`), deterministic synthetic fixtures, baselines, the HistoDelib protocol, local caching and budget protection, CLI commands, and reproducible smoke tests. Real API calls are disabled by default and require explicit environment configuration. The second delivery is only a future PDF-generation prompt stored under `docs/prompts/`; no paper, research result figure, or research result table is generated in this phase.

## Options considered

1. **Single monolithic runner.** Fastest to scaffold but couples API transport, protocol decisions, and reporting; this would make API mocking and ablations brittle.
2. **Layered package with explicit schemas and dependency-injected clients (recommended).** Separates data, API infrastructure, prompts, methods, metrics, and CLI. Mock and real clients implement the same protocol, so fixtures exercise the same orchestration path without paid calls.
3. **Workflow framework or local model stack.** Rejected: it adds unnecessary dependencies and conflicts with the API-only/no-local-model constraints.

## Architecture

`settings` loads explicit environment and YAML configuration. `data` owns schema validation, fixture construction, import validation, split and leakage checks. `api` owns request/response models, OpenAI-compatible transport, deterministic mock responses, redacted cache keys, retries, token/cost accounting, and a fail-closed budget manager.

`methods` owns shared API-only baselines and HistoDelib orchestration. Text and image agents are isolated: the text agent receives caption-only input, while the image agent receives image-only input. A short relation probe produces structured risk features. Rule or API routers decide whether targeted reinspection and bounded cross-examination are needed. A deferred judge performs a blind initial decision then returns `KEEP`, `REVISE`, or `ABSTAIN` with a final task label. Every decision is retained as structured evidence, not hidden chain-of-thought.

`runner` writes resolved configuration, JSONL predictions, per-call usage, and a synthetic-only report. `metrics` reads only structured predictions and returns `N/A` for unavailable formal metrics. `cli` exposes doctor, fixture, run, report, and data-import paths. Config files declare baseline and method variants without changing source code.

## Failure handling and safety

API errors are typed, retried only for transient failures, and redacted in logs. Cache entries exclude credentials and raw responses are off by default. A budget refusal, token/request limit, or estimated-cost limit prevents new API calls while preserving completed artifacts and marks the run `BUDGET_EXCEEDED`. Network clients are never selected in fixture mode. Image URLs are disabled by default; local images are Base64 encoded only when a configured VLM request is permitted.

## Testing and acceptance

Tests are test-first and run inside `histo-delib`. They cover schema validation, prompt hashing, response parsing, cache, budget boundaries, router decisions, reinspection and stop criteria, deferred judging, metrics, fixture labelling, mock API integration, budget-stop recovery, CLI commands, and an end-to-end mock smoke run. The smoke report must say `SYNTHETIC_FIXTURE` and `NOT_FOR_RESEARCH_RESULTS`; formal dataset and experiment status must remain `NOT_SELECTED` and `NOT_RUN`.

## Deferred PDF work

The repository will store `docs/prompts/generate_paper_writing_spec_pdf.md`. It will instruct a later, separate PDF build to generate a Chinese writing-specification document without research values, claims, or formal-paper text. The deferred prompt will require a local free-tool render and build report when that later task is explicitly requested.
