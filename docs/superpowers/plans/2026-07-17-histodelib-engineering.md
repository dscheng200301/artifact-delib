# HistoDelib Engineering Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the HistoDelib API-only Python 3.12 research-engineering package, using only synthetic fixtures and a pre-existing `histo-delib` Conda environment.

**Architecture:** A typed, layered Python package separates data and fixture validation, API infrastructure, prompts, methods, metrics, and CLI orchestration. OpenAI-compatible and deterministic mock clients implement the same client protocol; all runs write structured artifacts and safety status.

**Tech Stack:** Python 3.12, Pydantic, pydantic-settings, httpx, tenacity, Typer, PyYAML, Pillow, JSONL, pytest, ruff, mypy, reportlab for future local PDF rendering only.

## Global Constraints

- Use the existing Conda environment `histo-delib` with Python 3.12; do not create it.
- All intelligent models are remote API calls; never install or use local model weights, Torch, Transformers, Ollama, vLLM, or GPU jobs.
- No formal dataset search, download, or experiment; test fixtures must carry `SYNTHETIC_FIXTURE` and `NOT_FOR_RESEARCH_RESULTS`.
- Paid API calls are off by default and require `API_ALLOW_PAID_CALLS=true`; no other paid services are used.
- Do not commit `.env` or credentials; redact authorization and API keys.
- Formal results must derive from structured prediction files; current formal-data metrics are `N/A` and experiments are `NOT_RUN`.

---

### Task 1: Repository metadata and reproducible runtime

**Files:** Create project metadata, environment files, installation scripts, root documentation, policy files, and configuration skeletons.

**Interfaces:** Produces an installable `histodelib` package and the `make verify` command contract used by all later tasks.

- [ ] Write tests that assert `environment.yml` pins Python 3.12, dependency files exclude local-model packages, and `.env` is ignored.
- [ ] Run `conda run -n histo-delib python -m pytest tests/test_project_contract.py -v`; expect failure because project files do not exist.
- [ ] Add `pyproject.toml`, requirements, `environment.yml`, scripts, Makefile, policy/docs, configs, and empty package paths.
- [ ] Re-run the contract test and `conda run -n histo-delib python -m pip check`; expect passing contract assertions and no dependency conflicts.
- [ ] Commit with `chore: scaffold api-only project metadata`.

### Task 2: Typed schemas, settings, and run artifacts

**Files:** Create `src/histodelib/{schemas,settings,constants,exceptions,logging_config}.py`, `tests/test_schemas.py`, and `tests/test_settings.py`.

**Interfaces:** `Sample`, `Label`, `ModelRequest`, `ModelResponse`, `TokenUsage`, `CallRecord`, `Prediction`, and `RunStatus` are Pydantic models consumed by every package layer.

- [ ] Write tests for accepted labels, rejected labels, required fixture markers, safe settings defaults, and redacted settings serialization.
- [ ] Run those tests; expect import/function failures.
- [ ] Implement immutable typed models, safe defaults (`API_ALLOW_PAID_CALLS=false`), `Path` roots, and redaction helpers.
- [ ] Re-run schema/settings tests; expect pass.
- [ ] Commit with `feat: add typed settings and run schemas`.

### Task 3: Data interfaces and deterministic synthetic fixtures

**Files:** Create `src/histodelib/data/{schema,fixture_builder,validator,importer,splitter,leakage}.py`, fixture data, `tests/test_data.py`, and dataset documentation.

**Interfaces:** `build_fixture(root) -> list[Sample]`, `validate_samples(samples) -> ValidationReport`, and `validate_import_manifest(path, image_root) -> ValidationReport`.

- [ ] Write tests that build labelled synthetic images, reject missing image/caption/duplicate IDs, reject invalid paths, and flag a raw-image group crossing splits.
- [ ] Run the tests; expect missing modules/functions.
- [ ] Implement deterministic Pillow fixtures, manifest validation, group-aware splitting, and leakage reporting without downloading data.
- [ ] Re-run data tests; expect pass.
- [ ] Commit with `feat: add fixture and dataset validation interfaces`.

### Task 4: API safety infrastructure

**Files:** Create `src/histodelib/api/{base,mock,openai_compatible,retry,cache,budget,token_usage,response_parser}.py`, API YAML configs, and `tests/test_api_*.py`.

**Interfaces:** `ModelClient.generate(request) -> ModelResponse`; `BudgetManager.reserve(estimate)`; `ResponseCache.get_or_set(key, producer)`; `parse_structured_response(text, schema)`.

- [ ] Write tests for mock determinism, OpenAI message construction, malformed JSON repair/rejection, cache hit/miss, retry eligibility, safe header redaction, request/token/cost limits, and budget-stop preservation.
- [ ] Run API tests; expect failures due to absent implementations.
- [ ] Implement dependency-injected mock/transport clients, atomic file cache, token records, retry policy, fail-closed budget checks, and response parsing.
- [ ] Re-run API tests using no network access; expect pass.
- [ ] Commit with `feat: add safe mock and openai-compatible api layer`.

### Task 5: Versioned prompts and modality-isolated agents

**Files:** Create YAML prompts, `src/histodelib/prompts/{loader,renderer}.py`, `src/histodelib/methods/{agents,probe}.py`, and focused tests.

**Interfaces:** `render_prompt(name, context) -> RenderedPrompt`; `TextAgent.analyze(caption)`; `ImageAgent.analyze(image_path)`; `RelationProbe.assess(text_evidence, image_evidence)`.

- [ ] Write tests ensuring text prompts never receive image input, image prompts never receive captions, required prompt metadata is validated, and prompt hashes change with content.
- [ ] Run tests; expect absent modules/functions.
- [ ] Implement YAML prompt loading/rendering and structured agent/probe outputs with bounded output-token settings.
- [ ] Re-run tests; expect pass.
- [ ] Commit with `feat: add versioned isolated agents and relation probe`.

### Task 6: Routing, targeted reinspection, cross-examination, and deferred judge

**Files:** Create `src/histodelib/methods/{router,reinspection,cross_exam,judge,histodelib}.py`, method configs, and focused tests.

**Interfaces:** `RuleRouter.route(probe) -> RouteDecision`; `ApiRouter.route(probe) -> RouteDecision`; `HistoDelibMethod.run(sample) -> Prediction`.

- [ ] Write tests for risk-based rule routes, API-router schema validation, text/glyph/panoramic/patch reinspection selection, maximum-round and abstention stop rules, and `KEEP`/`REVISE`/`ABSTAIN` deferred-judge outcomes.
- [ ] Run tests; expect failing imports/behaviors.
- [ ] Implement the bounded protocol with structured evidence and no access to gold labels at routing or judging time.
- [ ] Re-run method tests; expect pass.
- [ ] Commit with `feat: add disagreement-triggered deliberation protocol`.

### Task 7: API-only baselines and run orchestration

**Files:** Create `src/histodelib/methods/baselines.py`, `src/histodelib/runner/{run_manager,artifacts,resume}.py`, baseline/run configs, and integration tests.

**Interfaces:** `create_method(name, clients, config) -> VerificationMethod`; `RunManager.run(samples, method) -> RunSummary`.

- [ ] Write tests covering direct VLM, structured reasoning, self-consistency, self-reflection, sequential context, fixed multi-view, generic MAD, Always-Full, HistoDelib Rule, and HistoDelib API Router dispatch.
- [ ] Run tests; expect missing factory/orchestrator behavior.
- [ ] Implement baseline factories and JSONL artifact/resume behavior that records every call and never reports fixture output as research output.
- [ ] Re-run integration tests; expect pass.
- [ ] Commit with `feat: add baselines and resumable run orchestration`.

### Task 8: Metrics, reporting, and CLI

**Files:** Create `src/histodelib/{metrics,reporting,cli}.py`, CLI tests, report docs, and config resolution tests.

**Interfaces:** `compute_metrics(predictions, reference) -> MetricsReport`; `typer` commands `doctor`, `fixture build`, `fixture validate`, `run`, `report run`, and `data import`.

- [ ] Write tests for Accuracy/Macro-F1/Misc-F1, token saving versus Always-Full, correction/harm rate denominators, `N/A` unavailable metrics, synthetic report banners, and command exit codes.
- [ ] Run tests; expect failures.
- [ ] Implement metric calculation from structured predictions only, synthetic-safe reports, config resolution, and CLI command wiring.
- [ ] Re-run metrics and CLI tests; expect pass.
- [ ] Commit with `feat: add safe metrics reports and command line interface`.

### Task 9: Mock smoke verification and quality gates

**Files:** Create smoke-test runner/config, comprehensive tests, CI/config docs, `STATUS.md`, `CHANGELOG.md`, and `reports/smoke_test_report.md`.

**Interfaces:** `make doctor`, `make fixture`, `make smoke-mock`, `make smoke-api`, and `make verify` are documented user-facing commands.

- [ ] Write an end-to-end test that builds fixtures, runs each required mock method, verifies structured artifacts and `SYNTHETIC_FIXTURE`/`NOT_FOR_RESEARCH_RESULTS`, then verifies a budget-interrupted run can resume.
- [ ] Run it; expect a meaningful failure before final wiring.
- [ ] Implement remaining command wiring and report updates without running a real paid API call.
- [ ] Run `conda run -n histo-delib python -m pytest`, `ruff check .`, `mypy src`, and `make verify`; expect all pass.
- [ ] Commit with `test: add mock smoke verification and quality gates`.

### Task 10: Deferred paper-writing prompt

**Files:** Create `docs/prompts/generate_paper_writing_spec_pdf.md` and `docs/PAPER_WORK_DEFERRED.md`.

**Interfaces:** The prompt is a standalone future input; it does not create a PDF or a research paper in the present phase.

- [ ] Write a test that asserts the prompt includes all required section headings and forbids experimental values, figures, result tables, SOTA, and submission claims.
- [ ] Run it; expect failure while the prompt is absent.
- [ ] Add the standalone future-PDF prompt and deferred-paper statement, preserving `NOT_RUN`/`N/A` status language.
- [ ] Re-run the test; expect pass.
- [ ] Commit with `docs: add deferred paper writing specification prompt`.

## Plan self-review

Coverage is mapped to the prompt: repository/runtime (Task 1), data schema/import/splits (Tasks 2--3), API client/cost/cache/privacy (Task 4), prompt isolation (Task 5), the complete deliberation protocol (Task 6), all named API-only baselines and run artifacts (Task 7), metrics/CLI/automation (Task 8), tests and smoke evidence (Task 9), and the requested independent future-PDF prompt (Task 10). The plan deliberately excludes formal datasets, formal experiments, results figures/tables, and a generated paper/PDF.
