# Runtime Completion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Close the remaining runtime and experiment-readiness gaps without selecting formal data or claiming research results.

**Architecture:** Keep the guarded provider boundary as the only remote-call path. Add explicit artifact validation and config validation at the runner boundary, then add a dry-run experiment matrix layer that consumes YAML but never downloads data. Preserve synthetic-only safeguards.

**Tech Stack:** Python 3.12, Pydantic, Typer, YAML, pytest, Ruff, mypy, existing Conda environment `histo-delib`.

## Global Constraints

- Remote APIs only; no local weights, formal datasets, or paid services beyond explicitly authorized APIs.
- Paid calls remain disabled by default; every call remains cached, budget-limited, logged, and token-accounted.
- Synthetic artifacts remain `SYNTHETIC_FIXTURE` and `NOT_FOR_RESEARCH_RESULTS`.

### Task 1: Correct API metadata and CLI reporting

Files: `src/histodelib/runner/run_manager.py`, `src/histodelib/cli.py`, tests for runner/CLI.

Add an explicit `mode` argument to `RunManager.run`, persist it in `run_metadata.json`, and print `remote API synthetic smoke complete` for API mode while retaining the fixture marker.

### Task 2: Add smoke artifact validation

Files: create `src/histodelib/validation/smoke.py`, `src/histodelib/cli.py`, `scripts/run_qwen_smoke.ps1`, tests.

Implement `validate_smoke_artifacts(run_dir: Path) -> SmokeValidation` checking prediction count, completed status, legal labels, call-log errors, provider presence, and mode. The script must exit nonzero when validation fails.

### Task 3: Validate runtime YAML contracts

Files: create `src/histodelib/config_schema.py`, modify `src/histodelib/cli.py`, tests and config examples.

Validate known fields, reject negative call limits, require supported router names, and pass validated values into baseline construction. Keep unknown fields available in resolved artifacts for provenance.

### Task 4: Exercise complete API deliberation smoke

Files: tests and smoke documentation.

Add a guarded API-mode test using a deterministic recording provider for `histodelib_api_router` with `enable_api_deliberation=true`; assert bounded reinspection, cross-exam, judge calls and structured predictions. No real additional paid call is required for this test.

### Task 5: Add concurrency and rate-limit boundary

Files: `src/histodelib/api/guarded.py` or a focused client wrapper, settings, tests.

Implement a bounded semaphore using `API_MAX_CONCURRENCY` around provider calls and preserve budget reservation before acquisition. Add a provider-independent 429 retry test without changing the existing retry contract.

### Task 6: Add dry-run experiment matrix

Files: create `src/histodelib/experiments/matrix.py`, CLI command/tests/docs.

Load a YAML matrix of methods/configs, emit planned run records, and refuse formal execution unless an explicit dataset authorization flag is supplied. Dry-run must never call an API or download data.

### Task 7: Final verification and status

Run pytest, Ruff, mypy, pip check, fixture smoke, and config/matrix dry-run. Update `STATUS.md` and `CHANGELOG.md`; commit each coherent task with a conventional message.
