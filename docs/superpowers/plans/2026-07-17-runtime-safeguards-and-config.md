# Runtime Safeguards and Config Integration Implementation Plan

> **For agentic workers:** Execute task-by-task with TDD checkpoints.

**Goal:** Wire budget, retry, cache, error recovery, and YAML configuration into the existing API-only fixture/remote-client path without making real API calls.

**Architecture:** Add a guarded client decorator around the provider-neutral `ModelClient`. It will derive a deterministic request cache key, reserve budget before provider calls when an estimate is available, retry transient failures, and append redacted audit records. Extend the CLI runner to load a resolved YAML config and preserve resumable artifacts.

**Tech Stack:** Python 3.12, Pydantic, httpx, pytest/respx, PyYAML, local JSON cache.

## Global Constraints

- Use the existing `histo-delib` Conda environment.
- Keep all intelligent model calls API-only; tests use deterministic MockModelClient.
- Do not download formal datasets or run formal experiments.
- Keep real API calls disabled by default.
- Fixture outputs remain `SYNTHETIC_FIXTURE` and `NOT_FOR_RESEARCH_RESULTS`.

### Task 1: Guarded client contract

**Files:**
- Create: `src/histodelib/api/guarded.py`
- Modify: `tests/test_api.py`

- [ ] Write failing tests for cache hit, budget refusal before provider call, transient retry, and `BUDGET_EXCEEDED` audit status.
- [ ] Run the focused tests and confirm they fail because `GuardedModelClient` does not exist.
- [ ] Implement the minimal decorator using `ResponseCache`, `BudgetManager`, `retry_call`, `CallLogStore`, and `estimate_cost`.
- [ ] Run focused tests and the existing API tests.

### Task 2: Config-driven runner

**Files:**
- Modify: `src/histodelib/cli.py`
- Modify: `src/histodelib/runner/run_manager.py`
- Modify: `tests/test_runner.py`
- Modify: `tests/test_cli_module.py`

- [ ] Write failing tests showing `--config` loads a local YAML mapping and persists its resolved content.
- [ ] Run the focused tests and confirm the config is currently ignored.
- [ ] Implement safe local config loading and pass resolved settings into run metadata.
- [ ] Run focused tests and the full suite.

### Task 3: Fixture and schema coverage

**Files:**
- Modify: `src/histodelib/schemas.py`
- Modify: `src/histodelib/data/fixture_builder.py`
- Modify: `tests/test_data.py`

- [ ] Write failing tests for `fixture`/`unassigned` split values and at least ten deterministic fixture samples.
- [ ] Run the focused tests and confirm the current three-sample fixture fails the contract.
- [ ] Extend the schema and fixture definitions without changing formal-data status markers.
- [ ] Run the full verification script.

### Task 4: Documentation and acceptance

**Files:**
- Modify: `STATUS.md`
- Modify: `CHANGELOG.md`

- [ ] Record the new safeguards, config loading, and fixture count.
- [ ] Run pytest, Ruff, mypy, pip check, doctor, fixture validation, and mock smoke.
- [ ] Commit with a Conventional Commit message and verify a clean worktree.
