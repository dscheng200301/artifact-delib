# HistoDelib Project Status

## Project Version

0.1.0

## Python Version

3.12.13 verified in `histo-delib`.

## Conda Environment

Existing `histo-delib`; no environment was created. The editable package install was verified in-place.

## Completed

- Environment audit and engineering design/plan.
- Existing `histo-delib` environment is now the only documented runtime; no environment creation commands remain.
- API-only project metadata, typed schemas/settings, synthetic fixture builder, manifest import, split and leakage checks.
- Mock API, OpenAI-compatible HTTP normalization, retry policy, cache, budget guard, structured response parsing, token/cost accounting, audited redacted call logs, rule/API routers, deferred judge, bounded deliberation, metrics, resumable runner and CLI workflow.
- Versioned YAML prompts are loaded and hashed by the modality-isolated agents.
- Guarded client is now wired into fixture runs for cache hits/misses, conservative pre-call budgets, retries, and redacted call logs; YAML run configs are persisted as resolved artifacts.
- Fixture builder creates 12 synthetic samples (four per class); API-router call counts include the router request per sample.
- The default model configuration is now `qwen3.5-flash` across text, image, router, judge, and baseline requests.
- Mock smoke run generated only synthetic-labelled artifacts; the paper-writing specification was rendered to a four-page PDF and visually inspected.

## In Progress

- Formal dataset selection, real API smoke tests and formal experiments remain intentionally deferred.

## Blocked

- None.

## Tests

- 54 passed; Ruff, mypy and pip check passed in `histo-delib`.

## Mock Smoke Test

- PASSED with `histodelib_rule` on synthetic fixtures only.

## Real API Smoke Test

- NOT_RUN; paid calls are disabled.

## Current API Budget

- No API calls performed.

## Dataset Status

- NOT_SELECTED.

## Last Successful Command

- `PYTHONPATH=src conda run -n histo-delib python -m histodelib.cli run --method histodelib_rule --config fixture` (mock only)

## Last Failed Command

- Contract test before runtime metadata was added (expected TDD red state).

## Current Git Commit

- `40c600c` (qwen3.5-flash model default integration)

## Next Executable Step

- Before any formal result claim: obtain data authorization, freeze prompts/models/API settings, run all baselines and ablations, and generate numbers only from structured artifacts.
