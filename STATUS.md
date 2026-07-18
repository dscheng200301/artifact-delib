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
- The default model configuration is now the fixed `qwen3.5-flash-2026-02-23` snapshot across text, image, router, judge, and baseline requests; DashScope OpenAI-compatible endpoints and a CNY pricing example are recorded.
- Remote-client construction is fail-closed and guarded by cache, retries, budgets, redacted call logs, and an explicit paid-call gate; API reinspection, cross-examination, and deferred-judge paths are bounded and tested.
- Named baselines now declare modality payloads and call schedules; method/run YAML values drive model and deliberation limits into execution.
- Mock smoke run generated only synthetic-labelled artifacts; the paper-writing specification was rendered to a four-page PDF and visually inspected.

## In Progress

- Formal dataset selection and formal experiments remain intentionally deferred; one real API synthetic smoke has now been executed.

## Blocked

- None.

## Tests

- 66 passed; Ruff, mypy and pip check passed in `histo-delib`.

## Mock Smoke Test

- PASSED with `histodelib_rule` on synthetic fixtures only.

## Real API Smoke Test

- COMPLETED transport smoke on `direct_vlm`: 12/12 DashScope-compatible calls returned, 0 transport errors, 10,224 tokens, estimated 0.0178848 CNY, cache misses only. All 12 predictions were `INSUFFICIENT_EVIDENCE` because the provider returned prose instead of parseable JSON; this is not a research result.

## Current API Budget

- Last smoke budget accounting: 12 requests and 10,224 tokens; formal data remains `NOT_SELECTED`.

## Dataset Status

- NOT_SELECTED.

## Last Successful Command

- `powershell -ExecutionPolicy Bypass -File .\\scripts\\run_qwen_smoke.ps1` (transport succeeded; structured-output parsing issue remains)

## Last Failed Command

- Contract test before runtime metadata was added (expected TDD red state).

## Current Git Commit

- Latest commits: `feat: complete guarded qwen deliberation runtime`; guarded Qwen smoke entrypoints for Bash and PowerShell.

## Next Executable Step

- Fix and re-test structured JSON output enforcement for Qwen, then obtain data authorization, freeze prompts/models/API settings, run all baselines and ablations, and generate numbers only from structured artifacts.
