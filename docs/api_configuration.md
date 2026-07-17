# API Configuration

Copy `.env.example` to an untracked `.env` in the repository root, then put the DashScope key in `VLM_API_KEY` (and optionally `LLM_API_KEY` if the modalities use different credentials). Do not commit `.env`. Configure LLM and VLM endpoints independently or identically. Set `API_ALLOW_PAID_CALLS=true` only for an explicit real API run; wrap the provider in `GuardedModelClient` so cache, request, token, retry, and estimated-cost limits are applied before the call. Never log a key or authorization header. Fixture commands always use `MockModelClient` and never contact a provider.

The default model is the fixed snapshot `qwen3.5-flash-2026-02-23` for both text and vision requests (the provider alias `qwen3.5-flash` currently resolves to this snapshot). Keeping the same model ID for every baseline makes protocol comparisons fair; freeze the dated ID in the resolved run config before formal experiments. The example pricing file records mainland-China pricing in CNY and must be rechecked before a paid run.

Example local configuration run:

```powershell
conda run -n histo-delib python -m histodelib.cli run `
  --method histodelib_rule `
  --config configs/run/fixture.yaml
```

The run writes `config.resolved.yaml`, `run_metadata.json`, `predictions.jsonl`, a local cache, and `call_log.jsonl`. Formal data and experiments remain `NOT_SELECTED` and `NOT_RUN` until separately authorized.

Remote synthetic smoke (still not a formal experiment) requires both an API key and an explicit gate:

```powershell
$env:API_ALLOW_PAID_CALLS="true"
conda run -n histo-delib python -m histodelib.cli run --mode api --method direct_vlm --config configs/api/default.yaml
```

The command is intentionally not run by the repository test suite.
