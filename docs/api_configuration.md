# API Configuration

Copy `.env.example` to an untracked `.env`. Configure LLM and VLM endpoints independently or identically. Set `API_ALLOW_PAID_CALLS=true` only for an explicit real API run; wrap the provider in `GuardedModelClient` so cache, request, token, retry, and estimated-cost limits are applied before the call. Never log a key or authorization header. Fixture commands always use `MockModelClient` and never contact a provider.

The default research model is `qwen3.5-flash` for both text and vision requests. Keeping the same model ID for every baseline makes protocol comparisons fair; freeze an exact dated model ID in the resolved run config before formal experiments.

Example local configuration run:

```powershell
conda run -n histo-delib python -m histodelib.cli run `
  --method histodelib_rule `
  --config configs/run/fixture.yaml
```

The run writes `config.resolved.yaml`, `run_metadata.json`, `predictions.jsonl`, a local cache, and `call_log.jsonl`. Formal data and experiments remain `NOT_SELECTED` and `NOT_RUN` until separately authorized.
