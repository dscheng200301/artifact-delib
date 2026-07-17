# HistoDelib Project Rules

1. Use only Python 3.12 in the existing Conda environment `histo-delib`.
2. LLM and VLM capabilities may only be called through remote APIs.
3. Never download, train, or deploy local model weights; do not add Torch, Transformers, Ollama, vLLM, CLIP, or GPU jobs.
4. Use no paid service other than explicitly authorized LLM/VLM APIs.
5. Do not search for or download formal datasets without explicit user authorization.
6. Do not fabricate research results; formal results must come from structured prediction files.
7. Record token use, latency, errors, caching state, and budget decisions for every API call.
8. API calls must be cached and budget-limited; paid calls default to disabled.
9. Never commit `.env`, API keys, or authorization headers.
10. Do not tune prompts or thresholds on a test set.
11. Run relevant tests after every code change.
12. Do not write a formal paper or generate formal-paper result figures/tables in this phase.
13. Maintain `STATUS.md` and `CHANGELOG.md` truthfully.
