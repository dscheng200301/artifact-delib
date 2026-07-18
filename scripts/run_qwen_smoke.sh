#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="${ROOT_DIR}/.env"

if [[ ! -f "${ENV_FILE}" ]]; then
  echo "Missing ${ENV_FILE}. Copy .env.example to .env and fill VLM_API_KEY." >&2
  exit 1
fi

read_env_value() {
  local key="$1"
  grep -E "^${key}=" "${ENV_FILE}" | tail -n 1 | cut -d '=' -f 2- | tr -d '\r'
}

# Read only the gate and credential checks; Python Settings loads the full .env itself.
VLM_API_KEY="$(read_env_value VLM_API_KEY)"
LLM_API_KEY="$(read_env_value LLM_API_KEY)"
API_ALLOW_PAID_CALLS="$(read_env_value API_ALLOW_PAID_CALLS)"

if [[ -z "${VLM_API_KEY:-}" && -z "${LLM_API_KEY:-}" ]]; then
  echo "VLM_API_KEY or LLM_API_KEY is required in .env." >&2
  exit 1
fi

if [[ "${API_ALLOW_PAID_CALLS:-false}" != "true" ]]; then
  echo "Set API_ALLOW_PAID_CALLS=true in .env before running this paid smoke test." >&2
  exit 1
fi

cd "${ROOT_DIR}"
export PYTHONPATH="${ROOT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"
SMOKE_ROOT="outputs/qwen-smoke-$(date -u +%Y%m%d-%H%M%S)"

conda run --no-capture-output -n histo-delib \
  python -m histodelib.cli run \
  --mode api \
  --method direct_vlm \
  --config configs/api/default.yaml \
  --output-root "${SMOKE_ROOT}"

conda run --no-capture-output -n histo-delib \
  python -m histodelib.cli validate-smoke \
  "${SMOKE_ROOT}/direct_vlm-default-api-synthetic" \
  --expected-predictions 12
