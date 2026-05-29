#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${ZS_SIM_ENV:-zssim}"
NOTEBOOK_DIR="${1:-notebooks}"

if command -v mamba >/dev/null 2>&1; then
  exec mamba run -n "${ENV_NAME}" zs-sim launch-notebooks --notebook-dir "${NOTEBOOK_DIR}"
fi

if [ -x ".venv/bin/zs-sim" ]; then
  exec .venv/bin/zs-sim launch-notebooks --notebook-dir "${NOTEBOOK_DIR}"
fi

exec zs-sim launch-notebooks --notebook-dir "${NOTEBOOK_DIR}"
