#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${ZS_SIM_ENV:-zssim}"

if command -v mamba >/dev/null 2>&1; then
  exec mamba run -n "${ENV_NAME}" zs-sim sync-tags "$@"
fi

if [ -x ".venv/bin/zs-sim" ]; then
  exec .venv/bin/zs-sim sync-tags "$@"
fi

exec zs-sim sync-tags "$@"
