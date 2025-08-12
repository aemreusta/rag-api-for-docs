#!/usr/bin/env bash
set -euo pipefail

# Activate conda env if available (local dev)
if command -v conda >/dev/null 2>&1 && conda env list | grep -q "reportEnv"; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate reportEnv || true
fi

export C_FORCE_ROOT=1

exec celery -A app.core.jobs.celery_app worker \
  --loglevel=INFO \
  --hostname=worker@%h


