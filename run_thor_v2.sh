#!/usr/bin/env bash
set -euo pipefail
ROOT=/data/jw_workspace/trace-alfred
cd "$ROOT"
PY=/data/jw_workspace/FILM-run/.venv/bin/python
export PYTHONUNBUFFERED=1
mkdir -p results/thor_port_v2
log=results/thor_port_v2/matrix.log
{
  echo "START $(date) keys jev=${JEV_API_KEY:+set} openai=${OPENAI_API_KEY:+set}"
  echo "=== baseline host oracle ==="
} | tee -a "$log"
"$PY" -m experiments.run_thor_matrix --sources host oracle 2>&1 | tee -a "$log"
{
  echo "=== LIVE jev openai ==="
} | tee -a "$log"
"$PY" -m experiments.run_thor_matrix --sources jev openai_decisions 2>&1 | tee -a "$log"
echo "DONE $(date)" | tee -a "$log"
