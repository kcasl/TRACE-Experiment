#!/usr/bin/env bash
set -euo pipefail
ROOT=/data/jw_workspace/trace-alfred
cd "$ROOT"
PY=/data/jw_workspace/FILM-run/.venv/bin/python
export PYTHONUNBUFFERED=1
OUT="$ROOT/results/thor_port_v2_recheck"
mkdir -p "$OUT"
log="$OUT/recheck.log"
echo "START $(date) keys jev=${JEV_API_KEY:+set} openai=${OPENAI_API_KEY:+set}" | tee -a "$log"

echo "=== unseen baseline FILM+CAPEAM host/oracle ===" | tee -a "$log"
"$PY" -m experiments.run_thor_matrix --hosts film capeam --splits valid_unseen --sources host oracle --out "$OUT" 2>&1 | tee -a "$log"

echo "=== CAPEAM unseen Jev/OpenAI ===" | tee -a "$log"
"$PY" -m experiments.run_thor_matrix --hosts capeam --splits valid_unseen --sources jev openai_decisions --out "$OUT" 2>&1 | tee -a "$log"

echo "DONE $(date)" | tee -a "$log"
