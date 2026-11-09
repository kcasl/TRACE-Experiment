#!/usr/bin/env bash
# TRACE full-oct21 rerun with live pickup_not_ref on Diff pickups.
# Keys must already be in the environment. Do not write .env.
set -euo pipefail
ROOT=/data/jw_workspace/trace-alfred
cd "$ROOT"
PY=/data/jw_workspace/FILM-run/.venv/bin/python
export PYTHONUNBUFFERED=1
OUT="$ROOT/results/thor_port_v3_notref"
mkdir -p "$OUT"

echo "START $(date -Is) keys jev=${JEV_API_KEY:+set} openai=${OPENAI_API_KEY:+set}"
"$PY" - <<'PY'
from experiments.thor_rollout import live_goal_template, TRACE_ACTIONS
from trace.types import Relation, MemoryEntry
rel = Relation("DIFFERENT", MemoryEntry(key="x", cls="Apple", action="PickupObject", t=1, role="object"), "pick_two")
kind = live_goal_template("PickupObject", rel, holding=None, task_cls="Apple").kind
assert kind == "pickup_not_ref", kind
assert TRACE_ACTIONS == frozenset({"PickupObject", "PutObject"})
print("live_goal_template Diff pickup ->", kind)
print("TRACE_ACTIONS", sorted(TRACE_ACTIONS))
PY

run_one() {
  local gpu="$1"
  local host="$2"
  local log="$OUT/${host}.log"
  local pidf="$OUT/${host}.pid"
  echo "LAUNCH host=$host gpu=$gpu -> $log"
  nohup env CUDA_VISIBLE_DEVICES="$gpu" "$PY" -m experiments.run_thor_matrix \
    --hosts "$host" \
    --splits valid_seen valid_unseen \
    --sources jev openai_decisions \
    --out "$OUT" \
    >"$log" 2>&1 &
  echo $! >"$pidf"
  echo "PID $(cat "$pidf") host=$host"
}

run_one 0 film
run_one 1 capeam
sleep 2
echo "PIDS film=$(cat "$OUT/film.pid") capeam=$(cat "$OUT/capeam.pid")"
ps -p "$(cat "$OUT/film.pid")" -p "$(cat "$OUT/capeam.pid")" -o pid,etime,cmd || true
tail -n 20 "$OUT/film.log" || true
tail -n 20 "$OUT/capeam.log" || true
