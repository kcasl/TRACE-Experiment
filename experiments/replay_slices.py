"""Slice Metric-FF replay by action / j_true / N. No simulator.

    python -m experiments.replay_slices
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from trace.config import DEFERRAL_COST
from trace.consequence.ff import TERMINAL, costs_from_row
from trace.gate import decide, decision_utility, oracle_decision
from trace.types import Belief, normalize_mode

from .paths import REPLAY_SPLITS, ryankwon_root
from .replay import SELECT_N, load_beliefs, load_jsonl

ROOT = Path(__file__).resolve().parents[1]


def _truth(row: dict, n: int) -> int:
    return row["j_true"] if row.get("j_true") is not None else n


def inventory(costs_path: Path) -> dict:
    rows = [row for row in load_jsonl(costs_path) if "C" in row]
    by_action: dict[str, int] = defaultdict(int)
    by_n: dict[str, int] = defaultdict(int)
    n_ge2 = 0
    j_true_null_ge2 = 0
    j_true_in_ge2 = 0
    actions_ge2: dict[str, int] = defaultdict(int)
    tasks_ge2: dict[str, int] = defaultdict(int)
    n_hist: dict[int, int] = defaultdict(int)
    for row in rows:
        n = len(row["C"])
        by_action[row.get("action") or "?"] += 1
        bucket = "N=1" if n < 2 else ("N=2-8" if n <= 8 else "N>8")
        by_n[bucket] += 1
        if n < SELECT_N:
            continue
        n_ge2 += 1
        n_hist[n] += 1
        actions_ge2[row.get("action") or "?"] += 1
        tasks_ge2[row.get("task_type") or "?"] += 1
        if row.get("j_true") is None:
            j_true_null_ge2 += 1
        else:
            j_true_in_ge2 += 1
    return {
        "n_cost_rows": len(rows),
        "N_ge2": n_ge2,
        "j_true_in_candidates": j_true_in_ge2,
        "j_true_null": j_true_null_ge2,
        "j_true_null_rate": (j_true_null_ge2 / n_ge2) if n_ge2 else 0.0,
        "all_events_by_action": dict(sorted(by_action.items())),
        "all_events_by_N": dict(by_n),
        "N_ge2_by_action": dict(sorted(actions_ge2.items())),
        "N_ge2_by_task": dict(sorted(tasks_ge2.items())),
        "N_ge2_by_N": {str(k): v for k, v in sorted(n_hist.items())},
    }


def slice_trace(costs_path: Path, beliefs_dir: Path, *, source: str) -> dict:
    costs_rows = [row for row in load_jsonl(costs_path) if "C" in row]
    beliefs = load_beliefs(beliefs_dir)
    rename = {"appearance": "dinov2"}
    mapping = None
    for name, table in beliefs.items():
        if rename.get(name, name) == source or name == source:
            mapping = table
            break
    if mapping is None:
        return {"error": f"no beliefs for {source}"}

    buckets: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for row in costs_rows:
        try:
            matrix, reachable = costs_from_row(row)
        except (KeyError, ValueError):
            continue
        n = matrix.C.shape[0]
        if n < SELECT_N:
            continue
        packed = mapping.get(row["event_key"])
        if packed is None or "p" not in packed:
            continue
        p_identity = np.asarray(packed["p"], dtype=np.float64)
        if p_identity.size != n + 1:
            continue
        truth = _truth(row, n)
        relation = normalize_mode(row.get("relation") or "SAME")
        oracle = oracle_decision(matrix.C, reachable, truth, DEFERRAL_COST, TERMINAL)
        u_oracle = decision_utility(oracle, truth, matrix.C, reachable, DEFERRAL_COST, TERMINAL)
        decision = decide(
            Belief(p_identity=p_identity, source=source),
            matrix,
            mode=relation,
            deferral_cost=DEFERRAL_COST,
            rule="trace",
            scores=p_identity[:n].tolist(),
        )
        utility = decision_utility(decision, truth, matrix.C, reachable, DEFERRAL_COST, TERMINAL)
        if relation == "DIFFERENT":
            wrong = int(decision.action == "SELECT" and truth < n and decision.idx == truth)
        else:
            wrong = int(decision.action == "SELECT" and decision.idx != truth)
        keys = [
            "all",
            f"action={row.get('action') or '?'}",
            "j_true=" + ("null" if row.get("j_true") is None else "in_list"),
            "N<=8" if n <= 8 else "N>8",
        ]
        for key in keys:
            b = buckets[key]
            b["n"] += 1
            b["regret"] += utility - u_oracle
            b["wrong"] += wrong
            b["returned"] += int(decision.action == "RETURN")
            b["select"] += int(decision.action == "SELECT")
    out = {}
    for key, b in sorted(buckets.items()):
        n = max(b["n"], 1)
        out[key] = {
            "n": int(b["n"]),
            "CR": b["regret"] / n,
            "wrong": b["wrong"] / n,
            "return": b["returned"] / n,
            "select": b["select"] / n,
        }
    return out


def run() -> dict:
    root = ryankwon_root()
    report: dict = {"root": str(root), "deferral_cost": DEFERRAL_COST, "min_n": SELECT_N, "splits": {}}
    for spec in REPLAY_SPLITS:
        costs = root / spec["costs"]
        beliefs = root / spec["beliefs"]
        if not costs.exists() or not beliefs.exists():
            report["splits"][spec["name"]] = {
                "error": "missing logs",
                "costs": str(costs),
                "beliefs": str(beliefs),
            }
            continue
        report["splits"][spec["name"]] = {
            "inventory": inventory(costs),
            "jev_trace": slice_trace(costs, beliefs, source="jev"),
            "openai_trace": slice_trace(costs, beliefs, source="openai_decisions"),
            "oracle_trace": slice_trace(costs, beliefs, source="oracle"),
        }
    missing = {
        "capeam/valid_unseen": {
            "error": "missing logs",
            "costs": str(root / "results/offline/capeam/valid_unseen_host_v2_cand/costs.jsonl"),
            "beliefs": str(root / "results/beliefs/capeam/valid_unseen_host_v2"),
        }
    }
    report["splits"].update(missing)
    return report


def main() -> None:
    payload = run()
    out = ROOT / "results" / "replay_slices.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
