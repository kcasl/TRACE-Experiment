"""Replay TRACE on FF-critic cost matrices and logged beliefs.

The C(i,j) files were produced by ryankwon's Metric-FF critic on the host
pre-states. Beliefs are Jev, OpenAI Decisions, DINOv2 appearance, and GT-ID
oracle from the same mask events. OpenAI images are rebuilt from the Jev
bbox/overlay log. No simulator is started.

    python -m experiments.replay
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

SELECT_N = 2  # paper Table 2: select-eligible events, N >= 2


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_beliefs(directory: Path) -> dict[str, dict[str, dict]]:
    by_source: dict[str, dict[str, dict]] = {}
    for path in directory.glob("*.jsonl"):
        source = path.stem
        mapping = {}
        for row in load_jsonl(path):
            if "p" in row and "event_key" in row:
                mapping[row["event_key"]] = row
        by_source[source] = mapping
    return by_source


def replay_split(costs_path: Path, beliefs_dir: Path, *, deferral: float = DEFERRAL_COST, min_n: int = SELECT_N) -> dict:
    costs_rows = [row for row in load_jsonl(costs_path) if "C" in row]
    beliefs = load_beliefs(beliefs_dir)
    rename = {"appearance": "dinov2"}
    aggregate: dict[tuple[str, str], dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for row in costs_rows:
        try:
            matrix, reachable = costs_from_row(row)
        except (KeyError, ValueError):
            continue
        costs = matrix.C
        n = costs.shape[0]
        if n < min_n:
            continue
        truth = row["j_true"] if row.get("j_true") is not None else n
        relation = normalize_mode(row.get("relation") or "SAME")
        oracle = oracle_decision(costs, reachable, truth, deferral, TERMINAL)
        u_oracle = decision_utility(oracle, truth, costs, reachable, deferral, TERMINAL)
        for source, mapping in beliefs.items():
            packed = mapping.get(row["event_key"])
            if packed is None:
                continue
            p_identity = np.asarray(packed["p"], dtype=np.float64)
            if p_identity.size != n + 1:
                continue
            label = rename.get(source, source)
            belief = Belief(p_identity=p_identity, source=label)
            scores = p_identity[:n].tolist()
            host_idx = row.get("host_choice")
            rules = {
                "trace": dict(rule="trace"),
                "threshold": dict(rule="threshold", tau=0.5),
                "force_best": dict(rule="force_best"),
            }
            if host_idx is not None and 0 <= int(host_idx) < n:
                rules["host_score"] = dict(rule="host_score")
            for name, kwargs in rules.items():
                extra_scores = scores
                if name == "host_score":
                    extra_scores = [1.0 if i == int(host_idx) else 0.0 for i in range(n)]
                decision = decide(
                    belief,
                    matrix,
                    mode=relation,
                    deferral_cost=deferral,
                    scores=extra_scores,
                    **kwargs,
                )
                utility = decision_utility(decision, truth, costs, reachable, deferral, TERMINAL)
                if relation == "DIFFERENT":
                    wrong = int(decision.action == "SELECT" and truth < n and decision.idx == truth)
                else:
                    wrong = int(decision.action == "SELECT" and decision.idx != truth)
                returned = int(decision.action == "RETURN")
                unrec = int(
                    decision.action == "SELECT"
                    and decision.idx is not None
                    and not bool(reachable[decision.idx, truth])
                )
                bucket = aggregate[(label, name)]
                bucket["n"] += 1
                bucket["regret"] += utility - u_oracle
                bucket["wrong"] += wrong
                bucket["returned"] += returned
                bucket["unrecoverable"] += unrec
    summary = {}
    for (source, rule), bucket in sorted(aggregate.items()):
        n = max(bucket["n"], 1)
        summary[f"{source}|{rule}"] = {
            "n": int(bucket["n"]),
            "commitment_regret": bucket["regret"] / n,
            "wrong_instance_rate": bucket["wrong"] / n,
            "return_rate": bucket["returned"] / n,
            "unrecoverable_rate": bucket["unrecoverable"] / n,
            "critic": "Metric-FF (precomputed C)",
        }
    return summary


def run(*, root: Path | None = None, deferral: float = DEFERRAL_COST) -> dict:
    root = root or ryankwon_root()
    report = {"root": str(root), "deferral_cost": deferral, "min_n": SELECT_N, "splits": {}}
    for spec in REPLAY_SPLITS:
        costs = root / spec["costs"]
        beliefs = root / spec["beliefs"]
        if not costs.exists() or not beliefs.exists():
            report["splits"][spec["name"]] = {"error": "missing logs", "costs": str(costs), "beliefs": str(beliefs)}
            continue
        report["splits"][spec["name"]] = {
            "n_episodes": spec["n_episodes"],
            "table": replay_split(costs, beliefs, deferral=deferral),
        }
    return report


def main() -> None:
    payload = run()
    out = Path(__file__).resolve().parents[1] / "results" / "log_replay.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
