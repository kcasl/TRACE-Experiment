"""Exact expected-cost comparison. No network and no simulator."""

from __future__ import annotations

from typing import Optional

import numpy as np

from trace.config import DEFERRAL_COST
from trace.gate import decide
from trace.types import Belief, CostMatrix

RULES = ("trace", "threshold_055", "threshold_075", "host_score")


def binary_cost(wrong: float) -> np.ndarray:
    """Two candidates. Acting on the reference costs 0; every mistake costs ``wrong``."""

    return np.array(
        [
            [0.0, wrong, wrong],
            [wrong, 0.0, wrong],
        ],
        dtype=np.float64,
    )


def _matrix(wrong: float) -> CostMatrix:
    costs = binary_cost(wrong)
    return CostMatrix(C=costs, total=costs, base=np.zeros(3), candidate_ids=("a", "b"), action="PutObject")


def realized_expected_cost(
    decision_action: str,
    decision_index: Optional[int],
    costs: np.ndarray,
    truth: np.ndarray,
    deferral_cost: float,
) -> float:
    """Expected extra cost when the true column is drawn from ``truth``."""

    if decision_action == "RETURN":
        return float(deferral_cost)
    if decision_index is None:
        raise ValueError("a selection needs an index")
    return float(costs[decision_index] @ truth)


def heterogeneous_table(*, deferral_cost: float = DEFERRAL_COST) -> dict:
    """Same identity belief, two mistake costs, three commitment rules.

    The belief puts probability 0.6 on candidate 0. A threshold of 0.55 therefore
    always commits, and a threshold of 0.75 always defers. TRACE commits only in
    the cheap world. The host's detector prefers candidate 1, which is usually
    the wrong object under this belief.
    """

    belief = Belief(p_identity=np.array([0.6, 0.4, 0.0]), source="example", confidence=0.6)
    truth = belief.p_identity
    rows = []
    for name, wrong in (("cheap", 2.0), ("expensive", 20.0)):
        costs = _matrix(wrong)
        decisions = {
            "trace": decide(belief, costs, mode="SAME", deferral_cost=deferral_cost, rule="trace"),
            "threshold_055": decide(belief, costs, mode="SAME", deferral_cost=deferral_cost, rule="threshold", tau=0.55),
            "threshold_075": decide(belief, costs, mode="SAME", deferral_cost=deferral_cost, rule="threshold", tau=0.75),
            "host_score": decide(
                belief, costs, mode="SAME", deferral_cost=deferral_cost, rule="host_score", scores=[0.4, 0.9]
            ),
        }
        for rule, decision in decisions.items():
            rows.append(
                {
                    "world": name,
                    "wrong_cost": wrong,
                    "rule": rule,
                    "action": decision.action,
                    "index": decision.idx,
                    "min_expected_regret": round(float(np.min(decision.expected_regret)), 4),
                    "realized_expected_cost": round(
                        realized_expected_cost(decision.action, decision.idx, costs.C, truth, deferral_cost),
                        4,
                    ),
                }
            )
    means = {}
    for rule in RULES:
        chosen = [row["realized_expected_cost"] for row in rows if row["rule"] == rule]
        means[rule] = round(sum(chosen) / len(chosen), 4)
    return {"deferral_cost": deferral_cost, "belief": truth.tolist(), "rows": rows, "mean_realized_cost": means}
