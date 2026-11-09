"""Select a mask or return control.

TRACE commits to mask i only when its expected extra cost is strictly smaller
than the cost of handing control back:

    R(i) = sum_j C(i, j) p_identity(j)
    select argmin_i R(i)    if that value is a unique minimum and < c_defer
    return control          otherwise

Equality defers, both against ``c_defer`` and among masks. A deferred step
keeps the option to look again; committing at the same expected cost cannot.
Tied minima used to break by array order (``argmin`` = 0) and overrode a
correct host Put on FloorPlan424 cabinets.

``p_identity`` is P(this candidate is the reference), aligned with the columns
of C. Threshold and force-best rules instead use pick probabilities from
``same_to_pick``, because those rules ask "how sure is the chosen mask?"
rather than "what does a mistake cost?".
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from .belief.base import normalize, same_to_pick
from .types import Belief, CostMatrix, GateDecision

RULES = frozenset({"trace", "threshold", "force_best", "host_score"})


def expected_regret(costs: np.ndarray, p_identity: np.ndarray) -> np.ndarray:
    probability = normalize(p_identity)
    matrix = np.asarray(costs, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] != probability.size:
        raise ValueError(
            f"cost columns ({matrix.shape[1] if matrix.ndim == 2 else '?'}) "
            f"!= identity length ({probability.size})"
        )
    # 0 * inf is nan; unreachable cells must still contribute 0 when p_j = 0.
    finite = np.where(np.isfinite(matrix), matrix, 1e6)
    return finite @ probability


def decide(
    belief: Belief,
    costs: CostMatrix,
    *,
    mode: str,
    deferral_cost: float,
    rule: str = "trace",
    tau: float = 0.75,
    scores: Optional[Sequence[float]] = None,
) -> GateDecision:
    if rule not in RULES:
        raise ValueError(f"unknown decision rule {rule!r}")
    if deferral_cost < 0:
        raise ValueError("deferral_cost must be non-negative")
    if belief.refusal and rule != "host_score":
        width = costs.C.shape[0]
        return GateDecision("RETURN", None, np.full(width, np.nan), rule, "belief_refusal")

    regret = expected_regret(costs.C, belief.p_identity)
    if rule == "trace":
        index = int(np.argmin(regret))
        min_r = float(regret[index])
        if min_r >= deferral_cost:
            return GateDecision("RETURN", None, regret, rule, "regret_not_below_deferral")
        n_ties = int(np.count_nonzero(np.isclose(regret, min_r, rtol=0.0, atol=1e-8)))
        if n_ties > 1:
            return GateDecision("RETURN", None, regret, rule, "regret_tied")
        return GateDecision("SELECT", index, regret, rule, "regret_below_deferral")

    if rule == "host_score":
        if not scores or len(scores) != costs.C.shape[0]:
            raise ValueError("host_score requires one detector score per candidate")
        index = int(np.argmax(np.asarray(scores, dtype=np.float64)))
        return GateDecision("SELECT", index, regret, rule, "max_detector_score")

    pick = same_to_pick(belief.p_identity, mode)
    candidate_pick = pick[:-1]
    if candidate_pick.size == 0:
        return GateDecision("RETURN", None, regret, rule, "no_candidate")
    index = int(np.argmax(candidate_pick))
    if rule == "force_best":
        return GateDecision("SELECT", index, regret, rule, "max_pick_probability")
    if rule == "threshold":
        # A candidate must beat both the threshold and the "none of these" mass.
        # Otherwise the confident answer is to leave the mask list alone.
        confident = float(candidate_pick[index])
        if confident >= tau and confident >= float(pick[-1]):
            return GateDecision("SELECT", index, regret, rule, "confidence_above_tau")
        return GateDecision("RETURN", None, regret, rule, "confidence_below_tau")
    raise AssertionError(rule)


def decision_utility(
    decision: GateDecision,
    truth_j: int,
    costs: np.ndarray,
    reachable: np.ndarray,
    deferral_cost: float,
    terminal_cost: float,
) -> float:
    """Realized extra cost given the true identity column. Used for log replay."""

    if decision.action == "RETURN" or decision.idx is None:
        return float(deferral_cost)
    index = decision.idx
    if not bool(reachable[index, truth_j]):
        return float(terminal_cost)
    value = costs[index, truth_j]
    if not np.isfinite(value):
        return float(terminal_cost)
    return float(value)


def oracle_decision(
    costs: np.ndarray,
    reachable: np.ndarray,
    truth_j: int,
    deferral_cost: float,
    terminal_cost: float,
) -> GateDecision:
    """Best decision with the true identity known."""

    n = costs.shape[0]
    values = np.array(
        [
            terminal_cost if not bool(reachable[index, truth_j]) or not np.isfinite(costs[index, truth_j]) else float(costs[index, truth_j])
            for index in range(n)
        ],
        dtype=np.float64,
    )
    index = int(np.argmin(values))
    if values[index] < deferral_cost:
        return GateDecision("SELECT", index, values, "oracle", "oracle_select")
    return GateDecision("RETURN", None, values, "oracle", "oracle_return")
