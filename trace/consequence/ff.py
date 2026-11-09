"""Metric-FF extra-cost matrices from host mask logs.

The live ALFRED PDDL critic (Metric-FF, then BFS) is the one that wrote
``costs.jsonl``. TRACE consumes those matrices so replay does not need the
planner binary. ``live_metric_ff()`` returns that critic when the host tree
and Metric-FF are on this machine.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Optional

import numpy as np

from ..types import CostMatrix

TERMINAL = 100.0


def costs_from_row(row: dict) -> tuple[CostMatrix, np.ndarray]:
    """``C`` as written by Metric-FF. ``None`` cells are unreachable (inf)."""

    raw = row["C"]
    costs = np.array([[np.inf if cell is None else cell for cell in line] for line in raw], dtype=np.float64)
    reachable = np.array(row["reachable"], dtype=bool)
    n = costs.shape[0]
    ids = tuple(row.get("cand_gt") or [f"c{i}" for i in range(n)])
    matrix = CostMatrix(
        C=costs,
        total=costs,
        base=np.zeros(n + 1),
        candidate_ids=ids,
        action=row.get("action") or "PutObject",
    )
    return matrix, reachable


def load_cost_log(path: Path) -> dict[str, dict]:
    import json

    rows = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if "C" in row and "event_key" in row:
                rows[row["event_key"]] = row
    return rows


def live_metric_ff(*, root: Optional[Path] = None) -> Any:
    """Host Metric-FF critic, or ``None`` if the binary / package is absent."""

    search = root
    if search is None:
        env = os.environ.get("RYANKWON_ROOT")
        search = Path(env) if env else None
    if search is None or not search.exists():
        return None
    inserted = str(search)
    if inserted not in sys.path:
        sys.path.insert(0, inserted)
    try:
        from cail.consequence.critic import ConsequenceCritic

        return ConsequenceCritic(planner="ff")
    except Exception:
        return None
