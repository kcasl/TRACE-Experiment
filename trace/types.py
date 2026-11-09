"""Shared contracts.

Belief columns are identity hypotheses, not pick-correctness.
``p_identity[j]`` for ``j < N`` is P(candidate j is the reference object).
``p_identity[N]`` is P(the reference is not among the candidates).
The critic's cost matrix uses the same column order. Mixing this with
pick probabilities inverts DIFFERENT decisions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional

import numpy as np

Mode = Literal["SAME", "DIFFERENT", "NONE"]
GateAction = Literal["SELECT", "RETURN"]


def normalize_mode(mode: str) -> Mode:
    key = mode.strip().upper()
    if key == "DIFF":
        key = "DIFFERENT"
    if key not in ("SAME", "DIFFERENT", "NONE"):
        raise ValueError(f"unknown relation mode {mode!r}")
    return key  # type: ignore[return-value]


def require_positions(candidates: list[Candidate]) -> None:
    """``idx`` is the position in the mask list. Belief row i must be that mask."""

    for position, candidate in enumerate(candidates):
        if candidate.idx != position:
            raise ValueError(f"candidate {candidate.object_id!r} has idx {candidate.idx}, expected {position}")


@dataclass
class Candidate:
    """One same-class mask the host is about to choose among."""

    idx: int
    object_id: str
    cls: str
    score: float = 0.0
    room: str = "here"
    bbox: Optional[tuple[int, int, int, int]] = None
    dist: Optional[float] = None
    crop: Optional[np.ndarray] = None


@dataclass
class MemoryEntry:
    key: str
    cls: str
    action: str
    t: int
    role: Literal["object", "receptacle"] = "object"
    receptacle_cls: Optional[str] = None


@dataclass
class Relation:
    mode: Mode
    ref: Optional[MemoryEntry]
    rule: str


@dataclass
class Observation:
    """Multimodal evidence shown to a decision model. Pixels are optional."""

    frame: Optional[np.ndarray] = None
    reference_frame: Optional[np.ndarray] = None
    context_frame: Optional[np.ndarray] = None
    goal_text: str = ""
    subgoal_text: str = ""
    history: tuple[str, ...] = ()
    task_type: str = ""
    notes: str = ""


@dataclass
class Belief:
    """Identity distribution. Length is always N + 1."""

    p_identity: np.ndarray
    source: str
    confidence: Optional[float] = None
    refusal: bool = False
    raw: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.p_identity = np.asarray(self.p_identity, dtype=np.float64)
        if self.p_identity.ndim != 1 or self.p_identity.size < 1:
            raise ValueError("p_identity must be a non-empty 1-D distribution")

    @property
    def n_candidates(self) -> int:
        return int(self.p_identity.size - 1)


@dataclass
class CostMatrix:
    """``C[i, j]`` is the extra remaining cost of acting on i when the reference is j.

    Column ``N`` is the hypothesis that the reference is absent.
    ``total[i, j]`` is the absolute remaining cost, including the committed action.
    ``base[j]`` is the best absolute cost under hypothesis j, including doing nothing.
    """

    C: np.ndarray
    total: np.ndarray
    base: np.ndarray
    candidate_ids: tuple[str, ...]
    action: str

    def __post_init__(self) -> None:
        self.C = np.asarray(self.C, dtype=np.float64)
        self.total = np.asarray(self.total, dtype=np.float64)
        self.base = np.asarray(self.base, dtype=np.float64)
        n = len(self.candidate_ids)
        if self.C.shape != (n, n + 1):
            raise ValueError(f"C shape {self.C.shape} does not match N={n}")


@dataclass
class GateDecision:
    action: GateAction
    idx: Optional[int]
    expected_regret: np.ndarray
    rule: str
    reason: str


@dataclass
class MaskDecision:
    """What the host should do with its current mask list."""

    defer: bool
    index: Optional[int]
    rule: str
    relation: Mode
    relation_rule: str
    expected_regret: tuple[float, ...]
    p_identity: tuple[float, ...]
    reason: str
    source: str
