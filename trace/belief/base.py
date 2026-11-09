"""Identity beliefs and the conversion to pick probabilities.

Decision models are asked which candidate *is the reference*. The gate's
regret uses that distribution directly. Pick probabilities are a derived
view for confidence thresholds:

* SAME: the correct mask is the reference, so the two distributions match.
* DIFFERENT: the correct mask is a candidate that is not the reference.
* NONE: identity is irrelevant; callers should not commit on it.
"""

from __future__ import annotations

from typing import Optional, Protocol

import numpy as np

from ..types import Belief, Candidate, Observation, Relation

_EPS = 1e-12


def normalize(probabilities: np.ndarray) -> np.ndarray:
    values = np.asarray(probabilities, dtype=np.float64).clip(min=0.0)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("probabilities must be a non-empty vector")
    total = float(values.sum())
    if total <= _EPS:
        return np.full(values.shape, 1.0 / values.size, dtype=np.float64)
    return values / total


def same_to_pick(p_identity: np.ndarray, mode: str) -> np.ndarray:
    """Map P(candidate is the reference) onto P(picking the candidate is correct)."""

    p_identity = normalize(p_identity)
    n = p_identity.size - 1
    if mode == "SAME":
        return p_identity
    if mode == "DIFFERENT":
        if n == 0:
            return np.array([1.0], dtype=np.float64)
        weights = 1.0 - p_identity[:n]
        # A single visible candidate that might be the forbidden reference
        # leaves the correct object unseen. That mass stays on "none".
        none = float(p_identity[0]) if n == 1 else 0.0
        return normalize(np.concatenate([weights, [none]]))
    if n == 0:
        return np.array([1.0], dtype=np.float64)
    return np.concatenate([np.full(n, 1.0 / n), [0.0]])


class BeliefSource(Protocol):
    name: str

    def identify(self, relation: Relation, candidates: list[Candidate], observation: Observation) -> Belief:
        """Return an identity distribution of length ``len(candidates) + 1``."""


class FixedIdentity:
    """Test double. ``p_identity`` is copied on every call."""

    name = "fixed"

    def __init__(self, p_identity: np.ndarray, *, refusal: bool = False, confidence: Optional[float] = None):
        self._p = normalize(np.asarray(p_identity, dtype=np.float64))
        self._refusal = refusal
        self._confidence = confidence

    def identify(self, relation: Relation, candidates: list[Candidate], observation: Observation) -> Belief:
        if self._p.size != len(candidates) + 1:
            raise ValueError(
                f"fixed identity length {self._p.size} != N+1 ({len(candidates) + 1})"
            )
        return Belief(
            p_identity=self._p.copy(),
            source=self.name,
            confidence=self._confidence,
            refusal=self._refusal,
            raw={"mode": relation.mode},
        )
