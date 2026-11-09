"""Shared reading of choice answers from Jev and OpenAI Decisions."""

from __future__ import annotations

from typing import Optional

import numpy as np

from .base import normalize
from ..types import Belief


def option_label(index: int) -> str:
    """0 -> A, 1 -> B, 25 -> Z, 26 -> AA."""

    if index < 0:
        raise ValueError("option index must be non-negative")
    label = ""
    value = index + 1
    while value:
        value, remainder = divmod(value - 1, 26)
        label = chr(65 + remainder) + label
    return label


def find_answer(payload: dict, name: str) -> dict:
    answers = payload.get("answers")
    if isinstance(answers, dict):
        try:
            answer = answers[name]
        except KeyError as exc:
            raise KeyError(f"response has no answer {name!r}") from exc
        if not isinstance(answer, dict):
            raise TypeError(f"answer {name!r} is not an object")
        return answer
    if isinstance(answers, list):
        for answer in answers:
            if isinstance(answer, dict) and answer.get("name") == name:
                return answer
        raise KeyError(f"response has no answer {name!r}")
    raise TypeError("response.answers must be an object or a list")


def probability_map(answer: dict) -> dict[str, float]:
    raw = answer.get("probabilities")
    if isinstance(raw, dict):
        return {str(key): float(value) for key, value in raw.items()}
    if isinstance(raw, list):
        mapped = {}
        for item in raw:
            mapped[str(item["value"])] = float(item["probability"])
        return mapped
    return {}


def belief_from_choice(answer: dict, labels: list[str], *, source: str, raw: Optional[dict] = None) -> Belief:
    """Align a choice answer onto ``[p(label_0), ..., p(label_n), p(none)]``."""

    if answer.get("type") == "refusal":
        width = len(labels) + 1
        return Belief(
            p_identity=np.full(width, 1.0 / width),
            source=source,
            confidence=None,
            refusal=True,
            raw={**(raw or {}), "refusal": True},
        )
    mapped = probability_map(answer)
    if not mapped and answer.get("choice") is not None:
        mapped = {str(answer["choice"]): 1.0}
    values = [mapped.get(label, 0.0) for label in labels]
    values.append(mapped.get("none", 0.0))
    confidence = answer.get("confidence")
    return Belief(
        p_identity=normalize(np.asarray(values, dtype=np.float64)),
        source=source,
        confidence=None if confidence is None else float(confidence),
        refusal=False,
        raw=raw or {},
    )
