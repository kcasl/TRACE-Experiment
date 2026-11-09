"""OpenAI Decisions as an identity belief source.

The question is the same choice the Jev client asks: which labelled candidate
is the reference, or ``none``. Images ride in the shared ``input`` as
``input_image`` data URLs. The key is read from ``OPENAI_API_KEY``.
"""

from __future__ import annotations

import base64
import os
from typing import Optional

from ..types import Candidate, Observation, Relation, require_positions
from .http import post_json
from .images import png_bytes
from .parse import belief_from_choice, find_answer, option_label
from .prompt import attach_reference_images, identity_instructions

DECISIONS_URL = "https://api.openai.com/v1/decisions"
NONE_KEY = "none"


class OpenAIDecisionSource:
    name = "openai_decisions"

    def __init__(
        self,
        *,
        model: str = "gpt-6-luna",
        api_key: Optional[str] = None,
        timeout: float = 90.0,
        transport=None,
    ):
        self.model = model
        self.api_key = api_key
        self.timeout = timeout
        self.transport = transport

    def identify(self, relation: Relation, candidates: list[Candidate], observation: Observation):
        require_positions(candidates)
        labels = [option_label(candidate.idx) for candidate in candidates]
        payload = post_json(
            DECISIONS_URL,
            _body(self.model, relation, candidates, observation, labels),
            self._headers(),
            timeout=self.timeout,
            transport=self.transport,
        )
        answer = find_answer(payload, "identity")
        return belief_from_choice(
            answer,
            labels,
            source=self.name,
            raw={"model": payload.get("model", self.model), "usage": payload.get("usage")},
        )

    def _headers(self) -> dict[str, str]:
        key = self.api_key or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise RuntimeError("OPENAI_API_KEY is not set")
        return {"Authorization": f"Bearer {key}"}


def _body(model, relation: Relation, candidates: list[Candidate], observation: Observation, labels: list[str]) -> dict:
    lines = [
        f"Relation required by the task: {relation.mode}.",
        f"Goal: {observation.goal_text}" if observation.goal_text else "",
        f"Subgoal: {observation.subgoal_text}" if observation.subgoal_text else "",
        "History: " + " | ".join(observation.history) if observation.history else "",
        identity_instructions(relation),
    ]
    if observation.notes:
        lines.append(observation.notes)
    content: list[dict] = [{"type": "input_text", "text": "\n".join(line for line in lines if line)}]
    if attach_reference_images(relation, observation):
        content.append({"type": "input_text", "text": "Reference object crop, from the earlier interaction:"})
        content.append({"type": "input_image", "image_url": _data_url(observation.reference_frame)})
        if observation.context_frame is not None:
            content.append({"type": "input_text", "text": "Reference context crop (box expanded 50%):"})
            content.append({"type": "input_image", "image_url": _data_url(observation.context_frame)})
    if observation.frame is not None:
        content.append({"type": "input_text", "text": "Current view with candidates labelled A, B, ...:"})
        content.append({"type": "input_image", "image_url": _data_url(observation.frame)})
    choices = []
    for label, candidate in zip(labels, candidates):
        description = (
            f"Candidate {label}, class {candidate.cls}, detector score {candidate.score:.3f}, room {candidate.room}."
        )
        choices.append({"value": label, "description": description})
    choices.append({"value": NONE_KEY, "description": "The reference object is not among the labelled candidates."})
    return {
        "model": model,
        "input": [{"role": "user", "content": content}],
        "questions": [
            {
                "type": "choice",
                "name": "identity",
                "instructions": identity_instructions(relation),
                "choices": choices,
            }
        ],
    }


def _data_url(frame) -> str:
    encoded = base64.b64encode(png_bytes(frame)).decode("ascii")
    return f"data:image/png;base64,{encoded}"
