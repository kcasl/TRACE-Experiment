"""Jev (TypeSafe System One) as an identity belief source.

One choice question per interaction. The options are the labelled candidates
plus ``none``. Images are experimental on this API: a request that contains
them sends ``TypeSafe-Danger-Zone-Images: true``. The key is read from
``JEV_API_KEY`` and is never written into the cached request.
"""

from __future__ import annotations

import json
import os
import uuid
from typing import Optional

import numpy as np

from ..types import Candidate, Observation, Relation, require_positions
from .http import post_bytes, post_json
from .images import png_bytes
from .parse import belief_from_choice, find_answer, option_label
from .prompt import attach_reference_images, identity_instructions

JEV_URL = "https://api.typesafe.ai/v1/systemone"
IMAGE_HEADER = "TypeSafe-Danger-Zone-Images"
NONE_KEY = "none"


class JevSource:
    name = "jev"

    def __init__(
        self,
        *,
        model: str = "jev-latest",
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
        images = _collect_images(observation, relation)
        body, placeholders = _body(self.model, relation, candidates, observation, labels, images)
        payload = _post(JEV_URL, body, placeholders, images, self._headers(bool(images)), self.timeout, self.transport)
        answer = find_answer(payload, "identity")
        belief = belief_from_choice(
            answer,
            labels,
            source=self.name,
            raw={"model": payload.get("model", self.model), "usage": payload.get("usage"), "images": bool(images)},
        )
        return belief

    def _headers(self, with_images: bool) -> dict[str, str]:
        key = self.api_key or os.environ.get("JEV_API_KEY") or os.environ.get("TYPESAFE_API_KEY")
        if not key:
            raise RuntimeError("JEV_API_KEY is not set")
        headers = {"Authorization": f"Bearer {key}"}
        if with_images:
            headers[IMAGE_HEADER] = "true"
        return headers


def _collect_images(observation: Observation, relation: Relation) -> dict[str, np.ndarray]:
    images = {}
    if attach_reference_images(relation, observation):
        images["reference"] = observation.reference_frame
    if observation.frame is not None:
        images["current"] = observation.frame
    return images


def _body(model, relation, candidates, observation, labels, images):
    token = uuid.uuid4().hex[:12]
    placeholders = {}
    assets = {}
    for name in images:
        placeholder = f"img-{token}-{name}"
        placeholders[name] = placeholder
        assets[placeholder] = {"type": "image", "part": placeholder}
    criteria = {}
    described = []
    for label, candidate in zip(labels, candidates):
        text = (
            f"Candidate {label} is the same physical {candidate.cls} as the reference. "
            f"Detector score {candidate.score:.3f}"
        )
        if candidate.dist is not None:
            text += f", distance {candidate.dist:.2f} m"
        criteria[label] = text
        described.append({"label": label, "object_class": candidate.cls, "detector_score": candidate.score, "room": candidate.room})
    criteria[NONE_KEY] = "None of the labelled candidates is the reference object."
    state = {
        "task": observation.goal_text,
        "subgoal": observation.subgoal_text,
        "task_type": observation.task_type,
        "history": list(observation.history),
        "relation": relation.mode,
        "reference_interaction": None if relation.ref is None else relation.ref.action,
        "reference_class": None if relation.ref is None else relation.ref.cls,
        "candidates": described,
    }
    if "reference" in placeholders:
        state["reference_image"] = placeholders["reference"]
    if "current" in placeholders:
        state["current_image"] = placeholders["current"]
    body = {
        "model": model,
        "state": state,
        "questions": {
            "identity": {
                "type": "choice",
                "instructions": identity_instructions(relation),
                "criteria": criteria,
            }
        },
    }
    if assets:
        body["assets"] = assets
    return body, placeholders


def _post(url, body, placeholders, images, headers, timeout, transport) -> dict:
    if not images:
        return post_json(url, body, headers, timeout=timeout, transport=transport)
    encoded, boundary = _multipart(body, {placeholders[name]: png_bytes(frame) for name, frame in images.items()})
    multipart_headers = {**headers, "Content-Type": f"multipart/form-data; boundary={boundary}"}
    return post_bytes(url, encoded, multipart_headers, timeout=timeout, transport=transport)


def _multipart(body: dict, files: dict[str, bytes]) -> tuple[bytes, str]:
    boundary = "----trace" + uuid.uuid4().hex
    chunks: list[bytes] = []

    def part(name: str, content: bytes, content_type: str, filename: Optional[str] = None) -> None:
        disposition = f'form-data; name="{name}"'
        if filename:
            disposition += f'; filename="{filename}"'
        head = (
            f"--{boundary}\r\nContent-Disposition: {disposition}\r\nContent-Type: {content_type}\r\n\r\n"
        ).encode("utf-8")
        chunks.append(head + content + b"\r\n")

    part("request", json.dumps(body).encode("utf-8"), "application/json")
    for name, payload in files.items():
        part(name, payload, "image/png", f"{name}.png")
    chunks.append(f"--{boundary}--\r\n".encode("utf-8"))
    return b"".join(chunks), boundary
