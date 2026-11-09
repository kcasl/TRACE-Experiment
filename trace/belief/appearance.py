"""DINOv2 (fallback CLIP) appearance identity belief.

p_identity[:N] = (1 - p_none) * softmax(cosine / temperature)
p_none         = sigmoid((threshold - max cosine) / none_scale)

The backbone is loaded lazily. Tests inject ``embedder``.
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np

from ..types import Belief, Candidate, Observation, Relation, require_positions
from .base import normalize

Embedder = Callable[[list[np.ndarray]], np.ndarray]


def softmax(values: np.ndarray, temperature: float) -> np.ndarray:
    scaled = np.asarray(values, dtype=np.float64) / max(temperature, 1e-8)
    shifted = scaled - scaled.max()
    exp = np.exp(shifted)
    return exp / exp.sum()


def sigmoid(value: float) -> float:
    if value >= 0:
        z = np.exp(-value)
        return float(1.0 / (1.0 + z))
    z = np.exp(value)
    return float(z / (1.0 + z))


def load_embedder(backbone: str = "dinov2", device: Optional[str] = None) -> Embedder:
    if backbone == "dinov2":
        try:
            return load_dinov2(device=device)
        except Exception:
            return load_clip(device=device)
    if backbone == "clip":
        return load_clip(device=device)
    raise ValueError(f"unknown backbone {backbone!r}")


def load_dinov2(model_name: str = "dinov2_vits14", device: Optional[str] = None) -> Embedder:
    import torch

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = torch.hub.load("facebookresearch/dinov2", model_name, verbose=False).to(device).eval()
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    @torch.no_grad()
    def embed(images: list[np.ndarray]) -> np.ndarray:
        batch = torch.stack([_to_tensor(image, 224) for image in images]).to(device)
        features = model((batch - mean) / std)
        return torch.nn.functional.normalize(features, dim=-1).cpu().numpy()

    embed.backend = f"dinov2/{model_name}"  # type: ignore[attr-defined]
    return embed


def load_clip(model_name: str = "ViT-B-32", pretrained: str = "openai", device: Optional[str] = None) -> Embedder:
    import open_clip
    import torch
    from PIL import Image

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
    model = model.to(device).eval()

    @torch.no_grad()
    def embed(images: list[np.ndarray]) -> np.ndarray:
        batch = torch.stack([preprocess(Image.fromarray(np.uint8(image))) for image in images]).to(device)
        features = model.encode_image(batch)
        return torch.nn.functional.normalize(features.float(), dim=-1).cpu().numpy()

    embed.backend = f"clip/{model_name}/{pretrained}"  # type: ignore[attr-defined]
    return embed


def _to_tensor(image: np.ndarray, size: int):
    import torch
    from PIL import Image

    pil = Image.fromarray(np.uint8(image)).convert("RGB").resize((size, size), Image.BICUBIC)
    array = np.asarray(pil, dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1)


class AppearanceSource:
    """Cosine identity over the reference crop and each candidate crop."""

    name = "dinov2"

    def __init__(
        self,
        *,
        temperature: float = 0.05,
        threshold: float = 0.6,
        none_scale: float = 0.1,
        embedder: Optional[Embedder] = None,
    ):
        self.temperature = temperature
        self.threshold = threshold
        self.none_scale = none_scale
        self._embedder = embedder

    @property
    def embedder(self) -> Embedder:
        if self._embedder is None:
            self._embedder = load_embedder()
        return self._embedder

    def identify(self, relation: Relation, candidates: list[Candidate], observation: Observation) -> Belief:
        require_positions(candidates)
        reference = observation.reference_frame
        if reference is None:
            raise ValueError("appearance belief needs observation.reference_frame")
        crops = []
        for candidate in candidates:
            if candidate.crop is None:
                raise ValueError(f"candidate {candidate.object_id!r} has no crop")
            crops.append(candidate.crop)
        features = self.embedder([reference, *crops])
        similarities = features[1:] @ features[0]
        p_none = sigmoid((self.threshold - float(similarities.max())) / self.none_scale)
        body = softmax(similarities, self.temperature) * (1.0 - p_none)
        p_identity = normalize(np.concatenate([body, [p_none]]))
        backend = getattr(self.embedder, "backend", "custom")
        return Belief(
            p_identity=p_identity,
            source=self.name,
            confidence=float(p_identity.max()),
            raw={"sim": similarities.tolist(), "backend": backend, "mode": relation.mode},
        )
