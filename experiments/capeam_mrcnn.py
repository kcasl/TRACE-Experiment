"""CAPEAM's extra Mask R-CNN (GitHub pretrained weight) as a host scorer.

The checkpoint is torchvision MaskRCNN-ResNet50-FPN, 119 classes, the sliced-object
weight linked from https://github.com/snumprlab/capeam. Scores overlap THOR instance
boxes so the CAPEAM host ranks lookalikes by the detector, not expert pose.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

WEIGHT = Path("/data/jw_workspace/CAPEAM/weight_maskrcnn.pt")


def iou(a, b) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return float(inter / ua) if ua > 0 else 0.0


class CapeamMaskRCNN:
    def __init__(self, weight: Path | None = None, device: str | None = None):
        import torch
        from torchvision.models.detection import maskrcnn_resnet50_fpn

        path = Path(weight or WEIGHT)
        if not path.is_file():
            raise FileNotFoundError(str(path))
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        state = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        model = maskrcnn_resnet50_fpn(weights=None, num_classes=119)
        model.load_state_dict(state, strict=False)
        model.eval()
        self.model = model.to(self.device)
        self._torch = torch

    def detect(self, frame: np.ndarray) -> list[dict]:
        tensor = self._torch.from_numpy(np.ascontiguousarray(frame)).permute(2, 0, 1).float() / 255.0
        tensor = tensor.to(self.device)
        with self._torch.no_grad():
            out = self.model([tensor])[0]
        boxes = out["boxes"].detach().cpu().numpy()
        scores = out["scores"].detach().cpu().numpy()
        labels = out["labels"].detach().cpu().numpy()
        hits = []
        for box, score, label in zip(boxes, scores, labels):
            if float(score) < 0.05:
                continue
            hits.append({"bbox": [float(box[0]), float(box[1]), float(box[2]), float(box[3])], "score": float(score), "label": int(label)})
        return hits

    def scores_for(self, frame: np.ndarray | None, bboxes: list, fallback: list[float]) -> list[float]:
        if frame is None or not bboxes:
            return fallback
        hits = self.detect(frame)
        out = []
        for i, box in enumerate(bboxes):
            if box is None:
                out.append(fallback[i] if i < len(fallback) else 1e-6)
                continue
            best = 0.0
            for hit in hits:
                overlap = iou(box, hit["bbox"])
                if overlap >= 0.2:
                    best = max(best, hit["score"] * overlap)
            out.append(best if best > 0 else (fallback[i] if i < len(fallback) else 1e-6))
        return out
