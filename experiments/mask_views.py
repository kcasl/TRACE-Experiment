"""Rebuild the three images Jev saw for each mask event.

The host rollouts (frames/*.png + masks/*.npz) are gitignored and not on this
machine. The Jev request log still has every candidate bbox, the reference bbox,
and the A/B/C legend. That is enough to redraw:

  * reference object crop
  * reference context crop (bbox grown 50%)
  * current frame with labelled candidate outlines

If a real RGB frame is present under ``rollout_dir``, it is used as the canvas.
Otherwise the canvas is a schematic of the same boxes (layout is the identity
cue for identical drawers).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

PALETTE = [
    (255, 64, 64),
    (64, 200, 64),
    (64, 128, 255),
    (255, 200, 0),
    (255, 64, 255),
    (0, 220, 220),
    (255, 140, 0),
    (160, 100, 255),
]


def load_jev_log(path: Path) -> dict[str, dict]:
    by_key = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            key = row.get("cache_key")
            if key and row.get("ok"):
                by_key[key] = row
    return by_key


def event_key(episode_id: str, t: int) -> str:
    return episode_id.replace("/", "__") + f"__t{int(t):04d}"


def load_rollout_index(rollout_dir: Path) -> dict[str, dict]:
    """Map event_key -> event dict, with ``_base`` = directory of events.jsonl."""

    index = {}
    if not rollout_dir.exists():
        return index
    for path in rollout_dir.rglob("events.jsonl"):
        base = path.parent
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            row["_base"] = str(base)
            index[event_key(row["episode_id"], row["t"])] = row
    return index


def load_rgb(event: dict) -> np.ndarray:
    from PIL import Image

    rel = event.get("frame_path")
    if not rel:
        raise FileNotFoundError(f"event {event.get('episode_id')} t={event.get('t')} has no frame_path")
    path = Path(event["_base"]) / rel
    if not path.is_file():
        raise FileNotFoundError(str(path))
    return np.array(Image.open(path).convert("RGB"))


def load_masks(event: dict):
    rel = event.get("masks_path")
    if not rel:
        return None
    path = Path(event["_base"]) / rel
    if not path.is_file():
        return None
    return np.load(path)["masks"].astype(bool)


def overlay_rgb(frame: np.ndarray, bboxes: list, *, masks=None, labels: list[str] | None = None, scale: int = 2) -> np.ndarray:
    canvas = np.asarray(frame, dtype=np.uint8).copy()
    for index, bbox in enumerate(bboxes):
        color = PALETTE[index % len(PALETTE)]
        mask = None if masks is None or index >= len(masks) else masks[index]
        if mask is not None and mask.shape[:2] == canvas.shape[:2] and np.any(mask):
            _outline_mask(canvas, mask, color, width=2)
        else:
            _outline_box(canvas, bbox, color, width=2)
    return _draw_labels(_upscale(canvas, scale), bboxes, labels or [], scale)


def overlay_instances(
    frame: np.ndarray,
    bboxes: list,
    *,
    masks=None,
    labels: list[str] | None = None,
    selected: int | None = None,
    fill: float = 0.45,
    scale: int = 1,
) -> np.ndarray:
    """FILM/Mask R-CNN style: tint instance pixels, outline, A/B/C labels."""
    canvas = np.asarray(frame, dtype=np.uint8).copy()
    overlay = canvas.astype(np.float32)
    n = max(len(bboxes), 0 if masks is None else len(masks))
    for index in range(n):
        mask = None if masks is None or index >= len(masks) else masks[index]
        if mask is None or mask.shape[:2] != canvas.shape[:2] or not np.any(mask):
            continue
        color = np.array(PALETTE[index % len(PALETTE)], dtype=np.float32)
        alpha = fill + (0.12 if index == selected else 0.0)
        overlay[mask] = (1.0 - alpha) * overlay[mask] + alpha * color
    canvas = np.clip(overlay, 0, 255).astype(np.uint8)
    for index in range(n):
        color = PALETTE[index % len(PALETTE)]
        mask = None if masks is None or index >= len(masks) else masks[index]
        bbox = bboxes[index] if index < len(bboxes) else None
        thick = 4 if index == selected else 2
        edge = (255, 255, 255) if index == selected else color
        if mask is not None and mask.shape[:2] == canvas.shape[:2] and np.any(mask):
            _outline_mask(canvas, mask, edge, width=thick)
        elif bbox:
            _outline_box(canvas, bbox, color, width=thick)
    points = []
    for index in range(n):
        mask = None if masks is None or index >= len(masks) else masks[index]
        bbox = bboxes[index] if index < len(bboxes) else None
        if mask is not None and np.any(mask):
            ys, xs = np.where(mask)
            points.append((int(xs.min()), int(ys.min())))
        elif bbox:
            points.append((int(bbox[0]), int(bbox[1])))
        else:
            points.append(None)
    return _draw_labels_xy(_upscale(canvas, scale), points, labels or [], scale)


def views_from_rollout(belief_row: dict, events: dict[str, dict], jev_log: dict[str, dict], *, scale: int = 2) -> dict:
    """Original THOR RGB + mask overlay. Fails if the current or reference frame is missing."""

    current = events.get(belief_row["event_key"])
    if current is None:
        raise FileNotFoundError(f"no rollout event {belief_row['event_key']}")
    frame = load_rgb(current)
    masks = load_masks(current)
    cache_key = ((belief_row.get("raw") or {}).get("cache_key"))
    record = jev_log.get(cache_key) if cache_key else None
    state = record["request"]["body"]["state"] if record else _state_from_event(current, belief_row)
    bboxes = [candidate["bbox_xyxy"] for candidate in (state.get("candidates") or {}).values()]
    if not bboxes:
        bboxes = [c["bbox"] for c in current.get("candidates") or []]
    labels = list((state.get("candidates") or {}).keys()) or [chr(65 + i) for i in range(len(bboxes))]
    overlay = overlay_rgb(frame, bboxes, masks=masks, labels=labels, scale=scale)
    ref_key = None
    if belief_row.get("episode_id") is not None and belief_row.get("ref_t") is not None:
        ref_key = event_key(belief_row["episode_id"], belief_row["ref_t"])
    ref_event = events.get(ref_key) if ref_key else None
    if ref_event is None:
        raise FileNotFoundError(f"no reference frame for {belief_row['event_key']} ref_t={belief_row.get('ref_t')}")
    ref_frame = load_rgb(ref_event)
    ref_bbox = None
    selected = (ref_event.get("extra") or {}).get("selected")
    ref_cands = ref_event.get("candidates") or []
    if selected is not None and 0 <= int(selected) < len(ref_cands):
        ref_bbox = ref_cands[int(selected)]["bbox"]
    if ref_bbox is None and record:
        ref_bbox = state["reference_object"].get("bbox_xyxy_in_its_own_frame")
    object_crop = crop_box(ref_frame, ref_bbox, 0.0)
    context_crop = crop_box(ref_frame, ref_bbox, 0.5)
    return {
        "event_key": belief_row["event_key"],
        "state": state,
        "overlay": overlay,
        "object_crop": object_crop,
        "context_crop": context_crop,
        "n": len(bboxes),
        "images_used": True,
        "source": "thor_rgb",
    }


def views_from_belief_row(belief_row: dict, jev_log: dict[str, dict], *, scale: int = 2) -> dict:
    cache_key = ((belief_row.get("raw") or {}).get("cache_key"))
    record = jev_log.get(cache_key) if cache_key else None
    if record is None:
        raise KeyError(f"no Jev image log for {belief_row.get('event_key')}")
    state = record["request"]["body"]["state"]
    height, width = _frame_size(state)
    overlay = labeled_frame(state, (height, width), scale=scale)
    ref_bbox = state["reference_object"].get("bbox_xyxy_in_its_own_frame")
    base = labeled_frame(state, (height, width), scale=1, highlight=ref_bbox)
    object_crop = crop_box(base, ref_bbox, 0.0)
    context_crop = crop_box(base, ref_bbox, 0.5)
    return {
        "event_key": belief_row["event_key"],
        "state": state,
        "overlay": overlay,
        "object_crop": object_crop,
        "context_crop": context_crop,
        "n": len(state.get("candidates") or {}),
        "images_used": True,
        "source": "jev_log_bbox",
    }


def labeled_frame(state: dict, shape: tuple[int, int], *, scale: int = 2, highlight=None) -> np.ndarray:
    height, width = shape
    canvas = np.full((height, width, 3), 36, dtype=np.uint8)
    ref_bbox = state["reference_object"].get("bbox_xyxy_in_its_own_frame")
    if highlight is None:
        highlight = ref_bbox
    if highlight:
        _fill(canvas, highlight, (80, 80, 80))
    candidates = list((state.get("candidates") or {}).items())
    for index, (_label, candidate) in enumerate(candidates):
        color = PALETTE[index % len(PALETTE)]
        _outline_box(canvas, candidate["bbox_xyxy"], color, width=2)
    labels = [label for label, _ in candidates]
    bboxes = [candidate["bbox_xyxy"] for _, candidate in candidates]
    return _draw_labels(_upscale(canvas, scale), bboxes, labels, scale)


def crop_box(frame: np.ndarray, bbox, margin: float) -> np.ndarray:
    if not bbox:
        return frame
    height, width = frame.shape[:2]
    x0, y0, x1, y1 = [float(value) for value in bbox]
    bw, bh = max(x1 - x0, 1.0), max(y1 - y0, 1.0)
    dx, dy = bw * margin / 2.0, bh * margin / 2.0
    x0 = int(max(0, np.floor(x0 - dx)))
    y0 = int(max(0, np.floor(y0 - dy)))
    x1 = int(min(width, np.ceil(x1 + dx)))
    y1 = int(min(height, np.ceil(y1 + dy)))
    x1, y1 = max(x1, x0 + 2), max(y1, y0 + 2)
    return np.ascontiguousarray(frame[y0:y1, x0:x1])


def save_views(views: dict, directory: Path) -> dict[str, str]:
    from PIL import Image

    directory.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name in ("overlay", "object_crop", "context_crop"):
        path = directory / f"{views['event_key']}__{name}.png"
        Image.fromarray(np.uint8(views[name])).save(path)
        paths[name] = str(path)
    return paths


def _frame_size(state: dict) -> tuple[int, int]:
    max_x = max_y = 300
    for candidate in (state.get("candidates") or {}).values():
        _, _, x1, y1 = candidate["bbox_xyxy"]
        max_x = max(max_x, int(np.ceil(x1)))
        max_y = max(max_y, int(np.ceil(y1)))
    ref = state["reference_object"].get("bbox_xyxy_in_its_own_frame")
    if ref:
        max_x = max(max_x, int(np.ceil(ref[2])))
        max_y = max(max_y, int(np.ceil(ref[3])))
    return max(300, max_y), max(300, max_x)


def _fill(canvas: np.ndarray, bbox, color) -> None:
    x0, y0, x1, y1 = [int(value) for value in bbox]
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(canvas.shape[1], x1), min(canvas.shape[0], y1)
    if x1 > x0 and y1 > y0:
        canvas[y0:y1, x0:x1] = color


def _outline_box(canvas: np.ndarray, bbox, color, width: int = 2) -> None:
    x0, y0, x1, y1 = [int(value) for value in bbox]
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(canvas.shape[1] - 1, x1), min(canvas.shape[0] - 1, y1)
    canvas[y0 : min(y0 + width, y1 + 1), x0 : x1 + 1] = color
    canvas[max(y1 - width + 1, y0) : y1 + 1, x0 : x1 + 1] = color
    canvas[y0 : y1 + 1, x0 : min(x0 + width, x1 + 1)] = color
    canvas[y0 : y1 + 1, max(x1 - width + 1, x0) : x1 + 1] = color


def _upscale(frame: np.ndarray, scale: int) -> np.ndarray:
    if scale == 1:
        return frame
    return np.kron(frame, np.ones((scale, scale, 1), dtype=np.uint8))


def _draw_labels_xy(image: np.ndarray, points: list, labels: list[str], scale: int) -> np.ndarray:
    from PIL import Image, ImageDraw, ImageFont

    picture = Image.fromarray(np.uint8(image))
    draw = ImageDraw.Draw(picture)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14 * scale)
    except OSError:
        try:
            font = ImageFont.truetype("arial.ttf", 14 * scale)
        except OSError:
            font = ImageFont.load_default()
    for index, point in enumerate(points):
        if point is None:
            continue
        label = labels[index] if index < len(labels) else chr(65 + index)
        color = PALETTE[index % len(PALETTE)]
        x0, y0 = int(point[0]) * scale, int(point[1]) * scale
        box = draw.textbbox((0, 0), label, font=font)
        tw, th = box[2] - box[0], box[3] - box[1]
        y = max(0, y0 - th - 4)
        draw.rectangle([x0, y, x0 + tw + 8, y + th + 4], fill=color)
        draw.text((x0 + 4, y + 2), label, fill=(0, 0, 0), font=font)
    return np.asarray(picture)


def _draw_labels(image: np.ndarray, bboxes: list, labels: list[str], scale: int) -> np.ndarray:
    from PIL import Image, ImageDraw, ImageFont

    picture = Image.fromarray(np.uint8(image))
    draw = ImageDraw.Draw(picture)
    try:
        font = ImageFont.truetype("arial.ttf", 14 * scale)
    except OSError:
        font = ImageFont.load_default()
    for index, bbox in enumerate(bboxes):
        label = labels[index] if index < len(labels) else chr(65 + index)
        color = PALETTE[index % len(PALETTE)]
        x0, y0 = int(bbox[0]) * scale, int(bbox[1]) * scale
        box = draw.textbbox((0, 0), label, font=font)
        tw, th = box[2] - box[0], box[3] - box[1]
        y = max(0, y0 - th - 4)
        draw.rectangle([x0, y, x0 + tw + 8, y + th + 4], fill=color)
        draw.text((x0 + 4, y + 2), label, fill=(0, 0, 0), font=font)
    return np.asarray(picture)


def _outline_mask(canvas: np.ndarray, mask: np.ndarray, color, width: int = 2) -> None:
    binary = np.asarray(mask, dtype=bool)
    eroded = binary.copy()
    for _ in range(width):
        eroded = eroded & np.roll(eroded, 1, 0) & np.roll(eroded, -1, 0) & np.roll(eroded, 1, 1) & np.roll(eroded, -1, 1)
    canvas[binary & ~eroded] = color


def _state_from_event(event: dict, belief_row: dict) -> dict:
    labels = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    candidates = {}
    for index, candidate in enumerate(event.get("candidates") or []):
        label = labels[index] if index < 26 else f"C{index}"
        candidates[label] = {
            "bbox_xyxy": candidate["bbox"],
            "detector_score": candidate.get("score"),
            "distance_m": candidate.get("dist"),
            "position_relative_to_reference": "",
        }
    return {
        "task": {"type": event.get("task_type"), "goal": (event.get("extra") or {}).get("goal"), "current_subgoal": event.get("subgoal_text")},
        "upcoming_action": {"action": event.get("action"), "target_class": event.get("target_cls")},
        "relation_mode": belief_row.get("relation"),
        "action_history": [],
        "reference_object": {"bbox_xyxy_in_its_own_frame": None},
        "candidates": candidates,
    }
