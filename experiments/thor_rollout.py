"""Ported ALFRED on THOR 5 CloudRendering + TRACE.

2019 object names do not exist in THOR 5, so poses are applied by object type.
This is not the official ALFRED 10-failure / 1000-step protocol. It follows
oct21 high-pddl on all seven task types (valid_seen 820 / valid_unseen 821).

FILM host: greedy nearest-to-agent. CAPEAM host: GitHub Mask R-CNN
(``weight_maskrcnn.pt`` from snumprlab/capeam) scores THOR instance boxes.
Jev and OpenAI Decisions are separate TRACE sources on the same RGB.

Closed-loop: unique-min SELECT, planner extra C, sink-shell Put penalty,
furniture preserved under SetObjectPoses. Off-type Puts are not unreachable.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from experiments.capeam_mrcnn import CapeamMaskRCNN
from experiments.mask_views import PALETTE, _outline_box, _outline_mask, crop_box, overlay_instances, overlay_rgb
from trace.belief.base import normalize
from trace.belief.jev import JevSource
from trace.belief.openai_decisions import OpenAIDecisionSource
from trace.consequence.critic import ConsequenceCritic, GoalTemplate, matches_task_type
from trace.consequence.world import Obj, World
from trace.redact import load_env, redact
from trace.config import OVERRIDE_BUDGET
from trace.relation import EpisodeMemory, infer_relation
from trace.selector import MaskRequest, TraceSelector
from trace.types import Belief, Candidate, MemoryEntry, Observation, Relation

ROOT = Path(__file__).resolve().parents[1]
ALFRED = Path("/data/jw_workspace/ALFRED/data")
OUT = ROOT / "results" / "thor_port_v2"
MAX_CANDIDATES = 8  # identity APIs and the critic stay tractable; do not pin the expert instance


def type_of(name: str) -> str:
    return name.rsplit("_", 1)[0]


def dist(a: dict, b: dict) -> float:
    return math.sqrt((a["x"] - b["x"]) ** 2 + (a["z"] - b["z"]) ** 2)


def parse_oid(oid: str) -> tuple[str, dict | None]:
    parts = oid.split("|")
    if len(parts) >= 4:
        try:
            return parts[0], {"x": float(parts[1]), "y": float(parts[2]), "z": float(parts[3])}
        except ValueError:
            return parts[0], None
    return parts[0], None


class SoftmaxScore:
    name = "distance"

    def identify(self, relation, candidates, observation) -> Belief:
        scores = np.array([max(c.score, 1e-6) for c in candidates], dtype=np.float64)
        body = scores / scores.sum()
        p = np.concatenate([0.9 * body, [0.1]])
        return Belief(p_identity=normalize(p), source=self.name, raw={"mode": relation.mode})


class OracleIdentity:
    name = "oracle"

    def __init__(self):
        self.target_id: str | None = None

    def identify(self, relation, candidates, observation) -> Belief:
        p = np.zeros(len(candidates) + 1, dtype=np.float64)
        if self.target_id is None:
            p[-1] = 1.0
        else:
            hits = [i for i, c in enumerate(candidates) if c.object_id == self.target_id]
            if hits:
                p[hits[0]] = 0.95
                p[-1] = 0.05
            else:
                p[-1] = 1.0
        return Belief(p_identity=normalize(p), source=self.name, raw={"mode": relation.mode})


def load_split(split: str) -> list[dict]:
    items = json.loads((ALFRED / "splits/oct21.json").read_text())[split]
    out = []
    for item in items:
        rel = item["task"] if isinstance(item, dict) else item
        path = ALFRED / "json_2.1.0" / split / rel / "traj_data.json"
        if path.exists():
            out.append({"rel": rel, "repeat_idx": item.get("repeat_idx", 0), "path": path})
    return out


def load_traj(path: Path) -> dict:
    return json.loads(path.read_text())


def expert_targets(traj: dict) -> dict[str, dict]:
    found: dict[str, dict] = {}
    for step in traj.get("plan", {}).get("low_actions", []):
        api = step.get("api_action") or step.get("discrete_action") or {}
        action = api.get("action")
        oid = api.get("objectId")
        if not action or not oid:
            continue
        cls, pos = parse_oid(oid)
        if pos is None:
            continue
        key = f"{action}:{cls}"
        found.setdefault(key, pos)
    return found


def map_poses(event, poses: list[dict]) -> list[dict]:
    """Move ALFRED pickupables, but keep THOR furniture in the scene.

    SetObjectPoses replaces the object set with the provided list. Mapping only
    pickupables used to delete Safe / Dresser / GarbageCan, so Put fell back to
    nearby shelves and cabinets.
    """

    by_type: dict[str, list] = defaultdict(list)
    for obj in event.metadata["objects"]:
        if obj.get("pickupable") or obj.get("moveable"):
            by_type[obj["objectType"]].append(obj)
    used: set[str] = set()
    mapped = []
    for pose in poses:
        typ = type_of(pose["objectName"])
        pool = [o for o in by_type.get(typ, []) if o["name"] not in used]
        if not pool:
            continue
        obj = pool[0]
        used.add(obj["name"])
        mapped.append(
            {"objectName": obj["name"], "position": pose["position"], "rotation": pose["rotation"]}
        )
    used_names = {item["objectName"] for item in mapped}
    for obj in event.metadata["objects"]:
        name = obj.get("name")
        if not name or name in used_names:
            continue
        mapped.append(
            {
                "objectName": name,
                "position": obj["position"],
                "rotation": obj.get("rotation") or {"x": 0, "y": 0, "z": 0},
            }
        )
        used_names.add(name)
    return mapped


PARENT_ALIASES = {
    "CoffeeTable": ("CoffeeTable", "DiningTable", "SideTable", "Desk"),
    "SideTable": ("SideTable", "CoffeeTable", "Desk", "DiningTable", "Dresser"),
    "DiningTable": ("DiningTable", "CoffeeTable", "Desk", "SideTable"),
    "ArmChair": ("ArmChair", "Chair", "Sofa"),
    "Sofa": ("Sofa", "ArmChair"),
    "Ottoman": ("Ottoman", "ArmChair", "Sofa"),
    "Dresser": ("Dresser", "Drawer", "Cabinet"),
    "Safe": ("Safe", "Cabinet", "Drawer"),
    "Shelf": ("Shelf", "SideTable", "Desk"),
    "Desk": ("Desk", "SideTable", "DiningTable"),
    "Bed": ("Bed",),
    "CounterTop": ("CounterTop", "DiningTable", "SideTable", "Desk", "Shelf"),
    "SinkBasin": ("SinkBasin", "Sink", "BathtubBasin"),
    "Sink": ("Sink", "SinkBasin"),
    "GarbageCan": ("GarbageCan",),
    "Microwave": ("Microwave",),
    "Fridge": ("Fridge",),
    "Cabinet": ("Cabinet", "Drawer", "Dresser"),
    "Drawer": ("Drawer", "Cabinet", "Dresser"),
    "Toilet": ("Toilet",),
    "BathtubBasin": ("BathtubBasin", "SinkBasin"),
    "Cart": ("Cart", "Shelf", "SideTable"),
    "StoveBurner": ("StoveBurner",),
    "CoffeeMachine": ("CoffeeMachine",),
}

FURNITURE_TYPES = frozenset(
    {
        "CoffeeTable",
        "SideTable",
        "DiningTable",
        "Desk",
        "Dresser",
        "Drawer",
        "Cabinet",
        "Shelf",
        "Sofa",
        "ArmChair",
        "Bed",
        "Safe",
        "CounterTop",
        "SinkBasin",
        "Sink",
        "GarbageCan",
        "Microwave",
        "Fridge",
        "Toilet",
        "BathtubBasin",
        "Cart",
        "Ottoman",
        "StoveBurner",
        "CoffeeMachine",
    }
)

OBJECT_ALIASES = {
    "Knife": ("Knife", "ButterKnife"),
    "ButterKnife": ("ButterKnife", "Knife"),
}

TASK_FAMILIES = (
    "pick_and_place_simple",
    "pick_and_place_with_movable_recep",
    "pick_two_obj_and_place",
    "look_at_obj_in_light",
    "pick_clean_then_place_in_recep",
    "pick_heat_then_place_in_recep",
    "pick_cool_then_place_in_recep",
)

CRITIC_ACTIONS = frozenset({"PickupObject", "PutObject", "OpenObject", "CloseObject"})
TRACE_ACTIONS = frozenset({"PickupObject", "PutObject"})


def task_family(rel: str) -> str:
    text = str(rel)
    for name in TASK_FAMILIES:
        if text.startswith(name):
            return name
    return text.split("-")[0] if text else ""


def objects_of(event, typ: str) -> list[dict]:
    return [o for o in event.metadata["objects"] if o.get("objectType") == typ]


def objects_of_target(event, typ: str, *, prefer_sliced: bool = False) -> list[dict]:
    if not typ:
        return []
    if prefer_sliced:
        sliced_name = typ if str(typ).endswith("Sliced") else f"{typ}Sliced"
        sliced = objects_of(event, sliced_name)
        if sliced:
            return sliced
    seen: set[str] = set()
    out = []
    for name in OBJECT_ALIASES.get(typ, (typ,)):
        for obj in objects_of(event, name):
            if obj["objectId"] not in seen:
                seen.add(obj["objectId"])
                out.append(obj)
    return out


def objects_of_parent(event, typ: str) -> list[dict]:
    seen: set[str] = set()
    out = []
    for name in PARENT_ALIASES.get(typ, (typ,)):
        for obj in objects_of(event, name):
            if obj["objectId"] not in seen:
                seen.add(obj["objectId"])
                out.append(obj)
    if out:
        return out
    if typ not in FURNITURE_TYPES and typ not in PARENT_ALIASES:
        return []
    furn = [
        o
        for o in event.metadata["objects"]
        if o.get("receptacle") and o.get("objectType") in FURNITURE_TYPES
    ]
    if furn:
        return furn
    return [
        o
        for o in event.metadata["objects"]
        if o.get("receptacle") and not o.get("pickupable")
    ]


def nearest(cands: list[dict], pos: dict | None, agent: dict | None) -> dict | None:
    if not cands:
        return None
    if pos is not None:
        return min(cands, key=lambda o: dist(o["position"], pos))
    if agent is not None:
        return min(cands, key=lambda o: dist(o["position"], agent))
    return cands[0]


def world_from_event(event, candidate_ids: list[str], opened: set[str]) -> World:
    agent = event.metadata["agent"]["position"]
    objects = {}
    for obj in event.metadata["objects"]:
        oid = obj["objectId"]
        if oid not in candidate_ids and obj.get("objectType") not in {"Apple", "Potato", "Egg"}:
            if oid not in opened:
                continue
        d = dist(obj["position"], agent)
        objects[oid] = Obj(
            id=oid,
            type=obj["objectType"],
            room="here" if d < 2.5 else "far",
            pickupable=bool(obj.get("pickupable")),
            receptacle=bool(obj.get("receptacle")),
            openable=bool(obj.get("openable")),
            is_open=bool(obj.get("isOpen")) or oid in opened,
        )
    for cid in candidate_ids:
        if cid not in objects:
            objects[cid] = Obj(cid, "Unknown", "here", receptacle=True, openable=True, is_open=cid in opened)
    holding = None
    for obj in event.metadata["objects"]:
        if obj.get("isPickedUp"):
            holding = obj["objectId"]
            objects.setdefault(
                holding,
                Obj(holding, obj["objectType"], "here", pickupable=True),
            )
    return World(objects=objects, holding=holding, room="here")


def cap_candidates(
    cands: list[Candidate],
    pin_types: frozenset[str] | None = None,
) -> list[Candidate]:
    """Keep the instruction's class even when it is farther than MAX_CANDIDATES others.

    Expert instance identifiers are not pinned. TRACE and the host share this
    truncated list; the GT-ID oracle may pick the expert mask only if it is
    already here.
    """

    if len(cands) <= MAX_CANDIDATES:
        for i, cand in enumerate(cands):
            cand.idx = i
        return cands
    pin_types = pin_types or frozenset()

    def pinned(cand: Candidate) -> bool:
        return any(matches_task_type(cand.object_id, name, cls=cand.cls) for name in pin_types)

    # Pin every exact-type mask, then fill with highest host scores.
    keep_pin = [cand for cand in cands if pinned(cand)]
    rest = sorted((cand for cand in cands if not pinned(cand)), key=lambda c: c.score, reverse=True)
    if len(keep_pin) >= MAX_CANDIDATES:
        keep_pin = sorted(keep_pin, key=lambda c: c.score, reverse=True)[:MAX_CANDIDATES]
        rest = []
    keep = keep_pin + rest[: MAX_CANDIDATES - len(keep_pin)]
    out = []
    for i, cand in enumerate(keep):
        cand.idx = i
        out.append(cand)
    return out


def candidates_from(objs: list[dict], agent: dict, expert_pos: dict | None = None, host: str = "film") -> list[Candidate]:
    """Score is always agent distance. CAPEAM later overwrites with Mask R-CNN.

    ``expert_pos`` and ``host`` are kept for callers; they do not change the
    geometric score, so a missed detection cannot leak the expert pose.
    """

    out = []
    for i, obj in enumerate(objs):
        d_agent = dist(obj["position"], agent)
        score = 1.0 / (d_agent + 0.05)
        out.append(
            Candidate(
                idx=i,
                object_id=obj["objectId"],
                cls=obj["objectType"],
                score=score,
                room="here" if d_agent < 2.5 else "far",
                dist=d_agent,
            )
        )
    return out


def infer_live_relation(
    action: str,
    prompt_cls: str,
    memory: EpisodeMemory,
    *,
    family: str,
    n_cands: int,
) -> tuple[Relation | None, bool]:
    """Adapter Ground override and pick-two Diff flag, matching ``choose_id``."""

    pick_two_second = False
    override = None
    remembered = any(entry.cls == prompt_cls for entry in memory.entries)
    if (
        family == "pick_two_obj_and_place"
        and action == "PickupObject"
        and memory.last(action="PickupObject", cls=prompt_cls, role="object") is not None
    ):
        pick_two_second = True
    elif n_cands >= 2 and not remembered:
        role = "receptacle" if action == "PutObject" else "object"
        override = Relation(
            "SAME",
            MemoryEntry(key="task", cls=prompt_cls, action=action, t=0, role=role),
            "task_instance",
        )
    return override, pick_two_second


def resolve_live_relation(
    action: str,
    prompt_cls: str,
    memory: EpisodeMemory,
    *,
    family: str,
    n_cands: int,
) -> tuple[Relation, Relation | None, bool]:
    override, pick_two_second = infer_live_relation(
        action, prompt_cls, memory, family=family, n_cands=n_cands
    )
    relation = override or infer_relation(
        action, prompt_cls, memory, pick_two_second=pick_two_second
    )
    return relation, override, pick_two_second


def live_goal_template(
    action: str,
    relation: Relation,
    *,
    holding: str | None,
    task_cls: str | None,
) -> GoalTemplate:
    """Bind critic goal from the high-level action and the task relation ``r_t``."""

    if action == "PutObject":
        return GoalTemplate(
            kind="put_into_ref",
            movable_id=holding,
            movable_type=None,
            reference_type=task_cls or None,
        )
    if relation.mode == "DIFFERENT":
        return GoalTemplate(
            kind="pickup_not_ref",
            movable_type=task_cls or None,
            reference_type=task_cls or None,
        )
    return GoalTemplate(kind="pickup_ref", reference_type=task_cls or None)


def _reset_scene(controller, floor: str, retries: int = 4) -> None:
    last: Exception | None = None
    for attempt in range(retries):
        try:
            controller.reset(floor, renderInstanceSegmentation=True)
            return
        except (TimeoutError, RuntimeError) as exc:
            last = exc
            print(f"reset {floor} failed ({attempt + 1}/{retries}): {exc}", flush=True)
            time.sleep(8 * (attempt + 1))
    raise last  # type: ignore[misc]


def step_retry(controller, retries: int = 4, **kwargs):
    last: Exception | None = None
    action = kwargs.get("action")
    for attempt in range(retries):
        try:
            return controller.step(**kwargs)
        except (TimeoutError, RuntimeError) as exc:
            last = exc
            print(f"step {action} failed ({attempt + 1}/{retries}): {exc}", flush=True)
            time.sleep(4 * (attempt + 1))
    raise last  # type: ignore[misc]


def teleport_near(controller, target: dict, horizon: float = 30.0) -> bool:
    ev = step_retry(controller, action="GetReachablePositions")
    pts = ev.metadata.get("actionReturn") or []
    if not pts:
        ev = step_retry(
            controller,
            action="TeleportFull",
            position=target["position"],
            rotation={"x": 0, "y": 0, "z": 0},
            horizon=horizon,
            standing=True,
            forceAction=True,
        )
        return bool(ev.metadata.get("lastActionSuccess"))
    best = min(pts, key=lambda p: dist(p, target["position"]))
    dx = target["position"]["x"] - best["x"]
    dz = target["position"]["z"] - best["z"]
    yaw = math.degrees(math.atan2(dx, dz))
    ev = step_retry(
        controller,
        action="TeleportFull",
        position=best,
        rotation={"x": 0, "y": yaw, "z": 0},
        horizon=horizon,
        standing=True,
        forceAction=True,
    )
    return bool(ev.metadata.get("lastActionSuccess"))


def act(controller, action: str, object_id: str):
    return controller.step(action=action, objectId=object_id, forceAction=True)


def rgb_frame(event) -> np.ndarray | None:
    frame = getattr(event, "frame", None)
    if frame is None:
        return None
    return np.asarray(frame, dtype=np.uint8)


def bbox_of(event, object_id: str) -> list[int] | None:
    dets = getattr(event, "instance_detections2D", None) or {}
    box = dets.get(object_id)
    if box is None:
        return None
    if hasattr(box, "tolist"):
        box = box.tolist()
    if len(box) < 4:
        return None
    return [int(box[0]), int(box[1]), int(box[2]), int(box[3])]


def labelled_view(event, object_ids: list[str]) -> tuple[np.ndarray | None, list]:
    frame = rgb_frame(event)
    bboxes = [bbox_of(event, oid) for oid in object_ids]
    if frame is None:
        return None, bboxes
    usable = [box if box is not None else [0, 0, 2, 2] for box in bboxes]
    labels = [chr(65 + i) if i < 26 else f"C{i}" for i in range(len(usable))]
    masks = [instance_mask(event, oid) for oid in object_ids]
    return overlay_rgb(frame, usable, masks=masks, labels=labels, scale=2), bboxes


def dump_instance_overlay(
    dump_dir: Path | None,
    action: str,
    event,
    candidate_ids: list[str],
    selected_id: str | None,
) -> None:
    if dump_dir is None:
        return
    frame = rgb_frame(event)
    if frame is None:
        return
    bboxes = [bbox_of(event, oid) for oid in candidate_ids]
    masks = [instance_mask(event, oid) for oid in candidate_ids]
    labels = [chr(65 + i) if i < 26 else f"C{i}" for i in range(len(candidate_ids))]
    selected = candidate_ids.index(selected_id) if selected_id in candidate_ids else None
    save_png(
        dump_dir / f"{action.lower()}_instances.png",
        overlay_instances(frame, bboxes, masks=masks, labels=labels, selected=selected, fill=0.35, scale=1),
    )


def save_png(path: Path | None, frame: np.ndarray | None) -> None:
    if path is None or frame is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    from PIL import Image

    Image.fromarray(np.asarray(frame, dtype=np.uint8)).save(path)


def dump_choice(dump_dir: Path | None, action: str, event, candidate_ids: list[str]) -> tuple[np.ndarray | None, list]:
    overlay, bboxes = labelled_view(event, candidate_ids)
    if dump_dir is None:
        return overlay, bboxes
    save_png(dump_dir / f"{action.lower()}_choice_rgb.png", rgb_frame(event))
    save_png(dump_dir / f"{action.lower()}_choice_overlay.png", overlay)
    (dump_dir / f"{action.lower()}_choice.json").write_text(
        json.dumps({"action": action, "ids": candidate_ids, "bboxes": bboxes}, indent=2),
        encoding="utf-8",
    )
    return overlay, bboxes


def instance_mask(event, object_id: str) -> np.ndarray | None:
    """Pixel mask for one THOR object. Never synthesize a rectangle."""
    frame = rgb_frame(event)
    if frame is None:
        return None
    height, width = frame.shape[:2]
    seg = getattr(event, "instance_segmentation_frame", None)
    masks = getattr(event, "instance_masks", None)
    if masks is not None:
        try:
            if object_id in masks:
                mask = np.asarray(masks[object_id])
                if mask.ndim > 2:
                    mask = mask[..., 0]
                if mask.shape[:2] == (height, width) and np.any(mask):
                    return mask.astype(bool)
        except (TypeError, KeyError, ValueError):
            pass
    colors = getattr(event, "object_id_to_color", None) or {}
    color = None
    try:
        if object_id in colors:
            color = colors[object_id]
    except TypeError:
        color = None
    if seg is None or color is None:
        return None
    rgb = np.array(list(color)[:3], dtype=np.int16)
    return np.all(np.abs(np.asarray(seg).astype(np.int16) - rgb) <= 1, axis=-1)


def _mask_short_name(oid: str) -> str:
    parts = oid.split("|")
    if any(part == "SinkBasin" for part in parts):
        return "SinkBasin"
    if len(parts) > 1 and parts[-1][:1] not in "+-":
        return parts[-1]
    return parts[0]


def render_mask_panel(
    event,
    candidate_ids: list[str],
    selected_id: str | None,
    *,
    crop: str = "none",
) -> np.ndarray | None:
    """RGB + pixel instance masks. Selected is filled; others are outlines. No bbox fill."""
    from PIL import Image, ImageDraw, ImageFont

    frame = rgb_frame(event)
    if frame is None:
        return None
    height, width = frame.shape[:2]
    canvas = np.asarray(frame, dtype=np.uint8).copy()
    resolved: list[tuple[str, np.ndarray | None, list[int] | None]] = []
    for index, oid in enumerate(candidate_ids):
        mask = instance_mask(event, oid)
        box = bbox_of(event, oid)
        resolved.append((oid, mask, box))
        color = PALETTE[index % len(PALETTE)]
        if mask is not None and np.any(mask):
            if oid == selected_id:
                overlay = canvas.astype(np.float32)
                overlay[mask] = 0.38 * overlay[mask] + 0.62 * np.array(color, dtype=np.float32)
                canvas = np.clip(overlay, 0, 255).astype(np.uint8)
            edge = (255, 255, 255) if oid == selected_id else color
            _outline_mask(canvas, mask, edge, width=3 if oid == selected_id else 2)
        elif box is not None:
            _outline_box(canvas, box, color, width=2)
    if crop == "selected":
        selected_mask = next(
            (mask for oid, mask, _ in resolved if oid == selected_id and mask is not None and np.any(mask)),
            None,
        )
        if selected_mask is not None:
            ys, xs = np.where(selected_mask)
            x0, y0, x1, y1 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
        else:
            box = next((item[2] for item in resolved if item[0] == selected_id and item[2]), None)
            if box is None:
                x0, y0, x1, y1 = 0, 0, width, height
            else:
                x0, y0, x1, y1 = [int(v) for v in box]
        pad = 64
        x0, y0 = max(0, x0 - pad), max(0, y0 - pad)
        x1, y1 = min(width, x1 + pad), min(height, y1 + pad)
        canvas = canvas[y0:y1, x0:x1]
    image = Image.fromarray(canvas)
    bar = 32
    panel = Image.new("RGB", (image.width, image.height + bar), (18, 18, 18))
    panel.paste(image, (0, bar))
    draw = ImageDraw.Draw(panel)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
    except OSError:
        try:
            font = ImageFont.truetype("arial.ttf", 16)
        except OSError:
            font = ImageFont.load_default()
    x = 8
    for index, oid in enumerate(candidate_ids):
        mask = resolved[index][1]
        box = resolved[index][2]
        if (mask is None or not np.any(mask)) and box is None:
            continue
        letter = chr(65 + index) if index < 26 else f"C{index}"
        tag = f"{letter} {_mask_short_name(oid)}"
        if oid == selected_id:
            tag += " SELECT"
        color = tuple(int(v) for v in PALETTE[index % len(PALETTE)])
        text_box = draw.textbbox((0, 0), tag, font=font)
        tw = text_box[2] - text_box[0] + 12
        draw.rectangle([x, 6, x + tw, 26], fill=color)
        draw.text((x + 6, 7), tag, fill=(0, 0, 0), font=font)
        x += tw + 8
    return np.asarray(panel)


def dump_selected_masks(
    dump_dir: Path | None,
    action: str,
    event,
    candidate_ids: list[str],
    selected_id: str | None,
    *,
    crop: str = "none",
) -> None:
    if dump_dir is None:
        return
    save_png(dump_dir / f"{action.lower()}_masks.png", render_mask_panel(event, candidate_ids, selected_id, crop=crop))
    stats = []
    for oid in candidate_ids:
        mask = instance_mask(event, oid)
        stats.append(
            {
                "id": oid,
                "pixels": int(np.sum(mask)) if mask is not None else 0,
                "bbox": bbox_of(event, oid),
            }
        )
    (dump_dir / f"{action.lower()}_masks.json").write_text(
        json.dumps(
            {"action": action, "ids": candidate_ids, "selected": selected_id, "crop": crop, "masks": stats},
            indent=2,
        ),
        encoding="utf-8",
    )
    print(
        f"  masks {action}: "
        + ", ".join(f"{item['id'].split('|')[0]}={item['pixels']}px" for item in stats),
        flush=True,
    )


def task_goal(traj: dict) -> str:
    anns = (traj.get("turk_annotations") or {}).get("anns") or []
    if anns and anns[0].get("task_desc"):
        return str(anns[0]["task_desc"])
    params = traj.get("pddl_params") or {}
    return f"put the {params.get('object_target')} on the {params.get('parent_target')}"


def high_actions(traj: dict) -> list[tuple[str, list[str]]]:
    out = []
    for step in (traj.get("plan") or {}).get("high_pddl") or []:
        discrete = step.get("discrete_action") or {}
        action = str(discrete.get("action") or "")
        if not action or action == "NoOp":
            continue
        args = [str(item) for item in (discrete.get("args") or [])]
        out.append((action, args))
    return out


def resolve_cls(event, arg: str, fallback: str = "") -> str:
    raw = (arg or "").strip()
    if not raw:
        return fallback
    if "|" in raw:
        return raw.split("|", 1)[0]
    key = raw.replace(" ", "").replace("_", "").lower()
    for obj in event.metadata.get("objects") or []:
        typ = str(obj.get("objectType") or "")
        if typ.lower() == key:
            return typ
    if raw[:1].isupper():
        return raw
    return fallback or raw


def receptacle_ids(obj: dict) -> list[str]:
    recs = list(obj.get("parentReceptacles") or [])
    parent = obj.get("parentReceptacle")
    if parent and parent not in recs:
        recs.append(parent)
    return [str(item) for item in recs if item]


def holding_id(event) -> str | None:
    for obj in event.metadata.get("objects") or []:
        if obj.get("isPickedUp"):
            return obj["objectId"]
    inv = event.metadata.get("inventoryObjects") or []
    if inv:
        return inv[0].get("objectId")
    return None


def goal_satisfied(event, family: str, params: dict) -> bool:
    target = str(params.get("object_target") or "")
    parent = str(params.get("parent_target") or "")
    mrecep = str(params.get("mrecep_target") or "")
    toggle = str(params.get("toggle_target") or "")
    sliced = bool(params.get("object_sliced"))
    objects = event.metadata.get("objects") or []

    def type_set(name: str) -> set[str]:
        if not name:
            return set()
        names = {name}
        if sliced and not name.endswith("Sliced"):
            names.add(f"{name}Sliced")
        names.update(OBJECT_ALIASES.get(name, ()))
        if name in ("SinkBasin", "Sink"):
            names.update({"SinkBasin", "Sink"})
        if name in ("BathtubBasin", "Bathtub"):
            names.update({"BathtubBasin", "Bathtub"})
        return names

    def in_types(obj: dict, names: set[str]) -> bool:
        recs = set(receptacle_ids(obj))
        for other in objects:
            if other.get("objectType") in names and other.get("objectId") in recs:
                return True
        return False

    if family == "look_at_obj_in_light":
        held = next((obj for obj in objects if obj.get("isPickedUp")), None)
        lamp_types = type_set(toggle) | {"DeskLamp", "FloorLamp"}
        lamp_on = any(obj.get("isToggled") and obj.get("objectType") in lamp_types for obj in objects)
        return bool(held and lamp_on and held.get("objectType") in type_set(target))

    targets = [obj for obj in objects if obj.get("objectType") in type_set(target)]
    parent_types = type_set(parent)
    need = 2 if family == "pick_two_obj_and_place" else 1
    if mrecep:
        movable = [obj for obj in objects if obj.get("objectType") in type_set(mrecep)]
        placed = 0
        for item in targets:
            recs = set(receptacle_ids(item))
            for bowl in movable:
                if bowl.get("objectId") not in recs:
                    continue
                if in_types(bowl, parent_types):
                    placed += 1
                    break
        return placed >= need

    placed = [obj for obj in targets if in_types(obj, parent_types)]
    if len(placed) < need:
        return False
    if "heat" in family:
        return any(obj.get("isCooked", True) for obj in placed)
    if "clean" in family:
        return any(not obj.get("isDirty", False) for obj in placed)
    return True


def choose_id(
    selector_name: str,
    selector: TraceSelector | None,
    oracle: OracleIdentity,
    controller,
    objs: list[dict],
    action: str,
    memory: EpisodeMemory,
    opened: set[str],
    *,
    host: str,
    expert_pos: dict | None,
    expert_obj: dict | None,
    traj: dict,
    detector: CapeamMaskRCNN | None = None,
    dump_dir: Path | None = None,
    force_id: str | None = None,
    ask_cls: str | None = None,
    family: str = "",
) -> tuple[str | None, dict]:
    event = controller.last_event
    agent = event.metadata["agent"]["position"]
    params = traj.get("pddl_params") or {}
    if not ask_cls:
        ask_cls = str(params.get("parent_target") if action == "PutObject" else params.get("object_target") or "")
    task_cls = ask_cls
    critic_action = action if action in CRITIC_ACTIONS else "PickupObject"
    cands = candidates_from(objs, agent, expert_pos, host)
    if host == "capeam" and detector is not None:
        boxes = [bbox_of(event, c.object_id) for c in cands]
        scored = detector.scores_for(rgb_frame(event), boxes, [c.score for c in cands])
        for cand, score in zip(cands, scored):
            cand.score = score
    cands = cap_candidates(
        cands,
        pin_types=frozenset({task_cls}) if task_cls else None,
    )
    info = {"n": len(cands), "n_raw": len(objs), "relation": "NONE", "defer": False, "reason": "", "host": host}
    if not cands:
        return None, info
    need_view = dump_dir is not None or (
        selector_name not in ("host", "oracle")
        and selector is not None
        and len(cands) > 1
        and action in TRACE_ACTIONS
    )
    overlay, bboxes = (None, [])
    if need_view:
        overlay, bboxes = dump_choice(dump_dir, action, event, [c.object_id for c in cands])
    ids = [c.object_id for c in cands]
    cost_info: dict = {}

    def finish(pick: str | None, extra: dict) -> tuple[str | None, dict]:
        extra = {**cost_info, "cand_ids": ids, **extra}
        info.update(extra)
        dump_selected_masks(dump_dir, action, event, ids, pick)
        dump_instance_overlay(dump_dir, action, event, ids, pick)
        return pick, info

    def host_among(pool: list[Candidate]) -> Candidate:
        return max(pool, key=lambda c: c.score)

    def attach_costs(template: GoalTemplate | None = None) -> None:
        nonlocal cost_info
        cost_info = event_cost_fields(
            cands,
            critic_action,
            expert_obj,
            controller,
            opened,
            reference_type=task_cls or None,
            template=template,
        )

    if force_id:
        attach_costs()
        return finish(force_id, {"reason": "logged_replay", "force_id": force_id})
    if selector_name == "oracle":
        attach_costs()
        if oracle.target_id and any(c.object_id == oracle.target_id for c in cands):
            return finish(oracle.target_id, {"reason": "oracle_match"})
        pick = host_among(cands)
        return finish(pick.object_id, {"reason": "oracle_absent"})
    if selector_name == "host" or selector is None:
        attach_costs()
        pick = host_among(cands)
        return finish(pick.object_id, {"reason": "host_score"})
    if len(cands) == 1:
        attach_costs()
        return finish(cands[0].object_id, {"reason": "host_score"})
    if action not in TRACE_ACTIONS:
        attach_costs()
        pick = host_among(cands)
        return finish(pick.object_id, {"reason": "host_non_trace_action"})
    for cand, box in zip(cands, bboxes):
        cand.bbox = tuple(box) if box else None
        if overlay is not None and box:
            cand.crop = crop_box(np.asarray(event.frame, dtype=np.uint8), box, 0.0)

    world = world_from_event(event, [c.object_id for c in cands], opened)
    prompt_cls = task_cls or cands[0].cls
    relation, override, pick_two_second = resolve_live_relation(
        action, prompt_cls, memory, family=family, n_cands=len(cands)
    )
    template = live_goal_template(action, relation, holding=world.holding, task_cls=prompt_cls)
    attach_costs(template)
    history = tuple(f"{entry.action} {entry.cls}" for entry in memory.entries)
    observation = Observation(
        frame=overlay,
        goal_text=task_goal(traj),
        subgoal_text=f"{action} {prompt_cls}",
        history=history,
        task_type=family or "pick_and_place_simple",
        notes=(
            "THOR 5 CloudRendering port; instance crops from instance_detections2D. "
            "First pickup/put use task grounding, not an expert reference crop. "
            "Receptacle parts (SinkBasin vs Sink rim) are different candidates. "
            f"The instruction names {prompt_cls}."
        ),
    )
    request = MaskRequest(
        action=critic_action,
        target_cls=prompt_cls,
        candidates=cands,
        memory=memory,
        world=world,
        goal=template,
        observation=observation,
        relation_override=override,
        pick_two_second=pick_two_second,
    )
    try:
        decision = selector.choose(request)
    except Exception as exc:
        pick = host_among(cands)
        return finish(pick.object_id, {"reason": "api_error_host_fallback", "error": redact(str(exc))})
    packed = {
        "relation": decision.relation,
        "relation_rule": decision.relation_rule,
        "defer": decision.defer,
        "reason": decision.reason,
        "p": list(decision.p_identity),
        "source": decision.source,
        "B_ov": OVERRIDE_BUDGET,
        "c_defer": OVERRIDE_BUDGET,
        "critic_kind": template.kind,
    }
    if decision.defer or decision.index is None:
        pick = host_among(cands)
        packed["reason"] = f"defer:{decision.reason}"
        return finish(pick.object_id, packed)
    return finish(cands[decision.index].object_id, packed)


def _object_by_id(event, object_id: str) -> dict | None:
    return next((obj for obj in event.metadata.get("objects") or [] if obj.get("objectId") == object_id), None)


def _interact(
    controller,
    log: dict,
    memory: EpisodeMemory,
    opened: set[str],
    *,
    action: str,
    object_id: str,
    cls: str,
    info: dict | None,
    dump_dir: Path | None,
    role: str,
) -> bool:
    obj = _object_by_id(controller.last_event, object_id)
    if obj is None:
        log["error"] = f"missing:{object_id}"
        log["events"].append({"action": action, "id": object_id, "ok": False, **(info or {})})
        return False
    horizon = 45.0 if action == "PutObject" else 30.0
    teleport_near(controller, obj, horizon=horizon)
    controller.step(action="Pass")
    dump_selected_masks(
        dump_dir,
        f"{action}_near",
        controller.last_event,
        (info or {}).get("cand_ids") or [object_id],
        object_id,
        crop="selected",
    )
    dump_instance_overlay(
        dump_dir,
        f"{action}_near",
        controller.last_event,
        (info or {}).get("cand_ids") or [object_id],
        object_id,
    )
    if action == "PutObject":
        rec_now = _object_by_id(controller.last_event, object_id) or obj
        if rec_now.get("openable") and not rec_now.get("isOpen"):
            ev = act(controller, "OpenObject", object_id)
            open_ok = bool(ev.metadata.get("lastActionSuccess"))
            log["events"].append({"action": "OpenObject", "id": object_id, "ok": open_ok})
            if open_ok:
                opened.add(object_id)
                memory.record("OpenObject", cls, True, t=len(memory.entries) + 1, role="receptacle")
    api = "ToggleObjectOn" if action == "ToggleObject" else action
    ev = act(controller, api, object_id)
    ok = bool(ev.metadata.get("lastActionSuccess"))
    if api == "ToggleObjectOn" and not ok:
        now = _object_by_id(controller.last_event, object_id)
        ok = bool(now and now.get("isToggled"))
    log["events"].append({"action": action, "id": object_id, "ok": ok, **(info or {})})
    if ok:
        if api in {"PickupObject", "PutObject", "OpenObject", "CloseObject", "ToggleObjectOn", "ToggleObjectOff"}:
            memory.record(api, cls, True, t=len(memory.entries) + 1, role=role)
        if action == "PickupObject":
            log["pickup_ok"] = ok
        if action == "PutObject":
            log["put_ok"] = ok
        if action in ("ToggleObject", "ToggleObjectOn", "ToggleObjectOff"):
            log["toggle_ok"] = ok
        if action == "SliceObject":
            log["slice_ok"] = True
    return ok


def _choose_and_act(
    controller,
    traj: dict,
    selector_name: str,
    selector: TraceSelector | None,
    oracle: OracleIdentity,
    objs: list[dict],
    action: str,
    memory: EpisodeMemory,
    opened: set[str],
    log: dict,
    *,
    host: str,
    detector: CapeamMaskRCNN | None,
    dump_dir: Path | None,
    force_ids: dict[str, str] | None,
    ask_cls: str,
    family: str,
    expert_key: str,
    role: str,
) -> bool:
    if not objs:
        log["error"] = f"no_candidates:{action}:{ask_cls}"
        return False
    experts = expert_targets(traj)
    expert_pos = experts.get(expert_key)
    agent = controller.last_event.metadata["agent"]["position"]
    expert_obj = nearest(objs, expert_pos, agent)
    if selector_name == "oracle" and expert_obj is not None:
        oracle.target_id = expert_obj["objectId"]
    chosen, info = choose_id(
        selector_name,
        selector,
        oracle,
        controller,
        objs,
        action,
        memory,
        opened,
        host=host,
        expert_pos=expert_pos,
        expert_obj=expert_obj,
        traj=traj,
        detector=detector,
        dump_dir=dump_dir,
        force_id=(force_ids or {}).get(f"{action}:{ask_cls}") or (force_ids or {}).get(action),
        ask_cls=ask_cls,
        family=family,
    )
    if chosen is None:
        log["error"] = f"no_choice:{action}:{ask_cls}"
        return False
    return _interact(
        controller,
        log,
        memory,
        opened,
        action=action,
        object_id=chosen,
        cls=ask_cls,
        info=info,
        dump_dir=dump_dir,
        role=role,
    )


def _pickup_from_receptacle(controller, recep_id: str, prefer: set[str]) -> tuple[bool, str | None]:
    rec = _object_by_id(controller.last_event, recep_id)
    ids = list((rec or {}).get("receptacleObjectIds") or [])
    chosen = None
    for object_id in ids:
        obj = _object_by_id(controller.last_event, object_id)
        if obj and obj.get("objectType") in prefer:
            chosen = object_id
            break
    if chosen is None and ids:
        chosen = ids[0]
    if chosen is None:
        held = holding_id(controller.last_event)
        return bool(held), held
    ev = act(controller, "PickupObject", chosen)
    return bool(ev.metadata.get("lastActionSuccess")), chosen


def _process_held(
    controller,
    log: dict,
    memory: EpisodeMemory,
    opened: set[str],
    *,
    kind: str,
    prefer: set[str],
) -> bool:
    event = controller.last_event
    agent = event.metadata["agent"]["position"]
    held = holding_id(event)
    held_obj = _object_by_id(event, held) if held else None
    if held_obj is not None:
        prefer = set(prefer) | {held_obj["objectType"]}
    if kind == "heat":
        appliance_type, toggles = "Microwave", True
    elif kind == "cool":
        appliance_type, toggles = "Fridge", False
    else:
        appliance_type, toggles = "SinkBasin", True
    pool = objects_of(event, appliance_type) or objects_of_parent(event, appliance_type)
    rec = nearest(pool, None, agent)
    if rec is None:
        log["error"] = f"no_{kind}_appliance"
        return False
    rec_id = rec["objectId"]
    teleport_near(controller, rec, horizon=45.0)
    controller.step(action="Pass")
    rec_now = _object_by_id(controller.last_event, rec_id) or rec
    if rec_now.get("openable") and not rec_now.get("isOpen"):
        ev = act(controller, "OpenObject", rec_id)
        if not ev.metadata.get("lastActionSuccess"):
            log["events"].append({"action": "OpenObject", "id": rec_id, "ok": False, "role": kind})
            return False
        opened.add(rec_id)
        memory.record("OpenObject", appliance_type, True, t=len(memory.entries) + 1, role="receptacle")
        log["events"].append({"action": "OpenObject", "id": rec_id, "ok": True, "role": kind})
    ev = act(controller, "PutObject", rec_id)
    put_ok = bool(ev.metadata.get("lastActionSuccess"))
    log["events"].append({"action": "PutObject", "id": rec_id, "ok": put_ok, "role": kind})
    log["put_ok"] = put_ok
    if not put_ok:
        return False
    rec_now = _object_by_id(controller.last_event, rec_id) or rec_now
    if rec_now.get("openable") and rec_now.get("isOpen"):
        ev = act(controller, "CloseObject", rec_id)
        log["events"].append({"action": "CloseObject", "id": rec_id, "ok": bool(ev.metadata.get("lastActionSuccess")), "role": kind})
        opened.discard(rec_id)
    if toggles:
        if kind == "clean":
            faucet = nearest(objects_of(controller.last_event, "Faucet"), rec["position"], None)
            toggle_id = faucet["objectId"] if faucet is not None else rec_id
            toggle_cls = "Faucet"
        else:
            toggle_id, toggle_cls = rec_id, appliance_type
        ev = act(controller, "ToggleObjectOn", toggle_id)
        log["events"].append({"action": "ToggleObjectOn", "id": toggle_id, "ok": bool(ev.metadata.get("lastActionSuccess")), "role": kind})
        controller.step(action="Pass")
        if kind == "clean":
            sink = _object_by_id(controller.last_event, rec_id)
            for object_id in list((sink or {}).get("receptacleObjectIds") or []):
                obj = _object_by_id(controller.last_event, object_id)
                if obj and obj.get("dirtyable") and obj.get("isDirty"):
                    act(controller, "CleanObject", object_id)
        ev = act(controller, "ToggleObjectOff", toggle_id)
        log["events"].append({"action": "ToggleObjectOff", "id": toggle_id, "ok": bool(ev.metadata.get("lastActionSuccess")), "role": kind})
    else:
        controller.step(action="Pass")
        controller.step(action="Pass")
    rec_now = _object_by_id(controller.last_event, rec_id) or rec_now
    if rec_now.get("openable") and not rec_now.get("isOpen"):
        ev = act(controller, "OpenObject", rec_id)
        if ev.metadata.get("lastActionSuccess"):
            opened.add(rec_id)
        log["events"].append({"action": "OpenObject", "id": rec_id, "ok": bool(ev.metadata.get("lastActionSuccess")), "role": kind})
    ok, picked = _pickup_from_receptacle(controller, rec_id, prefer)
    log["events"].append({"action": "PickupObject", "id": picked, "ok": ok, "role": kind})
    log["pickup_ok"] = ok
    if ok:
        memory.record("PickupObject", next(iter(prefer), ""), True, t=len(memory.entries) + 1, role="object")
    rec_now = _object_by_id(controller.last_event, rec_id)
    if rec_now and rec_now.get("openable") and rec_now.get("isOpen"):
        ev = act(controller, "CloseObject", rec_id)
        opened.discard(rec_id)
        log["events"].append({"action": "CloseObject", "id": rec_id, "ok": bool(ev.metadata.get("lastActionSuccess")), "role": kind})
    return ok


def run_episode(
    controller,
    traj: dict,
    selector_name: str,
    selector: TraceSelector | None,
    oracle: OracleIdentity,
    *,
    host: str,
    detector: CapeamMaskRCNN | None = None,
    dump_dir: Path | None = None,
    force_ids: dict[str, str] | None = None,
    rel: str = "",
) -> dict:
    scene = traj["scene"]
    params = traj["pddl_params"]
    target_cls = params.get("object_target") or ""
    parent_cls = params.get("parent_target") or ""
    family = task_family(rel or traj.get("task_id") or "")
    memory = EpisodeMemory()
    opened: set[str] = set()
    log = {
        "task_id": traj.get("task_id"),
        "floor": scene["floor_plan"],
        "selector": selector_name,
        "host": host,
        "target": target_cls,
        "parent": parent_cls,
        "task": family,
        "pickup_ok": False,
        "put_ok": False,
        "toggle_ok": False,
        "success": False,
        "n_target": 0,
        "n_parent": 0,
        "events": [],
    }

    _reset_scene(controller, scene["floor_plan"])
    mapped = map_poses(controller.last_event, scene.get("object_poses") or [])
    if mapped:
        controller.step(action="SetObjectPoses", objectPoses=mapped)
    init = scene["init_action"]
    rot = init.get("rotation")
    if isinstance(rot, (int, float)):
        rot = {"x": 0, "y": rot, "z": 0}
    controller.step(
        action="TeleportFull",
        position={"x": init["x"], "y": init["y"], "z": init["z"]},
        rotation=rot or {"x": 0, "y": 0, "z": 0},
        horizon=init.get("horizon", 30),
        standing=True,
        forceAction=True,
    )
    ev = controller.last_event
    save_png((dump_dir / "init_rgb.png") if dump_dir else None, rgb_frame(ev))
    targets = objects_of_target(ev, target_cls) or objects_of(ev, target_cls)
    parents = objects_of_parent(ev, parent_cls) if parent_cls else []
    log["n_target"] = len(targets)
    log["n_parent"] = len(parents)

    oracle.target_id = None
    sliced_done = False
    steps = high_actions(traj)
    if not steps:
        steps = [("PickupObject", [target_cls.lower()]), ("PutObject", [target_cls.lower(), parent_cls.lower()])]

    for action, args in steps:
        event = controller.last_event
        if action == "GotoLocation":
            continue
        if action == "PickupObject":
            ask = resolve_cls(event, args[0] if args else "", target_cls)
            pool = objects_of_target(event, ask, prefer_sliced=sliced_done) or objects_of(event, ask)
            if not _choose_and_act(
                controller, traj, selector_name, selector, oracle, pool, "PickupObject",
                memory, opened, log, host=host, detector=detector, dump_dir=dump_dir,
                force_ids=force_ids, ask_cls=ask, family=family,
                expert_key=f"PickupObject:{ask}", role="object",
            ):
                if dump_dir is not None:
                    (dump_dir / "episode.json").write_text(json.dumps(log, indent=2), encoding="utf-8")
                return log
            continue
        if action == "PutObject":
            rec_arg = args[1] if len(args) > 1 else (args[0] if args else parent_cls)
            ask = resolve_cls(event, rec_arg, parent_cls)
            pool = objects_of_parent(event, ask) or objects_of(event, ask)
            if not _choose_and_act(
                controller, traj, selector_name, selector, oracle, pool, "PutObject",
                memory, opened, log, host=host, detector=detector, dump_dir=dump_dir,
                force_ids=force_ids, ask_cls=ask, family=family,
                expert_key=f"PutObject:{ask}", role="receptacle",
            ):
                if dump_dir is not None:
                    (dump_dir / "episode.json").write_text(json.dumps(log, indent=2), encoding="utf-8")
                return log
            continue
        if action == "SliceObject":
            ask = resolve_cls(event, args[0] if args else "", target_cls)
            pool = objects_of(event, ask) or objects_of_target(event, ask)
            if not _choose_and_act(
                controller, traj, selector_name, selector, oracle, pool, "SliceObject",
                memory, opened, log, host=host, detector=detector, dump_dir=dump_dir,
                force_ids=force_ids, ask_cls=ask, family=family,
                expert_key=f"SliceObject:{ask}", role="object",
            ):
                if dump_dir is not None:
                    (dump_dir / "episode.json").write_text(json.dumps(log, indent=2), encoding="utf-8")
                return log
            sliced_done = True
            continue
        if action in ("ToggleObject", "ToggleObjectOn", "ToggleObjectOff"):
            ask = resolve_cls(event, args[0] if args else "", str(params.get("toggle_target") or ""))
            pool = objects_of(event, ask)
            if not pool:
                pool = objects_of(event, "DeskLamp") + objects_of(event, "FloorLamp")
            if not _choose_and_act(
                controller, traj, selector_name, selector, oracle, pool, action,
                memory, opened, log, host=host, detector=detector, dump_dir=dump_dir,
                force_ids=force_ids, ask_cls=ask or "Lamp", family=family,
                expert_key=f"ToggleObjectOn:{ask}", role="object",
            ):
                if dump_dir is not None:
                    (dump_dir / "episode.json").write_text(json.dumps(log, indent=2), encoding="utf-8")
                return log
            continue
        if action == "HeatObject":
            prefer = {target_cls, f"{target_cls}Sliced"} if target_cls else set()
            if not _process_held(controller, log, memory, opened, kind="heat", prefer=prefer):
                if dump_dir is not None:
                    (dump_dir / "episode.json").write_text(json.dumps(log, indent=2), encoding="utf-8")
                return log
            continue
        if action == "CoolObject":
            prefer = {target_cls, f"{target_cls}Sliced"} if target_cls else set()
            if not _process_held(controller, log, memory, opened, kind="cool", prefer=prefer):
                if dump_dir is not None:
                    (dump_dir / "episode.json").write_text(json.dumps(log, indent=2), encoding="utf-8")
                return log
            continue
        if action == "CleanObject":
            prefer = {target_cls, f"{target_cls}Sliced"} if target_cls else set()
            if not _process_held(controller, log, memory, opened, kind="clean", prefer=prefer):
                if dump_dir is not None:
                    (dump_dir / "episode.json").write_text(json.dumps(log, indent=2), encoding="utf-8")
                return log
            continue
        log["error"] = f"unsupported_high:{action}"
        if dump_dir is not None:
            (dump_dir / "episode.json").write_text(json.dumps(log, indent=2), encoding="utf-8")
        return log

    log["success"] = bool(goal_satisfied(controller.last_event, family, params))
    save_png((dump_dir / "final.png") if dump_dir else None, rgb_frame(controller.last_event))
    if dump_dir is not None:
        (dump_dir / "episode.json").write_text(json.dumps(log, indent=2), encoding="utf-8")
    return log


def success_cost_matrix(n: int, *, fail: float = 100.0) -> tuple[list[list[float]], list[list[int]]]:
    costs = [[fail] * (n + 1) for _ in range(n)]
    reachable = [[1] * (n + 1) for _ in range(n)]
    for i in range(n):
        costs[i][i] = 0.0
    return costs, reachable


def event_cost_fields(
    cands: list[Candidate],
    action: str,
    expert_obj: dict | None,
    controller,
    opened: set[str],
    reference_type: str | None = None,
    template: GoalTemplate | None = None,
) -> dict:
    """Identity alignment + critic matrix for later CR replay (no live API)."""

    ids = [c.object_id for c in cands]
    scores = [float(c.score) for c in cands]
    expert_id = expert_obj["objectId"] if expert_obj else None
    j_true = ids.index(expert_id) if expert_id in ids else None
    host_choice = int(max(range(len(cands)), key=lambda i: cands[i].score)) if cands else None
    payload: dict = {
        "cand_ids": ids,
        "scores": scores,
        "expert_id": expert_id,
        "j_true": j_true,
        "host_choice": host_choice,
    }
    n = len(cands)
    if n < 1 or controller is None:
        return payload
    try:
        world = world_from_event(controller.last_event, ids, opened)
        if template is None:
            if action == "PutObject":
                if not world.holding:
                    payload["cost_error"] = "no_holding"
                    return payload
                template = GoalTemplate(
                    kind="put_into_ref",
                    movable_id=world.holding,
                    reference_type=reference_type,
                )
            else:
                template = GoalTemplate(kind="pickup_ref", reference_type=reference_type)
        matrix = ConsequenceCritic().cost_matrix(world, action, ids, template)
        payload["C"] = [[None if not np.isfinite(x) else float(x) for x in row] for row in matrix.C]
        payload["reachable"] = np.isfinite(matrix.C).astype(int).tolist()
        payload["critic"] = "planner extra"
        payload["critic_kind"] = template.kind
    except Exception as exc:
        payload["cost_error"] = redact(str(exc))
    return payload


def make_controller(*, width: int = 300, height: int = 300):
    from ai2thor.controller import Controller
    from ai2thor.platform import CloudRendering

    return Controller(
        platform=CloudRendering,
        scene="FloorPlan1",
        width=width,
        height=height,
        gridSize=0.25,
        renderInstanceSegmentation=True,
    )


def failed_episode(idx: int, name: str, host: str, exc: Exception) -> dict:
    return {
        "idx": idx,
        "selector": name,
        "host": host,
        "floor": "",
        "target": "",
        "parent": "",
        "task": "",
        "pickup_ok": False,
        "put_ok": False,
        "success": False,
        "n_target": 0,
        "n_parent": 0,
        "events": [],
        "error": redact(str(exc)),
    }


def _put_success_critic() -> ConsequenceCritic:
    """v2 ablation: wrong Put = episode fail 100. Not used by the reported selector."""

    return ConsequenceCritic(success_actions=frozenset({"PutObject"}))


def _planner_critic() -> ConsequenceCritic:
    """Paper critic: Put extra is remaining-plan length, same as pickup."""

    return ConsequenceCritic()


def make_selector(name: str, oracle_src: OracleIdentity) -> TraceSelector | None:
    if name == "host":
        return None
    if name == "oracle":
        return TraceSelector(oracle_src, rule="force_best", critic=_planner_critic())
    if name == "trace":
        return TraceSelector(SoftmaxScore(), rule="trace", critic=_planner_critic())
    if name == "jev":
        return TraceSelector(JevSource(), rule="trace", critic=_planner_critic())
    if name == "openai_decisions":
        return TraceSelector(OpenAIDecisionSource(), rule="trace", critic=_planner_critic())
    raise ValueError(f"unknown source {name}")


def summarize(rows: list[dict], names: list[str]) -> dict:
    summary = {}
    for name in names:
        part = [r for r in rows if r["selector"] == name]
        n = len(part) or 1
        by_task: dict[str, dict] = {}
        for row in part:
            family = str(row.get("task") or "unknown")
            slot = by_task.setdefault(family, {"n": 0, "ok": 0})
            slot["n"] += 1
            slot["ok"] += int(bool(row.get("success")))
        summary[name] = {
            "n": len(part),
            "sr": sum(bool(r["success"]) for r in part) / n,
            "pickup": sum(bool(r["pickup_ok"]) for r in part) / n,
            "put": sum(bool(r["put_ok"]) for r in part) / n,
            "api_errors": sum(1 for r in part for e in r.get("events") or [] if e.get("error")),
            "by_task": {k: {"n": v["n"], "sr": v["ok"] / v["n"]} for k, v in sorted(by_task.items())},
        }
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="valid_seen", choices=("valid_seen", "valid_unseen"))
    parser.add_argument("--host", default="film", choices=("film", "capeam"))
    parser.add_argument(
        "--source",
        action="append",
        dest="sources",
        choices=("host", "oracle", "trace", "jev", "openai_decisions"),
        help="repeat to run more than one source in this process",
    )
    parser.add_argument("--limit", type=int, default=0, help="0 = all oct21 episodes")
    parser.add_argument("--from-idx", type=int, default=0)
    parser.add_argument("--out", default="", help="results root; default results/thor_port_v2")
    parser.add_argument("--dump-frames", default="", help="directory for RGB/overlay PNGs")
    parser.add_argument("--width", type=int, default=300)
    parser.add_argument("--height", type=int, default=300)
    args = parser.parse_args(argv)
    load_env(ROOT / ".env")
    sources = args.sources or ["host", "oracle", "trace"]
    out_root = Path(args.out) if args.out else OUT

    tasks = load_split(args.split)
    if args.limit:
        tasks = tasks[args.from_idx : args.from_idx + args.limit]
    else:
        tasks = tasks[args.from_idx :]
    out_dir = out_root / args.host / args.split
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{args.from_idx}_{args.from_idx + len(tasks)}"
    print(f"host={args.host} split={args.split} sources={sources} n={len(tasks)} -> {out_dir}", flush=True)

    dump_root = Path(args.dump_frames) if args.dump_frames else None
    if dump_root is not None:
        dump_root.mkdir(parents=True, exist_ok=True)
    controller = make_controller(width=args.width, height=args.height)
    oracle_src = OracleIdentity()
    detector = CapeamMaskRCNN() if args.host == "capeam" else None
    if detector is not None:
        print(f"CAPEAM Mask R-CNN on {detector.device}", flush=True)
    selectors = {name: make_selector(name, oracle_src) for name in sources}
    handles = {}
    rows = []
    for name in sources:
        path = out_dir / f"{name}_{tag}.jsonl"
        handles[name] = path.open("w", encoding="utf-8")
        print(f"logging {path}", flush=True)
    try:
        for i, task in enumerate(tasks):
            idx = args.from_idx + i
            try:
                traj = load_traj(task["path"])
            except Exception as exc:
                print(f"{idx:03d} {args.host:6s} LOAD FAIL {redact(str(exc))}", flush=True)
                for name in sources:
                    row = failed_episode(idx, name, args.host, exc)
                    row["rel"] = task.get("rel", "")
                    rows.append(row)
                    handle = handles[name]
                    if not handle.closed:
                        handle.write(json.dumps(row) + "\n")
                        handle.flush()
                continue
            for name, sel in selectors.items():
                oracle_src.target_id = None
                dump_dir = None
                if dump_root is not None:
                    floor = traj["scene"]["floor_plan"]
                    params = traj["pddl_params"]
                    dump_dir = dump_root / (
                        f"{args.host}_{args.split}_{idx:03d}_"
                        f"{floor}_{params['object_target']}_{params['parent_target']}_{name}"
                    )
                try:
                    row = run_episode(
                        controller,
                        traj,
                        name,
                        sel,
                        oracle_src,
                        host=args.host,
                        detector=detector,
                        dump_dir=dump_dir,
                        rel=task.get("rel") or "",
                    )
                except Exception as exc:
                    print(f"{idx:03d} {args.host:6s} {name:18s} FAIL {redact(str(exc))}", flush=True)
                    row = failed_episode(idx, name, args.host, exc)
                    try:
                        controller.stop()
                    except Exception:
                        pass
                    time.sleep(5)
                    controller = make_controller(width=args.width, height=args.height)
                row["idx"] = idx
                row["rel"] = task["rel"]
                rows.append(row)
                handle = handles[name]
                if not handle.closed:
                    handle.write(json.dumps(row) + "\n")
                    handle.flush()
                print(
                    f"{row['idx']:03d} {args.host:6s} {name:18s} {row.get('floor') or '?':12s} "
                    f"{(row.get('task') or '')[:22]:22s} "
                    f"pick={int(row.get('pickup_ok') or 0)} put={int(row.get('put_ok') or 0)} "
                    f"ok={int(row.get('success') or 0)} nT={row.get('n_target') or 0} nP={row.get('n_parent') or 0}",
                    flush=True,
                )
    finally:
        for handle in handles.values():
            if not handle.closed:
                handle.close()
        try:
            controller.stop()
        except Exception:
            pass
    summary = summarize(rows, sources)
    summary_path = out_dir / f"summary_{'_'.join(sources)}_{tag}.json"
    summary_path.write_text(json.dumps({"host": args.host, "split": args.split, "sources": summary}, indent=2), encoding="utf-8")
    print("SUMMARY", json.dumps({"host": args.host, "split": args.split, "sources": summary}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
