"""Identity calls against Jev and OpenAI Decisions, then the TRACE gate.

The images are synthetic and the correct label is known from how they were drawn.
Cost worlds are the same symbolic tasks the unit tests check. Slot 0 is the
task-consistent object there, and it is also the visually matching object in
the cases that have one.

Sources whose API key is missing are skipped, not failed, so the script still
produces the oracle rows offline. Run from the jw directory.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np

from trace.belief.base import FixedIdentity
from trace.belief.jev import JevSource
from trace.belief.openai_decisions import OpenAIDecisionSource
from trace.config import DEFERRAL_COST
from trace.consequence.critic import ConsequenceCritic
from trace.consequence.planner import PUT
from trace.gate import decide
from trace.redact import has_env, load_env, redact
from trace.relation import EpisodeMemory
from trace.types import Candidate, Observation, Relation

from .scenes import far_fridge, same_room_drawers, split_fridges

ROOT = Path(__file__).resolve().parents[1]

# Each live source and the environment variables that unlock it.
LIVE_SOURCES = (
    ("jev", JevSource, ("JEV_API_KEY", "TYPESAFE_API_KEY")),
    ("openai_decisions", OpenAIDecisionSource, ("OPENAI_API_KEY",)),
)


def run(*, include_live: bool = True, sources: Sequence[str] | None = None, out: Path | None = None) -> dict:
    load_env(ROOT / ".env")
    wanted = list(sources) if sources is not None else [name for name, _, _ in LIVE_SOURCES]
    unknown = [name for name in wanted if name not in {item[0] for item in LIVE_SOURCES}]
    if unknown:
        raise ValueError(f"unknown live source {unknown!r}")
    cases = _cases()
    critic = ConsequenceCritic()
    cost_matrices = {
        "same_room_drawers": critic.cost_matrix(*_unpack(same_room_drawers(), PUT)),
        "far_fridge": critic.cost_matrix(*_unpack(far_fridge(), PUT)),
        "split_fridges": critic.cost_matrix(*_unpack(split_fridges(), PUT)),
    }
    rows = []
    for case in cases:
        rows.append(_oracle_row(case, cost_matrices))
        if not include_live:
            continue
        for name, factory, keys in LIVE_SOURCES:
            if name not in wanted:
                continue
            if not has_env(*keys):
                rows.append({"case": case["name"], "truth": case["truth"], "source": name, "ok": False, "skipped": "no_api_key"})
                continue
            rows.append(_live_row(case, name, factory(), cost_matrices))

    for row in rows:
        status = "ok" if row.get("ok") else (row.get("skipped") or "error")
        detail = row.get("p_identity") if row.get("ok") else row.get("error", "")
        print(f"{row['case']:18} {row['source']:18} {status:10} {detail}")

    live_names = [name for name in wanted if include_live]
    report = {
        "experiment": "+".join(live_names) if live_names else "oracle",
        "sources": live_names,
        "cases": [case["name"] for case in cases],
        "rows": rows,
        "identity_accuracy": _accuracy(rows),
    }
    path = out or (ROOT / "results" / "live_identity.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote {path}")
    return report


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Run one identity experiment, or both.")
    parser.add_argument(
        "--source",
        choices=[name for name, _, _ in LIVE_SOURCES],
        action="append",
        help="repeat to run more than one source; default is both, separately preferred via run_source",
    )
    args = parser.parse_args()
    run(include_live=True, sources=args.source)


def _unpack(scene, action):
    world, template, ids = scene
    return world, action, ids, template


def _belief_row(case: dict, source_name: str, belief, cost_matrices: dict) -> dict:
    top = int(np.argmax(belief.p_identity))
    decisions = {}
    for world_name, matrix in cost_matrices.items():
        trace = decide(belief, matrix, mode="SAME", deferral_cost=DEFERRAL_COST, rule="trace")
        threshold = decide(belief, matrix, mode="SAME", deferral_cost=DEFERRAL_COST, rule="threshold", tau=0.75)
        regret = trace.expected_regret
        decisions[world_name] = {
            "trace": trace.action if trace.idx is None else f"SELECT {trace.idx}",
            "threshold_075": threshold.action if threshold.idx is None else f"SELECT {threshold.idx}",
            "min_expected_regret": None
            if regret.size == 0 or np.all(np.isnan(regret))
            else round(float(np.nanmin(regret)), 4),
        }
    return {
        "case": case["name"],
        "truth": case["truth"],
        "source": source_name,
        "ok": True,
        "refusal": belief.refusal,
        "p_identity": [round(float(value), 4) for value in belief.p_identity],
        "confidence": belief.confidence,
        "top": top,
        "identity_correct": top == case["truth"],
        "decisions": decisions,
    }


def _oracle_row(case: dict, cost_matrices: dict) -> dict:
    belief = FixedIdentity(_one_hot(case["truth"], 2)).identify(case["relation"], case["candidates"], case["observation"])
    return _belief_row(case, "oracle", belief, cost_matrices)


def _live_row(case: dict, source_name: str, source, cost_matrices: dict) -> dict:
    try:
        belief = source.identify(case["relation"], case["candidates"], case["observation"])
        return _belief_row(case, source_name, belief, cost_matrices)
    except Exception as exc:  # network / schema / auth
        return {"case": case["name"], "truth": case["truth"], "source": source_name, "ok": False, "error": redact(str(exc))}


def _accuracy(rows: list[dict]) -> dict:
    accuracy = {}
    for source in sorted({row["source"] for row in rows}):
        graded = [row for row in rows if row["source"] == source and row.get("ok")]
        if graded:
            accuracy[source] = round(sum(row["identity_correct"] for row in graded) / len(graded), 4)
    return accuracy


def _one_hot(truth: int, n_candidates: int) -> np.ndarray:
    probability = np.zeros(n_candidates + 1, dtype=np.float64)
    probability[truth] = 1.0
    return probability


def _cases() -> list[dict]:
    visible = _pair(marked=True)
    absent = _pair(marked=False)
    lookalike = _pair(marked=True, lookalike=True)
    return [
        _case("visible_match", visible, truth=0),
        _case("reference_absent", absent, truth=2),
        _case("lookalike", lookalike, truth=0),
    ]


def _case(name: str, images: tuple[np.ndarray, np.ndarray], truth: int) -> dict:
    memory = EpisodeMemory()
    memory.record("PickupObject", "Apple", True, 1)
    # A lone pickup is NONE for the task chain. These pictures are a direct identity
    # question, so the relation is set to SAME against that pickup.
    relation = Relation("SAME", memory.entries[0], "visual")
    candidates = [
        Candidate(0, "slot0", "Apple", score=0.51, room="kitchen"),
        Candidate(1, "slot1", "Apple", score=0.49, room="kitchen"),
    ]
    reference, current = images
    frames = ROOT / "results" / "frames"
    frames.mkdir(parents=True, exist_ok=True)
    from PIL import Image

    Image.fromarray(reference).save(frames / f"{name}_reference.png")
    Image.fromarray(current).save(frames / f"{name}_current.png")
    return {
        "name": name,
        "truth": truth,
        "relation": relation,
        "candidates": candidates,
        "observation": Observation(
            frame=current,
            reference_frame=reference,
            goal_text="Pick up the apple that was picked up earlier.",
            history=("PickupObject Apple",),
            task_type="pick_and_place",
        ),
    }


def _pair(*, marked: bool, lookalike: bool = False) -> tuple[np.ndarray, np.ndarray]:
    reference = _canvas()
    _disk(reference, (160, 160), 48, (210, 40, 40))
    if marked:
        _disk(reference, (160, 160), 14, (250, 250, 250))
    current = _canvas()
    _label(current, "A", (40, 30))
    _label(current, "B", (200, 30))
    if marked and not lookalike:
        _disk(current, (80, 170), 48, (210, 40, 40))
        _disk(current, (80, 170), 14, (250, 250, 250))
        _box(current, (200, 130), (280, 220), (40, 90, 210))
    elif lookalike:
        _disk(current, (80, 170), 48, (210, 40, 40))
        _disk(current, (80, 170), 14, (250, 250, 250))
        _disk(current, (240, 170), 48, (210, 40, 40))
    else:
        _box(current, (40, 130), (130, 220), (40, 160, 70))
        _box(current, (200, 130), (280, 220), (40, 90, 210))
    return reference, current


def _canvas() -> np.ndarray:
    image = np.zeros((320, 320, 3), dtype=np.uint8)
    image[:, :] = (235, 235, 230)
    return image


def _disk(image: np.ndarray, center: tuple[int, int], radius: int, color: tuple[int, int, int]) -> None:
    cy, cx = center[1], center[0]
    yy, xx = np.ogrid[: image.shape[0], : image.shape[1]]
    image[(yy - cy) ** 2 + (xx - cx) ** 2 <= radius ** 2] = color


def _box(image, top_left, bottom_right, color) -> None:
    x0, y0 = top_left
    x1, y1 = bottom_right
    image[y0:y1, x0:x1] = color


def _label(image: np.ndarray, text: str, origin: tuple[int, int]) -> None:
    from PIL import Image, ImageDraw, ImageFont

    picture = Image.fromarray(image)
    draw = ImageDraw.Draw(picture)
    try:
        font = ImageFont.truetype("arial.ttf", 28)
    except OSError:
        font = ImageFont.load_default()
    draw.text(origin, text, fill=(0, 0, 0), font=font)
    image[:, :] = np.asarray(picture)


if __name__ == "__main__":
    main()
