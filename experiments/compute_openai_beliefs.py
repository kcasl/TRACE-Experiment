"""Call OpenAI Decisions on every FILM/CAPEAM select-eligible mask event.

    python -m experiments.compute_openai_beliefs --rgb   # original THOR frames/masks
    python -m experiments.compute_openai_beliefs         # Jev bbox overlay fallback
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from trace.belief.openai_decisions import OpenAIDecisionSource
from trace.redact import load_env, redact
from trace.types import Candidate, Observation, Relation, normalize_mode

from .mask_views import load_jev_log, load_rollout_index, save_views, views_from_belief_row, views_from_rollout
from .paths import REPLAY_SPLITS, ryankwon_root

ROOT = Path(__file__).resolve().parents[1]


def run(*, split: str | None = None, limit: int = 0, rgb: bool = False) -> dict:
    load_env(ROOT / ".env")
    source = OpenAIDecisionSource()
    root = ryankwon_root()
    summary = {}
    for spec in REPLAY_SPLITS:
        if split and spec["name"] != split:
            continue
        summary[spec["name"]] = _run_split(root, spec, source, limit=limit, rgb=rgb)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default=None)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--rgb", action="store_true", help="require original THOR frames/masks")
    args = parser.parse_args()
    print(json.dumps(run(split=args.split, limit=args.limit, rgb=args.rgb), indent=2))


def _run_split(root: Path, spec: dict, source: OpenAIDecisionSource, *, limit: int, rgb: bool = False) -> dict:
    beliefs_dir = root / spec["beliefs"]
    jev_path = beliefs_dir / "jev.jsonl"
    log_path = beliefs_dir / "jev_log" / "jev_requests.jsonl"
    costs_path = root / spec["costs"]
    tag = "rgb" if rgb else "bbox"
    out_path = beliefs_dir / ("openai_decisions_rgb.jsonl" if rgb else "openai_decisions.jsonl")
    preview_dir = ROOT / "results" / "mask_views" / f"{spec['name'].replace('/', '_')}_{tag}"
    if not jev_path.exists() or not log_path.exists():
        return {"error": "missing jev log", "jev": str(jev_path), "log": str(log_path)}
    events = load_rollout_index(root / spec["rollout"]) if rgb else {}
    if rgb and not events:
        return {"error": "missing THOR RGB rollouts", "rollout": str(root / spec["rollout"])}
    eligible = _cost_keys(costs_path)
    jev_log = load_jev_log(log_path)
    done = _existing_keys(out_path)
    written = skipped = errors = 0
    with out_path.open("a", encoding="utf-8") as handle:
        for row in _load(jev_path):
            if "p" not in row or "event_key" not in row:
                continue
            if eligible and row["event_key"] not in eligible:
                continue
            if row["event_key"] in done:
                skipped += 1
                continue
            try:
                views = views_from_rollout(row, events, jev_log) if rgb else views_from_belief_row(row, jev_log)
                save_views(views, preview_dir)
                belief = source.identify(*_identify_args(row, views))
                payload = {
                    "event_key": row["event_key"],
                    "episode_id": row.get("episode_id"),
                    "t": row.get("t"),
                    "relation": row.get("relation"),
                    "rule": row.get("rule"),
                    "ref_gt": row.get("ref_gt"),
                    "p": [float(value) for value in belief.p_identity],
                    "confidence": belief.confidence,
                    "source": "openai_decisions_rgb" if rgb else "openai_decisions",
                    "cand_gt": row.get("cand_gt"),
                    "host_choice": row.get("host_choice"),
                    "raw": {
                        **(belief.raw or {}),
                        "images": views["source"],
                        "n": views["n"],
                    },
                }
            except Exception as exc:
                payload = {"event_key": row["event_key"], "source": "openai_decisions", "error": redact(str(exc))}
                errors += 1
            handle.write(json.dumps(payload) + "\n")
            handle.flush()
            written += 1
            status = "error" if payload.get("error") else "ok"
            print(f"{spec['name']:22} {row['event_key'][-24:]:24} {status}")
            if limit and written >= limit:
                break
            time.sleep(0.05)
    return {"written": written, "skipped": skipped, "errors": errors, "out": str(out_path), "previews": str(preview_dir)}


def _identify_args(row: dict, views: dict):
    state = views["state"]
    mode = normalize_mode(row.get("relation") or "SAME")
    cls_name = (state.get("upcoming_action") or {}).get("target_class") or "object"
    relation = Relation(mode, None, row.get("rule") or "log")
    candidates = []
    for index, (label, candidate) in enumerate((state.get("candidates") or {}).items()):
        bbox = tuple(float(value) for value in candidate["bbox_xyxy"])
        candidates.append(
            Candidate(
                idx=index,
                object_id=label,
                cls=cls_name,
                score=float(candidate.get("detector_score") or 0.0),
                bbox=bbox,
                dist=candidate.get("distance_m"),
            )
        )
    observation = Observation(
        frame=views["overlay"],
        reference_frame=views["object_crop"],
        context_frame=views["context_crop"],
        goal_text=str((state.get("task") or {}).get("goal") or ""),
        subgoal_text=str((state.get("task") or {}).get("current_subgoal") or ""),
        history=tuple(state.get("action_history") or ()),
        task_type=str((state.get("task") or {}).get("type") or ""),
        notes=_notes(state),
    )
    return relation, candidates, observation


def _notes(state: dict) -> str:
    lines = [str(state.get("relation_mode") or "")]
    upcoming = state.get("upcoming_action") or {}
    lines.append(f"Upcoming action: {upcoming.get('action')} on class {upcoming.get('target_class')}.")
    for label, candidate in (state.get("candidates") or {}).items():
        lines.append(
            f"{label}: bbox {candidate.get('bbox_xyxy')}, score {candidate.get('detector_score')}, "
            f"dist {candidate.get('distance_m')}, {candidate.get('position_relative_to_reference')}"
        )
    return "\n".join(line for line in lines if line)


def _cost_keys(path: Path) -> set[str]:
    keys = set()
    if not path.exists():
        return keys
    for row in _load(path):
        raw = row.get("C")
        if not raw or "event_key" not in row:
            continue
        if len(raw) >= 2:
            keys.add(row["event_key"])
    return keys


def _existing_keys(path: Path) -> set[str]:
    if not path.exists():
        return set()
    keys = set()
    for row in _load(path):
        if row.get("event_key") and "p" in row:
            keys.add(row["event_key"])
    return keys


def _load(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


if __name__ == "__main__":
    main()
