"""Re-execute logged pickup/put ids in THOR 5. No live API.

Confirms that jsonl success is a real simulator outcome, not a copied host row.

    python -m experiments.replay_logged --jsonl-root results/thor_port_v2 --out results/thor_port_v2_idreplay
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.capeam_mrcnn import CapeamMaskRCNN
from experiments.paths import JW_ROOT, THOR_COST_SPLITS
from experiments.replay import load_jsonl
from experiments.thor_rollout import OracleIdentity, load_split, load_traj, run_episode, summarize

SOURCES = ("host", "oracle", "jev", "openai_decisions")


def force_ids_from_row(row: dict) -> dict[str, str]:
    out: dict[str, str] = {}
    for item in row.get("events") or []:
        action = item.get("action")
        oid = item.get("id")
        if action in ("PickupObject", "PutObject") and oid:
            out[str(action)] = str(oid)
    return out


def jsonl_path(directory: Path, source: str) -> Path | None:
    hits = sorted(directory.glob(f"{source}_*.jsonl"))
    return hits[0] if hits else None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl-root", default=str(JW_ROOT / "results" / "thor_port_v2"))
    parser.add_argument("--out", default=str(JW_ROOT / "results" / "thor_port_v2_idreplay"))
    parser.add_argument("--hosts", nargs="*", default=["film", "capeam"])
    parser.add_argument("--splits", nargs="*", default=["valid_seen", "valid_unseen"])
    parser.add_argument("--sources", nargs="*", default=["jev", "openai_decisions"])
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--from-idx", type=int, default=0)
    args = parser.parse_args(argv)

    from ai2thor.controller import Controller
    from ai2thor.platform import CloudRendering

    jsonl_root = Path(args.jsonl_root)
    out_root = Path(args.out)
    controller = Controller(
        platform=CloudRendering,
        scene="FloorPlan1",
        width=300,
        height=300,
        gridSize=0.25,
        renderInstanceSegmentation=True,
    )
    oracle_src = OracleIdentity()
    detectors: dict[str, CapeamMaskRCNN | None] = {}
    report = {"splits": {}}
    try:
        for spec in THOR_COST_SPLITS:
            if spec["host"] not in args.hosts or spec["split"] not in args.splits:
                continue
            tasks = load_split(spec["split"])
            start = args.from_idx
            stop = start + args.limit if args.limit else len(tasks)
            tasks = tasks[start:stop]
            src_dir = jsonl_root / spec["host"] / spec["split"]
            out_dir = out_root / spec["host"] / spec["split"]
            out_dir.mkdir(parents=True, exist_ok=True)
            if spec["host"] not in detectors:
                detectors[spec["host"]] = CapeamMaskRCNN() if spec["host"] == "capeam" else None
            detector = detectors[spec["host"]]
            split_stats = {}
            for source in args.sources:
                path = jsonl_path(src_dir, source)
                if path is None:
                    split_stats[source] = {"error": "missing jsonl"}
                    continue
                logged = {int(row["idx"]): row for row in load_jsonl(path) if row.get("idx") is not None}
                out_path = out_dir / f"{source}_{start}_{start + len(tasks)}.jsonl"
                rows = []
                match = 0
                with out_path.open("w", encoding="utf-8") as handle:
                    for i, task in enumerate(tasks):
                        idx = start + i
                        old = logged.get(idx)
                        if old is None:
                            continue
                        force = force_ids_from_row(old)
                        traj = load_traj(task["path"])
                        oracle_src.target_id = None
                        row = run_episode(
                            controller,
                            traj,
                            "host",
                            None,
                            oracle_src,
                            host=spec["host"],
                            detector=detector,
                            force_ids=force,
                        )
                        row["idx"] = idx
                        row["rel"] = task["rel"]
                        row["logged_success"] = bool(old.get("success"))
                        row["success_match"] = bool(row.get("success")) == bool(old.get("success"))
                        match += int(row["success_match"])
                        rows.append(row)
                        handle.write(json.dumps(row) + "\n")
                        handle.flush()
                        print(
                            f"{idx:03d} {spec['host']:6s} {source:18s} logged={int(old.get('success') or 0)} "
                            f"replay={int(row.get('success') or 0)} match={int(row['success_match'])}",
                            flush=True,
                        )
                n = len(rows) or 1
                split_stats[source] = {
                    **summarize(rows, ["host"])["host"],
                    "logged_sr": (sum(bool(r.get("logged_success")) for r in rows) / n),
                    "success_match": match / n,
                    "n": len(rows),
                }
            report["splits"][spec["name"]] = split_stats
    finally:
        try:
            controller.stop()
        except Exception:
            pass
    out_root.mkdir(parents=True, exist_ok=True)
    summary_path = out_root / "idreplay_summary.json"
    summary_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    print(f"wrote {summary_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
