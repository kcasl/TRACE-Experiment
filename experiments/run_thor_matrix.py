"""Run the ported THOR matrix: FILM/CAPEAM × seen/unseen × Jev/OpenAI.

    python -m experiments.run_thor_matrix --limit 2          # smoke
    python -m experiments.run_thor_matrix                    # full oct21 (820 / 821)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .thor_rollout import OUT, main as rollout_main

HOSTS = ("film", "capeam")
SPLITS = ("valid_seen", "valid_unseen")
LIVE = ("jev", "openai_decisions")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--from-idx", type=int, default=0)
    parser.add_argument("--hosts", nargs="*", default=list(HOSTS))
    parser.add_argument("--splits", nargs="*", default=list(SPLITS))
    parser.add_argument("--sources", nargs="*", default=["host", "oracle", *LIVE])
    parser.add_argument("--out", default="", help="results root passed to thor_rollout")
    args = parser.parse_args(argv)
    out_root = Path(args.out) if args.out else OUT

    failures = []
    for host in args.hosts:
        for split in args.splits:
            cmd = [
                "--host",
                host,
                "--split",
                split,
                "--from-idx",
                str(args.from_idx),
                "--limit",
                str(args.limit),
            ]
            if args.out:
                cmd.extend(["--out", args.out])
            for source in args.sources:
                cmd.extend(["--source", source])
            print("RUN", " ".join(cmd), flush=True)
            try:
                code = rollout_main(cmd)
            except Exception as exc:
                print(f"SPLIT FAIL {host} {split}: {exc}", flush=True)
                failures.append({"host": host, "split": split, "error": str(exc)})
                continue
            if code:
                failures.append({"host": host, "split": split, "code": code})
    combined = _combine(out_root)
    path = out_root / "matrix_summary.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"failures": failures, "table": combined}, indent=2), encoding="utf-8")
    print(f"wrote {path}", flush=True)
    _write_markdown(combined, failures, out_root)
    return 1 if failures else 0


def _combine(out_root: Path) -> list[dict]:
    rows = []
    if not out_root.exists():
        return rows
    for path in sorted(out_root.glob("*/*/summary_*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        host = payload.get("host") or path.parts[-3]
        split = payload.get("split") or path.parts[-2]
        for source, stats in (payload.get("sources") or {}).items():
            rows.append({"host": host, "split": split, "source": source, **stats})
    return rows


def _write_markdown(table: list[dict], failures: list[dict], out_root: Path) -> None:
    lines = [
        "# Ported THOR 5 TRACE matrix",
        "",
        "Full oct21 (all seven ALFRED types). FILM host = nearest to agent; CAPEAM host = Mask R-CNN.",
        "Jev/OpenAI: unique-min + planner extra C (c_defer=4). No off-type Put prior.",
        "",
        "| host | split | source | n | SR | pickup | put | api_errors |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for row in table:
        lines.append(
            f"| {row['host']} | {row['split']} | {row['source']} | {row['n']} | "
            f"{row['sr']:.3f} | {row['pickup']:.3f} | {row['put']:.3f} | {row.get('api_errors', 0)} |"
        )
    if failures:
        lines += ["", "Failed runs: " + json.dumps(failures)]
    (out_root / "matrix_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
