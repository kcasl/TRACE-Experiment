"""One command: run the tests, run every experiment, write the report.

    python -m experiments.run_all            # tests + offline + scenes + live
    python -m experiments.run_all --offline  # skip the network calls
    python -m experiments.run_all --no-tests # skip pytest

Artifacts land in results/: offline_rules.json, scene_costs.json,
live_identity.json, results.json (everything), and REPORT.md.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from trace.redact import has_env, load_env

from . import report
from .compare import heterogeneous_table
from .compare_sources import compare as compare_live_sources
from .compare_sources import render_markdown as render_source_compare
from .live import LIVE_SOURCES, run as run_live
from .protocol import official_table
from .replay import run as run_replay
from .scene_costs import scene_cost_tables

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def main(argv: list[str] | None = None) -> int:
    args = _parse(argv)
    load_env(ROOT / ".env")
    RESULTS.mkdir(parents=True, exist_ok=True)

    tests_passed = None if args.no_tests else _run_tests()

    offline = heterogeneous_table()
    _write("offline_rules.json", offline)

    scenes = scene_cost_tables()
    _write("scene_costs.json", scenes)

    live = {"jev": None, "openai_decisions": None}
    source_compare = None
    if not args.offline:
        key_map = {name: keys for name, _, keys in LIVE_SOURCES}
        for name in live:
            if has_env(*key_map[name]):
                live[name] = run_live(
                    include_live=True,
                    sources=[name],
                    out=RESULTS / name / "live.json",
                )
            else:
                print(f"{name}: no API key; oracle-only rows")
                live[name] = run_live(
                    include_live=False,
                    sources=[name],
                    out=RESULTS / name / "live.json",
                )
        source_compare = compare_live_sources(live["jev"], live["openai_decisions"])
        _write("source_compare.json", source_compare)
        (RESULTS / "source_compare.md").write_text(render_source_compare(source_compare), encoding="utf-8")
        print(f"wrote {RESULTS / 'source_compare.md'}")

    official = official_table()
    _write("official_sr.json", official)

    replay = run_replay()
    _write("log_replay.json", replay)

    _write(
        "results.json",
        {
            "tests_passed": tests_passed,
            "offline": offline,
            "scenes": scenes,
            "live": live,
            "source_compare": source_compare,
            "official_sr": official,
            "replay": replay,
        },
    )
    markdown = report.render(
        offline,
        scenes,
        live.get("jev") if isinstance(live, dict) else live,
        tests_passed=tests_passed,
        replay=replay,
        official_sr=official,
        source_compare=source_compare,
        live_openai=live.get("openai_decisions") if isinstance(live, dict) else None,
    )
    (RESULTS / "REPORT.md").write_text(markdown, encoding="utf-8")
    print(f"wrote {RESULTS / 'REPORT.md'}")

    if tests_passed is False:
        return 1
    return 0


def _parse(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the TRACE tests and experiments.")
    parser.add_argument("--offline", action="store_true", help="skip the live API calls")
    parser.add_argument("--no-tests", action="store_true", help="skip pytest")
    return parser.parse_args(argv)


def _run_tests() -> bool:
    print("running pytest ...")
    completed = subprocess.run([sys.executable, "-m", "pytest", "-q"], cwd=ROOT)
    return completed.returncode == 0


def _write(name: str, payload: dict) -> None:
    path = RESULTS / name
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {path}")


if __name__ == "__main__":
    raise SystemExit(main())
