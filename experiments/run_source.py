"""Run one identity experiment: Jev or OpenAI Decisions.

    python -m experiments.run_source jev
    python -m experiments.run_source openai_decisions
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from trace.redact import has_env, load_env

from .live import LIVE_SOURCES, run as run_live

ROOT = Path(__file__).resolve().parents[1]
KEYS = {name: keys for name, _, keys in LIVE_SOURCES}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run TRACE identity with exactly one decision model.")
    parser.add_argument("source", choices=[name for name, _, _ in LIVE_SOURCES])
    args = parser.parse_args(argv)
    load_env(ROOT / ".env")
    if not has_env(*KEYS[args.source]):
        print(f"{args.source}: API key is missing", file=sys.stderr)
        return 2
    out = ROOT / "results" / args.source / "live.json"
    run_live(include_live=True, sources=[args.source], out=out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
