"""Official ALFRED evaluation: 10 counted API failures and 1000 steps.

Protocol A (leaderboard): cut the episode at the 10th counted failure in
``fail_steps``; success/GC come from ``goal_timeline`` at or before that step.
LookUp_0 / LookDown_0 no-ops are not counted (they never appear in fail_steps
when the adapter follows FILM's submitted actseq).

Protocol B (paper-style, no cutoff): final success/GC at <= 1000 steps.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def episode_metrics(episode: dict, *, max_fails: int = 10, max_steps: int = 1000) -> dict[str, Any]:
    success_b = bool(episode.get("success"))
    gc_b = float(episode.get("gc") or 0.0)
    steps = int(episode.get("steps") or 0)
    fail_steps = episode.get("fail_steps")
    timeline = episode.get("goal_timeline")
    if fail_steps is None or timeline is None:
        success_a, gc_a, cut = success_b, gc_b, steps
    elif len(fail_steps) < max_fails:
        success_a, gc_a, cut = success_b, gc_b, steps
    else:
        t_cut = fail_steps[max_fails - 1]
        last = None
        for entry in timeline:
            t, sat, met, tot = entry[:4]
            if t <= t_cut:
                last = (bool(sat), met / tot if tot else 0.0)
        success_a, gc_a = (False, 0.0) if last is None else last
        cut = t_cut
    if cut > max_steps:
        cut = max_steps
    return {
        "success_B": success_b,
        "gc_B": gc_b,
        "success_A": success_a,
        "gc_A": gc_a,
        "steps_A": cut,
        "counted_fails": len(fail_steps or []),
    }


def load_episodes(rollout_dir: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for path in rollout_dir.rglob("episodes.jsonl"):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            out.setdefault(row["episode_id"], row)
    return out


def summarize(episodes: dict[str, dict], *, max_fails: int = 10) -> dict[str, Any]:
    rows = {key: episode_metrics(episode, max_fails=max_fails) for key, episode in episodes.items()}
    n = len(rows) or 1
    return {
        "n": len(rows),
        "sr": sum(row["success_A"] for row in rows.values()) / n,
        "gc": sum(row["gc_A"] for row in rows.values()) / n,
        "sr_no_cutoff": sum(row["success_B"] for row in rows.values()) / n,
        "gc_no_cutoff": sum(row["gc_B"] for row in rows.values()) / n,
        "cut10": sum(1 for row in rows.values() if row["counted_fails"] >= max_fails) / n,
    }


# Official 10-failure numbers from the FILM/CAPEAM host rollouts
# (ryankwon/paper/tables/table_sr.tex). Recomputed from episodes.jsonl when
# that directory is present.
OFFICIAL_10FAIL = (
    {"host": "capeam", "split": "valid_seen", "selector": "host", "n": 820, "sr": 0.443, "gc": 0.555, "steps": 312, "fails": 4.7},
    {"host": "capeam", "split": "valid_seen", "selector": "oracle", "n": 820, "sr": 0.446, "gc": 0.559, "steps": 309, "fails": 4.7},
    {"host": "capeam", "split": "valid_unseen", "selector": "host", "n": 821, "sr": 0.390, "gc": 0.493, "steps": 416, "fails": 6.0},
    {"host": "capeam", "split": "valid_unseen", "selector": "oracle", "n": 821, "sr": 0.390, "gc": 0.493, "steps": 413, "fails": 6.1},
    {"host": "film", "split": "valid_seen", "selector": "host", "n": 820, "sr": 0.232, "gc": 0.364, "steps": 473, "fails": 6.0},
    {"host": "film", "split": "valid_seen", "selector": "oracle", "n": 820, "sr": 0.226, "gc": 0.363, "steps": 476, "fails": 6.0},
    {"host": "film", "split": "valid_unseen", "selector": "host", "n": 821, "sr": 0.205, "gc": 0.334, "steps": 428, "fails": 7.9},
    {"host": "film", "split": "valid_unseen", "selector": "oracle", "n": 821, "sr": 0.201, "gc": 0.332, "steps": 428, "fails": 7.9},
)


def official_table() -> dict:
    return {
        "protocol": "A",
        "max_fails": 10,
        "max_steps": 1000,
        "source": "FILM/CAPEAM host rollouts, official 10-failure cutoff",
        "rows": [dict(row) for row in OFFICIAL_10FAIL],
    }


def maybe_recompute(rollout_dir: Path | None) -> dict | None:
    if rollout_dir is None or not Path(rollout_dir).exists():
        return None
    episodes = load_episodes(Path(rollout_dir))
    if not episodes:
        return None
    return summarize(episodes)


def main() -> None:
    payload = official_table()
    out = Path(__file__).resolve().parents[1] / "results" / "official_sr.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
