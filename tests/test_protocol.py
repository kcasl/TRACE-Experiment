import json
from pathlib import Path

from experiments.protocol import OFFICIAL_10FAIL, episode_metrics, official_table, summarize


def test_ten_failure_cutoff_reads_timeline_not_final_success():
    episode = {
        "success": True,
        "gc": 1.0,
        "steps": 80,
        "fail_steps": [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 40],
        "goal_timeline": [[18, False, 0, 2], [20, False, 0, 2], [80, True, 2, 2]],
    }
    metrics = episode_metrics(episode)
    assert metrics["success_B"] is True
    assert metrics["gc_B"] == 1.0
    assert metrics["success_A"] is False
    assert metrics["gc_A"] == 0.0
    assert metrics["steps_A"] == 20
    assert metrics["counted_fails"] == 11


def test_fewer_than_ten_failures_keeps_final_success():
    episode = {
        "success": True,
        "gc": 0.5,
        "steps": 30,
        "fail_steps": [3, 9],
        "goal_timeline": [[30, True, 1, 2]],
    }
    metrics = episode_metrics(episode)
    assert metrics["success_A"] is True
    assert metrics["gc_A"] == 0.5


def test_official_film_capeam_sr_matches_table():
    by_key = {(row["host"], row["split"], row["selector"]): row for row in OFFICIAL_10FAIL}
    assert by_key[("film", "valid_seen", "host")]["n"] == 820
    assert by_key[("film", "valid_seen", "host")]["sr"] == 0.232
    assert by_key[("film", "valid_seen", "host")]["gc"] == 0.364
    assert by_key[("film", "valid_unseen", "host")]["sr"] == 0.205
    assert by_key[("capeam", "valid_seen", "host")]["n"] == 820
    assert by_key[("capeam", "valid_seen", "host")]["sr"] == 0.443
    table = official_table()
    assert table["max_fails"] == 10
    assert len(table["rows"]) == 8


def test_summarize_from_jsonl(tmp_path: Path):
    directory = tmp_path / "rollout" / "shard0"
    directory.mkdir(parents=True)
    lines = [
        {"episode_id": "a", "success": True, "gc": 1.0, "steps": 10, "fail_steps": [], "goal_timeline": [[10, True, 1, 1]]},
        {"episode_id": "b", "success": False, "gc": 0.0, "steps": 40, "fail_steps": list(range(10)), "goal_timeline": [[9, False, 0, 1]]},
    ]
    (directory / "episodes.jsonl").write_text("\n".join(json.dumps(row) for row in lines), encoding="utf-8")
    from experiments.protocol import load_episodes

    summary = summarize(load_episodes(tmp_path / "rollout"))
    assert summary["n"] == 2
    assert summary["sr"] == 0.5
    assert summary["cut10"] == 0.5
