import json
from pathlib import Path

import pytest

from experiments.paths import JW_ROOT
from experiments.thor_jsonl import (
    reconstruct_decision,
    report_directory,
    run,
    vs_host,
    write_results,
)


def _episode(idx: int, *, success: bool, events: list[dict], **extra) -> dict:
    row = {
        "idx": idx,
        "floor": extra.get("floor", "FloorPlan10"),
        "target": extra.get("target", "SaltShaker"),
        "parent": extra.get("parent", "Cabinet"),
        "task_id": extra.get("task_id", f"trial_{idx}"),
        "pickup_ok": True,
        "put_ok": success,
        "success": success,
        "events": events,
    }
    return row


def test_put_success_gate_selects_only_when_peaked():
    select = reconstruct_decision(
        {"n": 2, "p": [0.97, 0.02, 0.01], "relation": "SAME", "source": "jev"}
    )
    assert select["action"] == "SELECT"
    assert select["idx"] == 0
    defer = reconstruct_decision(
        {"n": 3, "p": [0.70, 0.18, 0.10, 0.02], "relation": "SAME", "source": "jev"}
    )
    assert defer["action"] == "RETURN"


def test_jsonl_report_on_synthetic_split(tmp_path: Path):
    host = [
        _episode(
            0,
            success=True,
            events=[
                {"action": "PickupObject", "id": "a", "ok": True, "n": 1, "defer": False, "reason": "host_score"},
                {"action": "PutObject", "id": "cab0", "ok": True, "n": 2, "defer": True, "reason": "defer:regret_not_below_deferral"},
            ],
        ),
        _episode(
            1,
            success=False,
            floor="FloorPlan219",
            target="Vase",
            parent="Safe",
            events=[
                {"action": "PickupObject", "id": "vase", "ok": True, "n": 1, "defer": False, "reason": "host_score"},
                {"action": "PutObject", "id": "safe0", "ok": False, "n": 2, "defer": True, "reason": "defer:regret_not_below_deferral"},
            ],
        ),
    ]
    jev = [
        _episode(
            0,
            success=True,
            events=[
                {"action": "PickupObject", "id": "a", "ok": True, "n": 1, "defer": False, "reason": "host_score"},
                {
                    "action": "PutObject",
                    "id": "cab0",
                    "ok": True,
                    "n": 2,
                    "relation": "SAME",
                    "defer": False,
                    "reason": "regret_below_deferral",
                    "p": [0.97, 0.02, 0.01],
                    "source": "jev",
                },
            ],
        ),
        _episode(
            1,
            success=False,
            floor="FloorPlan219",
            target="Vase",
            parent="Safe",
            events=[
                {"action": "PickupObject", "id": "vase", "ok": True, "n": 1, "defer": False, "reason": "host_score"},
                {
                    "action": "PutObject",
                    "id": "safe0",
                    "ok": False,
                    "n": 3,
                    "relation": "SAME",
                    "defer": True,
                    "reason": "defer:regret_not_below_deferral",
                    "p": [0.70, 0.18, 0.10, 0.02],
                    "source": "jev",
                },
            ],
        ),
    ]
    for name, rows in (("host", host), ("jev", jev)):
        (tmp_path / f"{name}_0_100.jsonl").write_text(
            "\n".join(json.dumps(row) for row in rows) + "\n",
            encoding="utf-8",
        )
    report = report_directory(tmp_path)
    assert report["sources"]["host"]["sr"] == 0.5
    assert report["sources"]["host"]["fail_idxs"] == [1]
    jev_trace = report["sources"]["jev"]["trace"]
    assert jev_trace["n_events"] == 2
    assert jev_trace["select"] == 1
    assert jev_trace["return"] == 1
    assert jev_trace["rebuild_match_rate"] == 1.0
    assert report["sources"]["jev"]["vs_host"]["gained"] == 0
    assert report["sources"]["jev"]["vs_host"]["lost"] == 0


def test_vs_host_counts_gained_episode():
    host = [_episode(0, success=False, events=[{"action": "PutObject", "id": "sink"}])]
    other = [_episode(0, success=True, events=[{"action": "PutObject", "id": "sinkbasin"}])]
    pack = vs_host(host, other)
    assert pack["gained"] == 1
    assert pack["lost"] == 0
    assert pack["put_id_diff"] == 1


def test_capeam_unseen_lab_jsonl_when_present():
    directory = JW_ROOT / "results/thor_port_v2/capeam/valid_unseen"
    if not (directory / "jev_0_100.jsonl").exists():
        pytest.skip("CAPEAM unseen THOR jsonl is not on this machine")
    report = run()
    split = report["splits"]["capeam/valid_unseen"]
    assert "error" not in split
    for source in ("host", "oracle", "jev", "openai_decisions"):
        pack = split["sources"][source]
        assert pack["n"] == 100
        assert pack["sr"] == 0.91
        assert pack["pickup"] == 1.0
        assert pack["fail_idxs"] == [21, 22, 23, 54, 55, 56, 87, 88, 89]
    jev = split["sources"]["jev"]["trace"]
    openai = split["sources"]["openai_decisions"]["trace"]
    assert jev["n_events"] == 63
    assert jev["select"] == 1
    assert jev["selects"][0]["idx"] == 36
    assert openai["select"] == 0
    assert split["sources"]["oracle"]["trace"]["select"] == 0
    assert split["sources"]["host"]["trace"]["select"] == 0
    assert jev["rebuild_match_rate"] == 1.0
    assert openai["rebuild_match_rate"] == 1.0
    assert split["sources"]["jev"]["vs_host"]["gained"] == 0
    assert split["sources"]["jev"]["vs_host"]["lost"] == 0
    recheck = report["splits"]["capeam/valid_unseen_recheck"]
    assert recheck["sources"]["host"]["sr"] == 0.91
    assert recheck["sources"]["jev"]["trace"]["select"] == 0
    assert recheck["sources"]["openai_decisions"]["trace"]["select"] == 0
    assert recheck["sources"]["host"]["fail_idxs"] == [21, 22, 23, 54, 55, 56, 87, 88, 89]
    json_path, md_path = write_results(report)
    assert json_path.exists()
    assert "CAPEAM Unseen" in md_path.read_text(encoding="utf-8")
