import json
from pathlib import Path

from experiments.thor_costs import build_split, costs_row_from_event, markdown, run, write_results
from experiments.thor_rollout import event_cost_fields, success_cost_matrix


def test_put_success_matrix_zero_on_true_mask():
    costs, reachable = success_cost_matrix(2, fail=100.0)
    assert costs == [[0.0, 100.0, 100.0], [100.0, 0.0, 100.0]]
    assert reachable[0][0] == 1


def test_event_cost_fields_put_does_not_need_controller():
    from trace.types import Candidate

    cands = [
        Candidate(idx=0, object_id="cab0", cls="Cabinet", score=0.8),
        Candidate(idx=1, object_id="cab1", cls="Cabinet", score=0.2),
    ]
    payload = event_cost_fields(cands, "PutObject", {"objectId": "cab1"}, controller=None, opened=set())
    assert payload["j_true"] == 1
    assert payload["host_choice"] == 0
    assert "C" not in payload


def test_build_split_matches_logged_p(tmp_path: Path):
    host = [
        {
            "idx": 0,
            "task_id": "t0",
            "floor": "FloorPlan10",
            "target": "SaltShaker",
            "parent": "Cabinet",
            "events": [
                {
                    "action": "PutObject",
                    "id": "cab0",
                    "ok": True,
                    "n": 2,
                    "relation": "SAME",
                    "cand_ids": ["cab0", "cab1"],
                    "expert_id": "cab0",
                    "j_true": 0,
                    "host_choice": 1,
                    "C": [[0.0, 100.0, 100.0], [100.0, 0.0, 100.0]],
                    "reachable": [[1, 1, 1], [1, 1, 1]],
                }
            ],
        }
    ]
    jev = [
        {
            "idx": 0,
            "task_id": "t0",
            "events": [
                {
                    "action": "PutObject",
                    "n": 2,
                    "p": [0.97, 0.02, 0.01],
                    "defer": False,
                    "reason": "regret_below_deferral",
                }
            ],
        }
    ]
    openai = [
        {
            "idx": 0,
            "task_id": "t0",
            "events": [
                {
                    "action": "PutObject",
                    "n": 2,
                    "p": [0.70, 0.20, 0.10],
                    "defer": True,
                    "reason": "defer:regret_not_below_deferral",
                }
            ],
        }
    ]
    packed = build_split("capeam/valid_unseen", host, {"jev": jev, "openai_decisions": openai})
    assert packed["n_cost_events"] == 1
    assert packed["matched"] == {"jev": 1, "openai_decisions": 1}
    row = packed["costs"][0]
    assert costs_row_from_event("capeam/valid_unseen", host[0], host[0]["events"][0])["j_true"] == 0
    assert row["cand_gt"] == ["cab0", "cab1"]
    from experiments.thor_costs import write_split, replay_table, summarize_table

    write_split(packed, tmp_path)
    table = summarize_table(replay_table(tmp_path))
    assert table["jev|trace"]["n"] == 1
    assert table["jev|trace"]["CR"] == 0.0
    assert table["openai_decisions|trace"]["return"] == 1.0
    assert table["openai_decisions|trace"]["CR"] == 4.0


def test_load_source_rows_merges_jsonl_by_idx(tmp_path: Path):
    from experiments.thor_costs import load_source_rows

    (tmp_path / "host_0_2.jsonl").write_text(
        json.dumps({"idx": 0, "task_id": "a"}) + "\n" + json.dumps({"idx": 1, "task_id": "b"}) + "\n",
        encoding="utf-8",
    )
    (tmp_path / "host_2_3.jsonl").write_text(json.dumps({"idx": 2, "task_id": "c"}) + "\n", encoding="utf-8")
    rows = load_source_rows(tmp_path, "host")
    assert [row["idx"] for row in rows] == [0, 1, 2]


def test_run_reports_missing_host_dump(tmp_path: Path):
    report = run(costs_root=tmp_path / "missing", beliefs_root=tmp_path / "beliefs")
    assert report["splits"]["capeam/valid_unseen"]["error"] == "missing host dump"
    text = markdown(report)
    assert "CAPEAM valid_unseen official" in text
    json_path, md_path = write_results(report, root=tmp_path)
    assert json.loads(json_path.read_text(encoding="utf-8"))["critic"]
    assert md_path.exists()
