from pathlib import Path

from experiments.write_paper_tables import collect, write_tex


def _episode(idx, success, *, source, put_id, n=2, p=None, reason="host_score"):
    event = {
        "action": "PutObject",
        "id": put_id,
        "ok": success,
        "n": n,
        "reason": reason,
        "cand_ids": ["a", "b"],
        "j_true": 0,
        "host_choice": 0 if put_id == "a" else 1,
        "C": [[0.0, 2.0, 2.0], [2.0, 0.0, 2.0]],
        "reachable": [[1, 1, 1], [1, 1, 1]],
        "relation": "SAME",
    }
    if p is not None:
        event["p"] = p
        event["source"] = source
    return {
        "idx": idx,
        "selector": source,
        "floor": "FloorPlan422",
        "target": "HandTowel",
        "parent": "SinkBasin",
        "success": success,
        "pickup_ok": True,
        "put_ok": success,
        "events": [{"action": "PickupObject", "id": "t", "ok": True, "n": 1, "reason": "host_score"}, event],
    }


def test_write_tex_from_tiny_jsonl(tmp_path: Path):
    root = tmp_path / "thor_port_v3"
    for host, split, n in (("film", "valid_seen", 2), ("film", "valid_unseen", 1), ("capeam", "valid_seen", 1), ("capeam", "valid_unseen", 1)):
        directory = root / host / split
        directory.mkdir(parents=True)
        host_rows = [_episode(i, True, source="host", put_id="a") for i in range(n)]
        jev_rows = [_episode(0, True, source="jev", put_id="a", p=[0.93, 0.05, 0.02], reason="regret_below_deferral")]
        if n > 1:
            jev_rows.append(_episode(1, True, source="jev", put_id="b", p=[0.93, 0.05, 0.02], reason="regret_below_deferral"))
            host_rows[1] = _episode(1, False, source="host", put_id="b")
        openai_rows = list(jev_rows)
        (directory / f"host_0_{n}.jsonl").write_text("\n".join(__import__("json").dumps(r) for r in host_rows) + "\n")
        (directory / f"jev_0_{n}.jsonl").write_text("\n".join(__import__("json").dumps(r) for r in jev_rows) + "\n")
        (directory / f"openai_decisions_0_{n}.jsonl").write_text("\n".join(__import__("json").dumps(r) for r in openai_rows) + "\n")
    tex_dir = tmp_path / "tex"
    write_tex(collect(root), tex_dir)
    thor = (tex_dir / "table_thor_sr.tex").read_text(encoding="utf-8")
    assert "TRACE + Jev" in thor
    assert r"\label{tab:thor}" in thor
    gain = (tex_dir / "table_gain.tex").read_text(encoding="utf-8")
    assert "Jev" in gain and "OpenAI" in gain
    assert r"\label{tab:gain}" in gain
    delta = (tex_dir / "table_v1v2.tex").read_text(encoding="utf-8")
    assert r"\Delta$SR" in delta or "net ep" in delta
