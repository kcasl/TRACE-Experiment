import json
from pathlib import Path

import numpy as np
import pytest

from experiments.paths import ryankwon_root
from experiments.replay import replay_split, run
from experiments.replay_slices import inventory, run as run_slices
from trace.consequence.ff import costs_from_row
from trace.gate import decide, decision_utility, oracle_decision
from trace.types import Belief


def test_ff_log_row_becomes_cost_matrix():
    row = {
        "C": [[0.0, 2.0, None], [4.0, 0.0, 1.0]],
        "reachable": [[1, 1, 0], [1, 1, 1]],
        "cand_gt": ["a", "b"],
        "action": "PutObject",
    }
    matrix, reachable = costs_from_row(row)
    assert matrix.C.shape == (2, 3)
    assert np.isinf(matrix.C[0, 2])
    assert bool(reachable[0, 2]) is False
    belief = Belief(p_identity=np.array([0.9, 0.1, 0.0]), source="fixed")
    decision = decide(belief, matrix, mode="SAME", deferral_cost=4.0)
    assert decision.action == "SELECT"
    assert decision.idx == 0
    costs, reach = matrix.C, reachable
    oracle = oracle_decision(costs, reach, 0, 4.0, 100.0)
    assert oracle.action == "SELECT" and oracle.idx == 0
    assert decision_utility(oracle, 0, costs, reach, 4.0, 100.0) == 0.0
    returned = decide(belief, matrix, mode="SAME", deferral_cost=4.0, rule="threshold", tau=0.99)
    assert returned.action == "RETURN"
    assert decision_utility(returned, 0, costs, reach, 4.0, 100.0) == 4.0


def test_replay_on_synthetic_mask_log(tmp_path: Path):
    costs = tmp_path / "costs.jsonl"
    beliefs = tmp_path / "beliefs"
    beliefs.mkdir()
    row = {
        "event_key": "ep__t0001",
        "relation": "SAME",
        "j_true": 0,
        "host_choice": 1,
        "action": "PutObject",
        "cand_gt": ["mug_a", "mug_b"],
        "C": [[0.0, 2.0, 2.0], [8.0, 0.0, 4.0]],
        "reachable": [[1, 1, 1], [1, 1, 1]],
    }
    costs.write_text(json.dumps(row) + "\n", encoding="utf-8")
    (beliefs / "jev.jsonl").write_text(
        json.dumps({"event_key": "ep__t0001", "p": [0.8, 0.2, 0.0]}) + "\n",
        encoding="utf-8",
    )
    (beliefs / "appearance.jsonl").write_text(
        json.dumps({"event_key": "ep__t0001", "p": [0.1, 0.8, 0.1]}) + "\n",
        encoding="utf-8",
    )
    table = replay_split(costs, beliefs)
    jev = table["jev|trace"]
    assert jev["n"] == 1
    assert jev["critic"] == "Metric-FF (precomputed C)"
    assert jev["wrong_instance_rate"] == 0.0
    dino = table["dinov2|trace"]
    assert dino["n"] == 1
    assert table["jev|host_score"]["wrong_instance_rate"] == 1.0


def test_film_capeam_ff_logs_replay_when_present():
    root = ryankwon_root()
    costs = root / "results/offline/film/valid_seen_host_v2_cand/costs.jsonl"
    if not costs.exists():
        pytest.skip("FILM FF cost log is not on this machine")
    report = run(root=root)
    film = report["splits"]["film/valid_seen"]["table"]
    assert film["jev|trace"]["n"] == 92
    assert film["dinov2|trace"]["n"] == 92
    assert film["oracle|trace"]["n"] == 92
    if "openai_decisions|trace" in film:
        assert film["openai_decisions|trace"]["n"] == 92
    assert "capeam/valid_seen" in report["splits"]
    assert report["splits"]["capeam/valid_seen"]["n_episodes"] == 820
    assert report["splits"]["film/valid_seen"]["n_episodes"] == 820


def test_replay_slices_j_true_null_when_present():
    root = ryankwon_root()
    costs = root / "results/offline/capeam/valid_seen_host_v2_cand/costs.jsonl"
    if not costs.exists():
        pytest.skip("CAPEAM FF cost log is not on this machine")
    capeam = inventory(costs)
    assert capeam["N_ge2"] == 84
    assert capeam["j_true_null"] == 42
    report = run_slices()
    assert report["splits"]["capeam/valid_unseen"]["error"] == "missing logs"
    film = report["splits"]["film/valid_unseen"]["jev_trace"]
    assert film["action=PickupObject"]["n"] == 13
