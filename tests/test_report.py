import numpy as np

from experiments.compare import heterogeneous_table
from experiments.report import render
from experiments.scene_costs import scene_cost_tables


def test_scene_cost_tables_match_documented_values():
    tables = scene_cost_tables()
    scenes = {scene["scene"]: scene for scene in tables["scenes"]}

    near = scenes["far_fridge"]
    # Columns are identity hypotheses. Mask 0 is the near fridge, mask 1 the far one.
    np.testing.assert_allclose(near["regret_matrix"], [[0.0, 2.0, 2.0], [22.0, 0.0, 12.0]])
    # Belief [0.6, 0.4, 0]: the near mask is cheap in expectation, so TRACE commits to it.
    np.testing.assert_allclose(near["expected_regret"], [0.8, 13.2])
    assert near["trace_action"] == "SELECT" and near["argmin_regret"] == 0

    both_far = scenes["split_fridges"]
    np.testing.assert_allclose(both_far["expected_regret"], [4.8, 7.2])
    assert both_far["trace_action"] == "RETURN"

    close = scenes["open_drawers_to_close"]
    np.testing.assert_allclose(close["regret_matrix"], [[0.0, 1.0, 1.0], [1.0, 0.0, 1.0]])


def test_report_renders_every_section():
    offline = heterogeneous_table()
    scenes = scene_cost_tables()
    live = {
        "cases": ["visible_match"],
        "rows": [
            {"case": "visible_match", "source": "oracle", "ok": True, "p_identity": [1.0, 0.0, 0.0], "top": 0, "identity_correct": True},
            {"case": "visible_match", "source": "jev", "ok": False, "skipped": "no_api_key"},
        ],
        "identity_accuracy": {"oracle": 1.0},
    }
    markdown = render(
        offline,
        scenes,
        live,
        tests_passed=True,
        official_sr={
            "max_fails": 10,
            "max_steps": 1000,
            "rows": [{"host": "film", "split": "valid_seen", "selector": "host", "n": 820, "sr": 0.232, "gc": 0.364, "steps": 473, "fails": 6.0}],
        },
        replay={
            "min_n": 2,
            "deferral_cost": 4.0,
            "splits": {"film/valid_seen": {"table": {"jev|trace": {"n": 92, "commitment_regret": 0.2, "wrong_instance_rate": 0.1, "return_rate": 0.05, "unrecoverable_rate": 0.0}}}},
        },
    )
    assert "# TRACE results" in markdown
    assert "Unit tests: passed." in markdown
    assert "## Decision rules on the same belief" in markdown
    assert "## Symbolic-planner regret by scene" in markdown
    assert "## Identity beliefs from decision models" in markdown
    assert "Jev identity experiment" not in markdown
    assert "## Official 10-failure SR (FILM / CAPEAM, 820 ep)" in markdown
    assert "## Mask-log replay (Metric-FF critic, DINOv2 / Jev / oracle)" in markdown
    assert "no_api_key" in markdown
    # The mean-cost ordering is the paper's headline result.
    assert offline["mean_realized_cost"]["trace"] < offline["mean_realized_cost"]["host_score"]
