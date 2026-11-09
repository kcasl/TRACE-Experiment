from experiments.compare_sources import compare, render_markdown
from experiments.live import run as run_live


def test_live_source_filter_keeps_only_that_experiment():
    report = run_live(include_live=False, sources=["jev"])
    assert {row["source"] for row in report["rows"]} == {"oracle"}
    assert "openai_decisions" not in {row["source"] for row in report["rows"]}


def test_compare_joins_the_two_experiments():
    jev = {
        "identity_accuracy": {"jev": 1.0},
        "cases": ["visible_match"],
        "rows": [
            {
                "case": "visible_match",
                "truth": 0,
                "source": "jev",
                "ok": True,
                "p_identity": [0.9, 0.05, 0.05],
                "top": 0,
                "identity_correct": True,
                "decisions": {
                    "same_room_drawers": {"trace": "SELECT 0"},
                    "far_fridge": {"trace": "SELECT 0"},
                    "split_fridges": {"trace": "RETURN"},
                },
            }
        ],
    }
    openai = {
        "identity_accuracy": {"openai_decisions": 1.0},
        "cases": ["visible_match"],
        "rows": [
            {
                "case": "visible_match",
                "truth": 0,
                "source": "openai_decisions",
                "ok": True,
                "p_identity": [0.6, 0.2, 0.2],
                "top": 0,
                "identity_correct": True,
                "decisions": {
                    "same_room_drawers": {"trace": "SELECT 0"},
                    "far_fridge": {"trace": "RETURN"},
                    "split_fridges": {"trace": "RETURN"},
                },
            }
        ],
    }
    payload = compare(jev, openai)
    assert payload["summary"]["top1_agree"] == 1.0
    assert payload["summary"]["mean_l1"] > 0
    assert payload["summary"]["trace_agree"]["same_room_drawers"] == 1.0
    assert payload["summary"]["trace_agree"]["far_fridge"] == 0.0
    markdown = render_markdown(payload)
    assert "SELECT 0 vs RETURN" in markdown
