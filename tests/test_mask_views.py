import json
from pathlib import Path

import numpy as np

from experiments.mask_views import labeled_frame, load_jev_log, overlay_instances, overlay_rgb, views_from_belief_row
from trace.belief.openai_decisions import OpenAIDecisionSource
from trace.types import Candidate, Observation, Relation


def test_overlay_instances_tints_mask_pixels():
    frame = np.full((40, 40, 3), 80, dtype=np.uint8)
    mask = np.zeros((40, 40), dtype=bool)
    mask[10:30, 10:20] = True
    out = overlay_instances(frame, [[10, 10, 20, 30]], masks=[mask], labels=["A"], selected=0, fill=0.5, scale=1)
    assert int(out[20, 15, 0]) > 120
    assert abs(int(out[2, 2, 0]) - 80) < 8


def test_overlay_rgb_keeps_the_original_pixels():
    frame = np.zeros((40, 40, 3), dtype=np.uint8)
    frame[:, :] = (10, 20, 30)
    frame[8:32, 8:32] = (200, 10, 10)
    out = overlay_rgb(frame, [[8, 8, 32, 32]], labels=["A"], scale=1)
    assert tuple(out[20, 20]) == (200, 10, 10)


def test_labeled_overlay_draws_every_candidate():
    state = {
        "reference_object": {"bbox_xyxy_in_its_own_frame": [10, 10, 40, 40]},
        "candidates": {
            "A": {"bbox_xyxy": [10, 10, 40, 40], "detector_score": 0.9, "distance_m": 1.0, "position_relative_to_reference": "same"},
            "B": {"bbox_xyxy": [80, 80, 120, 120], "detector_score": 0.8, "distance_m": 1.2, "position_relative_to_reference": "right"},
        },
    }
    frame = labeled_frame(state, (300, 300), scale=2)
    assert frame.shape[0] == 600 and frame.shape[2] == 3
    assert frame.max() > 40


def test_views_join_jev_log_by_cache_key(tmp_path: Path):
    cache = "abc123"
    record = {
        "cache_key": cache,
        "ok": True,
        "request": {
            "body": {
                "state": {
                    "task": {"type": "pick_and_place", "goal": "put apple", "current_subgoal": "PutObject"},
                    "upcoming_action": {"action": "PutObject", "target_class": "Drawer"},
                    "relation_mode": "SAME",
                    "action_history": ["OpenObject(Drawer)"],
                    "reference_object": {"class": "Drawer", "bbox_xyxy_in_its_own_frame": [20, 20, 60, 60]},
                    "candidates": {
                        "A": {"bbox_xyxy": [20, 20, 60, 60], "detector_score": 0.9, "distance_m": 1.0, "position_relative_to_reference": "same"},
                        "B": {"bbox_xyxy": [100, 100, 140, 140], "detector_score": 0.2, "distance_m": 2.0, "position_relative_to_reference": "right"},
                    },
                }
            }
        },
    }
    log_path = tmp_path / "jev_requests.jsonl"
    log_path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    belief = {"event_key": "ep__t0001", "raw": {"cache_key": cache}}
    views = views_from_belief_row(belief, load_jev_log(log_path))
    assert views["n"] == 2
    assert views["object_crop"].ndim == 3
    assert views["overlay"].shape[0] >= views["object_crop"].shape[0]


def test_openai_sends_reference_context_and_overlay(monkeypatch):
    calls = []

    def transport(url, data, headers, timeout):
        calls.append(json.loads(data.decode()))
        return 200, json.dumps(
            {
                "answers": [
                    {
                        "type": "choice",
                        "name": "identity",
                        "choice": "A",
                        "probabilities": [
                            {"value": "A", "probability": 0.7},
                            {"value": "B", "probability": 0.2},
                            {"value": "none", "probability": 0.1},
                        ],
                    }
                ]
            }
        )

    frame = np.zeros((32, 32, 3), dtype=np.uint8)
    source = OpenAIDecisionSource(api_key="sk-test", transport=transport)
    belief = source.identify(
        Relation("SAME", None, "visual"),
        [Candidate(0, "a", "Mug", score=0.5), Candidate(1, "b", "Mug", score=0.4)],
        Observation(frame=frame, reference_frame=frame, context_frame=frame, notes="bbox A [1,2,3,4]"),
    )
    np.testing.assert_allclose(belief.p_identity, [0.7, 0.2, 0.1])
    body = calls[0]
    kinds = [item.get("type") for item in body["input"][0]["content"]]
    assert kinds.count("input_image") == 3
    assert "bbox A" in body["input"][0]["content"][0]["text"]
