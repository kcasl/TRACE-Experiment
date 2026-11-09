from experiments.replay_logged import force_ids_from_row
from experiments.thor_rollout import failed_episode


def test_force_ids_reads_pickup_and_put():
    row = {
        "events": [
            {"action": "PickupObject", "id": "mug|a", "ok": True},
            {"action": "OpenObject", "id": "cab|a", "ok": True},
            {"action": "PutObject", "id": "cab|a", "ok": True, "n": 3},
        ]
    }
    assert force_ids_from_row(row) == {"PickupObject": "mug|a", "PutObject": "cab|a"}


def test_force_ids_empty_without_ids():
    assert force_ids_from_row({"events": [{"action": "PutObject", "ok": False}]}) == {}


def test_failed_episode_is_not_success():
    row = failed_episode(3, "jev", "film", RuntimeError("timeout"))
    assert row["success"] is False
    assert row["selector"] == "jev"
    assert row["idx"] == 3
