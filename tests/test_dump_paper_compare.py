from experiments.dump_paper_compare import action_event, force_from_logged


def test_force_selects_only_when_logged_select():
    row = {
        "events": [
            {
                "action": "PutObject",
                "id": "Sink|basin",
                "p": [0.99, 0.01, 0.0],
                "defer": False,
                "reason": "regret_below_deferral",
            }
        ]
    }
    assert force_from_logged(row, "PutObject") == {"PutObject": "Sink|basin"}


def test_force_skips_return_and_host():
    returned = {
        "events": [
            {
                "action": "PutObject",
                "id": "Sink|rim",
                "p": [0.5, 0.4, 0.1],
                "defer": True,
                "reason": "defer:regret_not_below_deferral",
            }
        ]
    }
    host = {"events": [{"action": "PutObject", "id": "Sink|rim", "reason": "host_score"}]}
    assert force_from_logged(returned, "PutObject") is None
    assert force_from_logged(host, "PutObject") is None


def test_action_event_prefers_belief_row():
    row = {
        "events": [
            {"action": "PutObject", "id": "a", "reason": "host_score"},
            {"action": "PutObject", "id": "b", "p": [0.9, 0.1], "reason": "regret_below_deferral"},
        ]
    }
    assert action_event(row, "PutObject")["id"] == "b"
