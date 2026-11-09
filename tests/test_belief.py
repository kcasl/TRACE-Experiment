import json

import numpy as np

from trace.belief.base import same_to_pick
from trace.belief.jev import JevSource
from trace.belief.openai_decisions import OpenAIDecisionSource
from trace.belief.parse import belief_from_choice, find_answer, option_label
from trace.redact import redact
from trace.relation import EpisodeMemory, OPEN, infer_relation
from trace.types import Candidate, Observation


def test_option_labels_and_same_to_pick():
    assert [option_label(i) for i in range(3)] == ["A", "B", "C"]
    assert option_label(26) == "AA"
    identity = np.array([0.8, 0.2, 0.0])
    np.testing.assert_allclose(same_to_pick(identity, "SAME"), identity)
    np.testing.assert_allclose(same_to_pick(identity, "DIFFERENT"), [0.2, 0.8, 0.0])
    single = same_to_pick(np.array([0.7, 0.3]), "DIFFERENT")
    np.testing.assert_allclose(single, [0.3, 0.7])


def test_parsers_accept_both_response_shapes():
    jev = {"answers": {"identity": {"choice": "B", "probabilities": {"A": 0.1, "B": 0.7, "none": 0.2}, "confidence": 0.7}}}
    openai = {
        "answers": [
            {
                "type": "choice",
                "name": "identity",
                "choice": "B",
                "confidence": 0.66,
                "probabilities": [
                    {"value": "A", "probability": 0.25},
                    {"value": "B", "probability": 0.25},
                    {"value": "none", "probability": 0.50},
                ],
            }
        ]
    }
    left = belief_from_choice(find_answer(jev, "identity"), ["A", "B"], source="jev")
    right = belief_from_choice(find_answer(openai, "identity"), ["A", "B"], source="openai_decisions")
    np.testing.assert_allclose(left.p_identity, [0.1, 0.7, 0.2])
    np.testing.assert_allclose(right.p_identity, [0.25, 0.25, 0.5])
    refused = belief_from_choice({"type": "refusal", "name": "identity"}, ["A", "B"], source="openai_decisions")
    assert refused.refusal is True
    assert abs(refused.p_identity.sum() - 1.0) < 1e-12


def test_clients_send_the_choice_question_without_embedding_the_key():
    frame = np.zeros((320, 320, 3), dtype=np.uint8)
    frame[:, :] = (180, 180, 180)
    observation = Observation(frame=frame, reference_frame=frame, goal_text="find the mug")
    memory = EpisodeMemory()
    memory.record(OPEN, "Mug", True, 1)
    relation = infer_relation("CloseObject", "Mug", memory)
    candidates = [
        Candidate(0, "mug_a", "Mug", score=0.4),
        Candidate(1, "mug_b", "Mug", score=0.6),
    ]
    answer = json.dumps(
        {"answers": {"identity": {"probabilities": {"A": 0.2, "B": 0.7, "none": 0.1}, "confidence": 0.7}}}
    )

    def transport(url, data, headers, timeout):
        transport.calls.append((url, data, headers, timeout))
        return 200, answer

    transport.calls = []
    token = "local-test-token"
    jev = JevSource(api_key=token, transport=transport).identify(relation, candidates, observation)
    openai = OpenAIDecisionSource(api_key=token, transport=transport).identify(relation, candidates, observation)
    np.testing.assert_allclose(jev.p_identity, openai.p_identity)
    jev_url, jev_body, jev_headers, _ = transport.calls[0]
    openai_url, openai_body, openai_headers, _ = transport.calls[1]
    assert jev_url.endswith("/v1/systemone")
    assert openai_url.endswith("/v1/decisions")
    assert jev_headers["TypeSafe-Danger-Zone-Images"] == "true"
    assert token.encode() not in jev_body
    assert token.encode() not in openai_body
    assert b'"type": "choice"' in jev_body or b'"type":"choice"' in jev_body
    decoded = json.loads(openai_body.decode())
    assert decoded["model"] == "gpt-6-luna"
    assert decoded["questions"][0]["choices"][-1]["value"] == "none"


def test_redact_strips_bearer_and_env_values(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-unit-test-secret")
    hidden = redact("Authorization: Bearer sk-unit-test-secret")
    assert "sk-unit-test-secret" not in hidden
    assert "Bearer [REDACTED]" in hidden
