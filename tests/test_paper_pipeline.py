"""The paper abstract, as tests.

Decision models give identity probabilities. TRACE multiplies those by extra
task cost, then commits only when expected regret is below the deferral cost.
The host calls this at interaction-mask selection.
"""

from __future__ import annotations

import numpy as np

from experiments.protocol import OFFICIAL_10FAIL
from experiments.scenes import far_fridge, same_room_drawers, split_fridges
from trace.belief.appearance import AppearanceSource
from trace.belief.base import FixedIdentity
from trace.consequence.critic import ConsequenceCritic
from trace.consequence.planner import PUT
from trace.relation import EpisodeMemory, OPEN, infer_relation
from trace.selector import MaskRequest, TraceSelector, choose_interaction_mask
from trace.types import Candidate, MemoryEntry, Observation, Relation


class RecordingBelief(FixedIdentity):
    name = "recording"

    def __init__(self, p_identity):
        super().__init__(p_identity)
        self.calls = []

    def identify(self, relation, candidates, observation):
        self.calls.append(
            {
                "mode": relation.mode,
                "history": observation.history,
                "has_current": observation.frame is not None,
                "has_reference": observation.reference_frame is not None,
                "goal": observation.goal_text,
                "n": len(candidates),
            }
        )
        return super().identify(relation, candidates, observation)


def _put_request(scene, cls: str, memory: EpisodeMemory, observation: Observation) -> MaskRequest:
    world, template, ids = scene
    candidates = [
        Candidate(idx=i, object_id=object_id, cls=cls, score=0.5, room=world.objects[object_id].room)
        for i, object_id in enumerate(ids)
    ]
    return MaskRequest(
        action=PUT,
        target_cls=cls,
        candidates=candidates,
        memory=memory,
        world=world,
        goal=template,
        observation=observation,
        deferral_cost=4.0,
    )


def test_history_and_observation_drive_identity_then_cost():
    """TRACE uses visual memory, the current frame, and interaction history."""

    memory = EpisodeMemory()
    memory.record(OPEN, "Drawer", True, 1)
    relation = infer_relation(PUT, "Drawer", memory)
    assert relation.mode == "SAME"
    assert relation.rule == "R1"

    frame = np.zeros((320, 320, 3), dtype=np.uint8)
    reference = np.full((320, 320, 3), 80, dtype=np.uint8)
    observation = Observation(
        frame=frame,
        reference_frame=reference,
        goal_text="put the apple in the drawer that was opened",
        history=("OpenObject Drawer",),
        task_type="pick_and_place_simple",
    )
    belief = RecordingBelief(np.array([0.6, 0.4, 0.0]))
    selector = TraceSelector(belief, rule="trace")
    decision = selector.choose(_put_request(same_room_drawers(), "Drawer", memory, observation))
    assert belief.calls and belief.calls[0]["has_current"] and belief.calls[0]["has_reference"]
    assert belief.calls[0]["history"] == ("OpenObject Drawer",)
    assert decision.defer is False
    assert decision.index == 0
    assert choose_interaction_mask(selector, _put_request(same_room_drawers(), "Drawer", memory, observation)) == 0


def test_same_belief_selects_when_cheap_and_returns_control_when_costly():
    """Expected regret is compared with deferral cost; the host keeps control on RETURN."""

    p = np.array([0.6, 0.4, 0.0])
    observation = Observation(goal_text="put the apple away")
    selector = TraceSelector(FixedIdentity(p), rule="trace")

    cheap_memory = EpisodeMemory()
    cheap_memory.record(OPEN, "Drawer", True, 1)
    cheap_idx = choose_interaction_mask(
        selector, _put_request(same_room_drawers(), "Drawer", cheap_memory, observation)
    )
    assert cheap_idx == 0

    far_memory = EpisodeMemory()
    far_memory.record(OPEN, "Fridge", True, 1)
    far = selector.choose(_put_request(far_fridge(), "Fridge", far_memory, observation))
    assert far.defer is False and far.index == 0
    assert far.expected_regret[0] < 4.0 < far.expected_regret[1]

    split_memory = EpisodeMemory()
    split_memory.record(OPEN, "Fridge", True, 1)
    split_idx = choose_interaction_mask(
        selector, _put_request(split_fridges(), "Fridge", split_memory, observation)
    )
    assert split_idx is None


def test_critic_extra_cost_is_not_a_probability_threshold():
    """The symbolic critic reports extra remaining cost, not a confidence cutoff."""

    world, template, ids = far_fridge()
    costs = ConsequenceCritic().cost_matrix(world, PUT, ids, template)
    # Wrong fridge in another room is much more expensive than the local one.
    assert costs.C[1, 0] > costs.C[0, 1]
    assert costs.C[1, 0] > 4.0


def test_relation_override_runs_identity_when_memory_would_be_none():
    """Ported THOR first-pick lookalikes keep SAME against the task instance crop."""

    memory = EpisodeMemory()
    frame = np.zeros((16, 16, 3), dtype=np.uint8)
    observation = Observation(frame=frame, reference_frame=frame, goal_text="pick the apple")
    belief = RecordingBelief(np.array([0.8, 0.1, 0.1]))
    selector = TraceSelector(belief, rule="trace")
    request = _put_request(same_room_drawers(), "Drawer", memory, observation)
    request.relation_override = Relation(
        "SAME",
        MemoryEntry(key="task", cls="Drawer", action="PutObject", t=0, role="receptacle"),
        "task_instance",
    )
    decision = selector.choose(request)
    assert belief.calls and belief.calls[0]["mode"] == "SAME"
    assert decision.defer is False


def test_official_protocol_and_appearance_source_are_wired():
    """FILM/CAPEAM 820ep uses the 10-failure protocol and DINOv2 appearance."""

    seen = [row for row in OFFICIAL_10FAIL if row["host"] == "film" and row["split"] == "valid_seen" and row["selector"] == "host"]
    assert seen and seen[0]["n"] == 820 and seen[0]["sr"] == 0.232
    assert AppearanceSource.name == "dinov2"
