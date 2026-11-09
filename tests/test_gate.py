import numpy as np

from experiments.compare import heterogeneous_table
from experiments.scenes import far_fridge, same_room_drawers, split_fridges
from trace.belief.base import FixedIdentity, same_to_pick
from trace.consequence.critic import ConsequenceCritic
from trace.consequence.planner import PUT
from trace.gate import decide, expected_regret
from trace.relation import EpisodeMemory, OPEN
from trace.selector import MaskRequest, TraceSelector
from trace.types import Belief, Candidate, CostMatrix, Observation


def _costs(matrix: np.ndarray) -> CostMatrix:
    n = matrix.shape[0]
    return CostMatrix(C=matrix, total=matrix, base=np.zeros(n + 1), candidate_ids=tuple(f"c{i}" for i in range(n)), action=PUT)


def test_trace_defers_only_when_the_same_belief_is_expensive():
    table = heterogeneous_table()
    means = table["mean_realized_cost"]
    assert means["trace"] == 2.4
    assert means["threshold_055"] == 4.4
    assert means["threshold_075"] == 4.0
    assert means["host_score"] == 6.6
    actions = {(row["world"], row["rule"]): row["action"] for row in table["rows"]}
    assert actions[("cheap", "trace")] == "SELECT"
    assert actions[("expensive", "trace")] == "RETURN"


def test_different_mode_uses_identity_columns_not_pick_probabilities():
    # Candidate 0 is probably the reference, so a DIFFERENT task must pick candidate 1.
    identity = np.array([0.8, 0.2, 0.0])
    costs = _costs(np.array([[10.0, 0.0, 10.0], [0.0, 10.0, 10.0]]))
    belief = Belief(p_identity=identity, source="fixed")
    decision = decide(belief, costs, mode="DIFFERENT", deferral_cost=4.0, rule="trace")
    assert decision.action == "SELECT"
    assert decision.idx == 1
    pick = same_to_pick(identity, "DIFFERENT")
    swapped = expected_regret(costs.C, pick)
    assert int(np.argmin(swapped)) == 0


def test_equal_regret_returns_control():
    costs = _costs(np.array([[4.0, 4.0, 0.0], [4.0, 4.0, 0.0]]))
    belief = Belief(p_identity=np.array([0.5, 0.5, 0.0]), source="fixed")
    decision = decide(belief, costs, mode="SAME", deferral_cost=4.0, rule="trace")
    assert decision.action == "RETURN"


def test_tied_minimum_below_deferral_returns_to_host():
    """Same-room cabinets: every mask has extra 1, so R is flat and < c_defer.

    Breaking the tie by argmin=0 overrode a correct host Put (FloorPlan424).
    """

    costs = _costs(np.ones((4, 5)))
    belief = Belief(p_identity=np.array([0.79, 0.10, 0.05, 0.01, 0.05]), source="jev")
    decision = decide(belief, costs, mode="SAME", deferral_cost=4.0, rule="trace")
    assert decision.action == "RETURN"
    assert decision.reason == "regret_tied"


def test_unique_cheap_minimum_still_selects_sinkbasin():
    costs = _costs(np.array([[0.0, 2.0, 2.0], [2.0, 0.0, 2.0]]))
    belief = Belief(p_identity=np.array([0.92, 0.08, 0.0]), source="jev")
    decision = decide(belief, costs, mode="SAME", deferral_cost=4.0, rule="trace")
    assert decision.action == "SELECT" and decision.idx == 0


def test_selector_commits_on_a_cheap_put_and_defers_on_a_far_put():
    belief = FixedIdentity(np.array([0.6, 0.4, 0.0]), confidence=0.6)
    selector = TraceSelector(belief, rule="trace")
    memory = EpisodeMemory()
    memory.record(OPEN, "Drawer", True, 1)
    cheap = selector.choose(_request(memory, same_room_drawers(), "Drawer"))
    assert cheap.defer is False and cheap.index == 0 and cheap.relation_rule == "R1"

    memory = EpisodeMemory()
    memory.record(OPEN, "Fridge", True, 1)
    near = selector.choose(_request(memory, far_fridge(), "Fridge"))
    # The near fridge stays recoverable even under the 0.4 chance it is wrong.
    # The far fridge does not: its expected regret is above the deferral cost,
    # so the gate commits to the near mask instead of refusing every mask.
    assert near.defer is False and near.index == 0
    assert near.expected_regret[0] < 4.0 < near.expected_regret[1]

    both_far = selector.choose(_request(memory, split_fridges(), "Fridge"))
    assert both_far.defer is True
    assert both_far.reason == "regret_not_below_deferral"


def test_none_relation_does_not_call_the_belief_model():
    class Boom:
        name = "boom"

        def identify(self, relation, candidates, observation):
            raise AssertionError("belief model must not run when identity is irrelevant")

    memory = EpisodeMemory()
    memory.record("PickupObject", "Apple", True, 1)
    request = _request(memory, same_room_drawers(), "Apple", action="PickupObject")
    decision = TraceSelector(Boom()).choose(request)
    assert decision.defer is True
    assert decision.reason == "relation_none"


def _request(memory, scene, cls: str, action: str = PUT) -> MaskRequest:
    world, template, ids = scene
    candidates = [
        Candidate(idx=i, object_id=object_id, cls=cls, score=0.5, room=world.objects[object_id].room)
        for i, object_id in enumerate(ids)
    ]
    return MaskRequest(
        action=action,
        target_cls=cls,
        candidates=candidates,
        memory=memory,
        world=world,
        goal=template,
        observation=Observation(goal_text="put the apple away"),
        deferral_cost=4.0,
    )
