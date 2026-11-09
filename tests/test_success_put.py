"""Put success-valued critic and task-grounding prompts."""

from __future__ import annotations

import numpy as np

from trace.belief.jev import _body as jev_body
from trace.belief.jev import _collect_images
from trace.belief.openai_decisions import _body as openai_body
from trace.belief.prompt import GROUNDING_RULE, identity_instructions, is_grounding
from trace.config import EPISODE_FAIL_COST
from experiments.thor_rollout import (
    cap_candidates,
    candidates_from,
    live_goal_template,
    map_poses,
    resolve_live_relation,
    TRACE_ACTIONS,
)
from trace.belief.base import normalize
from trace.relation import EpisodeMemory
from trace.consequence.critic import ConsequenceCritic, GoalTemplate, furniture_shell_rows, matches_task_type
from trace.consequence.world import Obj, World
from trace.consequence.planner import UNREACHABLE
from trace.gate import decide
from trace.types import Belief, Candidate, CostMatrix, MemoryEntry, Observation, Relation


def _put_world() -> World:
    return World(
        objects={
            "a": Obj("a", "Table", "here", receptacle=True),
            "b": Obj("b", "Table", "here", receptacle=True),
            "apple": Obj("apple", "Apple", "here", pickupable=True),
        },
        holding="apple",
    )


def test_success_matrix_is_zero_on_the_true_mask_and_fail_elsewhere():
    critic = ConsequenceCritic(success_actions=frozenset({"PutObject"}), episode_fail_cost=100.0)
    costs = critic.cost_matrix(
        _put_world(),
        "PutObject",
        ["a", "b"],
        GoalTemplate(kind="put_into_ref", movable_id="apple"),
    )
    np.testing.assert_allclose(costs.C, [[0, 100, 100], [100, 0, 100]])


def test_put_success_trace_selects_only_when_identity_is_peaked():
    critic = ConsequenceCritic(success_actions=frozenset({"PutObject"}))
    costs = critic.cost_matrix(
        _put_world(),
        "PutObject",
        ["a", "b"],
        GoalTemplate(kind="put_into_ref", movable_id="apple"),
    )
    peaked = decide(
        Belief(p_identity=np.array([0.97, 0.02, 0.01]), source="x"),
        costs,
        mode="SAME",
        deferral_cost=4.0,
    )
    assert peaked.action == "SELECT" and peaked.idx == 0
    # 100 * (1 - 0.95) = 5, not strictly below c_defer=4
    mild = decide(
        Belief(p_identity=np.array([0.95, 0.04, 0.01]), source="x"),
        costs,
        mode="SAME",
        deferral_cost=4.0,
    )
    assert mild.action == "RETURN"
    assert EPISODE_FAIL_COST == 100.0


def test_pickup_still_uses_step_costs_when_only_put_is_success_valued():
    critic = ConsequenceCritic(success_actions=frozenset({"PutObject"}))
    world = World(
        objects={
            "a": Obj("a", "Apple", "here", pickupable=True),
            "b": Obj("b", "Apple", "far", pickupable=True),
        }
    )
    costs = critic.cost_matrix(
        world,
        "PickupObject",
        ["a", "b"],
        GoalTemplate(kind="pickup_ref"),
    )
    assert float(costs.C[0, 0]) == 0.0
    assert float(costs.C[1, 0]) > 0.0


def test_planner_put_selects_at_jev_fp422_confidence():
    """Cheap same-room extra ≈ 2: Jev p=0.93 SELECTs; success-valued C would RETURN."""

    world = _put_world()
    planner = ConsequenceCritic().cost_matrix(
        world,
        "PutObject",
        ["a", "b"],
        GoalTemplate(kind="put_into_ref", movable_id="apple"),
    )
    belief = Belief(p_identity=np.array([0.93, 0.05, 0.02]), source="jev")
    planner_gate = decide(belief, planner, mode="SAME", deferral_cost=4.0)
    assert planner_gate.action == "SELECT" and planner_gate.idx == 0
    assert float(planner.C[1, 0]) < 10.0
    fail = ConsequenceCritic(success_actions=frozenset({"PutObject"})).cost_matrix(
        world,
        "PutObject",
        ["a", "b"],
        GoalTemplate(kind="put_into_ref", movable_id="apple"),
    )
    fail_gate = decide(belief, fail, mode="SAME", deferral_cost=4.0)
    assert fail_gate.action == "RETURN"


def test_sink_shell_is_not_a_listed_part():
    basin = "Sink|-00.11|+00.89|-02.01|SinkBasin"
    rim = "Sink|-00.11|+00.89|-02.01"
    assert furniture_shell_rows([basin, rim]) == (1,)
    assert furniture_shell_rows(["drawer_a", "drawer_b"]) == ()


def test_fp20_jev_selects_sinkbasin_not_the_rim():
    """Host picks the Sink rim (score 0.390 vs 0.388). Jev peaks on the basin.

    Planner extra for the illegal rim Put used to be FAIL_PENALTY=1, so R(rim)=1
    won unique-min SELECT and the episode failed. The basin is the only valid Put.
    """

    basin = "Sink|-00.11|+00.89|-02.01|SinkBasin"
    rim = "Sink|-00.11|+00.89|-02.01"
    world = World(
        objects={
            basin: Obj(basin, "SinkBasin", "here", receptacle=True),
            rim: Obj(rim, "Sink", "here", receptacle=False),
            "knife": Obj("knife", "Knife", "here", pickupable=True),
        },
        holding="knife",
        room="here",
    )
    costs = ConsequenceCritic().cost_matrix(
        world,
        "PutObject",
        [basin, rim],
        GoalTemplate(kind="put_into_ref", movable_id="knife"),
    )
    assert float(np.min(costs.C[1])) >= 40.0
    assert float(costs.C[0, 1]) == 0.0
    live = CostMatrix(
        C=np.array([[0.0, 0.0, 12.0], [UNREACHABLE, UNREACHABLE, UNREACHABLE]]),
        total=np.array([[0.0, 0.0, 12.0], [UNREACHABLE, UNREACHABLE, UNREACHABLE]]),
        base=np.zeros(3),
        candidate_ids=(basin, rim),
        action="PutObject",
    )
    jev = decide(
        Belief(p_identity=np.array([0.73, 0.16, 0.11]), source="jev"),
        live,
        mode="SAME",
        deferral_cost=4.0,
    )
    assert jev.action == "SELECT" and jev.idx == 0
    # OpenAI none-mass 0.43: 12*0.43 > c_defer. 0.30 commits.
    mild = decide(
        Belief(p_identity=np.array([0.59, 0.11, 0.30]), source="openai"),
        live,
        mode="SAME",
        deferral_cost=4.0,
    )
    assert mild.action == "SELECT" and mild.idx == 0
    unsure = decide(
        Belief(p_identity=np.array([0.53, 0.04, 0.43]), source="openai"),
        live,
        mode="SAME",
        deferral_cost=4.0,
    )
    assert unsure.action == "RETURN"


def test_map_poses_keeps_furniture_the_alfred_list_omits():
    from types import SimpleNamespace

    event = SimpleNamespace(
        metadata={
            "objects": [
                {
                    "name": "Vase_1",
                    "objectType": "Vase",
                    "pickupable": True,
                    "moveable": False,
                    "position": {"x": 0, "y": 0, "z": 0},
                    "rotation": {"x": 0, "y": 0, "z": 0},
                },
                {
                    "name": "Safe_1",
                    "objectType": "Safe",
                    "pickupable": False,
                    "moveable": False,
                    "receptacle": True,
                    "position": {"x": 1, "y": 0, "z": 0},
                    "rotation": {"x": 0, "y": 90, "z": 0},
                },
            ]
        }
    )
    mapped = map_poses(
        event,
        [{"objectName": "Vase_1", "position": {"x": 2, "y": 0, "z": 0}, "rotation": {"x": 0, "y": 0, "z": 0}}],
    )
    names = {item["objectName"] for item in mapped}
    assert names == {"Vase_1", "Safe_1"}
    safe = next(item for item in mapped if item["objectName"] == "Safe_1")
    assert safe["position"]["x"] == 1


def test_cap_candidates_pins_a_far_safe():
    drawers = [
        Candidate(idx=i, object_id=f"Drawer|{i}", cls="Drawer", score=1.0 - 0.01 * i) for i in range(8)
    ]
    drawers.append(Candidate(idx=8, object_id="Safe|far", cls="Safe", score=0.01))
    keep = cap_candidates(drawers, pin_types=frozenset({"Safe"}))
    assert any(cand.object_id == "Safe|far" for cand in keep)
    assert len(keep) == 8


def test_cap_candidates_does_not_pin_an_expert_instance():
    drawers = [
        Candidate(idx=i, object_id=f"Drawer|{i}", cls="Drawer", score=1.0 - 0.01 * i) for i in range(9)
    ]
    keep = cap_candidates(drawers, pin_types=frozenset({"Drawer"}))
    assert len(keep) == 8
    assert "Drawer|8" not in {cand.object_id for cand in keep}


def test_off_type_put_is_not_marked_unreachable():
    safe = "Safe|-02.30|+00.00|+00.27"
    shelf = "Shelf|-05.24|+01.97|+00.09"
    world = World(
        objects={
            safe: Obj(safe, "Safe", "here", receptacle=True, openable=True, is_open=True),
            shelf: Obj(shelf, "Shelf", "here", receptacle=True),
            "vase": Obj("vase", "Vase", "here", pickupable=True),
        },
        holding="vase",
        room="here",
    )
    costs = ConsequenceCritic().cost_matrix(
        world,
        "PutObject",
        [shelf, safe],
        GoalTemplate(kind="put_into_ref", movable_id="vase", reference_type="Safe"),
    )
    assert float(np.min(costs.C[0])) < 40.0
    assert matches_task_type(safe, "Safe", cls="Safe")
    assert not matches_task_type(shelf, "Safe", cls="Shelf")


def test_task_family_covers_all_alfred_types():
    from experiments.thor_rollout import task_family

    assert task_family("pick_and_place_simple-Vase-None-Safe-219") == "pick_and_place_simple"
    assert task_family("pick_heat_then_place_in_recep-Apple-None-Fridge-10") == "pick_heat_then_place_in_recep"
    assert task_family("look_at_obj_in_light-Laptop-None-FloorLamp-219") == "look_at_obj_in_light"


def test_objects_of_parent_does_not_steal_furniture_for_a_pan():
    from types import SimpleNamespace

    from experiments.thor_rollout import objects_of_parent

    event = SimpleNamespace(
        metadata={
            "objects": [
                {
                    "objectId": "DiningTable|0",
                    "objectType": "DiningTable",
                    "receptacle": True,
                    "pickupable": False,
                }
            ]
        }
    )
    assert objects_of_parent(event, "Pan") == []


def test_same_class_cabinets_are_not_off_type_penalized():
    world = _put_world()
    world.objects["a"].type = "Cabinet"
    world.objects["b"].type = "Cabinet"
    costs = ConsequenceCritic().cost_matrix(
        world,
        "PutObject",
        ["a", "b"],
        GoalTemplate(kind="put_into_ref", movable_id="apple", reference_type="Cabinet"),
    )
    assert float(costs.C[1, 0]) < 10.0
    assert float(np.min(costs.C[1])) < 40.0


def test_grounding_prompt_drops_expert_crop_and_asks_the_instruction():
    relation = Relation(
        "SAME",
        MemoryEntry(key="task", cls="Apple", action="PickupObject", t=0, role="object"),
        GROUNDING_RULE,
    )
    assert is_grounding(relation)
    assert "remembered instance" in identity_instructions(relation)
    observation = Observation(
        frame=np.zeros((64, 64, 3), dtype=np.uint8),
        reference_frame=np.full((64, 64, 3), 80, dtype=np.uint8),
        context_frame=np.full((64, 64, 3), 40, dtype=np.uint8),
        goal_text="put the apple on the table",
        subgoal_text="PickupObject Apple",
    )
    images = _collect_images(observation, relation)
    assert "reference" not in images
    assert "current" in images
    candidates = [
        Candidate(idx=0, object_id="a", cls="Apple", score=0.5),
        Candidate(idx=1, object_id="b", cls="Apple", score=0.4),
    ]
    labels = ["A", "B"]
    jev, _ = jev_body("jev-latest", relation, candidates, observation, labels, images)
    assert "not yet interacted" in jev["questions"]["identity"]["instructions"]
    assert "SinkBasin" in jev["questions"]["identity"]["instructions"]
    openai = openai_body("gpt-6-luna", relation, candidates, observation, labels)
    assert "not yet interacted" in openai["questions"][0]["instructions"]
    assert "SinkBasin" in openai["questions"][0]["instructions"]
    assert sum(1 for part in openai["input"][0]["content"] if part.get("type") == "input_image") == 1


def test_candidates_from_uses_agent_distance_for_both_hosts():
    objs = [
        {"objectId": "Apple|near", "objectType": "Apple", "position": {"x": 0, "y": 0, "z": 0}},
        {"objectId": "Apple|far", "objectType": "Apple", "position": {"x": 10, "y": 0, "z": 0}},
    ]
    agent = {"x": 0, "y": 0, "z": 0}
    expert = {"x": 10, "y": 0, "z": 0}
    film = candidates_from(objs, agent, expert, "film")
    capeam = candidates_from(objs, agent, expert, "capeam")
    assert film[0].score > film[1].score
    assert capeam[0].score == film[0].score
    assert capeam[1].score == film[1].score


def test_live_diff_pickup_binds_pickup_not_ref():
    relation = Relation("DIFFERENT", MemoryEntry(key="a", cls="Apple", action="PickupObject", t=1), "R5")
    template = live_goal_template("PickupObject", relation, holding=None, task_cls="Apple")
    assert template.kind == "pickup_not_ref"
    assert template.movable_type == "Apple"
    same = Relation("SAME", MemoryEntry(key="a", cls="Apple", action="PickupObject", t=1), "R1")
    assert live_goal_template("PickupObject", same, holding=None, task_cls="Apple").kind == "pickup_ref"
    ground = Relation(
        "SAME",
        MemoryEntry(key="task", cls="Drawer", action="PutObject", t=0, role="receptacle"),
        "task_instance",
    )
    put = live_goal_template("PutObject", ground, holding="apple", task_cls="Drawer")
    assert put.kind == "put_into_ref"


def test_pickup_not_ref_prices_avoiding_the_referent():
    world = World(
        objects={
            "a": Obj("a", "Apple", "here", pickupable=True),
            "b": Obj("b", "Apple", "here", pickupable=True),
        }
    )
    costs = ConsequenceCritic().cost_matrix(
        world,
        "PickupObject",
        ["a", "b"],
        GoalTemplate(kind="pickup_not_ref", movable_type="Apple"),
    )
    # Column 0: a is the referent to avoid, so picking b is extra 0.
    assert float(costs.C[1, 0]) < float(costs.C[0, 0])
    assert float(costs.C[1, 0]) == 0.0
    # Column 2 (none): any apple satisfies, so both extras are 0.
    assert float(costs.C[0, 2]) == 0.0
    assert float(costs.C[1, 2]) == 0.0


def test_pick_two_second_sets_different_without_object_side_put():
    memory = EpisodeMemory()
    memory.record("PickupObject", "Apple", True, t=1, role="object")
    relation, override, pick_two = resolve_live_relation(
        "PickupObject", "Apple", memory, family="pick_two_obj_and_place", n_cands=2
    )
    assert pick_two is True
    assert override is None
    assert relation.mode == "DIFFERENT"
    assert relation.rule == "R5"


def test_r4_does_not_fire_without_object_side_put():
    memory = EpisodeMemory()
    memory.record("PutObject", "Microwave", True, t=1, role="receptacle")
    relation, _, pick_two = resolve_live_relation(
        "PickupObject", "Apple", memory, family="pick_heat_then_place_in_recep", n_cands=2
    )
    assert pick_two is False
    assert relation.mode != "SAME" or relation.rule != "R4"


def test_trace_actions_are_pickup_and_put_only():
    assert TRACE_ACTIONS == frozenset({"PickupObject", "PutObject"})


def test_zero_probability_vector_falls_back_to_uniform():
    out = normalize(np.zeros(3))
    np.testing.assert_allclose(out, [1 / 3, 1 / 3, 1 / 3])
