"""Counterfactual regret of committing to one mask.

Let ``z = j`` mean "candidate j is the reference", and let the last column
mean the reference was not detected.

    total(i, j) = cost(a_i) + J*(T(s, a_i) | z = j)
    base(j)     = min( J*(s | z = j), min_i total(i, j) )
    C(i, j)     = total(i, j) - base(j)

``J*`` is the shortest remaining plan. ``C(i, j) >= 0`` is the extra cost of
mask i against the best option under that same identity hypothesis. A
rejected action leaves the state unchanged and costs ``FAIL_PENALTY + J*(s)``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np

from ..config import EPISODE_FAIL_COST
from ..types import CostMatrix
from .planner import FAIL_PENALTY, PUT, UNREACHABLE, j_star, navigate_and_act
from .world import Goal, Obj, World

# THOR furniture ids are `Sink|x|y|z`; the puttable basin is `Sink|x|y|z|SinkBasin`.
# If both are in the mask list, committing to the unsuffixed shell fails in THOR
# (FloorPlan20 Knife→SinkBasin). The planner used to charge FAIL_PENALTY=1 for
# that illegal Put, which is below c_defer, so TRACE selected the rim.
_RECEPTACLE_PARTS = frozenset({"SinkBasin", "BathtubBasin"})

TemplateKind = Literal["put_into_ref", "close_ref", "pickup_ref", "pickup_not_ref"]


@dataclass(frozen=True)
class GoalTemplate:
    kind: TemplateKind
    movable_id: Optional[str] = None
    movable_type: Optional[str] = None
    reference_type: Optional[str] = None


class ConsequenceCritic:
    def __init__(
        self,
        *,
        search_room: str = "elsewhere",
        success_actions: frozenset[str] | None = None,
        episode_fail_cost: float = EPISODE_FAIL_COST,
    ):
        self.search_room = search_room
        self.success_actions = frozenset(success_actions or ())
        self.episode_fail_cost = float(episode_fail_cost)

    def cost_matrix(
        self,
        world: World,
        action: str,
        candidate_ids: list[str],
        template: GoalTemplate,
    ) -> CostMatrix:
        if not candidate_ids:
            raise ValueError("cost_matrix requires at least one candidate")
        if len(set(candidate_ids)) != len(candidate_ids):
            raise ValueError("candidate ids must be unique")
        if action in self.success_actions:
            return _success_matrix(action, candidate_ids, self.episode_fail_cost)
        n = len(candidate_ids)
        totals = np.zeros((n, n + 1), dtype=np.float64)
        bases = np.zeros(n + 1, dtype=np.float64)
        hypotheses: list[Optional[str]] = [*candidate_ids, None]
        for column, reference in enumerate(hypotheses):
            hypothetical = world.copy()
            reference_id = self._reference_id(hypothetical, template, reference, candidate_ids)
            goal = bind_goal(template, reference_id)
            nothing = j_star(hypothetical, goal)
            for row, candidate_id in enumerate(candidate_ids):
                immediate, successor, ok = navigate_and_act(hypothetical, action, candidate_id)
                if ok:
                    totals[row, column] = immediate + j_star(successor, goal)
                else:
                    totals[row, column] = FAIL_PENALTY + nothing
            bases[column] = min(nothing, float(np.min(totals[:, column])))
        costs = totals - bases.reshape(1, -1)
        if np.any(costs < -1e-8):
            raise RuntimeError("regret is negative; the baseline was not the minimum")
        matrix = CostMatrix(
            C=np.maximum(costs, 0.0),
            total=totals,
            base=bases,
            candidate_ids=tuple(candidate_ids),
            action=action,
        )
        if action == PUT:
            _penalize_furniture_shells(matrix)
        return matrix

    def _reference_id(
        self,
        world: World,
        template: GoalTemplate,
        reference: Optional[str],
        candidate_ids: list[str],
    ) -> Optional[str]:
        if reference is not None:
            return reference
        if template.kind == "pickup_not_ref":
            # The object that must be avoided was not detected, so any visible
            # instance of the class satisfies the goal.
            return None
        return _add_virtual(world, template, candidate_ids[0], self.search_room)


def furniture_shell_rows(candidate_ids: list[str] | tuple[str, ...]) -> tuple[int, ...]:
    """Indices of furniture shells whose receptacle part is also listed."""

    rows = []
    for index, candidate_id in enumerate(candidate_ids):
        prefix = candidate_id + "|"
        for other in candidate_ids:
            if other == candidate_id or not other.startswith(prefix):
                continue
            if other.rsplit("|", 1)[-1] in _RECEPTACLE_PARTS:
                rows.append(index)
                break
    return tuple(rows)


def type_of_id(object_id: str, world: World | None = None) -> str:
    if world is not None and object_id in world.objects:
        return world.objects[object_id].type
    parts = object_id.split("|")
    for part in reversed(parts):
        if part and part[:1] not in "+-":
            return part
    return parts[0]


def matches_task_type(object_id: str, task_type: str | None, world: World | None = None, cls: str | None = None) -> bool:
    """True when this mask is the instruction's receptacle/object class."""

    if not task_type:
        return True
    got = cls or type_of_id(object_id, world)
    if got == task_type:
        return True
    if task_type in _RECEPTACLE_PARTS:
        return object_id.endswith("|" + task_type)
    return False


def _penalize_furniture_shells(matrix: CostMatrix) -> None:
    """A listed Sink rim is not a 1-step retry; the port cannot Put there.

    The matching basin extra against the rim column is 0: the object is already
    in the instruction's receptacle, even if the ALFRED expert pose sat on the rim.
    """

    ids = matrix.candidate_ids
    for shell_row in furniture_shell_rows(ids):
        shell_id = ids[shell_row]
        prefix = shell_id + "|"
        matrix.C[shell_row, :] = UNREACHABLE
        matrix.total[shell_row, :] = matrix.base + UNREACHABLE
        for part_row, other in enumerate(ids):
            if other.startswith(prefix) and other.rsplit("|", 1)[-1] in _RECEPTACLE_PARTS:
                matrix.C[part_row, shell_row] = 0.0
                matrix.total[part_row, shell_row] = float(matrix.base[shell_row])


def _success_matrix(action: str, candidate_ids: list[str], fail: float) -> CostMatrix:
    """Wrong mask = episode failure. Correct mask extra cost is 0.

    R(i) = fail * (1 - p_i), so TRACE selects only when identity is peaked
    enough that expected fail < c_defer.
    """

    n = len(candidate_ids)
    costs = np.full((n, n + 1), fail, dtype=np.float64)
    np.fill_diagonal(costs[:, :n], 0.0)
    return CostMatrix(
        C=costs,
        total=costs.copy(),
        base=np.zeros(n + 1, dtype=np.float64),
        candidate_ids=tuple(candidate_ids),
        action=action,
    )


def bind_goal(template: GoalTemplate, reference_id: Optional[str]) -> Goal:
    if template.kind == "put_into_ref":
        if not template.movable_id or not reference_id:
            raise ValueError("put_into_ref needs a movable object and a reference receptacle")
        return Goal(inside=((template.movable_id, reference_id),))
    if template.kind == "close_ref":
        if not reference_id:
            raise ValueError("close_ref needs a reference receptacle")
        return Goal(closed=(reference_id,))
    if template.kind == "pickup_ref":
        if not reference_id:
            raise ValueError("pickup_ref needs a reference object")
        return Goal(holding=reference_id)
    if template.kind == "pickup_not_ref":
        if not template.movable_type:
            raise ValueError("pickup_not_ref needs movable_type")
        excluded = frozenset() if reference_id is None else frozenset({reference_id})
        return Goal(holding_type=template.movable_type, holding_exclude=excluded)
    raise ValueError(f"unknown goal template {template.kind}")


def _add_virtual(world: World, template: GoalTemplate, sample_id: str, search_room: str) -> str:
    virtual_id = "virtual_ref"
    if virtual_id in world.objects:
        raise ValueError("world already contains virtual_ref")
    sample = world.objects[sample_id]
    if template.kind == "pickup_ref":
        created = Obj(
            id=virtual_id,
            type=template.movable_type or sample.type,
            room=search_room,
            pickupable=True,
        )
    elif template.kind in ("put_into_ref", "close_ref"):
        created = Obj(
            id=virtual_id,
            type=sample.type,
            room=search_room,
            receptacle=True,
            openable=True,
            is_open=True,
        )
    else:
        raise ValueError(f"no virtual reference for {template.kind}")
    world.objects[virtual_id] = created
    return virtual_id
