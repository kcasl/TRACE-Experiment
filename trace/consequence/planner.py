"""Uniform-cost remaining plan length on ``World``.

Each local interaction costs 1. Changing rooms costs ``world.nav_cost``.
If no plan exists within ``max_cost``, the result is ``UNREACHABLE`` rather
than infinity, so an expected cost stays finite when a hypothesis is impossible.
"""

from __future__ import annotations

import heapq
from typing import Optional

from .world import Goal, World

# Sentinel cost for a goal with no plan inside the search horizon. It is finite
# so an expected regret stays finite when one identity hypothesis is impossible,
# and it is well above any reachable plan length in these worlds.
UNREACHABLE = 40.0

# A rejected interaction consumes one failure of the host's budget and leaves
# the state unchanged.
FAIL_PENALTY = 1.0

PICKUP = "PickupObject"
PUT = "PutObject"
OPEN = "OpenObject"
CLOSE = "CloseObject"
SUPPORTED_ACTIONS = frozenset({PICKUP, PUT, OPEN, CLOSE})


def j_star(world: World, goal: Goal, *, max_cost: float = 24.0, max_expansions: int = 5000) -> float:
    if goal.satisfied(world):
        return 0.0
    counter = 0
    heap: list[tuple[float, int, World]] = [(0.0, counter, world)]
    best = {world.state_key(): 0.0}
    expansions = 0
    while heap:
        cost, _, state = heapq.heappop(heap)
        key = state.state_key()
        if cost > best.get(key, UNREACHABLE) + 1e-9:
            continue
        if goal.satisfied(state):
            return cost
        if cost >= max_cost:
            continue
        expansions += 1
        if expansions > max_expansions:
            break
        for step_cost, nxt in _successors(state):
            new_cost = cost + step_cost
            if new_cost > max_cost:
                continue
            nxt_key = nxt.state_key()
            if new_cost + 1e-9 < best.get(nxt_key, UNREACHABLE):
                best[nxt_key] = new_cost
                counter += 1
                heapq.heappush(heap, (new_cost, counter, nxt))
    return UNREACHABLE


def navigate_and_act(world: World, action: str, object_id: str) -> tuple[float, World, bool]:
    """Apply one committed interaction, navigating first if the target is elsewhere.

    A failed interaction leaves the world unchanged. The caller adds
    ``FAIL_PENALTY`` on top of the cost of replanning from that unchanged world.
    """

    if action not in SUPPORTED_ACTIONS:
        raise ValueError(f"unsupported interaction {action!r}; expected one of {sorted(SUPPORTED_ACTIONS)}")
    if object_id not in world.objects:
        raise KeyError(f"unknown object {object_id!r}")
    if not _preconditions(world, action, object_id):
        return 0.0, world.copy(), False
    nxt = world.copy()
    target = nxt.objects[object_id]
    cost = 1.0
    if nxt.room != target.room:
        cost += nxt.nav_cost
        _move_agent(nxt, target.room)
    _apply_local(nxt, action, object_id)
    return cost, nxt, True


def _successors(world: World):
    rooms = {obj.room for obj in world.objects.values()}
    for room in sorted(rooms):
        if room != world.room:
            nxt = world.copy()
            _move_agent(nxt, room)
            yield world.nav_cost, nxt
    for obj in world.objects.values():
        if obj.room != world.room:
            continue
        for action in (PICKUP, PUT, OPEN, CLOSE):
            if not _preconditions(world, action, obj.id):
                continue
            nxt = world.copy()
            _apply_local(nxt, action, obj.id)
            yield 1.0, nxt


def _preconditions(world: World, action: str, object_id: str) -> bool:
    obj = world.objects[object_id]
    if action == PICKUP:
        if world.holding is not None or not obj.pickupable:
            return False
        parent = world.inside.get(object_id)
        if parent is None:
            return True
        container = world.objects.get(parent)
        return container is not None and (not container.openable or container.is_open)
    if action == PUT:
        if world.holding is None or not obj.receptacle or world.holding == object_id:
            return False
        return (not obj.openable) or obj.is_open
    if action == OPEN:
        return obj.openable and not obj.is_open
    if action == CLOSE:
        return obj.openable and obj.is_open
    return False


def _move_agent(world: World, room: str) -> None:
    """The held object moves with the agent. Contained objects stay with their receptacle."""

    world.room = room
    if world.holding is not None:
        world.objects[world.holding].room = room


def _apply_local(world: World, action: str, object_id: str) -> None:
    if action == PICKUP:
        world.inside.pop(object_id, None)
        world.holding = object_id
        world.objects[object_id].room = world.room
        return
    if action == PUT:
        held = world.holding
        if held is None:
            raise RuntimeError("put applied with an empty hand")
        world.inside[held] = object_id
        world.holding = None
        world.objects[held].room = world.objects[object_id].room
        return
    obj = world.objects[object_id]
    if action == OPEN:
        obj.is_open = True
        return
    if action == CLOSE:
        obj.is_open = False
        return
    raise RuntimeError(f"cannot apply {action}")
