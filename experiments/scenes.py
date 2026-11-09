"""Symbolic states used by the planner tests and the live cost comparison."""

from __future__ import annotations

from trace.config import NAV_COST
from trace.consequence.critic import GoalTemplate
from trace.consequence.world import Obj, World


def same_room_drawers() -> tuple[World, GoalTemplate, list[str]]:
    """Put the held apple into the drawer that was opened. Both are in reach."""

    world = World(
        objects={
            "apple": Obj("apple", "Apple", "kitchen", pickupable=True),
            "drawer_a": Obj("drawer_a", "Drawer", "kitchen", receptacle=True, openable=True, is_open=True),
            "drawer_b": Obj("drawer_b", "Drawer", "kitchen", receptacle=True, openable=True, is_open=True),
        },
        holding="apple",
        room="kitchen",
        nav_cost=NAV_COST,
    )
    template = GoalTemplate(kind="put_into_ref", movable_id="apple", movable_type="Apple")
    return world, template, ["drawer_a", "drawer_b"]


def far_fridge() -> tuple[World, GoalTemplate, list[str]]:
    """Put the held apple into the fridge that already contains the task's object.

    One fridge is here. The other is a navigation away, so a wrong put is expensive to undo.
    """

    world = World(
        objects={
            "apple": Obj("apple", "Apple", "kitchen", pickupable=True),
            "fridge_here": Obj("fridge_here", "Fridge", "kitchen", receptacle=True, openable=True, is_open=True),
            "fridge_far": Obj("fridge_far", "Fridge", "garage", receptacle=True, openable=True, is_open=True),
        },
        holding="apple",
        room="kitchen",
        nav_cost=NAV_COST,
    )
    template = GoalTemplate(kind="put_into_ref", movable_id="apple", movable_type="Apple")
    return world, template, ["fridge_here", "fridge_far"]


def split_fridges() -> tuple[World, GoalTemplate, list[str]]:
    """Both candidate fridges are in other rooms.

    Whichever one is the reference, choosing the other wastes a round trip.
    With a moderate identity belief, every mask's expected regret exceeds a
    small deferral cost, so TRACE returns control.
    """

    world = World(
        objects={
            "apple": Obj("apple", "Apple", "kitchen", pickupable=True),
            "fridge_a": Obj("fridge_a", "Fridge", "garage", receptacle=True, openable=True, is_open=True),
            "fridge_b": Obj("fridge_b", "Fridge", "basement", receptacle=True, openable=True, is_open=True),
        },
        holding="apple",
        room="kitchen",
        nav_cost=NAV_COST,
    )
    template = GoalTemplate(kind="put_into_ref", movable_id="apple", movable_type="Apple")
    return world, template, ["fridge_a", "fridge_b"]


def open_drawers_to_close() -> tuple[World, GoalTemplate, list[str]]:
    world = World(
        objects={
            "drawer_a": Obj("drawer_a", "Drawer", "kitchen", receptacle=True, openable=True, is_open=True),
            "drawer_b": Obj("drawer_b", "Drawer", "kitchen", receptacle=True, openable=True, is_open=True),
        },
        room="kitchen",
        nav_cost=NAV_COST,
    )
    return world, GoalTemplate(kind="close_ref"), ["drawer_a", "drawer_b"]
