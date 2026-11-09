"""Small symbolic world used by the consequence critic.

The state is the agent's estimate: which room it is in, what it holds, and
the open/closed and containment facts it believes. It is not a copy of
AI2-THOR metadata, and it does not read ground-truth object ids at decision time.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Optional


@dataclass
class Obj:
    id: str
    type: str
    room: str
    pickupable: bool = False
    receptacle: bool = False
    openable: bool = False
    is_open: bool = False


@dataclass
class World:
    objects: dict[str, Obj]
    inside: dict[str, str] = field(default_factory=dict)
    holding: Optional[str] = None
    room: str = "kitchen"
    nav_cost: float = 10.0

    def copy(self) -> "World":
        return World(
            objects={key: replace(obj) for key, obj in self.objects.items()},
            inside=dict(self.inside),
            holding=self.holding,
            room=self.room,
            nav_cost=self.nav_cost,
        )

    def state_key(self) -> tuple:
        objects = tuple(
            sorted((obj.id, obj.room, obj.is_open) for obj in self.objects.values())
        )
        inside = tuple(sorted(self.inside.items()))
        return (self.room, self.holding, objects, inside)


@dataclass(frozen=True)
class Goal:
    """Facts that must all hold. Empty fields are unconstrained."""

    holding: Optional[str] = None
    holding_type: Optional[str] = None
    holding_exclude: frozenset[str] = frozenset()
    inside: tuple[tuple[str, str], ...] = ()
    closed: tuple[str, ...] = ()

    def satisfied(self, world: World) -> bool:
        if self.holding is not None and world.holding != self.holding:
            return False
        if self.holding_type is not None:
            held = None if world.holding is None else world.objects.get(world.holding)
            if held is None or held.type != self.holding_type or held.id in self.holding_exclude:
                return False
        for obj_id, receptacle_id in self.inside:
            if world.inside.get(obj_id) != receptacle_id or world.holding == obj_id:
                return False
        for obj_id in self.closed:
            obj = world.objects.get(obj_id)
            if obj is None or obj.is_open:
                return False
        return True
