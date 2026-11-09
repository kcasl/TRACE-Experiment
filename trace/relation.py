"""Symbolic identity relation from the task chain.

The upcoming interaction needs one of three relations to a remembered object:

* SAME — act on the object that was interacted with earlier
* DIFFERENT — act on another instance of that class
* NONE — identity does not change the correct mask; the host keeps control

Rules use only the action chain. They do not read simulator identifiers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .types import MemoryEntry, Relation

PROCESSING_RECEPTACLES = frozenset({"Microwave", "Fridge", "SinkBasin", "Sink", "BathtubBasin"})

OPEN = "OpenObject"
CLOSE = "CloseObject"
PUT = "PutObject"
PICK = "PickupObject"
TOGGLE_ON = "ToggleObjectOn"
TOGGLE_OFF = "ToggleObjectOff"


def canonical_action(action: str) -> str:
    text = action.replace("_15", "").replace("_25", "").replace("_90", "")
    return {"ToggleOn": TOGGLE_ON, "ToggleOff": TOGGLE_OFF}.get(text, text)


@dataclass
class EpisodeMemory:
    """Successful interactions in one episode, oldest first."""

    entries: list[MemoryEntry] = field(default_factory=list)
    _seq: int = 0

    def reset(self) -> None:
        self.entries.clear()
        self._seq = 0

    def record(
        self,
        action: str,
        cls: str,
        success: bool,
        t: int,
        *,
        role: str = "object",
        receptacle_cls: Optional[str] = None,
    ) -> Optional[MemoryEntry]:
        if not success:
            return None
        action = canonical_action(action)
        if action in (OPEN, CLOSE):
            role = "receptacle"
        elif action in (PICK, TOGGLE_ON, TOGGLE_OFF):
            role = "object"
        elif action == PUT:
            if role not in ("object", "receptacle"):
                raise ValueError("PutObject must record the object side and the receptacle side separately")
        else:
            raise ValueError(f"unsupported memory action {action!r}")
        entry = MemoryEntry(
            key=f"{cls}#{self._seq}",
            cls=cls,
            action=action,
            t=t,
            role=role,  # type: ignore[arg-type]
            receptacle_cls=receptacle_cls,
        )
        self._seq += 1
        self.entries.append(entry)
        return entry

    def last(self, *, action: Optional[str] = None, cls: Optional[str] = None, role: Optional[str] = None) -> Optional[MemoryEntry]:
        action = None if action is None else canonical_action(action)
        for entry in reversed(self.entries):
            if action is not None and entry.action != action:
                continue
            if cls is not None and entry.cls != cls:
                continue
            if role is not None and entry.role != role:
                continue
            return entry
        return None


def infer_relation(
    action: str,
    cls: str,
    memory: EpisodeMemory,
    *,
    pick_two_second: bool = False,
) -> Relation:
    """Return the identity constraint for the interaction the host is about to run."""

    action = canonical_action(action)
    if pick_two_second and action == PICK:
        ref = memory.last(action=PICK, cls=cls, role="object")
        if ref is None:
            return Relation("NONE", None, "")
        return Relation("DIFFERENT", ref, "R5")

    last = memory.entries[-1] if memory.entries else None
    if action in (PUT, CLOSE) and last is not None and last.action == OPEN and last.cls == cls and last.role == "receptacle":
        return Relation("SAME", last, "R1")

    if action == TOGGLE_OFF:
        ref = memory.last(action=TOGGLE_ON, cls=cls, role="object")
        if ref is not None:
            return Relation("SAME", ref, "R3")

    if action in (CLOSE, TOGGLE_ON, OPEN):
        put = _last_receptacle_put(memory, cls)
        if put is not None and _no_receptacle_event_since(memory, put):
            return Relation("SAME", put, "R2")

    if action == PICK:
        placed = _last_object_put(memory, cls)
        if placed is not None:
            if placed.receptacle_cls in PROCESSING_RECEPTACLES:
                return Relation("SAME", placed, "R4")
            return Relation("DIFFERENT", placed, "R5")

    return Relation("NONE", None, "")


def _last_receptacle_put(memory: EpisodeMemory, cls: str) -> Optional[MemoryEntry]:
    return memory.last(action=PUT, cls=cls, role="receptacle")


def _last_object_put(memory: EpisodeMemory, cls: str) -> Optional[MemoryEntry]:
    for entry in reversed(memory.entries):
        if entry.action == PUT and entry.role == "object" and entry.cls == cls:
            return entry
        if entry.action == PICK and entry.role == "object" and entry.cls == cls:
            return None
    return None


def _no_receptacle_event_since(memory: EpisodeMemory, put: MemoryEntry) -> bool:
    seen = False
    for entry in memory.entries:
        if entry is put:
            seen = True
            continue
        if seen and entry.role == "receptacle" and entry.action in (OPEN, CLOSE, TOGGLE_ON, TOGGLE_OFF):
            return False
    return True
