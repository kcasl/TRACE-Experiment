"""Identity vs task-grounding questions.

Paper identity asks which labelled candidate is the remembered object.
Ported first pickup/put has no memory, so ``relation.rule == task_instance``
asks which candidate satisfies the instruction instead, and drops the expert
reference crop so the model cannot match appearance to a leaked glance.
"""

from __future__ import annotations

from ..types import Observation, Relation

GROUNDING_RULE = "task_instance"


def is_grounding(relation: Relation) -> bool:
    return relation.rule == GROUNDING_RULE


def identity_instructions(relation: Relation) -> str:
    if is_grounding(relation):
        return (
            "The agent has not yet interacted with a remembered instance. "
            "Given the task instruction and the current subgoal, which labelled "
            "candidate should it use? If several labels are parts of one object "
            "(Sink basin vs Sink rim, Fridge shelf vs Fridge), pick the part the "
            "instruction names; prefer a Basin id when the parent is SinkBasin. "
            "Choose none if uncertain or if several look equally likely."
        )
    return (
        "The agent must find the physical object it interacted with earlier. "
        "Which labelled candidate in the current image is that reference? "
        "Choose none if it is not visible."
    )


def attach_reference_images(relation: Relation, observation: Observation) -> bool:
    return (not is_grounding(relation)) and observation.reference_frame is not None
