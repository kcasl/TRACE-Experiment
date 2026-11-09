"""Symbolic-planner cost matrices for the documented scenes.

These reproduce, without a network or a simulator, the regret numbers quoted in
METHOD.md. Each scene is a put/close task with two same-class candidates, so
``C`` is a 2x3 matrix (two masks, three identity hypotheses including "absent").
"""

from __future__ import annotations

import numpy as np

from trace.config import DEFERRAL_COST
from trace.consequence.critic import ConsequenceCritic
from trace.consequence.planner import CLOSE, PUT
from trace.gate import expected_regret

from .scenes import far_fridge, open_drawers_to_close, same_room_drawers, split_fridges

# A deliberately middling identity belief: candidate 0 is the likely reference.
EXAMPLE_BELIEF = np.array([0.6, 0.4, 0.0])

SCENES = {
    "same_room_drawers": (same_room_drawers, PUT),
    "far_fridge": (far_fridge, PUT),
    "split_fridges": (split_fridges, PUT),
    "open_drawers_to_close": (open_drawers_to_close, CLOSE),
}


def scene_cost_tables(belief: np.ndarray = EXAMPLE_BELIEF, *, deferral_cost: float = DEFERRAL_COST) -> dict:
    critic = ConsequenceCritic()
    scenes = []
    for name, (builder, action) in SCENES.items():
        world, template, ids = builder()
        costs = critic.cost_matrix(world, action, ids, template)
        regret = expected_regret(costs.C, belief)
        chosen = int(np.argmin(regret))
        scenes.append(
            {
                "scene": name,
                "action": action,
                "candidate_ids": list(ids),
                "regret_matrix": costs.C.round(3).tolist(),
                "expected_regret": regret.round(3).tolist(),
                "argmin_regret": chosen,
                "trace_action": "SELECT" if float(regret[chosen]) < deferral_cost else "RETURN",
            }
        )
    return {"belief": belief.tolist(), "deferral_cost": deferral_cost, "scenes": scenes}
