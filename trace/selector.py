"""Host integration at interaction-mask selection.

The host builds a ``MaskRequest`` from its own candidates and estimated
symbolic state, then calls ``choose``. A deferred decision means the host
executes the mask it would have used without TRACE. A selection replaces
that mask with ``candidates[index]``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .belief.base import BeliefSource
from .config import DEFERRAL_COST
from .consequence.critic import ConsequenceCritic, GoalTemplate
from .consequence.world import World
from .gate import decide
from .relation import EpisodeMemory, infer_relation
from .types import Candidate, MaskDecision, Observation, Relation, require_positions


@dataclass
class MaskRequest:
    action: str
    target_cls: str
    candidates: list[Candidate]
    memory: EpisodeMemory
    world: World
    goal: GoalTemplate
    observation: Observation
    deferral_cost: float = DEFERRAL_COST
    pick_two_second: bool = False
    relation_override: Optional[Relation] = None


class TraceSelector:
    def __init__(
        self,
        belief: BeliefSource,
        *,
        rule: str = "trace",
        tau: float = 0.75,
        critic: Optional[ConsequenceCritic] = None,
    ):
        if rule == "host_score":
            raise ValueError("host_score is a baseline, not a TRACE integration rule")
        self.belief = belief
        self.rule = rule
        self.tau = tau
        self.critic = critic or ConsequenceCritic()

    def choose(self, request: MaskRequest) -> MaskDecision:
        relation = request.relation_override or infer_relation(
            request.action,
            request.target_cls,
            request.memory,
            pick_two_second=request.pick_two_second,
        )
        if relation.mode == "NONE" or not request.candidates:
            reason = "relation_none" if relation.mode == "NONE" else "no_candidates"
            return _deferred(relation.mode, relation.rule, self.rule, self.belief.name, reason)
        require_positions(request.candidates)

        belief = self.belief.identify(relation, request.candidates, request.observation)
        if belief.p_identity.size != len(request.candidates) + 1:
            raise ValueError(
                f"{belief.source} returned length {belief.p_identity.size}, "
                f"expected {len(request.candidates) + 1}"
            )
        costs = self.critic.cost_matrix(
            request.world,
            request.action,
            [candidate.object_id for candidate in request.candidates],
            request.goal,
        )
        scores = [candidate.score for candidate in request.candidates]
        decision = decide(
            belief,
            costs,
            mode=relation.mode,
            deferral_cost=request.deferral_cost,
            rule=self.rule,
            tau=self.tau,
            scores=scores,
        )
        return MaskDecision(
            defer=decision.action == "RETURN",
            index=decision.idx,
            rule=decision.rule,
            relation=relation.mode,
            relation_rule=relation.rule,
            expected_regret=tuple(float(value) for value in decision.expected_regret),
            p_identity=tuple(float(value) for value in belief.p_identity),
            reason=decision.reason,
            source=belief.source,
        )


def choose_interaction_mask(selector: TraceSelector, request: MaskRequest) -> Optional[int]:
    """Return the candidate index that should supply the interaction mask.

    ``None`` returns control to the host.
    """

    decision = selector.choose(request)
    return None if decision.defer else decision.index


def _deferred(mode: str, relation_rule: str, rule: str, source: str, reason: str) -> MaskDecision:
    return MaskDecision(
        defer=True,
        index=None,
        rule=rule,
        relation=mode,  # type: ignore[arg-type]
        relation_rule=relation_rule,
        expected_regret=(),
        p_identity=(),
        reason=reason,
        source=source,
    )
