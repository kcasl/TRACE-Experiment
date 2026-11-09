"""Shared defaults.

Keeping these in one place means the closed-loop selector, the offline
comparison, and the live experiment all use the same numbers, so a change to
the deferral cost or the navigation cost cannot drift between them.
"""

from __future__ import annotations

# Extra-cost budget below which TRACE may override the host mask. RETURN still
# executes the host mask on this step; it is not a look-again cost. About two
# local recoveries (pickup then put = 2) plus slack.
OVERRIDE_BUDGET = 4.0
DEFERRAL_COST = OVERRIDE_BUDGET  # logs and gate kwargs keep this name

# Cost of moving between rooms in the symbolic planner, in the same step unit.
NAV_COST = 10.0

# Optional ablation: treat a wrong Put as episode failure. The reported
# critic does not use this; Put extra cost is the planner's remaining plan
# (wrong nearby sink ≈ 2, wrong far fridge ≈ 22). With planner C and
# c_defer=4, a cheap Put SELECTs at Jev's FP422 p≈0.92 because
# 2*(1-0.92)=0.16 < 4. Raising c_defer while keeping fail=100 would be a
# second, redundant knob and would over-commit on expensive Puts.
EPISODE_FAIL_COST = 100.0
