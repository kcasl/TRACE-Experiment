# TRACE results

Unit tests: passed.

## Decision rules on the same belief

Belief p_identity = [0.6, 0.4, 0], deferral cost = 4.
Realized expected extra cost, lower is better.

| rule | cheap world | expensive world | mean |
|---|---|---|---|
| trace | 0.8 | 4 | 2.4 |
| threshold_055 | 0.8 | 8 | 4.4 |
| threshold_075 | 4 | 4 | 4 |
| host_score | 1.2 | 12 | 6.6 |

## Symbolic-planner regret by scene

Belief p_identity = [0.6, 0.4, 0], deferral cost = 4.

| scene | action | regret per mask | TRACE |
|---|---|---|---|
| same_room_drawers | PutObject | [0.8, 1.2] | SELECT 0 |
| far_fridge | PutObject | [0.8, 13.2] | SELECT 0 |
| split_fridges | PutObject | [4.8, 7.2] | RETURN |
| open_drawers_to_close | CloseObject | [0.4, 0.6] | SELECT 0 |

## Identity beliefs from decision models

Top-1 identity accuracy over graded cases: jev 1.00, openai_decisions 1.00, oracle 1.00.

| case | source | p_identity | top | correct |
|---|---|---|---|---|
| visible_match | oracle | [1, 0, 0] | 0 | yes |
| visible_match | jev | [0.98, 0, 0.02] | 0 | yes |
| visible_match | openai_decisions | [0.98, 0.01, 0.01] | 0 | yes |
| reference_absent | oracle | [0, 0, 1] | 2 | yes |
| reference_absent | jev | [0.08, 0.01, 0.91] | 2 | yes |
| reference_absent | openai_decisions | [0.38, 0.06, 0.56] | 2 | yes |
| lookalike | oracle | [1, 0, 0] | 0 | yes |
| lookalike | jev | [0.77, 0.18, 0.05] | 0 | yes |
| lookalike | openai_decisions | [0.95, 0.03, 0.02] | 0 | yes |

