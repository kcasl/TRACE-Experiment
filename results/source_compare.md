# Jev vs OpenAI Decisions

Same identity cases, same TRACE cost worlds. Only the belief source changes.

- Jev identity accuracy: 1.0
- OpenAI Decisions identity accuracy: 1.0
- Top-1 agree: 1.0
- Mean L1 between p_identity: 0.36
- TRACE action agree: same_room_drawers 1.0, far_fridge 1.0, split_fridges 1.0

| case | truth | Jev p | OpenAI p | Jev top | OpenAI top | L1 | cheap TRACE | expensive TRACE | both-far TRACE |
|---|---|---|---|---|---|---|---|---|---|
| visible_match | 0 | [0.98, 0.00, 0.02] | [0.98, 0.01, 0.01] | 0 | 0 | 0.02 | SELECT 0 | SELECT 0 | SELECT 0 |
| reference_absent | 2 | [0.11, 0.01, 0.88] | [0.38, 0.06, 0.56] | 2 | 2 | 0.64 | SELECT 0 | SELECT 0 | RETURN |
| lookalike | 0 | [0.74, 0.21, 0.05] | [0.95, 0.03, 0.02] | 0 | 0 | 0.42 | SELECT 0 | SELECT 0 | SELECT 0 |
