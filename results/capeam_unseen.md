# CAPEAM Unseen (THOR-jsonl TRACE)

Closed-loop THOR 5 pick_and_place_simple, `n=100`.
Metric-FF `costs.jsonl` is still absent; this table is the put-success critic replayed from logged `p`.

Critic: Put success-valued C from logged p (no Metric-FF). `c_defer=4`, fail=100, SELECT only if `p_i>0.96`.

## capeam/valid_unseen

| source | n | SR | pickup | put | N≥2 | RETURN | SELECT | rebuild match | vs-host ΔSR |
|---|---|---|---|---|---|---|---|---|---|
| host | 100 | 0.910 | 1.000 | 0.910 | 63 | 63 | 0 | 1.00 | — |
| oracle | 100 | 0.910 | 1.000 | 0.910 | 63 | 63 | 0 | 1.00 | +0/-0 |
| jev | 100 | 0.910 | 1.000 | 0.910 | 63 | 62 | 1 | 1.00 | +0/-0 |
| openai_decisions | 100 | 0.910 | 1.000 | 0.910 | 63 | 63 | 0 | 1.00 | +0/-0 |

### Fail episodes (host = all sources)

| idx | floor | target | parent |
|---|---|---|---|
| 21 | FloorPlan219 | Vase | Safe |
| 22 | FloorPlan219 | Vase | Safe |
| 23 | FloorPlan219 | Vase | Safe |
| 54 | FloorPlan219 | Vase | Safe |
| 55 | FloorPlan219 | Vase | Safe |
| 56 | FloorPlan219 | Vase | Safe |
| 87 | FloorPlan219 | Vase | Safe |
| 88 | FloorPlan219 | Vase | Safe |
| 89 | FloorPlan219 | Vase | Safe |

### jev SELECT

| idx | floor | action | n | pmax | ok | ep |
|---|---|---|---|---|---|---|
| 36 | FloorPlan10 | PutObject | 8 | 0.97 | 1 | 1 |

### openai_decisions SELECT

(none)

## capeam/valid_unseen_recheck

| source | n | SR | pickup | put | N≥2 | RETURN | SELECT | rebuild match | vs-host ΔSR |
|---|---|---|---|---|---|---|---|---|---|
| host | 100 | 0.910 | 1.000 | 0.910 | 63 | 63 | 0 | 1.00 | — |
| oracle | 100 | 0.910 | 1.000 | 0.910 | 63 | 63 | 0 | 1.00 | +0/-0 |
| jev | 100 | 0.910 | 1.000 | 0.910 | 63 | 63 | 0 | 1.00 | +0/-0 |
| openai_decisions | 100 | 0.910 | 1.000 | 0.910 | 63 | 63 | 0 | 1.00 | +0/-0 |

### Fail episodes (host = all sources)

| idx | floor | target | parent |
|---|---|---|---|
| 21 | FloorPlan219 | Vase | Safe |
| 22 | FloorPlan219 | Vase | Safe |
| 23 | FloorPlan219 | Vase | Safe |
| 54 | FloorPlan219 | Vase | Safe |
| 55 | FloorPlan219 | Vase | Safe |
| 56 | FloorPlan219 | Vase | Safe |
| 87 | FloorPlan219 | Vase | Safe |
| 88 | FloorPlan219 | Vase | Safe |
| 89 | FloorPlan219 | Vase | Safe |

### jev SELECT

(none)

### openai_decisions SELECT

(none)

## Recheck boundary (idx 36)

Same episode `FloorPlan10` SaltShaker→Cabinet. v2 Jev SELECT at `pmax=0.97`.
Recheck Jev `pmax=0.96` so `R=4` equals `c_defer` and the gate RETURNS.
Logged v2 SELECT pmax=0.97. Recheck Jev SELECT count=0.
SR stays 0.910 both runs: the SELECT did not change Put success.

## Metric-FF

`capeam/valid_unseen` costs/beliefs are still missing on the lab and locally.
Official 820ep CAPEAM unseen host SR 0.390 is a campaign constant, not this run.
