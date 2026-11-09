# Ported THOR 5 TRACE matrix v2

Method: task grounding (no expert crop) + Put success-valued C (fail=100, `c_defer=4`).
TRACE SELECT only if `p_i > 0.96`. Else RETURN → host.

Baseline host/oracle remeasured on this code. Numbers match v1.

Pickup is 1.000 on every row. Gap is Put.

## SR vs v1 and baseline

| host | split | n | host | oracle | Jev v1 | OpenAI v1 | Jev v2 | OpenAI v2 |
|---|---|---|---|---|---|---|---|---|
| FILM distance | seen | 142 | 0.887 | 0.887 | **0.915** | 0.887 | 0.887 | **0.915** |
| FILM distance | unseen | 100 | 0.910 | 0.910 | 0.880 | 0.880 | **0.910** | **0.910** |
| CAPEAM Mask R-CNN | seen | 142 | 0.915 | 0.887 | 0.915 | 0.887 | 0.915 | **0.915** |
| CAPEAM Mask R-CNN | unseen | 100 | 0.910 | 0.910 | 0.880 | 0.880 | **0.910** | **0.910** |

## v2 SELECT vs RETURN (N≥2 TRACE events)

| host | split | source | TRACE events | RETURN | SELECT | SR |
|---|---|---|---|---|---|---|
| FILM | seen | Jev | 70 | 69 | 1 | 0.887 |
| FILM | seen | OpenAI | 70 | 59 | **11** | **0.915** |
| FILM | unseen | Jev | 63 | 63 | 0 | 0.910 |
| FILM | unseen | OpenAI | 63 | 63 | 0 | 0.910 |
| CAPEAM | seen | Jev | 70 | 70 | 0 | 0.915 |
| CAPEAM | seen | OpenAI | 70 | 59 | **11** | 0.915 |
| CAPEAM | unseen | Jev | 63 | 62 | 1 | 0.910 |
| CAPEAM | unseen | OpenAI | 63 | 63 | 0 | 0.910 |

## Raw v2 table

| host | split | source | n | SR | pickup | put | api_errors |
|---|---|---|---|---|---|---|---|
| film | valid_seen | host | 142 | 0.887 | 1.000 | 0.887 | 0 |
| film | valid_seen | oracle | 142 | 0.887 | 1.000 | 0.887 | 0 |
| film | valid_seen | jev | 142 | 0.887 | 1.000 | 0.887 | 0 |
| film | valid_seen | openai_decisions | 142 | 0.915 | 1.000 | 0.915 | 0 |
| film | valid_unseen | host | 100 | 0.910 | 1.000 | 0.910 | 0 |
| film | valid_unseen | oracle | 100 | 0.910 | 1.000 | 0.910 | 0 |
| film | valid_unseen | jev | 100 | 0.910 | 1.000 | 0.910 | 0 |
| film | valid_unseen | openai_decisions | 100 | 0.910 | 1.000 | 0.910 | 0 |
| capeam | valid_seen | host | 142 | 0.915 | 1.000 | 0.915 | 0 |
| capeam | valid_seen | oracle | 142 | 0.887 | 1.000 | 0.887 | 0 |
| capeam | valid_seen | jev | 142 | 0.915 | 1.000 | 0.915 | 0 |
| capeam | valid_seen | openai_decisions | 142 | 0.915 | 1.000 | 0.915 | 0 |
| capeam | valid_unseen | host | 100 | 0.910 | 1.000 | 0.910 | 0 |
| capeam | valid_unseen | oracle | 100 | 0.910 | 1.000 | 0.910 | 0 |
| capeam | valid_unseen | jev | 100 | 0.910 | 1.000 | 0.910 | 0 |
| capeam | valid_unseen | openai_decisions | 100 | 0.910 | 1.000 | 0.910 | 0 |

API errors: 0. Lab jsonl: `/data/jw_workspace/trace-alfred/results/thor_port_v2/`.
