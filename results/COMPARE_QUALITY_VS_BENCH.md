# Selection quality vs benchmark (baselines included)

두 표는 다른 질문이다. 섞어서 SR로 CR을 읽지 말 것.

- **Selection quality:** N≥2 mask event에서 어느 instance를 고르는가 (commitment regret, wrong instance, return). Simulator success가 아님.
- **Benchmark:** episode success rate (official 10-fail 또는 ported THOR 5 pick-and-place).

Baseline은 항상 같은 표에 둔다: `host`, `oracle`, `threshold`, `force_best`, `host_score`, `DINOv2`.

---

## A. Selection quality

### A1. Synthetic identity (known-label drawings)

같은 질문, Jev vs OpenAI만 바꿈. `results/source_compare.md`

| | Jev | OpenAI | baseline |
|---|---|---|---|
| identity accuracy | 1.00 | 1.00 | — (oracle labels) |
| top-1 agree | 1.00 | 1.00 | each other |
| mean L1(`p_identity`) | 0.36 | 0.36 | 0 = identical |
| TRACE action agree (3 cost worlds) | 1.00 | 1.00 | — |

| case | truth | Jev p | OpenAI p | L1 |
|---|---|---|---|---|
| visible_match | 0 | [0.98, 0.00, 0.02] | [0.98, 0.01, 0.01] | 0.02 |
| reference_absent | 2 | [0.11, 0.01, **0.88**] | [0.38, 0.06, **0.56**] | 0.64 |
| lookalike | 0 | [0.74, 0.21, 0.05] | [**0.95**, 0.03, 0.02] | 0.42 |

OpenAI는 lookalike에 더 peaked. Jev는 absent에서 `none`에 더 많이 둔다. Threshold 0.75는 lookalike에서 갈림: Jev RETURN, OpenAI SELECT.

### A2. Unit-test worlds (fixed p = [0.6, 0.4, 0], c_defer = 4)

`results/offline_rules.json`. Belief를 고정하고 **decision rule**만 비교.

| rule | mean extra cost | cheap (extra 2) | expensive (extra 20) |
|---|---|---|---|
| **TRACE** | **2.4** | SELECT 0 | RETURN |
| threshold τ=0.55 | 4.4 | SELECT 0 | SELECT 0 |
| threshold τ=0.75 | 4.0 | RETURN | RETURN |
| host_score | 6.6 | wrong mask | wrong mask |

### A3. FILM/CAPEAM mask-log replay (Metric-FF C, N≥2)

`results/log_replay.json`. 같은 C, belief source × rule. `host_score` n이 더 작은 건 `host_choice`가 있는 event만.

CR = commitment regret (extra steps vs oracle decision). 낮을수록 좋음.
wrong = 틀린 instance를 SELECT한 비율. return = 호스트에 되돌린 비율.

#### FILM valid_seen (n=92 events, 820 ep log)

| belief | rule | n | CR | wrong | return | unrec |
|---|---|---|---|---|---|---|
| oracle | trace | 92 | **0.130** | 0.250 | 0.054 | 0 |
| **jev** | **trace** | 92 | **0.152** | 0.261 | 0.054 | 0 |
| **openai** | **trace** | 92 | **0.196** | 0.293 | 0.033 | 0 |
| dinov2 | trace | 92 | 1.380 | 0.457 | 0.033 | 0 |
| jev | force_best | 92 | 2.707 | 0.272 | 0 | 0.022 |
| openai | force_best | 92 | 2.630 | 0.261 | 0 | 0.022 |
| dinov2 | force_best | 92 | 5.435 | 0.587 | 0 | 0.022 |
| jev | threshold 0.75 | 92 | 3.424 | 0.141 | 0.207 | 0.022 |
| openai | threshold 0.75 | 92 | 3.598 | 0.120 | 0.293 | 0.022 |
| dinov2 | threshold 0.75 | 92 | 5.685 | 0.217 | 0.554 | 0.022 |
| host_score | (detector) | 38 | 5.474 | 0.211 | 0 | 0.053 |

TRACE + Jev/OpenAI가 DINOv2 TRACE·threshold·force_best·host_score보다 CR이 낮다.

#### FILM valid_unseen (n=65)

| belief | rule | n | CR | wrong | return | unrec |
|---|---|---|---|---|---|---|
| openai | force_best | 65 | **0.000** | 0.138 | 0 | 0 |
| oracle | force_best | 65 | 0.062 | 0.000 | 0 | 0 |
| oracle | threshold | 65 | 0.062 | 0.000 | 0 | 0 |
| oracle | trace | 65 | 0.123 | 0.062 | 0.031 | 0 |
| **openai** | **trace** | 65 | **0.185** | 0.000 | 0.031 | 0 |
| **jev** | **trace** | 65 | **0.246** | 0.031 | 0.031 | 0 |
| host_score | (detector) | 52 | 0.250 | 0.019 | 0 | 0 |
| jev | force_best | 65 | 0.431 | 0.046 | 0 | 0 |
| jev | threshold | 65 | 0.431 | 0.000 | 0.108 | 0 |
| dinov2 | trace | 65 | 0.462 | 0.108 | 0.031 | 0 |
| openai | threshold | 65 | 0.492 | 0.077 | 0.123 | 0 |
| dinov2 | threshold | 65 | 1.723 | 0.000 | 0.431 | 0 |
| dinov2 | force_best | 65 | 2.723 | 0.292 | 0 | 0 |

#### CAPEAM valid_seen (n=84)

| belief | rule | n | CR | wrong | return | unrec |
|---|---|---|---|---|---|---|
| oracle | trace | 84 | **0.000** | 0.512 | 0 | 0 |
| **jev** | **trace** | 84 | **0.012** | 0.512 | 0 | 0 |
| **openai** | **trace** | 84 | **0.012** | 0.512 | 0 | 0 |
| dinov2 | trace | 84 | 0.405 | 0.536 | 0 | 0 |
| host_score | (detector) | 60 | 1.100 | 0.367 | 0 | 0 |
| openai | force_best | 84 | 1.119 | 0.548 | 0 | 0 |
| jev | force_best | 84 | 1.179 | 0.619 | 0 | 0 |
| openai | threshold | 84 | 1.548 | 0.310 | 0.238 | 0 |
| jev | threshold | 84 | 1.833 | 0.274 | 0.345 | 0 |
| dinov2 | force_best | 84 | 2.131 | 0.679 | 0 | 0 |

CAPEAM seen TRACE CR이 거의 0이어도 wrong=0.51. 틀린 mask의 extra FF cost가 싸다는 뜻이지, instance가 맞다는 뜻이 아님.

CAPEAM unseen replay: local `costs.jsonl` 없음. Extra run `replay_slices.md`: CAPEAM seen wrong=0.512는 N≥2의 50%가 `j_true=null` (reference 미검출). in_list 42건 CR=0, wrong=0.167. FILM unseen Jev/OpenAI 갭은 Pickup 13건 (CR 0.923 vs 0.615).

### A4. THOR 5 closed-loop v2 — SELECT vs RETURN (Put success C)

TRACE-eligible = N≥2. 나머지는 N=1이라 host_score로 바로 감.

| host | split | source | n ep | TRACE events | RETURN | SELECT | SR |
|---|---|---|---|---|---|---|---|
| FILM | seen | host | 142 | 0 | — | — | 0.887 |
| FILM | seen | Jev | 142 | 70 | 69 | **1** | 0.887 |
| FILM | seen | OpenAI | 142 | 70 | 59 | **11** | **0.915** |
| FILM | unseen | host | 100 | 0 | — | — | 0.910 |
| FILM | unseen | Jev | 100 | 63 | 63 | **0** | 0.910 |
| FILM | unseen | OpenAI | 100 | 63 | 63 | **0** | 0.910 |
| CAPEAM | seen | Jev | 142 | 70 | 70 | **0** | 0.915 |
| CAPEAM | seen | OpenAI | 142 | 70 | 59 | **11** | 0.915 |
| CAPEAM | unseen | Jev | 100 | 63 | 62 | 1 | 0.910 |
| CAPEAM | unseen | OpenAI | 100 | 63 | 63 | **0** | 0.910 |

v2는 unseen에서 거의 SELECT를 안 해서 host SR을 유지한다. Seen에서 OpenAI만 11번 override. FILM seen +4는 FloorPlan422 HandTowel→SinkBasin idx 108–111 (`episode_v2.md`). CAPEAM seen은 host가 이미 그 4개를 맞춰 SR 동일.

---

## B. Benchmark (episode SR)

### B1. Closed-loop SR (jw THOR 5 pnp v2)

ryankwon 10-fail 820ep 표를 이 랩 숫자로 교체. pick_and_place_simple, seen 142 / unseen 100.
Pickup 1.0. Unseen 0.910 = 91/100 (recheck). 출처: `RERUN.md`.

| host | split | n | host | oracle | Jev v2 | OpenAI v2 |
|---|---|---|---|---|---|---|
| FILM | seen | 142 | 0.887 | 0.887 | 0.887 | **0.915** |
| FILM | unseen | 100 | **0.910** | 0.910 | 0.910 | 0.910 |
| CAPEAM | seen | 142 | **0.915** | 0.887 | 0.915 | 0.915 |
| CAPEAM | unseen | 100 | **0.910** | 0.910 | 0.910 | 0.910 |

### B2. Ported THOR 5 pick-and-place — method v1 (expert crop + step-cost C)

`results/thor_port/matrix_summary.md`. pnp-simple only. Pickup 전부 1.0.

| host | split | n | host | oracle | Jev TRACE | OpenAI TRACE | vs host |
|---|---|---|---|---|---|---|---|
| FILM distance | seen | 142 | **0.887** | 0.887 | **0.915** | 0.887 | Jev +4 ep |
| FILM distance | unseen | 100 | **0.910** | 0.910 | 0.880 | 0.880 | both −3 ep |
| CAPEAM Mask R-CNN | seen | 142 | **0.915** | 0.887 | 0.915 | 0.887 | OpenAI −4 ep |
| CAPEAM Mask R-CNN | unseen | 100 | **0.910** | 0.910 | 0.880 | 0.880 | both −3 ep |

Jev seen API error 3건 → host fallback. Unseen 손해는 전부 Put.

### B3. Ported THOR 5 — method v2 (task grounding + Put success C=100)

Lab `/data/jw_workspace/trace-alfred/results/thor_port_v2/`. Baseline host/oracle은 v1과 동일 (재측정 일치).

| host | split | n | host | oracle | Jev v2 | OpenAI v2 | vs host | vs v1 Jev/OpenAI |
|---|---|---|---|---|---|---|---|---|
| FILM | seen | 142 | 0.887 | 0.887 | 0.887 | **0.915** | OpenAI +4 | Jev 0.915→0.887, OpenAI 0.887→0.915 |
| FILM | unseen | 100 | 0.910 | 0.910 | **0.910** | **0.910** | 0 | both 0.880→0.910 |
| CAPEAM | seen | 142 | 0.915 | 0.887 | 0.915 | **0.915** | 0 / OpenAI +0 vs host, +4 vs v1 OpenAI | Jev same, OpenAI 0.887→0.915 |
| CAPEAM | unseen | 100 | 0.910 | 0.910 | **0.910** | **0.910** | 0 | both 0.880→0.910 |

v2 읽기: unseen에서 Jev/OpenAI가 host SR로 회복 (v1 −3ep 제거). Seen FILM/CAPEAM에서 OpenAI만 11번 SELECT, FILM seen +4 ep. Jev는 거의 전부 RETURN.

---

## 한 줄로

| 질문 | Jev/OpenAI TRACE vs baseline |
|---|---|
| Mask 선택 extra cost (replay CR) | host_score / DINOv2 / threshold보다 낮음 |
| Closed-loop SR (jw) | `RERUN.md` / B1 |
| THOR 5 pnp v1 unseen SR | host 0.910보다 **낮음** (0.880) |
| THOR 5 pnp v2 unseen SR | host와 **같음** (0.910, FILM+CAPEAM) |
