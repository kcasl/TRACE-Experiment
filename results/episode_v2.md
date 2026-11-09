# THOR 5 v2 episode list (lab jsonl, 2026-11-05)

Lab `/data/jw_workspace/trace-alfred/results/thor_port_v2/`. Parsed after SSH recovered.

---

## FILM seen OpenAI +4 (host fail → OpenAI success)

전부 `FloorPlan422` `HandTowel → SinkBasin`. Host Put 실패, OpenAI는 SinkBasin id로 SELECT.

| idx | floor | target | parent | host put | OpenAI put |
|---|---|---|---|---|---|
| 108 | FloorPlan422 | HandTowel | SinkBasin | 0 | 1 |
| 109 | FloorPlan422 | HandTowel | SinkBasin | 0 | 1 |
| 110 | FloorPlan422 | HandTowel | SinkBasin | 0 | 1 |
| 111 | FloorPlan422 | HandTowel | SinkBasin | 0 | 1 |

OpenAI Put id = `Sink|-01.93|+00.77|+00.33|SinkBasin`. Host id = `Sink|-01.93|+00.77|+00.33` (SinkBasin suffix 없음). pmax=0.99, n=3. lost=0.

Jev는 이 4개를 못 살림 (gained=0).

---

## OpenAI SELECT 11 (FILM seen)

| idx | floor | action | vs host | ok | ep | pmax | n |
|---|---|---|---|---|---|---|---|
| 16 | FloorPlan7 | Put Fridge | same | 1 | 1 | 0.97 | 8 |
| 108–111 | FloorPlan422 | Put SinkBasin | **different id** | 1 | **1 (host 0)** | 0.99 | 3 |
| 112–114 | FloorPlan406 | Pickup ToiletPaper | same | 1 | 1 | 0.99 | 2 |
| 124–126 | FloorPlan4 | Put Fridge | same | 1 | 1 | 0.99–1.0 | 8 |

FILM unseen OpenAI SELECT **0**. CAPEAM unseen OpenAI SELECT **0**.

## OpenAI SELECT 11 (CAPEAM seen)

Host가 FloorPlan422를 이미 맞춤 → SR 변화 없음.

| idx | floor | action | vs host | pmax |
|---|---|---|---|---|
| 15–17 | FloorPlan7 | Put Fridge | same | 0.97 |
| 108–111 | FloorPlan422 | Put SinkBasin | same (이미 SinkBasin) | 1.0 |
| 112–114 | FloorPlan406 | Pickup ToiletPaper | **different** (host `|-01.25…`, OpenAI `|+00.23…`) | 0.91 |
| 126 | FloorPlan4 | Put CounterTop | same | 0.97 |

112–114는 둘 다 pickup 성공이라 SR에 안 잡힘.

---

## Jev SELECT

| host | split | idx | floor | action | ok | pmax |
|---|---|---|---|---|---|---|
| FILM | seen | 128 | FloorPlan406 | PutObject | 1 | 0.97 |
| FILM | unseen | — | — | — | — | — |
| CAPEAM | seen | — | — | — | — | — |
| CAPEAM | unseen | 36 | FloorPlan10 | PutObject | 1 | 0.97 |

둘 다 episode 성공, host도 성공 → SR 0.

---

## FILM vs CAPEAM host pickup ID (seen)

pickup_id_diff=**3**, 전부 성공 쌍:

| idx | floor | FILM pickup | CAPEAM pickup |
|---|---|---|---|
| 112 | FloorPlan406 | ToiletPaper\|+00.23\|+00.09\|+02.16 | ToiletPaper\|-01.25\|+00.70\|+01.08 |
| 113 | FloorPlan406 | same | same |
| 114 | FloorPlan406 | same | same |

put_id_diff=28이지만 SR 갭 4는 이 pickup 3개가 아니다.

SR 갭 0.887→0.915 = FILM host만 실패한 FloorPlan422 HandTowel→SinkBasin 4건. CAPEAM host는 그 4개를 맞춤.

---

## Seen host fail floors

| host | floor | target | parent | n | 합 |
|---|---|---|---|---|---|
| FILM | FloorPlan224 | Box | Dresser | 3 | |
| FILM | FloorPlan20 | Knife | SinkBasin | 3 | |
| FILM | FloorPlan422 | HandTowel | SinkBasin | **4** | OpenAI가 살림 |
| FILM | FloorPlan409 | SprayBottle | GarbageCan | 3 | |
| FILM | FloorPlan224 | Newspaper | Sofa | 3 | **16** |
| CAPEAM | FloorPlan224 | Box | Dresser | 3 | |
| CAPEAM | FloorPlan20 | Knife | SinkBasin | 3 | |
| CAPEAM | FloorPlan409 | SprayBottle | GarbageCan | 3 | |
| CAPEAM | FloorPlan224 | Newspaper | Sofa | 3 | **12** |

CAPEAM seen fail = FILM seen fail − FloorPlan422 4건.

---

## Unseen fail idxs (FILM host = Jev = OpenAI)

`21, 22, 23, 54, 55, 56, 87, 88, 89` — FloorPlan219 Vase→Safe. Recheck와 동일.
