# TRACE closed-loop rerun (2026-11-02)

Ported THOR 5, pick_and_place_simple only.
**이 표가 ryankwon Official 10-failure SR (820ep host log)를 대체한다.** 10-fail protocol이 아님 (action 2개). Pickup은 모든 행에서 1.000. 실패는 Put.

**v2 method:** first pickup/put = task grounding (expert crop 없음) + Put success-valued C (`fail=100`, `c_defer=4`). SELECT는 `p_i > 0.96`일 때만.

**테스트:** `jw/` pytest 52 passed.

---

## 1. 뭘 다시 돌렸나

| 런 | 폴더 | 내용 | 시각 (KST) |
|---|---|---|---|
| v1 | `results/thor_port/` | expert crop + step-cost C, 전체 matrix | 이전 |
| v2 | `results/thor_port_v2/` | v2 method, host/oracle 재측정 + Jev/OpenAI, FILM+CAPEAM seen/unseen | 14:33–14:48 |
| recheck | `results/thor_port_v2_recheck/` | unseen host/oracle 재확인 + CAPEAM unseen Jev/OpenAI | 15:17–15:20 |

Lab jsonl: `/data/jw_workspace/trace-alfred/results/thor_port_v2/` 및 `thor_port_v2_recheck/`.

---

## 2. Baseline 재측정

host / oracle은 v1 → v2 → recheck에서 **같은 숫자**.

| host | split | n | host SR | oracle SR | 확인 |
|---|---|---|---|---|---|
| FILM distance | seen | 142 | 0.887 (126/142) | 0.887 | v1 = v2 |
| FILM distance | unseen | 100 | **0.910 (91/100)** | 0.910 | v1 = v2 = recheck |
| CAPEAM Mask R-CNN | seen | 142 | 0.915 (130/142) | 0.887 (126/142) | v1 = v2 |
| CAPEAM Mask R-CNN | unseen | 100 | **0.910 (91/100)** | 0.910 | v1 = v2 = recheck |

Unseen 0.910은 반올림이 아니다. **91 success / 9 fail.** Recheck에서 FILM host, FILM oracle, CAPEAM host, CAPEAM oracle, CAPEAM Jev, CAPEAM OpenAI가 전부 같은 9 episode에서 실패한다.

실패 9개 (전부 `FloorPlan219`, `Vase → Safe`, pickup 성공, Put 실패):

`idx 21, 22, 23, 54, 55, 56, 87, 88, 89`

이 Safe Put은 identity SELECT와 무관하다. Host를 바꿔도 같다.

---

## 3. v1 vs v2 SR (Jev / OpenAI)

| host | split | n | host | Jev v1 | OpenAI v1 | Jev v2 | OpenAI v2 |
|---|---|---|---|---|---|---|---|
| FILM | seen | 142 | 0.887 | **0.915** | 0.887 | 0.887 | **0.915** |
| FILM | unseen | 100 | 0.910 | 0.880 | 0.880 | **0.910** | **0.910** |
| CAPEAM | seen | 142 | 0.915 | 0.915 | 0.887 | 0.915 | **0.915** |
| CAPEAM | unseen | 100 | 0.910 | 0.880 | 0.880 | **0.910** | **0.910** |

Recheck CAPEAM unseen Jev/OpenAI도 **0.910**. v2와 일치.

Δ vs host:

| | Jev v1 | OpenAI v1 | Jev v2 | OpenAI v2 |
|---|---|---|---|---|
| FILM seen | +4 ep | 0 | 0 | **+4 ep** |
| FILM unseen | −3 ep | −3 ep | 0 | 0 |
| CAPEAM seen | 0 | −4 ep | 0 | 0 |
| CAPEAM unseen | −3 ep | −3 ep | 0 | 0 |

v1 unseen −3은 v2에서 사라졌다. Seen에서 OpenAI만 host를 이긴다 (FILM +4). Jev v2는 거의 RETURN이라 host와 같다.

v1 Jev seen API error 3건. v2 API error **0**.

---

## 4. v2 SELECT / RETURN (N≥2)

N=1이면 TRACE를 안 타고 host_score.

| host | split | source | events | RETURN | SELECT | SR |
|---|---|---|---|---|---|---|
| FILM | seen | Jev | 70 | 69 | 1 | 0.887 |
| FILM | seen | OpenAI | 70 | 59 | **11** | **0.915** |
| FILM | unseen | Jev | 63 | 63 | 0 | 0.910 |
| FILM | unseen | OpenAI | 63 | 63 | 0 | 0.910 |
| CAPEAM | seen | Jev | 70 | 70 | 0 | 0.915 |
| CAPEAM | seen | OpenAI | 70 | 59 | **11** | 0.915 |
| CAPEAM | unseen | Jev | 63 | 62 | 1 | 0.910 |
| CAPEAM | unseen | OpenAI | 63 | 63 | 0 | 0.910 |

OpenAI seen SELECT 11번 → FILM +4 success. CAPEAM seen은 host가 이미 0.915라 SELECT가 SR을 더 안 올림.

---

## 5. CAPEAM unseen

THOR 5 closed-loop에는 **seen과 같이 있다.** Recheck까지 포함.

| source | CAPEAM unseen SR |
|---|---|
| host | 0.910 |
| oracle | 0.910 |
| Jev v2 | 0.910 |
| OpenAI v2 | 0.910 |

없는 것은 Metric-FF **log replay CR**이다. Lab/local 모두 `capeam/valid_unseen` `costs.jsonl`이 없다.

대신 closed-loop jsonl에서 put-success $C$를 재구성했다 (`experiments/thor_jsonl.py`, `results/capeam_unseen.md`).

| source | SR | N≥2 | RETURN | SELECT | vs host |
|---|---|---|---|---|---|
| host / oracle / OpenAI | 0.910 | 63 | 63 | 0 | 0 |
| Jev v2 | 0.910 | 63 | 62 | 1 (idx 36, pmax=0.97) | 0 |
| Jev recheck | 0.910 | 63 | 63 | 0 (같은 에피소드 pmax=0.96 → RETURN) | 0 |

실패 9건은 전부 FloorPlan219 Vase→Safe (`21–23, 54–56, 87–89`). Rebuild match 1.00.

---

## 6. Raw v2 (전체)

| host | split | source | n | SR | pickup | put | api_errors |
|---|---|---|---|---|---|---|---|
| film | seen | host | 142 | 0.887 | 1.000 | 0.887 | 0 |
| film | seen | oracle | 142 | 0.887 | 1.000 | 0.887 | 0 |
| film | seen | jev | 142 | 0.887 | 1.000 | 0.887 | 0 |
| film | seen | openai_decisions | 142 | 0.915 | 1.000 | 0.915 | 0 |
| film | unseen | host | 100 | 0.910 | 1.000 | 0.910 | 0 |
| film | unseen | oracle | 100 | 0.910 | 1.000 | 0.910 | 0 |
| film | unseen | jev | 100 | 0.910 | 1.000 | 0.910 | 0 |
| film | unseen | openai_decisions | 100 | 0.910 | 1.000 | 0.910 | 0 |
| capeam | seen | host | 142 | 0.915 | 1.000 | 0.915 | 0 |
| capeam | seen | oracle | 142 | 0.887 | 1.000 | 0.887 | 0 |
| capeam | seen | jev | 142 | 0.915 | 1.000 | 0.915 | 0 |
| capeam | seen | openai_decisions | 142 | 0.915 | 1.000 | 0.915 | 0 |
| capeam | unseen | host | 100 | 0.910 | 1.000 | 0.910 | 0 |
| capeam | unseen | oracle | 100 | 0.910 | 1.000 | 0.910 | 0 |
| capeam | unseen | jev | 100 | 0.910 | 1.000 | 0.910 | 0 |
| capeam | unseen | openai_decisions | 100 | 0.910 | 1.000 | 0.910 | 0 |

---

## 7. 따로 돌린 추가 분석 (local replay slices)

Metric-FF log를 action / `j_true`로 자른 extra run. Episode SELECT 목록은 §8.

커맨드: `python -m experiments.replay_slices`  
표: `replay_slices.md` · JSON: `replay_slices.json` · TeX: `results.tex`

| split | N≥2 | j_true in list | j_true null | null rate |
|---|---|---|---|---|
| FILM seen | 92 | 73 | 19 | 0.207 |
| FILM unseen | 65 | 63 | 2 | 0.031 |
| CAPEAM seen | 84 | 42 | 42 | **0.500** |
| CAPEAM unseen | — | — | — | **missing logs** |

CAPEAM seen TRACE wrong=0.512는 identity가 반반 틀려서가 아니다. N≥2의 **절반**이 reference가 candidate에 없고 (`j_true=null` → truth=`none`), TRACE가 SELECT해서 wrong으로 집계된다. in_list 42건만 보면 Jev/OpenAI CR **0**, wrong **0.167**. Oracle TRACE도 전체 wrong이 0.512.

FILM unseen Jev CR 0.246 vs OpenAI 0.185: 갭은 PickupObject 13건 (0.923 vs 0.615). Put 13 + ToggleOff 32는 둘 다 CR 0.

FILM seen OpenAI CR이 Jev보다 높은 쪽은 Put 39건 (0.462 vs 0.051). Pickup 14건은 반대 (OpenAI 0, Jev 0.857).

---

## 8. Lab jsonl episode list (2026-11-05)

SSH 회복 후 `thor_port_v2` jsonl 파싱. 표 전체: `episode_v2.md`.

**FILM seen OpenAI +4** = `FloorPlan422` `HandTowel→SinkBasin` idx **108, 109, 110, 111**. Host Put 실패, OpenAI는 `…|SinkBasin` id로 SELECT (pmax=0.99). Jev는 이 4개를 못 살림. lost=0.

나머지 OpenAI SELECT 7건(FILM seen)은 host와 같은 id, 이미 성공: FloorPlan7 Fridge idx 16, FloorPlan406 ToiletPaper pickup 112–114, FloorPlan4 Fridge 124–126.

CAPEAM seen OpenAI SELECT 11건은 host가 FloorPlan422를 이미 맞춰서 SR이 안 오른다. FloorPlan406 pickup 3건은 host와 ToiletPaper id가 다르지만 둘 다 성공.

**Jev SELECT:** FILM seen idx 128 FloorPlan406 Put (ok), CAPEAM unseen idx 36 FloorPlan10 Put (ok). SR 변화 없음. 그 외 SELECT 0.

**FILM vs CAPEAM pickup ID:** 3개, 전부 FloorPlan406 idx 112–114 ToiletPaper. 둘 다 성공. SR 갭 4는 이 pickup이 아니라 위의 FloorPlan422 Put 4건.

**Seen host fail:** FILM 16 = FloorPlan224 Box→Dresser 3 + FloorPlan20 Knife→SinkBasin 3 + FloorPlan422 HandTowel→SinkBasin 4 + FloorPlan409 SprayBottle→GarbageCan 3 + FloorPlan224 Newspaper→Sofa 3. CAPEAM 12 = 같은 목록에서 FloorPlan422 4건만 빠짐.

**Unseen FILM host=Jev=OpenAI fail idx:** 21–23, 54–56, 87–89 (FloorPlan219 Safe). Recheck와 동일.

CAPEAM unseen Metric-FF `costs.jsonl`은 랩 `trace-alfred/results`에도 없음.

---

## 한 줄

v2는 unseen에서 host를 깎지 않는다 (0.910 유지). OpenAI FILM seen +4는 FloorPlan422 SinkBasin SELECT 4건. 남은 unseen 실패 9개는 FloorPlan219 Safe Put. Replay wrong 0.51은 `j_true=null` 50%.
