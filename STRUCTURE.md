# TRACE 구현 구조 · 수행 task · 결과 (refine용)

이 문서는 현재 `jw/`에 실제로 들어 있는 방법, 호출 흐름, 실험 트랙, 숫자를 한 곳에 모은다.
논문 주장과 포팅 실험이 섞이지 않게 **무엇이 official이고 무엇이 아닌가**를 먼저 적었다.
수동으로 고칠 때 이 파일을 원본으로 쓰면 된다. 숫자 출처는 각 표 아래에 있다.

관련: `METHOD.md`(수식·규칙), `README.md`(실행 커맨드), `results/`(원 숫자 JSON).

---

## 0. 한 줄 요약

**주 수행 task는 내비게이션/언어/계획기가 아니다.** 같은 클래스 후보 마스크가 둘 이상일 때, 기억된(또는 task) 참조 인스턴스에 대해 **SELECT i 또는 RETURN(호스트 마스크 유지)** 를 내는 결정층이다.

결정식:

```
R(i) = Σ_j C(i,j) p_identity[j]
SELECT argmin_i R(i)  if min R < c_defer
RETURN                otherwise
```

- `c_defer = 4` (`trace/config.py`)
- `p_identity` 길이 = N+1 (마지막은 “후보에 참조 없음”)
- `C(i,j) = total(i,j) − base(j)` ≥ 0, 맞춘 마스크는 0
- Jev와 OpenAI Decisions는 **같은 질문, 별도 실험**. 키는 로그에 안 남김 (`trace/redact.py`)

FILM / CAPEAM은 호스트다. TRACE는 마스크 직전에만 끼워진다.

---

## 1. 무엇이 구현되어 있고, 무엇이 아닌가

| 층 | 구현 | 비고 |
|---|---|---|
| Relation R1–R5 | `trace/relation.py` | 행동 기록만. 시뮬레이터 ID 안 봄 |
| Belief (Jev) | `trace/belief/jev.py` | POST typesafe SystemOne `jev-latest` |
| Belief (OpenAI) | `trace/belief/openai_decisions.py` | POST `/v1/decisions` `gpt-6-luna` |
| Belief (DINOv2) | `trace/belief/appearance.py` | replay 비교용. T=0.05 |
| Live critic J* | `trace/consequence/{world,planner,critic}.py` | interact 1, nav 10, UNREACHABLE 40 |
| Offline critic Metric-FF | `trace/consequence/ff.py` | ryankwon `costs.jsonl` 읽기 |
| Gate TRACE / threshold / force_best | `trace/gate.py` | TRACE만 `p_identity`를 C 열에 곱함 |
| Host hook | `trace/selector.py` `choose()` | 네 단계를 유일하게 묶음 |
| Official 10-fail SR 표 | `experiments/protocol.py` | **재실행 아님**. `table_sr.tex` 숫자 |
| Mask-log replay Table-2 | `experiments/replay.py` | 시뮬레이터 없음 |
| Ported THOR 5 pnp | `experiments/thor_rollout.py` | **official ALFRED 아님** |
| CAPEAM Mask R-CNN host | `experiments/capeam_mrcnn.py` | GitHub `weight_maskrcnn.pt` |

**아직 없는 것**

- Official FILM/CAPEAM 820ep closed-loop (2019 THOR 2.1 + torch 1.6 + NVIDIA X). RTX 5060 Ti Blackwell에서 막힘.
- CAPEAM `valid_unseen` Metric-FF `costs.jsonl` / beliefs → Table-2 CR 없음 (lab 재확인 2026-11-05).
- CAPEAM unseen THOR-jsonl put-success replay는 있음: `experiments/thor_jsonl.py`, `results/capeam_unseen.md` (SR 0.910, Jev SELECT 1 / recheck 0).
- THOR 5 v2 episode SELECT/fail idx는 `results/episode_v2.md` (lab jsonl 2026-11-05).
- 원본 THOR RGB `results/rollouts/` → OpenAI `--rgb` replay 미실행. 현재 OpenAI replay는 Jev bbox overlay 재구성.

---

## 2. 방법 흐름 (런타임)

```
Host candidates (N same-class masks + scores)
        │
        ▼
  Relation  ── NONE ──────────────────────────────► RETURN (호스트 마스크)
  R1–R5     ── SAME / DIFFERENT
        │
        ├──────────────────────┐
        ▼                      ▼
  Belief p[N+1]           Critic C[N, N+1]
  Jev | OpenAI | DINOv2   live J* 또는 Metric-FF
        │                      │
        └──────────┬───────────┘
                   ▼
              Gate TRACE
         min R(i) < 4 ? SELECT i : RETURN
                   │
                   ▼
         SELECT → candidates[i]
         RETURN / API error → host argmax score
```

`selector.choose()`만 이 그래프를 탄다. Relation은 비용을 모르고, Belief는 planner를 모르고, Critic은 HTTP를 모른다.

Ported THOR의 **첫 pickup / put** 은 에피소드 기억이 비어 있어 논문 relation은 NONE이다. 구현은 lookalike가 2개 이상이고 expert glance crop이 있으면 `relation_override=SAME` (`task_instance`)을 넣는다. **논문 method에 넣을지 여기가 refine 포인트.**

### 2.1 Relation (기억, ID 아님)

실패한 상호작용은 기억에 안 남는다. Put은 object 쪽과 receptacle 쪽을 따로 `record` 해야 한다.

| 규칙 | 앞선 성공 | 이번 행동 | mode |
|---|---|---|---|
| R1 | Open receptacle R | Put/Close R | SAME |
| R2 | Put into R | Close/Toggle/Open R | SAME |
| R3 | ToggleOn X | ToggleOff X | SAME |
| R4 | Put into Microwave/Fridge/Sink | Pickup 같은 클래스 | SAME |
| R5 | 그 외 Put, 또는 pick-two 두 번째 | Pickup 같은 클래스 | DIFFERENT |
| else | — | — | NONE → belief/critic skip |

### 2.2 Belief

같은 질문: 선택지 A, B, … + `none`. 이미지 있으면 참조 crop, +50% context crop, 라벨된 현재 프레임.

- Jev: multipart + `TypeSafe-Danger-Zone-Images`. 한 변 280px 미만 / 총화소 78,400 미만이면 API 거절 → 보내기 전 리사이즈.
- OpenAI: `input_image` data URL. 답의 `confidence`는 게이트가 무시. `probabilities`만 씀.
- 분포 길이 ≠ N+1 이면 `choose()`가 ValueError.

### 2.3 Critic

```
total(i,j) = cost(a_i) + J*(T(s, a_i) | z=j)
base(j)    = min( J*(s|z=j), min_i total(i,j) )
C(i,j)     = total − base
```

`z=j` = “후보 j가 참조”. 마지막 열 = 참조가 후보에 없음. 비유한 cell은 1e6으로 바꿔 `0×inf` NaN을 막는다.

### 2.4 Gate vs 다른 규칙

| 규칙 | 보는 분포 | 동작 |
|---|---|---|
| `trace` | `p_identity` × C | min R < 4 이면 SELECT |
| `threshold` | `same_to_pick` (DIFFERENT면 뒤집힘) | max ≥ τ=0.75 이고 ≥ none |
| `force_best` | `same_to_pick` | 항상 SELECT argmax |
| `host_score` | detector score | TRACE 통합 규칙 아님. selector가 거부 |

DIFFERENT에서 “이 마스크가 참조다”와 “이 마스크를 집는 게 맞다”는 반대다. TRACE는 전자만 쓴다.

### 2.5 유닛테스트 세계 (같은 p=[0.6, 0.4, 0], c_defer=4)

`experiments/offline.py`

| World | TRACE | threshold 0.75 | host score |
|---|---|---|---|
| same-room drawers (extra 2) | SELECT 0, R=0.8 | RETURN (0.6 < 0.75) | 틀린 마스크 |
| far fridge (extra 22 vs 2) | SELECT kitchen, R=0.8 vs 13.2 | RETURN | 틀린 마스크 |
| both fridges far (extra 12) | RETURN, R=4.8 and 7.2 | RETURN | 틀린 마스크 |

평균 extra: TRACE 2.4, τ=0.55 → 4.4, τ=0.75 → 4.0, host 6.6.

---

## 3. 코드 맵

```
jw/
  trace/
    selector.py          MaskRequest + TraceSelector.choose
    relation.py          EpisodeMemory, infer_relation
    gate.py              decide, expected_regret, decision_utility
    config.py            DEFERRAL_COST=4
    redact.py            키/에러 문자열 마스킹, .env load
    types.py             Candidate, Belief, Observation, Relation
    belief/
      jev.py / openai_decisions.py / appearance.py
      images.py          crop 리사이즈, overlay
      parse.py           choice → p
      http.py            POST (키 로그 금지)
    consequence/
      world.py           Obj, World
      planner.py         J* uniform
      critic.py          live C
      ff.py              Metric-FF jsonl → C
  experiments/
    live.py, run_source.py, compare_sources.py   합성 identity
    offline.py, scenes.py, scene_costs.py        기호 critic
    replay.py, mask_views.py, compute_openai_beliefs.py
    protocol.py          10-fail SR
    thor_rollout.py      THOR 5 pnp + TRACE
    capeam_mrcnn.py      CAPEAM GitHub weight host
    run_thor_matrix.py   film|capeam × seen|unseen × sources
    thor_jsonl.py        CAPEAM unseen closed-loop jsonl replay
  tests/                 ~40 (lab에서 skip 1)
  results/
    log_replay.json, replay_compare.md
    source_compare.md
    thor_port/matrix_summary.md
    REPORT.md
```

Lab: `/data/jw_workspace/trace-alfred` (코드), `/data/jw_workspace/ALFRED/data` (json_2.1.0), `/data/jw_workspace/CAPEAM/weight_maskrcnn.pt`.
GitHub: `ryankwondev/ICEIC-2027-ALFRED-Decision` (kcasl).

---

## 4. 실험 트랙 세 개

서로 **다른 질문**이다. SR과 CR을 한 표에 섞지 말 것.

### Track A — 합성 live identity

**Task:** 라벨을 아는 그림에서 Jev vs OpenAI가 같은 p를 내는지, TRACE 행동이 같은지.

**구현:** `python -m experiments.run_source jev` / `openai_decisions` → `compare_sources`.

**결과** (`results/source_compare.md`)

| case | truth | Jev p | OpenAI p | L1 | cheap / expensive / both-far TRACE |
|---|---|---|---|---|---|
| visible_match | 0 | [0.98, 0.00, 0.02] | [0.98, 0.01, 0.01] | 0.02 | SELECT 0 / SELECT 0 / SELECT 0 |
| reference_absent | 2 | [0.11, 0.01, 0.88] | [0.38, 0.06, 0.56] | 0.64 | SELECT 0 / SELECT 0 / RETURN |
| lookalike | 0 | [0.74, 0.21, 0.05] | [0.95, 0.03, 0.02] | 0.42 | SELECT 0 / SELECT 0 / SELECT 0 |

- identity accuracy 둘 다 1.00, top-1 agree 1.00
- mean L1 **0.36**
- TRACE action agree 세 world 전부 1.00
- Jev는 absent에서 `none`에 더 많이 (0.88 vs 0.56). OpenAI는 lookalike에 더 peak (0.95 vs 0.74)
- threshold 0.75는 lookalike에서 갈림: Jev RETURN, OpenAI SELECT

### Track B — Official-protocol mask-log replay (시뮬레이터 없음)

**Task:** FILM/CAPEAM 호스트 로그의 N≥2 이벤트에서, 같은 Metric-FF C에 Jev / OpenAI / DINOv2 / oracle belief를 곱해 Table-2 CR을 잰다.

**구현:** `python -m experiments.replay`. OpenAI 이미지는 Jev bbox overlay에서 재구성 (`compute_openai_beliefs`). `--rgb`는 `results/rollouts/`가 있을 때만.

**CR 정의:** TRACE 결정의 실현 extra cost − oracle extra cost. Unrecoverable = unreachable SELECT.

#### B1. Closed-loop SR (jw THOR 5 pnp v2)

ryankwon 10-fail 820ep 대신 이 랩 숫자. 출처: `results/RERUN.md`.

| host | split | n | host | oracle | Jev v2 | OpenAI v2 |
|---|---|---|---|---|---|---|
| FILM | seen | 142 | 0.887 | 0.887 | 0.887 | **0.915** |
| FILM | unseen | 100 | 0.910 | 0.910 | 0.910 | 0.910 |
| CAPEAM | seen | 142 | 0.915 | 0.887 | 0.915 | 0.915 |
| CAPEAM | unseen | 100 | 0.910 | 0.910 | 0.910 | 0.910 |

#### B2. TRACE commitment regret (N≥2)

출처: `results/log_replay.json`, 요약 `results/replay_compare.md`. c_defer=4.

**FILM valid_seen (n=92 events, 820 ep 로그)**

| source\|rule | CR | wrong | return | unrec |
|---|---|---|---|---|
| jev\|trace | **0.152** | 0.261 | 0.054 | 0 |
| openai\|trace | **0.196** | 0.293 | 0.033 | 0 |
| dinov2\|trace | 1.380 | 0.457 | 0.033 | 0 |
| oracle\|trace | 0.130 | 0.250 | 0.054 | 0 |
| jev\|threshold | 3.424 | 0.141 | 0.207 | 0.022 |
| jev\|force_best | 2.707 | 0.272 | 0 | 0.022 |
| jev\|host_score | 5.474 (n=38) | 0.211 | 0 | 0.053 |

**FILM valid_unseen (n=65)**

| source\|rule | CR | wrong | return | unrec |
|---|---|---|---|---|
| jev\|trace | **0.246** | 0.031 | 0.031 | 0 |
| openai\|trace | **0.185** | 0.000 | 0.031 | 0 |
| dinov2\|trace | 0.462 | 0.108 | 0.031 | 0 |
| openai\|force_best | 0.000 | 0.138 | 0 | 0 |
| jev\|host_score | 0.250 (n=52) | 0.019 | 0 | 0 |

**CAPEAM valid_seen (n=84)**

| source\|rule | CR | wrong | return | unrec |
|---|---|---|---|---|
| jev\|trace | **0.012** | 0.512 | 0 | 0 |
| openai\|trace | **0.012** | 0.512 | 0 | 0 |
| dinov2\|trace | 0.405 | 0.536 | 0 | 0 |
| oracle\|trace | 0.000 | 0.512 | 0 | 0 |
| jev\|force_best | 1.179 | 0.619 | 0 | 0 |
| jev\|threshold | 1.833 | 0.274 | 0.345 | 0 |

읽기: CAPEAM seen에서 TRACE CR은 거의 0이지만 wrong_instance 0.51. Extra run (`replay_slices.md`): N≥2 84건 중 **42건이 `j_true=null`**. in_list만 보면 CR 0 / wrong 0.167. Oracle TRACE도 전체 wrong 0.512라서 belief 탓이 아니다.

CAPEAM unseen replay: 로컬 costs/beliefs 없음. 2026-11-02 extra run으로도 확인.

### Track C — Ported THOR 5 closed loop

**Task:** 2019 이름이 없는 THOR 5에서 pick_and_place_simple만 type-mapped pose로 돌리고, 첫 pickup/put에서 identity TRACE를 켠다.

**왜 이 트랙이 있나:** official FILM (torch 1.6, ai2thor 2.1, thor-201909061227, maskrcnn-benchmark) 이 Blackwell sm_120에서 로드가 안 됨.

**구현 한 에피소드** (`thor_rollout.run_episode`)

1. `reset(floor)` → `SetObjectPoses` (objectName을 type으로 매핑, 타입당 첫 미사용 인스턴스)
2. expert `init_action`으로 TeleportFull
3. target class 전부 후보. FILM host score = 1/(agent dist+0.05). CAPEAM이면 Mask R-CNN이 THOR instance box와 IoU≥0.2 overlay score로 덮어씀
4. N>8이면 score 상위 8, expert instance는 강제 포함 (`MAX_CANDIDATES=8`)
5. expert 물체로 glance → reference/context crop. 현재 프레임에 A,B,… overlay
6. `relation_override=SAME` + TRACE. defer/API error → host
7. teleport + PickupObject (forceAction)
8. parent: `PARENT_ALIASES` (CounterTop/Sink/Fridge/CoffeeTable 등) + 없으면 아무 non-pickupable receptacle
9. glance parent → TRACE PutObject. openable이면 Open 후 Put
10. success = pickup_ok ∧ put_ok. jsonl incremental write

**범위:** valid_seen **142**, valid_unseen **100**, pick_and_place_simple only. 7종 ALFRED task 아님. 10-fail protocol 해당 없음 (액션 2개짜리 에피소드).

**CAPEAM weight:** `snumprlab/capeam` `weight_maskrcnn.pt` 171MB, torchvision MaskRCNN-ResNet50-FPN 119-class. load missing=0 unexpected=0 (torch 2.11+cu128 / torchvision 0.26). score cutoff 0.05, IoU 0.2.

표시용 closed-loop SR은 **v2** (`results/RERUN.md`, `thor_port_v2/`). 아래 C 표는 method v1 (expert crop + step-cost C).

#### C 결과 v1 (`results/thor_port/matrix_summary.md`)

Pickup rate **전부 1.000**. 실패는 전부 Put.

| host | split | n | host SR | oracle | Jev TRACE | OpenAI TRACE | API err |
|---|---|---|---|---|---|---|---|
| FILM distance | seen | 142 | 0.887 | 0.887 | **0.915** | 0.887 | Jev 3 |
| FILM distance | unseen | 100 | 0.910 | 0.910 | 0.880 | 0.880 | 0 |
| CAPEAM Mask R-CNN | seen | 142 | **0.915** | 0.887 | **0.915** | 0.887 | Jev 3 |
| CAPEAM Mask R-CNN | unseen | 100 | 0.910 | 0.910 | 0.880 | 0.880 | 0 |

행 단위 (put = SR):

| host | split | source | SR | pickup | put | api_errors |
|---|---|---|---|---|---|---|
| film | seen | host | 0.887 | 1.000 | 0.887 | 0 |
| film | seen | oracle | 0.887 | 1.000 | 0.887 | 0 |
| film | seen | jev | 0.915 | 1.000 | 0.915 | 3 |
| film | seen | openai_decisions | 0.887 | 1.000 | 0.887 | 0 |
| film | unseen | host | 0.910 | 1.000 | 0.910 | 0 |
| film | unseen | oracle | 0.910 | 1.000 | 0.910 | 0 |
| film | unseen | jev | 0.880 | 1.000 | 0.880 | 0 |
| film | unseen | openai_decisions | 0.880 | 1.000 | 0.880 | 0 |
| capeam | seen | host | 0.915 | 1.000 | 0.915 | 0 |
| capeam | seen | oracle | 0.887 | 1.000 | 0.887 | 0 |
| capeam | seen | jev | 0.915 | 1.000 | 0.915 | 3 |
| capeam | seen | openai_decisions | 0.887 | 1.000 | 0.887 | 0 |
| capeam | unseen | host | 0.910 | 1.000 | 0.910 | 0 |
| capeam | unseen | oracle | 0.910 | 1.000 | 0.910 | 0 |
| capeam | unseen | jev | 0.880 | 1.000 | 0.880 | 0 |
| capeam | unseen | openai_decisions | 0.880 | 1.000 | 0.880 | 0 |

읽기:

- CAPEAM seen host vs FILM: pickup ID가 **3개** 다름, SR 0.887 → 0.915. Unseen 호스트는 100/100 동일.
- CAPEAM seen host SR 0.915 > oracle 0.887. Oracle은 expert pose에 가장 가까운 인스턴스. Mask R-CNN이 다른 인스턴스를 고르고 Put이 더 잘 된 케이스가 있다. **parent alias / glance / type-map 점검 포인트.**
- Jev seen +3 SR, API error 3건은 host fallback. Unseen에서 Jev/OpenAI 둘 다 host 0.910 → 0.880 (Put 3건 손해).
- 이 SR을 official 820ep와 비교하면 안 된다.

Lab jsonl: `/data/jw_workspace/trace-alfred/results/thor_port/{film,capeam}/{valid_seen,valid_unseen}/*.jsonl`

---

## 5. 실험이 답하는 질문 vs 안 답하는 질문

| 질문 | 답 | 어디서 |
|---|---|---|
| 같은 p에서 TRACE가 threshold/host보다 extra cost가 낮은가 | 예 (유닛 세계 2.4 vs 4.0–6.6) | offline.py |
| Jev와 OpenAI가 같은 identity 질문에 동의하는가 | top-1 yes, L1 0.36 | live |
| FILM/CAPEAM 로그에서 TRACE CR이 DINOv2보다 낮은가 | 예 (seen 0.15/0.20 vs 1.38) | replay |
| jw THOR 5 closed-loop SR | 표 B1 / `RERUN.md` | v2, unseen 0.910 = 91/100 |
| TRACE를 켠 official 820ep SR은? | **모름** | 2019 stack blocked |
| THOR 5 pnp에서 identity TRACE가 Put을 돕는가 | v1 unseen −3ep; v2 unseen 0. OpenAI FILM seen +4 | RERUN.md |
| CAPEAM Mask R-CNN host가 distance host와 다른가 | seen SR +4 ep (0.887→0.915). pickup-ID 목록은 lab jsonl | Track C |
| CAPEAM seen replay wrong 0.51은 identity 실패인가 | 아님. N≥2의 50%가 j_true=null | replay_slices.md |

---

## 6. 수동 refine 체크리스트

구조 쪽에서 논문/코드가 갈릴 수 있는 지점만.

1. **첫 pickup SAME override** — 논문 relation은 NONE. 포팅에서 identity를 돌리려고 넣음. 유지 / 제거 / “task-instance SAME”을 새 규칙으로 쓸지.
2. **CAPEAM seen CR 0.012 + wrong 0.51** — 비용이 싸서 CR이 낮음. Table-2에 wrong rate를 같이 둘지.
3. **OpenAI replay 이미지** — overlay vs 원본 RGB. 원본이 없으면 논문에 overlay라고 명시할지.
4. **THOR 5 type-map** — 2019 objectName → 타입 첫 인스턴스. 인스턴스 대응이 깨지면 oracle/host 해석이 흔들림.
5. **MAX_CANDIDATES=8** — FloorPlan13 nP=43 critic hang 때문에 넣음. 논문 N과 맞는지.
6. **PARENT_ALIASES + any-receptacle fallback** — CoffeeTable nP=0 때문에 확장. Put 실패/성공이 alias 때문에 바뀌는지.
7. **Oracle < CAPEAM host (seen 0.887 vs 0.915)** — oracle 정의(expert pose vs GT id)를 다시 쓸지.
8. **Jev API error 3** — FloorPlan / N 로그 확인.
9. **Official 820ep** — 다른 머신(2019 Xorg) 없으면 closed-loop SR을 안 쓰는 쪽으로 문장을 고정할지.
10. **c_defer=4, nav=10, unreachable=40** — 논문 하이퍼파라미터와 숫자 일치 확인.

---

## 7. 숫자 파일 위치

| 산출 | 경로 |
|---|---|
| Replay 전체 규칙 표 | `jw/results/log_replay.json` |
| Replay Jev vs OpenAI 요약 | `jw/results/replay_compare.md` |
| Live identity | `jw/results/source_compare.md` |
| THOR matrix | `jw/results/thor_port/matrix_summary.md` + `.json` |
| Official SR 상수 | `jw/experiments/protocol.py` `OFFICIAL_10FAIL` |
| 방법 수식 | `jw/METHOD.md` |
