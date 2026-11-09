# Remember, But Know When to Trust

Consequence-Aware Instance Association for Embodied Instruction Following

Oct 26, 2026

같은 class 후보가 여러 개일 때, 전에 만진 그 물체(SAME) 또는 그것과 다른 물체(DIFF)를 지금 고를지 보류할지를 정한다. identity 확률만으로 commit하지 않는다. 같은 확률이라도 잘못된 서랍을 닫는 비용과 critical object를 다른 방에 넣는 비용은 다르다. planner가 실행 전에 그 regret을 계산하고, 기대 비용이 보류보다 싸고 태스크를 망칠 확률이 낮을 때만 마스크를 고른다.

호스트의 인식과 행동 정책은 다시 학습하지 않는다. 이 모듈은 검출된 후보 중 선택과 보류만 맡는다.

```text
identity belief p(z | I_ref, I_t, H_t)
        |
        v
   r_t, reference          symbolic relation
        |
        v
   alpha_k                 admissibility
        |
        v
   E[C(i)]                 planner regret
        |
        +-- Select i*
        +-- Return
```

## 범위

| 안 | 밖 |
| --- | --- |
| 검출된 후보 중 선택 또는 보류 | 새 segmentation, detection, navigation, task plan 생성 |
| SAME / DIFF / NONE과 기대 regret | clean·hot 추론, Slice lineage |
| 성공한 상호작용만 내부 key로 기억 | 추론 시 AI2-THOR object id |

object id는 ALFRED-Commit 라벨과 채점에만 쓴다.

## 세 층

| 층 | 질문 | 답 | 출력 |
| --- | --- | --- | --- |
| 1. Symbolic relation | identity가 지금 필요한가, 방향은 | action chain + subgoal 규칙. API 없음 | `r_t` ∈ {SAME, DIFF, NONE}, reference key |
| 2. Identity belief | 어느 후보가 reference와 같은 물체인가 | Decisions `choice`. source는 교체 가능 | `p = (p_1 … p_N, p_None)`, confidence |
| 3. Consequence gate | i를 골랐는데 실제가 j면 얼마나 비싼가 | symbolic planner | Select i* 또는 Return |

호스트가 action과 target class를 내고 segmentation이 후보를 주면, `r_t ≠ NONE`이고 후보가 둘 이상일 때만 층 2를 부른다. Select는 마스크를 호스트에 넘기고, 성공하면 memory를 갱신한다. Return은 호스트의 기존 재관찰·탐색으로 제어를 넘긴다. `r_t = NONE`이거나 후보가 하나면 층 2를 생략하고, 층 3은 geometric 실패 확률과 실패 budget만 본다.

### 결정

```text
alpha_k = p_k            if SAME
        = 1 - p_k        if DIFF
        = 1              if NONE

E[C(i)] = sum_j p_j C(i, j) + (1 - q_i) C_fail
C(i, j) = J*(T(s, a_i) | z = j) - min_{i'} J*(T(s, a_i') | z = j)

Select i* iff E[C(i*)] < c_return  and  P_irr(i*) <= eps
i* = argmin_i E[C(i)]
```

`q_i`는 마스크 신뢰도, 거리, receptacle 상태로 학습한 geometric 성공 확률이다. identity 확률과 분리한다. `c_return`은 보류한 뒤 호스트가 같은 interaction을 다시 제안하기까지 걸린 step의 기댓값이다. `J`는 PDDL step에 simulator 최단 navigation을 더한 remaining cost다. 올바른 결과에도 음수가 나오는 `V(s') - V(s)`는 쓰지 않는다.

`P_irr(i)`는 unreachable이거나 실패 budget `β`가 0이 되는 확률이다. natural ALFRED에서 grounding 오류만으로는 거의 0이고, budget 소진에서 커진다. `β`는 state다. 실패할 때마다 1 줄고, 0이면 에피소드가 끝난다.

테스트 시점의 `J`는 GT metadata가 아니라 호스트의 symbolic state에서 계산한다. critic-on-GT는 oracle로만 두고 격차를 보고한다.

## 모듈 1 — 관계

관계를 확정할 수 없으면 `NONE`이고 belief를 부르지 않는다. memory는 성공한 상호작용만 올리고, 에피소드가 시작되면 비운다. 같은 class의 다른 인스턴스는 다른 key다. 현재 subgoal과 무관한 reference는 쓰지 않는다. key는 내부 참조다.

| 이전 | 현재 | r_t | reference |
| --- | --- | --- | --- |
| Open(R) 성공 | Put 또는 Close | SAME | R |
| Put(o, R) 성공 | Close / ToggleOn / Open | SAME | R |
| Put 이후 Open(R) | Pickup | SAME | o |
| ToggleOn(X) | ToggleOff | SAME | X |
| Heat / Cool / Clean 완료 | Pickup | SAME | 처리한 o |
| Pick Two 첫 subgoal 완료 | Pickup | DIFF | 첫 o |
| 그 외 | — | NONE | 호출 안 함 |

MOCA 입력은 예측 action 시퀀스와 class다. FILM 입력은 subgoal program과 action pointer다.

## 모듈 2 — belief

한 번의 `choice`. 선택지는 후보와 `None of the above`다. shared input은 reference context crop, reference object crop, 현재 이미지의 A/B/C mask overlay, action history, subgoal 문장이다. 후보 crop을 따로 보내지 않는다.

호출 조건은 `r_t ≠ NONE`, 후보 ≥ 2, reference 존재다. 불확실성 gate는 ablation이다. payload와 반환 확률은 저장한다. decision-rule 실험은 API를 다시 부르지 않고 replay한다.

Backend는 Decisions의 `gpt-6-luna`다. Jev는 belief-source 축의 교체 조건이다. 어느 쪽도 제목과 기여에 넣지 않는다.

## 모듈 3 — gate

Return은 새 탐색 정책을 만들지 않는다. 비교는 세 가지다.

| 변형 | 하는 일 |
| --- | --- |
| fallback-to-base | 호스트가 원래 고른 마스크 |
| Return | 호스트의 재관찰·탐색으로 제어를 넘김 |
| force-best | 확률이 가장 높은 후보를 강제 |

## 호스트

| 항목 | MOCA | FILM |
| --- | --- | --- |
| 후보 | Mask R-CNN | `segmented_dict` |
| 삽입 | `eval_task.py`의 Instance Association | `segmentation_helper.py`, `sem_exp_thor.py`의 `open_mask` |
| 관계 입력 | 예측 action + class | subgoal program + pointer |
| planner state | history로 만든 class, receptacle, holding, state flag | semantic map + pointer |
| 기존 selector | fallback과 baseline으로 보존 | 유효한 `open_mask`는 유지 |

adapter는 같은 class 후보를 뽑고, 공통 입력으로 바꾸고, 출력 마스크를 호스트 형식으로 되돌리고, 실행 성공 여부를 회신한다. 층 1–3은 호스트와 분리된 core로 한 번만 둔다.

## 기회가 있는 구간

통계는 expert 재생이 아니라 호스트 rollout에서 낸다.

| 관계 | 상황 | 시각만으로 부족한 이유 |
| --- | --- | --- |
| SAME | 연 서랍에 넣고 닫기 | 서랍 외형이 같고 cue는 주변과 history |
| SAME | 가열·세척 후 다시 집기 | hot·clean은 크롭에 안 보임 |
| DIFF | Pick Two의 두 번째 집기 | 같은 물체를 또 집으면 subgoal이 안 끝남 |
| NONE | 단순 Pick and Place | 아무 인스턴스나 됨 |

기회는 복수 receptacle(drawer, cabinet, countertop, sink), Pick Two, 상태 변화 후 재선택, 실패·side-step 이후 카메라가 움직인 구간이다. microwave와 fridge는 장면당 하나인 경우가 많아 후보가 생기지 않는다.

## ALFRED-Commit

interaction 직전 상태에서, 같은 action·같은 class의 visible 인스턴스만 성공과 실패로 분기한다. 다른 action type은 넣지 않는다. AI2-THOR는 state를 fork하지 않으므로 분기마다 prefix를 다시 재생한다.

범위는 valid_seen 820과 valid_unseen 821이다. train은 분기점 샘플만 쓴다.

한 분기의 필드는 `s_t`, `β_t`, `a_t`, `r_t`, `ref`, `k`, `isRef(k)`, `adm(k)`, `exec(k)`, `C_reg(k)`, `reachable(k)`, `camMoved`다. `adm`은 goal을 만족하면 1이고, class-level goal이면 여러 k가 동시에 1일 수 있다. `camMoved`는 reference 이후 카메라가 움직였는지이며 BaseError 분해의 축이다.

붙이기 전에 보는 네 숫자:

| 숫자 | 정의 |
| --- | --- |
| Coverage | `r_t ≠ NONE`이고 후보 ≥ 2인 interaction 비율 |
| CandRecall | GT reference가 후보 안에 있을 확률 |
| BaseError | GT가 있는데 호스트가 틀린 확률. task × camMoved로 분해 |
| OracleGain | GT-ID로 고른 SR − 호스트 SR |

## 실험

belief를 누가 내는가와, 그 확률로 어떻게 결정하는가를 분리한다. Decisions가 평범해도 gate가 좋으면 주장은 선다. 모든 source에서 gate 효과가 같으면 source에 무관하다는 주장이 선다.

| 축 | 조건 |
| --- | --- |
| A. Belief. gate 고정 | detector confidence, DINO/CLIP crop, context crop, spatial+appearance(FILM), 학습 classifier, Decisions choice, Jev, GT-ID oracle |
| B. Decision. belief 고정 | MOCA 2D 또는 FILM `open_mask`, CAPEAM EAM, fixed τ·δ, tuned threshold, force-best, consequence gate, budget 제거, GT state 위의 gate |

| 지표 축 | 내용 |
| --- | --- |
| Association | 선택 정확도(GT가 후보에 있을 때 / 전체), Identity Switch, Eligible Coverage, Acceptance |
| Belief | ECE, Brier, Risk–Coverage |
| Decision | Commitment Regret, wrong-instance rate, Return rate, Return precision, unrecoverable errors |
| Task | SR, GC, PLWSR, PLWGC. 전체와 개입 에피소드를 분리 |
| Critic | planner의 `C(i, j)`와, 그 분기를 호스트로 이어 돌린 extra cost의 상관 |
| 비용 | mean/p95 latency, calls/episode, tokens/call |

Killer 비교는 `p_A = 0.55 ± 0.03`으로 맞춘 뒤 regret만 바꾼다. Close(drawer)는 commit, 다른 방 receptacle에 Put은 return, Pick Two에서 `β = 1`이면 return이다. 그림의 X는 identity confidence, Y는 regret이다. threshold는 수직선, gate는 iso-risk 곡선이다. 낮은 confidence·낮은 regret에서 gate가 commit하는 점이 있어야 한다.

| 용도 | 데이터 |
| --- | --- |
| q와 classifier 학습 | train 분기점 샘플 |
| calibration | train held-out 장면, 또는 valid_seen 고정 subset |
| 개발 지표 | valid_unseen |
| 최종 SR | test_unseen leaderboard. RGB와 언어만. 7일에 한 번 |

## Go / No-Go

순서는 suite, oracle, belief source다. Decisions와 Jev는 마지막이다. 기준 수치는 pilot 분포로 다시 잡는다.

| 조건 | 측정 | 통과 |
| --- | --- | --- |
| A. 기회가 있는가 | Coverage, CandRecall, BaseError, OracleGain | BaseError가 있고 OracleGain ≥ 1–2%p. Coverage 약 3%에 OracleGain 약 0.2%p면 detection 문제로 NO-GO |
| B. 비용이 갈리는가 | `C(i, j)` 분포 | 분기의 5–10% 이상이 중앙값의 2배 이상 |
| C. Gate headroom | appearance 또는 classifier belief를 고정하고, GT consequence와 fixed threshold를 비교 | Commitment Regret 25% 이상 감소, 또는 wrong-instance가 의미 있게 감소. GT belief 0/1로는 재지 않음 |
| D. 독립성 | confidence × regret 사분면 | 고confidence·고regret과 저confidence·저regret이 둘 다 비어 있지 않음 |

C가 실패하면 Decisions를 붙이기 전에 멈춘다. planner 비용과 실제 extra cost의 상관이 낮으면 navigation 보정과 budget 항을 다시 본다. 그래도 낮으면 비용을 empirical 추정으로 바꾼다. 그때 남는 차이는 실행 전 commit뿐이다.

## 기여

1. Instance selection을 identity belief 아래의 commit 문제로 두고, 틀린 선택의 비용을 counterfactual regret으로 정의한다.
2. 관계 규칙, multimodal identity, planner gate를 MOCA와 FILM adapter로 붙인다. belief source는 교체할 수 있다.
3. 같은 confidence에서 regret에 따라 결정이 갈린다. gate는 threshold, CAPEAM 캐시, error classifier보다 wrong-instance, 불필요한 보류, Commitment Regret을 줄이고 SR을 올린다. ALFRED-Commit suite를 공개한다.

## 위치

| 선행 | 이미 한 것 | 이 모듈 |
| --- | --- | --- |
| MOCA | 2D 거리로 이전 마스크와 연관 | long-gap, context, 확률을 regret으로 소비 |
| CAPEAM | 마스크 재사용, relocation cache, 상태 변화 위치 캐시 | 복수 후보의 probabilistic re-ID와 regret gate. FILM 실험의 baseline |
| RESCUE | alarm 이후 intervention 가치를 rollout으로 평가 | 실행 전 identity commit. 비용은 planner가 online 계산 |
| RoVLaP, Seeing is Believing | VLM 확률을 symbolic belief로 | 확률의 소비 방식과 기존 agent 위의 adapter |
| TAMPURA | 전용 belief-space planner로 비가역 결과 회피 | 호스트 planner는 유지하고 critic만 둠 |
