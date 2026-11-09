# JEV를 기존 파이프라인에 넣는 계획

Oct 26, 2026

JEV는 identity belief source 하나다. P3의 마지막에, 저장된 rollout과 suite 위에서만 호출한다. 인식, 관계 규칙, planner, closed-loop 실행에는 넣지 않는다.

기존 코드의 계약은 `seyong/mma/belief/base.py`의 `BeliefSource.predict(event) -> Belief`다. `probs`는 후보 id와 `"None"` 키의 합이 1인 dict다. `seyong/`은 수정하지 않고, 클라이언트와 어댑터는 `kcasl/`에 둔다. 연결은 `predict`와 같은 시그니처의 함수 하나로 한다.

## 제약

import 경로에 올라가는 코드는 표준 라이브러리와 numpy만 쓴다. DINOv2, CLIP, XGBoost, Fast Downward는 그 환경에서 오프라인으로 돌리고, 결과 파일만 읽는다.

AI2-THOR 2.1.0은 state fork가 없다. counterfactual은 prefix replay다. replay는 `fast_actions`로 돌리고, 렌더링은 그 분기를 실제로 실행하는 시점에만 켠다.

`JEV_API_KEY`와 `VAST_API_KEY`는 환경 변수다. 코드, 로그, 커밋에 넣지 않는다. 이 세션은 샌드박스이므로 키를 읽거나 API를 호출하지 않는다.

## 언제 호출하는가

| 단계 | JEV |
| --- | --- |
| P0 호스트 재현과 interaction 로그 | 호출하지 않음. 프레임과 후보를 남긴다 |
| P1 relation, Coverage, CandRecall, BaseError, OracleGain | 호출하지 않음. Gate A 실패 시 중단 |
| P2 counterfactual과 `C(i, j)` | 호출하지 않음. Gate B |
| P3 belief | appearance, context, classifier 다음. 마지막 |
| P4 offline decision matrix | 저장된 `results/jev/*.jsonl`만 replay |
| P5 closed-loop | Gate C를 통과한 source만. 기본 경로는 저장된 확률 |
| P6 논문 | 표는 `experiments/metrics.py`가 생성 |

Gate 임계값은 pilot 분포로 정한다. Master Plan의 A–D와 같다.

## P0에서 JEV가 받을 로그

매 interaction 직전 JSONL. JEV 입력으로 쓰는 필드는 다음이다.

- episode id, task type, step, 남은 budget β
- 카메라 pose, subgoal (FILM은 pointer, MOCA는 예측 action index)
- host가 낸 action과 target class
- 같은 class 후보 전부: mask, bbox, detector score, 거리
- host가 고른 index, 실행 성공 또는 거부, 거부 사유
- reference 이후 `camMoved`
- RGB. reference context crop, object crop, 현재 프레임의 A/B/C overlay는 이 프레임에서 만든다

GT objectId는 같은 줄에 두되 oracle 전용 필드로 분리한다. `predict`로 넘어가는 객체에는 넣지 않는다. 산출물은 `results/rollouts/{host}/{split}/*.jsonl`과 frames다.

## P3 클라이언트

파일은 `kcasl/belief/jev.py`다. import 시 네트워크를 열지 않는다. `JEV_API_KEY`가 없으면 호출 함수만 실패하고, 나머지 모듈 import는 된다.

요청은 후보마다 보내지 않는다. eligible event당 한 번이다.

```text
POST https://api.typesafe.ai/v1/systemone
Authorization: Bearer $JEV_API_KEY
model: jev-latest
```

`questions.identity`는 `choice`다. criteria 키는 후보 수만큼 `A`, `B`, `C`와, 항상 `none` ("정답 후보가 없음")이다. 이미지는 danger-zone 헤더로 보낸다. 포맷과 assets map은 호출 전에 공식 문서를 다시 읽고, 확인한 헤더 이름만 코드에 적는다.

`state`에 넣는 것:

- reference context crop, reference object crop
- 현재 전체 이미지에 A/B/C mask overlay
- action history, subgoal text

기하 질문(미터 거리, 충돌, 경로)은 넣지 않는다. 거리는 로그에만 있고 `q_i` 학습용이디.

선택 질문 하나 외에, ablation용 `noul`을 같은 요청에 둘 수 있다. "현재 관측만으로 identity를 판단하기 충분한가." 메인 경로는 `choice`만 사용한다.

응답의 choice 확률을 `Belief.probs`로 맞춘다. 후보 id가 키이고 마지막이 `"None"`이다. 합이 1이 아니면 그 사실을 `raw`에 남기고 정규화한 값과 원본을 둘 다 저장한다. confidence, 모델 버전, 호출 시각, latency, 요청에서 키를 지운 payload를 `results/jev/*.jsonl`에 쓴다.

호출 대상은 P1에서 `r_t ≠ NONE`이고 후보가 둘 이상이며 reference가 있는 event뿐이다. 그 전 단계에서 JEV로 `r_t`를 정하지 않는다.

## 기존 인터페이스에 맞추는 자리

`seyong/mma/adapters/film_hook.py`의 `AssocEvent`는 지금 instance id와 선택 결과만 갖고, crop과 pose는 비어 있다. P0 로그가 그 필드를 채운 뒤에 JEV adapter가 로그 한 줄을 `AssocEvent`에 해당하는 입력으로 만든다.

```text
rollout jsonl + frames
        |
        v
kcasl.belief.jev.predict(relation, candidates, ctx) -> Belief
        |
        v
mma.belief.base.Belief 와 같은 probs dict
        |
        v
P4 Gate.decide  (저장된 Belief만 사용)
```

`ConsequenceCritic`과 `Gate`는 JEV를 import하지 않는다. offline replay는 확률 파일만 읽으므로 GPU와 API가 없다.

calibration은 train held-out 장면이거나 valid_seen의 고정 subset에서 temperature를 맞춘다. valid_unseen과 장면을 섞지 않는다. τ·δ를 맞추더라도 P4의 본문 규칙은 tuned threshold와 consequence gate다. fixed τ·δ는 확장 표로 남긴다.

## P4–P5에서 JEV의 위치

본문 비교는 호스트 둘, belief 넷, 규칙 셋 이내다.

| 축 | 본문 | 확장 |
| --- | --- | --- |
| Host | MOCA, FILM | — |
| Belief | host selector, DINOv2 appearance, JEV, GT-ID oracle | context crop, classifier, CLIP, Decisions |
| Rule | host 원본, tuned threshold, consequence gate | fixed τ·δ, force-best, gate − budget, gate on GT state, CAPEAM+gate |

Gate C는 appearance 또는 classifier belief에 GT consequence를 붙여 fixed threshold와 비교한다. GT belief 0/1로는 재지 않는다. C를 통과하기 전에는 JEV를 호출하지 않는다.

P5 closed-loop에서 JEV를 다시 부르지 않는 것이 기본이다. `results/jev`의 확률을 같은 event에 붙인다. 온라인 재호출은 로그에 없는 event가 생겼을 때로 한정하고, 그 전에 호출 수를 보고한다. Return은 호스트의 기존 재관찰로 제어를 넘긴다. `c_return`은 그 rollout에서 측정한다.

## 키를 요청하는 시점

| 키 | 시점 |
| --- | --- |
| `VAST_API_KEY` | P0에서 인스턴스를 만들기 직전 |
| `JEV_API_KEY` | appearance와 classifier 결과가 있고 Gate C를 본 뒤, 첫 호출 직전 |
| OpenAI Decisions | JEV 이미지 결과가 쓸 수 없거나 두 source 비교가 해석에 필요할 때만 |

## 호출 전에 코드에서 확인할 것

- MOCA `models/eval/eval_task.py`의 instance association과 mask를 고르는 줄
- FILM `segmentation_helper.py`의 `segmented_dict`, `sem_exp_thor.py`의 `open_mask`와 subgoal pointer
- `seyong/mma/adapters/film_hook.py`가 이미 거는 `va_interact` 지점. crop과 pose는 아직 이 hook까지 안 넘어온다
- ALFRED `gen/planner/ff_planner_handler.py`, `gen/game_states/task_game_state_full_knowledge.py`에서 임의 state의 PDDL 생성 가능 여부
- AI2-THOR 2.1.0 `GetReachablePositions`와 `gen/graph/graph_obj.py`
- CAPEAM EAM을 FILM 위에서 분리 실행할 수 있는지
- JEV 이미지 입력의 assets map, 포맷, danger-zone 헤더. 코드에 적기 전에 공식 문서를 다시 읽는다
