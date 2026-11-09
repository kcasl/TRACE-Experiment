# TRACE 방법론 구현

TRACE(Task-Regret-Aware Consequence Evaluation for Instance Association)는
같은 클래스의 후보 마스크 중 어느 것을 쓸지, 아니면 호스트 에이전트에게
제어를 돌려줄지를 정한다. 판단 모델이 준 동일성 확률만으로 실행하지 않고,
그 확률을 기호 상태에서 계산한 추가 비용과 곱한 뒤 보류 비용과 비교한다.

이 폴더는 그 결정 경로만 구현한다. FILM, MOCA, AI2-THOR 롤아웃은 포함하지 않는다.
호스트가 마스크를 고르기 직전에 `choose_interaction_mask`를 호출하면 된다.
반환값이 정수면 그 후보의 마스크를 쓰고, `None`이면 호스트가 원래 고른 마스크를 쓴다.

## 문제

이전에 연 서랍을 다시 닫거나, 가열한 사과를 다시 집거나, 이미 집은 사과와
다른 사과를 집어야 하는 순간이 있다. 후보가 둘 이상이면 어느 인스턴스인지가
과업의 성패를 가른다.

같은 확률이라도 틀렸을 때의 복구 비용은 같지 않다. 눈앞의 서랍을 잘못 닫는
것은 한 걸음이고, 다른 방의 냉장고에 물건을 넣는 것은 왕복이다. 확률 임계값
하나는 이 차이를 표현하지 못한다.

## 세 단계

관계는 모델 호출 없이 행동 기록만으로 정한다 (`trace/relation.py`).

| 규칙 | 앞선 성공 | 이번 행동 | 관계 |
|---|---|---|---|
| R1 | 용기 R을 염 | R에 넣기 또는 R을 닫기 | SAME |
| R2 | R에 넣음 | R을 닫기, 켜기, 열기 | SAME |
| R3 | X를 켬 | X를 끔 | SAME |
| R4 | 가열·냉각·세척 용기에 넣음 | 그 물체 클래스 집기 | SAME |
| R5 | 그 외 용기에 넣음, 또는 pick-two의 두 번째 집기 | 같은 클래스 집기 | DIFFERENT |
| 그 외 | | | NONE |

NONE이면 동일성이 마스크를 바꾸지 않는다. 믿음 모델과 비용 계산을 호출하지 않고
바로 제어를 돌려준다. 실패한 상호작용은 기억에 남지 않는다. Open과 Close는
용기 사건으로 기록된다. Put은 물체 쪽과 용기 쪽을 따로 기록해야 한다.

SAME은 기억 속 그 물체, DIFFERENT는 그것이 아닌 다른 인스턴스다.

믿음은 후보가 그 참조 물체일 확률이다 (`trace/belief`). 길이는 후보 수 + 1이고
마지막 칸은 "후보 중에 없다"이다. Jev와 OpenAI Decisions는 같은 질문을 받는다.
선택지는 라벨 A, B, …와 `none`이다. 이미지가 있으면 참조 장면과, 후보가
라벨된 현재 장면을 함께 보낸다.

Jev는 `POST https://api.typesafe.ai/v1/systemone`, 모델 `jev-latest`,
choice 질문이다. 이미지가 있으면 헤더 `TypeSafe-Danger-Zone-Images: true`가
필요하고, 이미지는 한 번만 `assets`에 넣은 뒤 state 안의 placeholder로
참조한다. 한 변이 280px 미만이거나 총 화소가 78,400 미만이면 API가 거절하므로
보내기 전에 그 범위로 맞춘다.

OpenAI Decisions는 `POST https://api.openai.com/v1/decisions`, 모델
`gpt-6-luna`이다. 공유 증거는 `input`의 텍스트와 `input_image`이고,
질문 하나의 `choices`가 후보와 `none`이다. 답의 `probabilities`는
`{value, probability}` 목록이다. `confidence`는 그 분포와 별개의 값이므로
게이트는 분포만 쓴다.

비용은 기호 상태 위의 최단 남은 계획이다 (`trace/consequence`).

```
total(i, j) = cost(a_i) + J*(T(s, a_i) | z = j)
base(j)     = min( J*(s | z = j),  min_i total(i, j) )
C(i, j)     = total(i, j) - base(j)
R(i)        = sum_j C(i, j) p_identity(j)
```

`z = j`는 "후보 j가 참조 물체"라는 가설이고, 마지막 열은 참조 물체가 후보에
없다는 가설이다. `J*`는 상호작용 1, 방 이동 `nav_cost`의 균일 비용 탐색이다.
탐색 한계 안에서는 계획이 없으면 40을 반환해 기댓값이 무한대가 되지 않게 한다.
행동 전제조건이 깨지면 상태는 그대로 두고 실패 비용 1을 `J*(s)`에 더한다.
`C`는 0 이상이다. 맞춘 마스크의 추가 비용은 0이다.

게이트 (`trace/gate.py`)는 `min R(i) < c_defer`일 때만 그 마스크를 고른다.
같으면 보류한다. 보류는 다시 볼 기회를 남기고, 같은 기댓값으로 확정할 이유는
없기 때문이다. 기본 보류 비용은 4걸음이다.

임계값 규칙은 다른 분포를 본다. DIFFERENT에서는 "이 후보가 참조 물체일 확률"과
"이 후보를 집는 것이 맞을 확률"이 반대다. 임계값은 후자(`same_to_pick`)의
최댓값이 `tau` 이상이고 `none` 확률 이상일 때만 고른다. TRACE의 기댓값은
반드시 전자, 즉 `C`의 열과 같은 순서인 `p_identity`로 계산한다. 순서를 바꾸면
피해야 할 물체를 집는다.

들고 있는 물체와 용기 안에 있는 물체는 그 방 정보를 따라간다. 넣어 둔 물체의
방이 처음에 본 방으로 남아 있으면, 계획기는 다른 방에서 그 물체를 집어 올리는
경로를 만들지 못하고 복구 비용을 과대평가한다.

## 기호 계획이 내는 비용

단위 테스트가 아래 값을 고정한다. 방 이동 비용은 10이다.

같은 부엌의 열린 서랍 둘에 사과를 넣는 경우, 틀린 서랍의 추가 비용은 2다.
집어 다시 올바른 서랍에 넣는 두 걸음이다. 틀린 서랍을 닫는 추가 비용은 1이다.

참조 냉장고가 부엌에 있고 다른 후보는 차고에 있으면 추가 비용은 비대칭이다.
차고에 넣는 실수는 22다. 가지 않아도 되는 왕복이다. 참조 냉장고가 이미 차고에
있으면 그 이동은 맞는 행동도 치르므로, 부엌 냉장고에 잘못 넣는 추가 비용은
2뿐이다. 그래서 믿음이 `[0.6, 0.4, 0]`이면 부엌 마스크의 기대 regret는 0.8,
차고 마스크는 13.2다. 보류 비용 4에서 TRACE는 부엌 마스크를 고르고 차고
마스크는 고르지 않는다. 확신도 0.6은 임계값 0.75보다 낮아서, 임계값 규칙은
둘 다 거절한다.

두 냉장고가 모두 다른 방에 있으면 어느 쪽을 잘못 골라도 추가 비용이 12다.
같은 믿음에서 기대 regret는 4.8과 7.2이고, 둘 다 4 이상이므로 TRACE는 보류한다.

## 같은 믿음, 다른 비용

믿음 `[0.6, 0.4, 0]`, 보류 비용 4, 두 세계를 같은 가중으로 평균한다.
싼 세계는 실수 비용 2, 비싼 세계는 20이다. 검출기 점수는 `[0.4, 0.9]`라서
호스트는 믿음이 낮은 후보를 고른다. `python -m experiments.offline`이
`results/offline_rules.json`에 같은 표를 쓴다.

| 규칙 | 싼 세계 | 비싼 세계 | 평균 추가 비용 |
|---|---|---|---|
| TRACE | 선택, 0.8 | 보류, 4 | 2.4 |
| 임계값 0.55 | 선택, 0.8 | 선택, 8 | 4.4 |
| 임계값 0.75 | 보류, 4 | 보류, 4 | 4.0 |
| 호스트 점수 | 틀린 후보, 1.2 | 틀린 후보, 12 | 6.6 |

낮은 임계값은 비싼 실수까지 실행한다. 높은 임계값은 싼 실수도 보류한다.
TRACE는 기대 추가 비용이 보류보다 쌀 때만 실행한다.

## Jev와 OpenAI Decisions

합성 장면 세 장을 두 모델에 같은 질문으로 보냈다. 정답은 그림을 만들 때
정해 두었다. 후보 0이 흰 점이 있는 빨간 원이고, 없는 장면은 `none`이 정답이다.
결과는 `results/live_identity.json`이다. 세 장면 모두 두 모델의 최댓값이
정답 칸에 있었다.

| 장면 | Jev `p` | OpenAI `p` |
|---|---|---|
| 참조가 보임 | 0.96, 0.00, 0.04 | 0.98, 0.01, 0.01 |
| 참조가 없음 | 0.12, 0.01, 0.87 | 0.38, 0.06, 0.56 |
| 비슷한 빨간 원 | 0.77, 0.18, 0.05 | 0.95, 0.03, 0.02 |

보이는 장면과 비슷한 장면에서 두 모델의 기대 regret는 0.04에서 0.46 사이다.
보류 비용 4보다 작아서 TRACE와 임계값 0.75가 모두 후보 0을 고른다.
Jev는 비슷한 장면에서 확신 필드가 0.66이지만, 게이트는 그 필드를 쓰지 않고
확률 0.77을 쓴다.

참조가 없을 때 임계값 0.75는 보류한다. 어느 후보도 0.75를 넘지 않기 때문이다.
TRACE는 후보 0을 고른다. 참조가 화면 밖에 있다는 가설에서 가까운 용기에 넣는
추가 비용이 2이고, 2는 보류 비용 4보다 싸기 때문이다. 먼 용기에 가는 이동은
맞는 행동도 치르므로 regret에 다시 더해지지 않는다. 이 차이는 확률 임계값으로는
보이지 않는다.

## 코드가 나뉘는 경계

`trace/relation.py`는 믿음과 비용을 모른다. `trace/belief`는 기호 상태를
모른다. `trace/consequence`는 HTTP를 모른다. `trace/gate.py`는 비용 행렬과
확률 벡터만 받는다. `trace/selector.py`가 네 단계를 한 번 호출로 묶는다.
닫힌 루프 실험과 오프라인 재생이 같은 `decide`를 써야 규칙 비교가 어긋나지
않는다.

키는 `trace/redact.py`가 예외 문자열과 로그에서 지운다. 요청 본문에는 키를
넣지 않는다. 테스트 `tests/test_secrets.py`는 소스 트리에 키 접두사가
들어 있으면 실패한다.

## FILM / CAPEAM 로그 재생

2019 THOR를 이 기계에서 다시 돌리지 않는다. 대신 호스트가 이미 남긴
마스크 이벤트를 쓴다.

- 공식 평가: 10 counted API failure, 1000 steps (`experiments/protocol.py`).
  FILM valid_seen host SR 0.232 / GC 0.364 (n=820), unseen 0.205 / 0.334 (n=821).
  CAPEAM valid_seen host SR 0.443 / GC 0.555 (n=820). `episodes.jsonl`이 있으면
  같은 프로토콜로 다시 계산하고, 없으면 `table_sr.tex` 수치를 적는다.
- 비용: Metric-FF가 남긴 `costs.jsonl`의 C(i,j). `trace/consequence/ff.py`가
  그 행렬을 읽고, Metric-FF 바이너리가 있으면 `live_metric_ff()`로 같은 critic을
  연다.
- 믿음: 같은 이벤트의 Jev, OpenAI Decisions, DINOv2 appearance, GT-ID oracle.
  호스트 RGB 프레임은 gitignore라 이 기계에 없다. Jev 요청 로그의 후보 bbox와
  A/B 오버레이로 참조 크롭·문맥 크롭·라벨 프레임 PNG를 다시 그린 뒤
  (`experiments/mask_views.py`) OpenAI Decisions에 넣는다.
  원본 THOR RGB가 있으면 `python -m experiments.compute_openai_beliefs --rgb`가
  `frames/*.png`와 `masks/*.npz`로 크롭을 자른다. 결과는 `openai_decisions_rgb.jsonl`.
- 게이트: 그 C와 p를 TRACE / threshold / force_best / host_score에 넣고
  Table 2 형식의 commitment regret, wrong-instance, return, unrecoverable을
  낸다. `python -m experiments.replay`. N≥2만 센다.

닫힌 루프 FILM 820ep 재실행은 2019 NVIDIA X가 있는 호스트에서만 된다.

지금 `J*` 단위 테스트 플래너는 ALFRED 전체 PDDL이 아니라 집기·놓기·열기·닫기와
방 이동만 있는 기호 상태의 최단 경로다. 토글, 가열, 슬라이스는 그 플래너에는
아직 행동이 아니다. 로그 재생의 C는 FF가 계산한 전체 PDDL 비용이다.

## FILM 통합 보류 (서버 하드웨어)

TRACE 결정 계층은 SSH 서버(`robust_lab@10.201.11.122`,
`/data/jw_workspace/trace-alfred`)에서 전부 돈다. 21개 테스트 통과,
Jev·OpenAI Decisions 실시간 호출 모두 성공, `results/REPORT.md` 생성.

그러나 FILM 전체 임베디드 롤아웃은 이 서버에서 돌릴 수 없다. GPU가
RTX 5060 Ti 2장(Blackwell, sm_120), 드라이버 595.99.02, Ubuntu 24.04다.
FILM은 torch 1.6.0 + CUDA 10.2 + ai2thor 2.1.0에 고정돼 있는데, CUDA 10.2는
Blackwell을 지원하지 않고 torch 1.6에는 sm_120 커널이 없다. FILM 인식부의
`maskrcnn-benchmark`도 CUDA 12.x 툴체인에서 빌드되지 않는다.

그래서 TRACE 결정 계층을 산출물로 확정하고, FILM 임베디드 통합은 호환 하드웨어
(또는 torch 2.x + CUDA 12.8로의 스택 현대화)로 미룬다. 통합 시 바꿀 경계는
위의 호스트 어댑터(`trace/selector.py`의 `choose_interaction_mask`)뿐이다.
