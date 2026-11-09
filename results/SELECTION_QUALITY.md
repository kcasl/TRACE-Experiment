# Selection quality (v1)

Instance 선택 품질만. Episode SR/GC는 [BENCH.md](BENCH.md)에 있다. 이 파일의 숫자를 bench와 한 표에 넣지 말 것.

thor_port_v2는 여기 없음.

---

## Metric

| 이름 | 정의 |
|---|---|
| CR (commitment regret) | TRACE/rule 결정의 실현 extra cost − oracle extra cost. 단위: extra steps (Metric-FF `C`) |
| wrong_instance_rate | SELECT한 mask가 true instance가 아닌 비율 |
| return_rate | RETURN (host에 제어를 넘김) 비율 |
| unrecoverable | unreachable SELECT 비율 |
| identity accuracy | 합성 그림에서 top-1 == truth |
| L1 | Jev `p_identity`와 OpenAI `p_identity`의 L1 |

`c_defer = 4`. Replay는 **N ≥ 2** mask event만. Simulator를 안 켠다.

`p_identity[j]` = P(candidate j가 reference). 마지막 칸 = reference 없음. Pick probability와 섞으면 DIFFERENT가 뒤집힌다.

---

## 1. Mask-log replay (Metric-FF C × logged belief)

출처: `log_replay.json`, 요약 `replay_compare.md`.
Critic: precomputed Metric-FF. OpenAI image는 Jev bbox overlay 재구성 (원본 THOR RGB 없음).

Host log episode 수: FILM seen 820 / unseen 821 / CAPEAM seen 820. 아래 **n은 N≥2 event**.
CAPEAM unseen: local `costs.jsonl` 없음 → replay 없음.

### 1.1 TRACE only (Jev vs OpenAI vs DINOv2)

| split | n | source | CR | wrong | return | unrec |
|---|---|---|---|---|---|---|
| FILM seen | 92 | jev | 0.152 | 0.261 | 0.054 | 0 |
| FILM seen | 92 | openai | 0.196 | 0.293 | 0.033 | 0 |
| FILM seen | 92 | dinov2 | 1.380 | 0.457 | 0.033 | 0 |
| FILM seen | 92 | oracle | 0.130 | 0.250 | 0.054 | 0 |
| FILM unseen | 65 | jev | 0.246 | 0.031 | 0.031 | 0 |
| FILM unseen | 65 | openai | 0.185 | 0.000 | 0.031 | 0 |
| FILM unseen | 65 | dinov2 | 0.462 | 0.108 | 0.031 | 0 |
| FILM unseen | 65 | oracle | 0.123 | 0.062 | 0.031 | 0 |
| CAPEAM seen | 84 | jev | 0.012 | 0.512 | 0.000 | 0 |
| CAPEAM seen | 84 | openai | 0.012 | 0.512 | 0.000 | 0 |
| CAPEAM seen | 84 | dinov2 | 0.405 | 0.536 | 0.000 | 0 |
| CAPEAM seen | 84 | oracle | 0.000 | 0.512 | 0.000 | 0 |

읽기: Jev/OpenAI TRACE CR이 DINOv2보다 낮다. CAPEAM seen CR ≈ 0이지만 wrong 0.51 — 아래 1.5에서 `j_true=null` 50%가 원인으로 확인.

### 1.2 FILM seen n=92 (host_score만 n=38)

| source \| rule | n | CR | wrong | return | unrec |
|---|---|---|---|---|---|
| jev \| trace | 92 | 0.152 | 0.261 | 0.054 | 0 |
| jev \| threshold | 92 | 3.424 | 0.141 | 0.207 | 0.022 |
| jev \| force_best | 92 | 2.707 | 0.272 | 0 | 0.022 |
| jev \| host_score | 38 | 5.474 | 0.211 | 0 | 0.053 |
| openai \| trace | 92 | 0.196 | 0.293 | 0.033 | 0 |
| openai \| threshold | 92 | 3.598 | 0.120 | 0.293 | 0.022 |
| openai \| force_best | 92 | 2.630 | 0.261 | 0 | 0.022 |
| openai \| host_score | 38 | 5.474 | 0.211 | 0 | 0.053 |
| dinov2 \| trace | 92 | 1.380 | 0.457 | 0.033 | 0 |
| dinov2 \| threshold | 92 | 5.685 | 0.217 | 0.554 | 0.022 |
| dinov2 \| force_best | 92 | 5.435 | 0.587 | 0 | 0.022 |
| dinov2 \| host_score | 38 | 5.474 | 0.211 | 0 | 0.053 |
| oracle \| trace | 92 | 0.130 | 0.250 | 0.054 | 0 |
| oracle \| threshold | 92 | 3.022 | 0.022 | 0.174 | 0.022 |
| oracle \| force_best | 92 | 2.609 | 0.163 | 0 | 0.022 |
| oracle \| host_score | 38 | 5.474 | 0.211 | 0 | 0.053 |

### 1.3 FILM unseen n=65 (host_score만 n=52)

| source \| rule | n | CR | wrong | return | unrec |
|---|---|---|---|---|---|
| jev \| trace | 65 | 0.246 | 0.031 | 0.031 | 0 |
| jev \| threshold | 65 | 0.431 | 0.000 | 0.108 | 0 |
| jev \| force_best | 65 | 0.431 | 0.046 | 0 | 0 |
| jev \| host_score | 52 | 0.250 | 0.019 | 0 | 0 |
| openai \| trace | 65 | 0.185 | 0.000 | 0.031 | 0 |
| openai \| threshold | 65 | 0.492 | 0.077 | 0.123 | 0 |
| openai \| force_best | 65 | 0.000 | 0.138 | 0 | 0 |
| openai \| host_score | 52 | 0.250 | 0.019 | 0 | 0 |
| dinov2 \| trace | 65 | 0.462 | 0.108 | 0.031 | 0 |
| dinov2 \| threshold | 65 | 1.723 | 0.000 | 0.431 | 0 |
| dinov2 \| force_best | 65 | 2.723 | 0.292 | 0 | 0 |
| dinov2 \| host_score | 52 | 0.250 | 0.019 | 0 | 0 |
| oracle \| trace | 65 | 0.123 | 0.062 | 0.031 | 0 |
| oracle \| threshold | 65 | 0.062 | 0.000 | 0 | 0 |
| oracle \| force_best | 65 | 0.062 | 0.000 | 0 | 0 |
| oracle \| host_score | 52 | 0.250 | 0.019 | 0 | 0 |

### 1.4 CAPEAM seen n=84 (host_score만 n=60)

| source \| rule | n | CR | wrong | return | unrec |
|---|---|---|---|---|---|
| jev \| trace | 84 | 0.012 | 0.512 | 0 | 0 |
| jev \| threshold | 84 | 1.833 | 0.274 | 0.345 | 0 |
| jev \| force_best | 84 | 1.179 | 0.619 | 0 | 0 |
| jev \| host_score | 60 | 1.100 | 0.367 | 0 | 0 |
| openai \| trace | 84 | 0.012 | 0.512 | 0 | 0 |
| openai \| threshold | 84 | 1.548 | 0.310 | 0.238 | 0 |
| openai \| force_best | 84 | 1.119 | 0.548 | 0 | 0 |
| openai \| host_score | 60 | 1.100 | 0.367 | 0 | 0 |
| dinov2 \| trace | 84 | 0.405 | 0.536 | 0 | 0 |
| dinov2 \| threshold | 84 | 2.119 | 0.333 | 0.345 | 0 |
| dinov2 \| force_best | 84 | 2.131 | 0.679 | 0 | 0 |
| dinov2 \| host_score | 60 | 1.100 | 0.367 | 0 | 0 |
| oracle \| trace | 84 | 0.000 | 0.512 | 0 | 0 |
| oracle \| threshold | 84 | 2.048 | 0.012 | 0.429 | 0 |
| oracle \| force_best | 84 | 1.190 | 0.440 | 0 | 0 |
| oracle \| host_score | 60 | 1.100 | 0.367 | 0 | 0 |

### 1.5 Extra run: action / j_true slices

출처: `replay_slices.json`, 표 `replay_slices.md`. `python -m experiments.replay_slices`.

`j_true=null` = reference가 candidate list에 없음. Replay는 truth=N (`none`). SELECT는 wrong.

| split | N≥2 | in list | null | null rate |
|---|---|---|---|---|
| FILM seen | 92 | 73 | 19 | 0.207 |
| FILM unseen | 65 | 63 | 2 | 0.031 |
| CAPEAM seen | 84 | 42 | 42 | **0.500** |
| CAPEAM unseen | — | — | — | missing |

TRACE, `j_true`로 자름:

| split | source | in_list CR | in_list wrong | null CR | null wrong |
|---|---|---|---|---|---|
| FILM seen | Jev | **0.027** | 0.151 | 0.632 | 0.684 |
| FILM seen | OpenAI | 0.247 | 0.192 | 0.000 | 0.684 |
| FILM unseen | Jev | 0.127 | 0.032 | 4.0 (RETURN) | 0 |
| FILM unseen | OpenAI | **0.063** | 0.000 | 4.0 (RETURN) | 0 |
| CAPEAM seen | Jev = OpenAI | **0.000** | 0.167 | 0.024 | **0.857** |

CAPEAM seen 전체 wrong 0.512 ≈ 0.5×0.857 + 0.5×0.167. Oracle TRACE도 같은 0.512.

Action (N≥2 TRACE CR):

| split | action | n | Jev CR | OpenAI CR |
|---|---|---|---|---|
| FILM seen | Pickup | 14 | 0.857 | **0.000** |
| FILM seen | Put | 39 | **0.051** | 0.462 |
| FILM unseen | Pickup | 13 | 0.923 | **0.615** |
| FILM unseen | Put | 13 | 0.000 | 0.000 |
| FILM unseen | ToggleOff | 32 | 0.000 | 0.000 |
| CAPEAM seen | Put | 50 | 0.020 | 0.020 |

FILM unseen 갭은 Pickup 13건. FILM seen Jev vs OpenAI는 Pickup/Put이 반대.

---

## 2. Synthetic live identity

출처: `source_compare.md`, `jev/live.json`, `openai_decisions/live.json`.
라벨을 아는 그림. 같은 질문, belief source만 다름.

| | Jev | OpenAI |
|---|---|---|
| identity accuracy | 1.00 | 1.00 |
| top-1 agree | 1.00 | 1.00 |
| mean L1 | 0.36 | 0.36 |
| TRACE action agree (3 cost world) | 1.00 | 1.00 |

| case | truth | Jev p | OpenAI p | L1 | cheap / expensive / both-far TRACE |
|---|---|---|---|---|---|
| visible_match | 0 | [0.98, 0.00, 0.02] | [0.98, 0.01, 0.01] | 0.02 | SELECT 0 / SELECT 0 / SELECT 0 |
| reference_absent | 2 | [0.11, 0.01, 0.88] | [0.38, 0.06, 0.56] | 0.64 | SELECT 0 / SELECT 0 / RETURN |
| lookalike | 0 | [0.74, 0.21, 0.05] | [0.95, 0.03, 0.02] | 0.42 | SELECT 0 / SELECT 0 / SELECT 0 |

Jev는 absent에서 `none` 0.88, OpenAI는 0.56. OpenAI는 lookalike에 0.95로 더 peaked. Threshold 0.75는 lookalike에서 갈림 (Jev RETURN, OpenAI SELECT) — TRACE는 둘 다 SELECT.

---

## 3. Unit-test extra cost (`offline_rules.json`)

같은 p = [0.6, 0.4, 0], `c_defer = 4`. Cheap extra 2, expensive extra 20.

| rule | cheap action | expensive action | mean realized extra |
|---|---|---|---|
| trace | SELECT 0 (R=0.8) | RETURN | **2.4** |
| threshold 0.55 | SELECT 0 | SELECT 0 | 4.4 |
| threshold 0.75 | RETURN | RETURN | 4.0 |
| host_score | SELECT 1 (wrong) | SELECT 1 (wrong) | 6.6 |

---

## 빠진 것

- CAPEAM unseen replay (`costs.jsonl` / beliefs 로컬·랩 둘 다 없음)
- Official 820ep closed-loop에서 TRACE가 고른 mask의 live quality
- THOR 5 episode SELECT idx는 `episode_v2.md` (2026-11-05 lab jsonl)
