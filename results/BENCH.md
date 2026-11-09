# Benchmark SR

Episode 성공률만. Mask 선택 CR은 [SELECTION_QUALITY.md](SELECTION_QUALITY.md).
Closed-loop 표는 ryankwon 10-fail 대신 **jw THOR 5** (`RERUN.md`).

---

## Metric

| 이름 | 정의 |
|---|---|
| SR | episode goal success rate |
| GC | goal-condition progress (official만) |
| pickup / put | THOR 5 pnp에서 각 action 성공률. success = pickup ∧ put |
| Protocol A | counted API failure 10번째에서 cut, max 1000 step |

---

## A. Closed-loop SR (jw THOR 5 pnp v2)

ryankwon 10-fail 820ep 대신 이 랩에서 잰 숫자. 출처: `RERUN.md`.
pick_and_place_simple, seen 142 / unseen 100. Pickup 1.0.

| host | split | n | host | oracle | Jev v2 | OpenAI v2 |
|---|---|---|---|---|---|---|
| FILM | seen | 142 | 0.887 | 0.887 | 0.887 | **0.915** |
| FILM | unseen | 100 | 0.910 | 0.910 | 0.910 | 0.910 |
| CAPEAM | seen | 142 | 0.915 | 0.887 | 0.915 | 0.915 |
| CAPEAM | unseen | 100 | 0.910 | 0.910 | 0.910 | 0.910 |

---

## B. Ported THOR 5 pick-and-place (v1 closed loop)

출처: `thor_port/matrix_summary.md`.
**Official ALFRED 아님.** Task = pick_and_place_simple only. Pose는 2019 name → type map. 10-fail 미적용 (action 2개).

- FILM host = agent 최근접
- CAPEAM host = GitHub `weight_maskrcnn.pt` Mask R-CNN IoU≥0.2
- Jev / OpenAI = TRACE identity + expert glance crop + `c_defer=4` (v1 method)

| host | split | n | source | SR | pickup | put | api_err | 성공 수 |
|---|---|---|---|---|---|---|---|---|
| FILM | seen | 142 | host | 0.887 | 1.000 | 0.887 | 0 | 126 |
| FILM | seen | 142 | oracle | 0.887 | 1.000 | 0.887 | 0 | 126 |
| FILM | seen | 142 | jev | 0.915 | 1.000 | 0.915 | 3 | 130 |
| FILM | seen | 142 | openai | 0.887 | 1.000 | 0.887 | 0 | 126 |
| FILM | unseen | 100 | host | 0.910 | 1.000 | 0.910 | 0 | 91 |
| FILM | unseen | 100 | oracle | 0.910 | 1.000 | 0.910 | 0 | 91 |
| FILM | unseen | 100 | jev | 0.880 | 1.000 | 0.880 | 0 | 88 |
| FILM | unseen | 100 | openai | 0.880 | 1.000 | 0.880 | 0 | 88 |
| CAPEAM | seen | 142 | host | 0.915 | 1.000 | 0.915 | 0 | 130 |
| CAPEAM | seen | 142 | oracle | 0.887 | 1.000 | 0.887 | 0 | 126 |
| CAPEAM | seen | 142 | jev | 0.915 | 1.000 | 0.915 | 3 | 130 |
| CAPEAM | seen | 142 | openai | 0.887 | 1.000 | 0.887 | 0 | 126 |
| CAPEAM | unseen | 100 | host | 0.910 | 1.000 | 0.910 | 0 | 91 |
| CAPEAM | unseen | 100 | oracle | 0.910 | 1.000 | 0.910 | 0 | 91 |
| CAPEAM | unseen | 100 | jev | 0.880 | 1.000 | 0.880 | 0 | 88 |
| CAPEAM | unseen | 100 | openai | 0.880 | 1.000 | 0.880 | 0 | 88 |

Pickup은 전부 1.000. Gap은 Put.

### vs host (pure baseline)

| split | host SR | Jev Δ | OpenAI Δ |
|---|---|---|---|
| FILM seen | 0.887 | **+0.028** (+4 ep) | 0 |
| FILM unseen | 0.910 | **−0.030** (−3 ep) | **−0.030** (−3 ep) |
| CAPEAM seen | 0.915 | 0 | −0.028 (−4 ep) |
| CAPEAM unseen | 0.910 | **−0.030** (−3 ep) | **−0.030** (−3 ep) |

Unseen에서 FILM host와 CAPEAM host는 100/100 동일. CAPEAM seen host는 FILM보다 pickup ID 3개 다름 (0.887 → 0.915). CAPEAM seen oracle 0.887 < host 0.915.

---

## C. Ported THOR 5 — method v2

출처: `thor_port_v2/matrix_summary.md`. Task grounding + Put success C=100.

| host | split | n | host | oracle | Jev v2 | OpenAI v2 | vs host |
|---|---|---|---|---|---|---|---|
| FILM | seen | 142 | 0.887 | 0.887 | 0.887 | **0.915** | OpenAI +4 ep |
| FILM | unseen | 100 | 0.910 | 0.910 | 0.910 | 0.910 | 0 |
| CAPEAM | seen | 142 | 0.915 | 0.887 | 0.915 | 0.915 | 0 |
| CAPEAM | unseen | 100 | 0.910 | 0.910 | 0.910 | 0.910 | 0 |

Unseen v1 −3ep는 회복. OpenAI seen만 host를 이긴다 (FILM). API error 0.

## 없는 bench

- Official 820ep + TRACE Jev/OpenAI closed-loop SR (2019 stack blocked)
- Protocol A를 THOR 5 pnp에 적용한 SR (episode가 너무 짧음)
- THOR 5 episode SELECT/fail idx는 `episode_v2.md` / `RERUN.md` §8
- CAPEAM unseen Metric-FF CR — `costs.jsonl` 로컬·랩 둘 다 없음 (`replay_slices.md`)
