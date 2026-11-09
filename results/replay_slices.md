# Replay slices (extra local run)

`python -m experiments.replay_slices` — simulator 없음. 기존 Metric-FF `costs.jsonl` × beliefs만 자른다.
원 JSON: `replay_slices.json`. CAPEAM unseen은 파일이 없어서 그대로 missing.

`j_true=null` = reference instance가 candidate list에 없음. Replay는 이 경우 truth = N (`none` 칸). SELECT는 전부 wrong으로 센다.

---

## Event inventory (N≥2만 Table-2)

| split | cost rows | N≥2 | j_true in list | j_true null | null rate |
|---|---|---|---|---|---|
| FILM seen | 625 | 92 | 73 | 19 | 0.207 |
| FILM unseen | 441 | 65 | 63 | 2 | 0.031 |
| CAPEAM seen | 615 | 84 | 42 | 42 | **0.500** |
| CAPEAM unseen | — | — | — | — | missing logs |

N≥2 action mix:

| split | Pickup | Put | Close | ToggleOff |
|---|---|---|---|---|
| FILM seen | 14 | 39 | 30 | 9 |
| FILM unseen | 13 | 13 | 7 | **32** |
| CAPEAM seen | 10 | **50** | 16 | 8 |

FILM unseen N≥2의 절반이 ToggleObjectOff (faucet). CAPEAM seen N≥2의 절반은 Put.

---

## TRACE CR by j_true

| split | source | in_list n | in_list CR | in_list wrong | null n | null CR | null wrong |
|---|---|---|---|---|---|---|---|
| FILM seen | Jev | 73 | **0.027** | 0.151 | 19 | 0.632 | 0.684 |
| FILM seen | OpenAI | 73 | 0.247 | 0.192 | 19 | **0.000** | 0.684 |
| FILM unseen | Jev | 63 | 0.127 | 0.032 | 2 | 4.000 | 0 (RETURN) |
| FILM unseen | OpenAI | 63 | **0.063** | 0.000 | 2 | 4.000 | 0 (RETURN) |
| CAPEAM seen | Jev | 42 | **0.000** | 0.167 | 42 | 0.024 | **0.857** |
| CAPEAM seen | OpenAI | 42 | **0.000** | 0.167 | 42 | 0.024 | **0.857** |
| CAPEAM seen | oracle | 42 | 0.000 | 0.167 | 42 | 0.000 | 0.857 |

CAPEAM seen wrong=0.512의 원인: N≥2의 절반이 `j_true=null`이고, TRACE가 거의 전부 SELECT라 `none`이 정답인 event를 wrong으로 친다. in_list만 보면 wrong 0.167, CR 0. Oracle TRACE도 전체 wrong이 0.512 — belief 문제가 아니라 **GT가 candidate에 없는 event 정의**다.

FILM seen에서 OpenAI가 Jev보다 CR이 높은 이유: in_list 73건 CR 0.247 vs 0.027. null 19건은 오히려 OpenAI CR 0.

---

## TRACE CR by action

| split | action | n | Jev CR | OpenAI CR |
|---|---|---|---|---|
| FILM seen | PickupObject | 14 | 0.857 | **0.000** |
| FILM seen | PutObject | 39 | **0.051** | 0.462 |
| FILM seen | CloseObject | 30 | 0.000 | 0.000 |
| FILM seen | ToggleObjectOff | 9 | 0.000 | 0.000 |
| FILM unseen | PickupObject | 13 | 0.923 | **0.615** |
| FILM unseen | PutObject | 13 | 0.000 | 0.000 |
| FILM unseen | CloseObject | 7 | 0.571 | 0.571 |
| FILM unseen | ToggleObjectOff | 32 | 0.000 | 0.000 |
| CAPEAM seen | PickupObject | 10 | 0.000 | 0.000 |
| CAPEAM seen | PutObject | 50 | 0.020 | 0.020 |
| CAPEAM seen | CloseObject | 16 | 0.000 | 0.000 |
| CAPEAM seen | ToggleObjectOff | 8 | 0.000 | 0.000 |

FILM unseen Jev 0.246 vs OpenAI 0.185 갭은 PickupObject 13건 (0.923 vs 0.615). Put/Toggle은 둘 다 CR 0.

FILM seen Jev vs OpenAI는 Pickup(Jev 손해)과 Put(OpenAI 손해)이 반대.

---

## 아직 없는 것

- CAPEAM unseen `costs.jsonl` / beliefs → 이 스크립트로도 CR 불가 (랩에도 없음)
- THOR 5 episode SELECT 목록은 `episode_v2.md` / `RERUN.md` §8
