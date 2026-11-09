# Unseen baseline recheck + CAPEAM unseen live

Fresh run 2026-11-02 15:17–15:20 KST. Same v2 code. `n=100` pick_and_place_simple valid_unseen.

**0.910 is 91/100.** 모든 source가 같은 9 episode에서 Put 실패.

실패는 전부 `FloorPlan219` `Vase -> Safe` (idx 21–23, 54–56, 87–89). Pickup은 100/100.

| host | source | n | ok | fail | SR |
|---|---|---|---|---|---|
| FILM | host | 100 | 91 | 9 | 0.910 |
| FILM | oracle | 100 | 91 | 9 | 0.910 |
| CAPEAM | host | 100 | 91 | 9 | 0.910 |
| CAPEAM | oracle | 100 | 91 | 9 | 0.910 |
| CAPEAM | Jev v2 | 100 | 91 | 9 | 0.910 |
| CAPEAM | OpenAI v2 | 100 | 91 | 9 | 0.910 |

FILM/CAPEAM host가 unseen에서 같은 SR인 이유: 실패 9개가 같은 FloorPlan219 Safe Put이고, host score가 달라도 그 episode는 전부 실패한다.

Replay Table-2의 CAPEAM unseen은 이 숫자와 무관하다. ryankwon `costs.jsonl`이 `capeam/valid_unseen`에 없어서 Metric-FF CR만 없다.
