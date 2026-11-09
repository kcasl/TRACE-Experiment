# THOR 5 paper frames (2026-11-05)

Ported pick-and-place, CloudRendering 300×300. Dumped by `experiments/thor_rollout.py --dump-frames`.

Recommended paper cuts (3 per episode):

| Episode | Pickup | Labelled candidates | After Put |
|---|---|---|---|
| FILM seen 108 FloorPlan422 HandTowel→SinkBasin | `..._host/pickup_before_act.png` | `..._host/putobject_choice_overlay.png` | `..._host/put_after.png` |
| CAPEAM unseen 36 FloorPlan10 SaltShaker→Cabinet | `..._host/init_rgb.png` | `..._host/putobject_choice_overlay.png` | `..._host/put_after.png` |
| FILM unseen 21 FloorPlan219 Vase→Safe | `..._host/pickup_before_act.png` | `..._host/putobject_choice_overlay.png` | `..._host/put_after.png` |

Notes:
- 422 Put candidates in view: A=`Sink|…|SinkBasin`, B=`Sink|…` (no suffix). Host uses B and Put fails. Live OpenAI dump hit an expired key and fell back to the host; use the v2 jsonl SELECT (`SinkBasin`) for the caption, not this dump's `put_after`.
- 36 Jev re-dump: `pmax=0.92` so the gate RETURNed (same as the recheck boundary). Episode still succeeds.
- 219: vase is picked; Put lands on a shelf, not the Safe. Identity SELECT cannot repair this.
- Overlays are the lettered view TRACE / the host choose among, not a cleaned teaser.
