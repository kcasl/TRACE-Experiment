# FILM/CAPEAM mask-log replay: Jev vs OpenAI Decisions

Same Metric-FF `C(i,j)`, same N≥2 events, TRACE `c_defer=4`.
OpenAI images are rebuilt from the Jev bbox/overlay log (reference crop, context crop, labelled current frame).

| split | n | Jev TRACE CR | OpenAI TRACE CR | Jev wrong | OpenAI wrong | Jev return | OpenAI return |
|---|---|---|---|---|---|---|---|
| FILM valid_seen | 92 | 0.152 | 0.196 | 0.261 | 0.293 | 0.054 | 0.033 |
| FILM valid_unseen | 65 | 0.246 | 0.185 | 0.031 | 0.000 | 0.031 | 0.031 |
| CAPEAM valid_seen | 84 | 0.012 | 0.012 | 0.512 | 0.512 | 0.000 | 0.000 |

Both sources beat DINOv2 TRACE on FILM seen (DINOv2 CR 1.38). Unrecoverable rate is 0 for TRACE on all three splits.
