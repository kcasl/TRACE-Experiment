# Paper figures

Two folders, always **Baseline | Jev | OpenAI**.

- `masks/` — FILM-style instance tint + A/B/C outlines. Jev/OpenAI columns add `p_identity` bars (`*` = mode). Title is SELECT or RETURN.
- `results/` — same three columns after the focused action (success/fail).

| file | scene | why |
|---|---|---|
| `01_fp422_handtowel` | FloorPlan422 bathroom | OpenAI SELECT SinkBasin (p=0.99) vs host/Jev Sink. Host fail, OpenAI success. |
| `02_fp7_fridge` | FloorPlan7 kitchen | 8 Put candidates. OpenAI SELECT Fridge p=0.97, Jev RETURN. |
| `03_fp4_fridge` | FloorPlan4 kitchen | Same pattern, different house. |
| `04_fp406_toiletpaper` | FloorPlan406 bathroom | CAPEAM host roll A vs OpenAI roll B (p=0.91). |
| `05_fp406_plunger` | FloorPlan406 bathroom | Jev SELECT cabinet A p=0.97, OpenAI RETURN. |
| `06_fp10_saltshaker` | FloorPlan10 unseen kitchen | Jev SELECT cabinet A p=0.97 among 8 receptacles. |
| `07_fp219_vase` | FloorPlan219 unseen | All RETURN / fail (Safe put). Belief mass on `none`. |
| `08_fp7_capeam_fridge` | FloorPlan7, CAPEAM host | OpenAI SELECT same Fridge. |

Logged-id replay. Jev/OpenAI ids and `p` come from `thor_port_v2` jsonl. RETURN plays the host.
