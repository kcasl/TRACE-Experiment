# Ported THOR 5 TRACE matrix

Full oct21 (all seven ALFRED types). FILM host = nearest to agent; CAPEAM host = Mask R-CNN.
Jev/OpenAI: unique-min + planner extra C (c_defer=4). No off-type Put prior.

| host | split | source | n | SR | pickup | put | api_errors |
|---|---|---|---|---|---|---|---|
| capeam | valid_seen | jev | 820 | 0.615 | 0.991 | 0.817 | 0 |
| capeam | valid_seen | openai_decisions | 820 | 0.659 | 0.994 | 0.822 | 0 |
| capeam | valid_unseen | jev | 821 | 0.649 | 0.989 | 0.739 | 0 |
| capeam | valid_unseen | openai_decisions | 821 | 0.642 | 0.989 | 0.741 | 0 |
| film | valid_seen | jev | 820 | 0.615 | 0.994 | 0.821 | 0 |
| film | valid_seen | openai_decisions | 820 | 0.649 | 0.994 | 0.826 | 0 |
| film | valid_unseen | jev | 821 | 0.636 | 0.989 | 0.743 | 0 |
| film | valid_unseen | openai_decisions | 821 | 0.633 | 0.989 | 0.743 | 0 |
