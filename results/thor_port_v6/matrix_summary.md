# Ported THOR 5 TRACE matrix

Full oct21 (all seven ALFRED types). FILM host = nearest to agent; CAPEAM host = Mask R-CNN.
Jev/OpenAI: unique-min + planner extra C (c_defer=4). No off-type Put prior.

| host | split | source | n | SR | pickup | put | api_errors |
|---|---|---|---|---|---|---|---|
| capeam | valid_seen | host | 820 | 0.528 | 0.994 | 0.804 | 0 |
| capeam | valid_seen | oracle | 820 | 0.512 | 0.994 | 0.807 | 0 |
| capeam | valid_seen | jev | 820 | 0.623 | 0.994 | 0.820 | 0 |
| capeam | valid_seen | openai_decisions | 820 | 0.652 | 0.994 | 0.823 | 0 |
| capeam | valid_unseen | host | 821 | 0.629 | 0.989 | 0.725 | 0 |
| capeam | valid_unseen | oracle | 821 | 0.621 | 0.989 | 0.732 | 0 |
| capeam | valid_unseen | jev | 821 | 0.654 | 0.989 | 0.736 | 0 |
| capeam | valid_unseen | openai_decisions | 821 | 0.655 | 0.989 | 0.736 | 0 |
| film | valid_seen | host | 820 | 0.500 | 0.994 | 0.809 | 0 |
| film | valid_seen | oracle | 820 | 0.509 | 0.994 | 0.809 | 0 |
| film | valid_seen | jev | 820 | 0.598 | 0.994 | 0.821 | 0 |
| film | valid_seen | openai_decisions | 820 | 0.633 | 0.994 | 0.828 | 0 |
| film | valid_unseen | host | 821 | 0.614 | 0.989 | 0.736 | 0 |
| film | valid_unseen | oracle | 821 | 0.621 | 0.989 | 0.732 | 0 |
| film | valid_unseen | jev | 821 | 0.638 | 0.989 | 0.743 | 0 |
| film | valid_unseen | openai_decisions | 821 | 0.642 | 0.989 | 0.743 | 0 |
