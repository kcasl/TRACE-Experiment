# Ported THOR 5 TRACE matrix

Pick-and-place simple only (valid_seen 142, valid_unseen 100). Not the official 820-ep protocol.

- FILM host: nearest to the agent
- CAPEAM host: `snumprlab/capeam` `weight_maskrcnn.pt` Mask R-CNN scores on THOR RGB (CUDA)
- Jev / OpenAI Decisions: TRACE identity on THOR RGB crops, deferral cost 4

CAPEAM seen host SR 0.915 vs FILM 0.887 (3 different pickup IDs). Unseen hosts agreed on all 100 episodes.

| host | split | source | n | SR | pickup | put | api_errors |
|---|---|---|---|---|---|---|---|
| film | valid_seen | host | 142 | 0.887 | 1.000 | 0.887 | 0 |
| film | valid_seen | oracle | 142 | 0.887 | 1.000 | 0.887 | 0 |
| film | valid_seen | jev | 142 | 0.915 | 1.000 | 0.915 | 3 |
| film | valid_seen | openai_decisions | 142 | 0.887 | 1.000 | 0.887 | 0 |
| film | valid_unseen | host | 100 | 0.910 | 1.000 | 0.910 | 0 |
| film | valid_unseen | oracle | 100 | 0.910 | 1.000 | 0.910 | 0 |
| film | valid_unseen | jev | 100 | 0.880 | 1.000 | 0.880 | 0 |
| film | valid_unseen | openai_decisions | 100 | 0.880 | 1.000 | 0.880 | 0 |
| capeam | valid_seen | host | 142 | 0.915 | 1.000 | 0.915 | 0 |
| capeam | valid_seen | oracle | 142 | 0.887 | 1.000 | 0.887 | 0 |
| capeam | valid_seen | jev | 142 | 0.915 | 1.000 | 0.915 | 3 |
| capeam | valid_seen | openai_decisions | 142 | 0.887 | 1.000 | 0.887 | 0 |
| capeam | valid_unseen | host | 100 | 0.910 | 1.000 | 0.910 | 0 |
| capeam | valid_unseen | oracle | 100 | 0.910 | 1.000 | 0.910 | 0 |
| capeam | valid_unseen | jev | 100 | 0.880 | 1.000 | 0.880 | 0 |
| capeam | valid_unseen | openai_decisions | 100 | 0.880 | 1.000 | 0.880 | 0 |
