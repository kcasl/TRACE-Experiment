# TRACE

Task-regret-aware instance association for embodied instruction following.
The package decides whether a host agent should commit to one interaction mask
or keep its own choice. It does not run AI2-THOR.

## Run everything

```powershell
python -m pip install numpy pillow pytest
python -m experiments.run_all            # tests + offline + scenes + live
```

`run_all` runs the unit tests, the decision-rule comparison, the symbolic-planner
scene costs, and the Jev / OpenAI Decisions identity calls, then writes
`results/REPORT.md` plus one JSON file per experiment and a combined
`results/results.json`.

```powershell
python -m experiments.run_all --offline  # no network; oracle rows only
python -m experiments.run_all --no-tests # skip pytest
```

Individual pieces still run on their own:

```powershell
python -m pytest
python -m experiments.offline            # results/offline_rules.json
python -m experiments.run_source jev              # results/jev/live.json
python -m experiments.run_source openai_decisions # results/openai_decisions/live.json
python -m experiments.compare_sources             # results/source_compare.md
python -m experiments.compute_openai_beliefs  # rebuild crop PNGs, call OpenAI on mask events
python -m experiments.replay             # FILM/CAPEAM FF cost × Jev/OpenAI/DINOv2 beliefs
python -m experiments.protocol           # official 10-failure SR table
```

Replay reads `RYANKWON_ROOT` (or the sibling `ICEIC-2027-ALFRED-Decision/ryankwon`
tree) for `results/offline/*/costs.jsonl` and `results/beliefs/*`. It does not
start THOR. Official SR is protocol A: 10 counted failures, 1000 steps.

`JEV_API_KEY` and `OPENAI_API_KEY` are read from the environment or from `.env`.
That file is gitignored. `.env.example` lists the names only. A source whose key
is missing is skipped, not failed, so `run_all` still produces the oracle rows.

The method, the cost formulas, and the experiment numbers are in `METHOD.md`.
