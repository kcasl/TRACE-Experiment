"""Where the FILM/CAPEAM logs and FF critic matrices live."""

from __future__ import annotations

import os
from pathlib import Path


def ryankwon_root() -> Path:
    env = os.environ.get("RYANKWON_ROOT")
    if env:
        return Path(env)
    for parent in Path(__file__).resolve().parents:
        candidate = parent / "ICEIC-2027-ALFRED-Decision" / "ryankwon"
        if (candidate / "results").exists():
            return candidate
    return Path(r"C:/Project/ICEIC-2027-ALFRED-Decision/ryankwon")


JW_ROOT = Path(__file__).resolve().parents[1]

# Closed-loop THOR 5 pnp. Host dump with cand_ids/C × logged Jev/OpenAI p.
THOR_COST_SPLITS = (
    {"name": "film/valid_seen", "host": "film", "split": "valid_seen", "n_episodes": 142},
    {"name": "film/valid_unseen", "host": "film", "split": "valid_unseen", "n_episodes": 100},
    {"name": "capeam/valid_seen", "host": "capeam", "split": "valid_seen", "n_episodes": 142},
    {"name": "capeam/valid_unseen", "host": "capeam", "split": "valid_unseen", "n_episodes": 100},
)

# Closed-loop THOR jsonl (ported THOR 5). Used when Metric-FF costs.jsonl is absent.
THOR_JSONL_SPLITS = (
    {
        "name": "capeam/valid_unseen",
        "directory": "results/thor_port_v2/capeam/valid_unseen",
        "n_episodes": 100,
    },
    {
        "name": "capeam/valid_unseen_recheck",
        "directory": "results/thor_port_v2_recheck/capeam/valid_unseen",
        "n_episodes": 100,
    },
)

# FF critic matrices + logged beliefs from the host rollouts (no simulator).
REPLAY_SPLITS = (
    {
        "name": "film/valid_seen",
        "costs": "results/offline/film/valid_seen_host_v2_cand/costs.jsonl",
        "beliefs": "results/beliefs/film/valid_seen_host_v2",
        "rollout": "results/rollouts/film/valid_seen/host",
        "n_episodes": 820,
    },
    {
        "name": "film/valid_unseen",
        "costs": "results/offline/film/valid_unseen_host_v2_cand/costs.jsonl",
        "beliefs": "results/beliefs/film/valid_unseen_host_v2",
        "rollout": "results/rollouts/film/valid_unseen/host",
        "n_episodes": 821,
    },
    {
        "name": "capeam/valid_seen",
        "costs": "results/offline/capeam/valid_seen_host_v2_cand/costs.jsonl",
        "beliefs": "results/beliefs/capeam/valid_seen_host_v2",
        "rollout": "results/rollouts/capeam/valid_seen/host",
        "n_episodes": 820,
    },
)
