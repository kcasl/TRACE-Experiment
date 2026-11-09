"""Put-success CR from THOR 5 host dumps × logged Jev/OpenAI p.

Official CAPEAM unseen Metric-FF ``costs.jsonl`` was never written (the 821-ep
host campaign did not keep rollouts). Closed-loop v2 already logged ``p``. A
host-only THOR dump adds ``cand_ids`` / ``j_true`` / Put success C so the same
replay as Table-2 can run on every pnp split, including CAPEAM unseen.

    python -m experiments.thor_costs
    python -m experiments.thor_costs --costs-root results/thor_costs --beliefs-root results/thor_port_v2
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from trace.config import DEFERRAL_COST

from .paths import JW_ROOT, THOR_COST_SPLITS
from .replay import SELECT_N, load_jsonl, replay_split
from .thor_rollout import success_cost_matrix

CRITIC = "planner extra C (THOR 5 pnp v3)"
SOURCES = ("jev", "openai_decisions")


def event_key(name: str, row: dict, action: str) -> str:
    task = row.get("task_id") or row.get("rel") or "task"
    return f"{name}__{task}__{row.get('idx')}__{action}"


def _action_event(row: dict, action: str) -> dict | None:
    for item in row.get("events") or []:
        if item.get("action") == action:
            return item
    return None


def load_source_rows(directory: Path, source: str) -> list[dict]:
    hits = sorted(directory.glob(f"{source}_*.jsonl"))
    by: dict[int, dict] = {}
    extra = []
    for path in hits:
        for row in load_jsonl(path):
            if row.get("idx") is not None:
                by[int(row["idx"])] = row
            else:
                extra.append(row)
    return [by[key] for key in sorted(by)] + extra


def by_idx(rows: list[dict]) -> dict[int, dict]:
    return {int(row["idx"]): row for row in rows if row.get("idx") is not None}


def costs_row_from_event(name: str, row: dict, event: dict) -> dict | None:
    n = event.get("n")
    if n is None or int(n) < SELECT_N:
        return None
    n = int(n)
    cand_ids = list(event.get("cand_ids") or [])
    if len(cand_ids) != n:
        return None
    costs = event.get("C")
    reachable = event.get("reachable")
    if not costs:
        if event.get("action") != "PutObject":
            return None
        costs, reachable = success_cost_matrix(n)
    j_true = event.get("j_true")
    host_choice = event.get("host_choice")
    if host_choice is None and event.get("id") in cand_ids:
        host_choice = cand_ids.index(event["id"])
    return {
        "event_key": event_key(name, row, str(event.get("action"))),
        "episode_id": row.get("rel") or row.get("task_id"),
        "idx": row.get("idx"),
        "floor": row.get("floor"),
        "t": 0,
        "relation": event.get("relation") or "SAME",
        "ref_gt": event.get("expert_id"),
        "cand_gt": cand_ids,
        "host_choice": host_choice,
        "action": event.get("action"),
        "target_cls": row.get("parent") if event.get("action") == "PutObject" else row.get("target"),
        "task_type": "pick_and_place_simple",
        "j_true": j_true,
        "C": costs,
        "reachable": reachable,
        "critic": CRITIC,
    }


def belief_row(name: str, row: dict, event: dict, source: str) -> dict | None:
    n = event.get("n")
    p = event.get("p")
    if n is None or not p:
        return None
    n = int(n)
    if n < SELECT_N or len(p) != n + 1:
        return None
    return {
        "event_key": event_key(name, row, str(event.get("action"))),
        "p": list(p),
        "source": source,
        "idx": row.get("idx"),
        "action": event.get("action"),
    }


def oracle_belief(cost_row: dict) -> dict:
    n = len(cost_row["cand_gt"])
    p = [0.0] * (n + 1)
    truth = cost_row.get("j_true")
    p[n if truth is None else int(truth)] = 1.0
    return {"event_key": cost_row["event_key"], "p": p, "source": "oracle"}


def build_split(
    name: str,
    host_rows: list[dict],
    belief_rows: dict[str, list[dict]],
) -> dict:
    costs: list[dict] = []
    beliefs: dict[str, list[dict]] = {source: [] for source in (*SOURCES, "oracle")}
    matched = defaultdict(int)
    skipped = defaultdict(int)
    host_by_idx = by_idx(host_rows)
    belief_idx = {source: by_idx(rows) for source, rows in belief_rows.items()}
    for idx, host in host_by_idx.items():
        for event in host.get("events") or []:
            packed = costs_row_from_event(name, host, event)
            if packed is None:
                continue
            action = packed["action"]
            n = len(packed["cand_gt"])
            costs.append(packed)
            beliefs["oracle"].append(oracle_belief(packed))
            for source in SOURCES:
                other = belief_idx.get(source, {}).get(idx)
                item = _action_event(other, action) if other else None
                if item is None or not item.get("p") or len(item["p"]) != n + 1:
                    skipped[source] += 1
                    continue
                row = belief_row(name, other, item, source)
                if row is None:
                    skipped[source] += 1
                    continue
                beliefs[source].append(row)
                matched[source] += 1
    return {
        "name": name,
        "n_cost_events": len(costs),
        "matched": dict(matched),
        "skipped": dict(skipped),
        "costs": costs,
        "beliefs": beliefs,
    }


def write_split(payload: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    costs_path = out_dir / "costs.jsonl"
    with costs_path.open("w", encoding="utf-8") as handle:
        for row in payload["costs"]:
            handle.write(json.dumps(row) + "\n")
    beliefs_dir = out_dir / "beliefs"
    beliefs_dir.mkdir(parents=True, exist_ok=True)
    for source, rows in payload["beliefs"].items():
        path = beliefs_dir / f"{source}.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row) + "\n")


def replay_table(out_dir: Path) -> dict:
    table = replay_split(out_dir / "costs.jsonl", out_dir / "beliefs")
    for stats in table.values():
        stats["critic"] = CRITIC
    return table


def summarize_table(table: dict) -> dict:
    keep = {}
    for key, stats in table.items():
        source, rule = key.split("|", 1)
        if source not in (*SOURCES, "oracle") or rule not in ("trace", "host_score", "threshold", "force_best"):
            continue
        keep[key] = {
            "n": stats["n"],
            "CR": stats["commitment_regret"],
            "wrong": stats["wrong_instance_rate"],
            "return": stats["return_rate"],
            "unrec": stats["unrecoverable_rate"],
        }
    return keep


def load_split_rows(directory: Path) -> dict[str, list[dict]]:
    return {source: load_source_rows(directory, source) for source in ("host", "oracle", *SOURCES)}


def run(
    *,
    costs_root: Path | None = None,
    beliefs_root: Path | None = None,
) -> dict:
    costs_root = costs_root or (JW_ROOT / "results" / "thor_costs")
    beliefs_root = beliefs_root or (JW_ROOT / "results" / "thor_port_v2")
    report = {
        "critic": CRITIC,
        "deferral_cost": DEFERRAL_COST,
        "min_n": SELECT_N,
        "why_metric_ff_missing": (
            "CAPEAM valid_unseen official 821-ep host logs were never copied off the "
            "campaign VMs, so Metric-FF costs.jsonl / beliefs do not exist on the lab."
        ),
        "splits": {},
    }
    for spec in THOR_COST_SPLITS:
        host_dir = costs_root / spec["host"] / spec["split"]
        belief_dir = beliefs_root / spec["host"] / spec["split"]
        if not host_dir.exists():
            report["splits"][spec["name"]] = {"error": "missing host dump", "directory": str(host_dir)}
            continue
        host_rows = load_split_rows(host_dir).get("host") or []
        if not host_rows:
            report["splits"][spec["name"]] = {"error": "empty host jsonl", "directory": str(host_dir)}
            continue
        belief_rows = load_split_rows(belief_dir if belief_dir.exists() else host_dir)
        built = build_split(spec["name"], host_rows, belief_rows)
        out_dir = costs_root / spec["host"] / spec["split"]
        write_split(built, out_dir)
        table = replay_table(out_dir) if built["n_cost_events"] else {}
        report["splits"][spec["name"]] = {
            "n_episodes": len(host_rows),
            "n_cost_events": built["n_cost_events"],
            "matched": built["matched"],
            "skipped": built["skipped"],
            "table": summarize_table(table),
            "critic": CRITIC,
        }
    return report


def _md_row(stats: dict | None) -> str:
    if not stats:
        return "— | — | — | — | —"
    return (
        f"{stats['n']} | {stats['CR']:.3f} | {stats['wrong']:.3f} | "
        f"{stats['return']:.3f} | {stats['unrec']:.3f}"
    )


def markdown(report: dict) -> str:
    lines = [
        "# TRACE Jev / OpenAI (THOR 5 put-success CR)",
        "",
        report["why_metric_ff_missing"],
        "",
        f"Critic: `{report['critic']}`. `c_defer={report['deferral_cost']:.0f}`, N≥{report['min_n']}.",
        "Beliefs are the closed-loop logged `p` (no new API calls).",
        "",
        "## TRACE vs host (Jev / OpenAI)",
        "",
        "| split | source | n | CR | wrong | return | unrec |",
        "|---|---|---|---|---|---|---|",
    ]
    for spec in THOR_COST_SPLITS:
        split = report["splits"].get(spec["name"]) or {}
        if split.get("error"):
            lines.append(f"| {spec['name']} | — | — | missing: {split['error']} | — | — | — |")
            continue
        table = split.get("table") or {}
        for source, label in (("jev", "Jev"), ("openai_decisions", "OpenAI"), ("oracle", "oracle")):
            stats = table.get(f"{source}|trace")
            lines.append(f"| {spec['name']} | {label} TRACE | {_md_row(stats)} |")
        host = table.get("jev|host_score") or table.get("openai_decisions|host_score")
        lines.append(f"| {spec['name']} | host_score | {_md_row(host)} |")
    lines.extend(
        [
            "",
            "## Same belief, other rules (TRACE vs threshold / force_best)",
            "",
            "| split | source | rule | n | CR | wrong | return | unrec |",
            "|---|---|---|---|---|---|---|---|",
        ]
    )
    for spec in THOR_COST_SPLITS:
        split = report["splits"].get(spec["name"]) or {}
        table = split.get("table") or {}
        if not table:
            continue
        for source, label in (("jev", "Jev"), ("openai_decisions", "OpenAI")):
            for rule in ("trace", "threshold", "force_best", "host_score"):
                stats = table.get(f"{source}|{rule}")
                if not stats:
                    continue
                lines.append(f"| {spec['name']} | {label} | {rule} | {_md_row(stats)} |")
    lines.extend(["", "## Build inventory", ""])
    for spec in THOR_COST_SPLITS:
        split = report["splits"].get(spec["name"]) or {}
        lines.append(f"- `{spec['name']}`: {json.dumps({k: split.get(k) for k in ('n_episodes','n_cost_events','matched','skipped','error') if k in split or split.get('error')})}")
    lines.append("")
    return "\n".join(lines)


def write_results(report: dict, *, root: Path | None = None) -> tuple[Path, Path]:
    out_dir = (root or JW_ROOT) / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "thor_costs.json"
    md_path = out_dir / "thor_costs.md"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    md_path.write_text(markdown(report), encoding="utf-8")
    return json_path, md_path


def main() -> None:
    report = run()
    json_path, md_path = write_results(report)
    print(json.dumps(report, indent=2))
    print(f"wrote {json_path}")
    print(f"wrote {md_path}")


if __name__ == "__main__":
    main()
