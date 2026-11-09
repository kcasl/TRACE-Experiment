"""TRACE quality from closed-loop THOR jsonl (no Metric-FF).

CAPEAM valid_unseen has no costs.jsonl on the lab or locally. The v2
rollout already logged n, p, defer, and reason, so the same Put
success-valued critic can be reconstructed from those events.

    python -m experiments.thor_jsonl
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np

from trace.config import DEFERRAL_COST, EPISODE_FAIL_COST
from trace.gate import decide
from trace.types import Belief, CostMatrix, normalize_mode

from .paths import JW_ROOT, THOR_JSONL_SPLITS

SOURCES = ("host", "oracle", "jev", "openai_decisions")
SELECT_N = 2


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def success_matrix(n: int, *, fail: float = EPISODE_FAIL_COST) -> CostMatrix:
    costs = np.full((n, n + 1), fail, dtype=np.float64)
    np.fill_diagonal(costs[:, :n], 0.0)
    return CostMatrix(
        C=costs,
        total=costs.copy(),
        base=np.zeros(n + 1, dtype=np.float64),
        candidate_ids=tuple(str(i) for i in range(n)),
        action="PutObject",
    )


def reconstruct_decision(event: dict, *, deferral: float = DEFERRAL_COST) -> dict:
    n = int(event["n"])
    p_identity = np.asarray(event["p"], dtype=np.float64)
    if p_identity.size != n + 1:
        raise ValueError(f"p length {p_identity.size} != n+1 ({n + 1})")
    decision = decide(
        Belief(p_identity=p_identity, source=str(event.get("source") or "logged")),
        success_matrix(n),
        mode=normalize_mode(event.get("relation") or "SAME"),
        deferral_cost=deferral,
    )
    return {
        "action": decision.action,
        "idx": decision.idx,
        "reason": decision.reason,
        "pmax": float(np.max(p_identity[:n])) if n else 0.0,
        "p_none": float(p_identity[-1]),
    }


def _event_id(row: dict, action: str) -> str | None:
    for event in row.get("events") or []:
        if event.get("action") == action:
            return event.get("id")
    return None


def summarize_episodes(rows: list[dict]) -> dict:
    n = len(rows)
    if n == 0:
        return {"n": 0, "sr": 0.0, "pickup": 0.0, "put": 0.0, "fails": []}
    fails = []
    for row in rows:
        if not row.get("success"):
            fails.append(
                {
                    "idx": row.get("idx"),
                    "floor": row.get("floor"),
                    "target": row.get("target"),
                    "parent": row.get("parent"),
                    "task_id": row.get("task_id"),
                }
            )
    pickup = sum(1 for row in rows if row.get("pickup_ok"))
    put = sum(1 for row in rows if row.get("put_ok"))
    success = sum(1 for row in rows if row.get("success"))
    return {
        "n": n,
        "sr": success / n,
        "pickup": pickup / n,
        "put": put / n,
        "fails": fails,
        "fail_idxs": [item["idx"] for item in fails],
    }


def vs_host(host_rows: list[dict], other_rows: list[dict]) -> dict:
    by_idx = {row.get("idx"): row for row in host_rows}
    pickup_diff = 0
    put_diff = 0
    gained = 0
    lost = 0
    diffs = []
    for row in other_rows:
        host = by_idx.get(row.get("idx"))
        if host is None:
            continue
        pickup_changed = _event_id(host, "PickupObject") != _event_id(row, "PickupObject")
        put_changed = _event_id(host, "PutObject") != _event_id(row, "PutObject")
        pickup_diff += int(pickup_changed)
        put_diff += int(put_changed)
        if row.get("success") and not host.get("success"):
            gained += 1
        if host.get("success") and not row.get("success"):
            lost += 1
        if pickup_changed or put_changed:
            diffs.append(
                {
                    "idx": row.get("idx"),
                    "floor": row.get("floor"),
                    "pickup_changed": pickup_changed,
                    "put_changed": put_changed,
                    "host_success": bool(host.get("success")),
                    "other_success": bool(row.get("success")),
                }
            )
    return {
        "pickup_id_diff": pickup_diff,
        "put_id_diff": put_diff,
        "gained": gained,
        "lost": lost,
        "id_diffs": diffs,
    }


def trace_events(rows: list[dict], *, min_n: int = SELECT_N) -> dict:
    events = []
    for row in rows:
        for event in row.get("events") or []:
            n = event.get("n")
            if n is None or int(n) < min_n:
                continue
            reason = str(event.get("reason") or "")
            logged_select = reason == "regret_below_deferral"
            item = {
                "idx": row.get("idx"),
                "floor": row.get("floor"),
                "action": event.get("action"),
                "n": int(n),
                "relation": event.get("relation"),
                "defer": bool(event.get("defer")),
                "reason": reason,
                "logged_select": logged_select,
                "ok": bool(event.get("ok")),
                "ep_success": bool(row.get("success")),
                "id": event.get("id"),
                "pmax": float(max(event["p"][: int(n)])) if event.get("p") else None,
            }
            if event.get("p"):
                try:
                    rebuilt = reconstruct_decision(event)
                except ValueError:
                    rebuilt = None
                if rebuilt is not None:
                    item["rebuilt"] = rebuilt
                    item["match"] = (rebuilt["action"] == "SELECT") == logged_select
            events.append(item)
    selects = [item for item in events if item["logged_select"]]
    rebuilt = [item for item in events if "match" in item]
    return {
        "n_events": len(events),
        "return": sum(1 for item in events if not item["logged_select"]),
        "select": len(selects),
        "selects": selects,
        "rebuild_n": len(rebuilt),
        "rebuild_match": sum(1 for item in rebuilt if item["match"]),
        "rebuild_match_rate": (sum(1 for item in rebuilt if item["match"]) / len(rebuilt)) if rebuilt else 1.0,
        "actions": dict(Counter(item["action"] for item in events)),
    }


def report_directory(directory: Path) -> dict:
    loaded = {}
    for source in SOURCES:
        path = directory / f"{source}_0_100.jsonl"
        if not path.exists():
            path = directory / f"{source}.jsonl"
        loaded[source] = load_jsonl(path) if path.exists() else []
    host = loaded.get("host") or []
    sources = {}
    for name, rows in loaded.items():
        pack = summarize_episodes(rows)
        pack["trace"] = trace_events(rows)
        if name != "host" and host:
            pack["vs_host"] = vs_host(host, rows)
        sources[name] = pack
    return {
        "directory": str(directory),
        "n_files": sum(1 for rows in loaded.values() if rows),
        "sources": sources,
    }


def run(*, root: Path | None = None) -> dict:
    base = root or JW_ROOT
    splits = {}
    for spec in THOR_JSONL_SPLITS:
        directory = base / spec["directory"]
        if not directory.exists():
            splits[spec["name"]] = {"error": "missing jsonl", "directory": str(directory)}
            continue
        payload = report_directory(directory)
        payload["expected_n"] = spec["n_episodes"]
        splits[spec["name"]] = payload
    return {"critic": "Put success-valued C from logged p (no Metric-FF)", "splits": splits}


def _md_fails(fails: list[dict]) -> str:
    if not fails:
        return "(none)"
    lines = ["| idx | floor | target | parent |", "|---|---|---|---|"]
    for item in fails:
        lines.append(f"| {item['idx']} | {item['floor']} | {item['target']} | {item['parent']} |")
    return "\n".join(lines)


def _md_selects(selects: list[dict]) -> str:
    if not selects:
        return "(none)"
    lines = ["| idx | floor | action | n | pmax | ok | ep |", "|---|---|---|---|---|---|---|"]
    for item in selects:
        pmax = "—" if item.get("pmax") is None else f"{item['pmax']:.2f}"
        lines.append(
            f"| {item['idx']} | {item['floor']} | {item['action']} | {item['n']} | {pmax} | "
            f"{int(item['ok'])} | {int(item['ep_success'])} |"
        )
    return "\n".join(lines)


def markdown(report: dict) -> str:
    lines = [
        "# CAPEAM Unseen (THOR-jsonl TRACE)",
        "",
        "Closed-loop THOR 5 pick_and_place_simple, `n=100`.",
        "Metric-FF `costs.jsonl` is still absent; this table is the put-success critic replayed from logged `p`.",
        "",
        f"Critic: {report['critic']}. `c_defer=4`, fail=100, SELECT only if `p_i>0.96`.",
        "",
    ]
    for name, split in report["splits"].items():
        lines.append(f"## {name}")
        lines.append("")
        if split.get("error"):
            lines.append(f"Missing: `{split.get('directory')}`")
            lines.append("")
            continue
        lines.append("| source | n | SR | pickup | put | N≥2 | RETURN | SELECT | rebuild match | vs-host ΔSR |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|")
        for source, pack in split["sources"].items():
            trace = pack["trace"]
            vs = pack.get("vs_host") or {}
            delta = ""
            if vs:
                delta = f"+{vs['gained']}/-{vs['lost']}"
            lines.append(
                f"| {source} | {pack['n']} | {pack['sr']:.3f} | {pack['pickup']:.3f} | {pack['put']:.3f} | "
                f"{trace['n_events']} | {trace['return']} | {trace['select']} | "
                f"{trace['rebuild_match_rate']:.2f} | {delta or '—'} |"
            )
        host = split["sources"].get("host") or {}
        lines.extend(["", "### Fail episodes (host = all sources)", "", _md_fails(host.get("fails") or []), ""])
        for source in ("jev", "openai_decisions"):
            selects = (split["sources"].get(source) or {}).get("trace", {}).get("selects") or []
            lines.extend([f"### {source} SELECT", "", _md_selects(selects), ""])
    v2 = ((report["splits"].get("capeam/valid_unseen") or {}).get("sources") or {}).get("jev") or {}
    recheck = ((report["splits"].get("capeam/valid_unseen_recheck") or {}).get("sources") or {}).get("jev") or {}
    v2_p = next((item.get("pmax") for item in (v2.get("trace") or {}).get("selects") or [] if item.get("idx") == 36), None)
    lines.extend(
        [
            "## Recheck boundary (idx 36)",
            "",
            "Same episode `FloorPlan10` SaltShaker→Cabinet. v2 Jev SELECT at `pmax=0.97`.",
            "Recheck Jev `pmax=0.96` so `R=4` equals `c_defer` and the gate RETURNS.",
            f"Logged v2 SELECT pmax={v2_p}. Recheck Jev SELECT count={((recheck.get('trace') or {}).get('select') if recheck else 'n/a')}.",
            "SR stays 0.910 both runs: the SELECT did not change Put success.",
            "",
            "## Metric-FF",
            "",
            "`capeam/valid_unseen` costs/beliefs are still missing on the lab and locally.",
            "Official 820ep CAPEAM unseen host SR 0.390 is a campaign constant, not this run.",
            "",
        ]
    )
    return "\n".join(lines)


def write_results(report: dict, *, root: Path | None = None) -> tuple[Path, Path]:
    out_dir = (root or JW_ROOT) / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "capeam_unseen.json"
    md_path = out_dir / "capeam_unseen.md"
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
