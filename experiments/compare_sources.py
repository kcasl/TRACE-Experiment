"""Side-by-side comparison of the Jev and OpenAI Decisions experiments.

    python -m experiments.compare_sources
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
WORLDS = ("same_room_drawers", "far_fridge", "split_fridges")


def compare(jev: dict, openai: dict) -> dict:
    cases = jev.get("cases") or openai.get("cases") or []
    rows = []
    for case in cases:
        left = _find(jev, case, "jev")
        right = _find(openai, case, "openai_decisions")
        row = {"case": case, "truth": (left or right or {}).get("truth")}
        row["jev"] = _slice(left)
        row["openai_decisions"] = _slice(right)
        if left and right and left.get("ok") and right.get("ok"):
            p_jev = np.asarray(left["p_identity"], dtype=np.float64)
            p_openai = np.asarray(right["p_identity"], dtype=np.float64)
            row["top1_agree"] = int(left["top"] == right["top"])
            row["both_correct"] = int(left["identity_correct"] and right["identity_correct"])
            row["l1"] = round(float(np.abs(p_jev - p_openai).sum()), 4)
            row["trace_agree"] = {
                world: int(
                    (left.get("decisions") or {}).get(world, {}).get("trace")
                    == (right.get("decisions") or {}).get(world, {}).get("trace")
                )
                for world in WORLDS
                if world in (left.get("decisions") or {}) and world in (right.get("decisions") or {})
            }
        else:
            row["top1_agree"] = None
            row["both_correct"] = None
            row["l1"] = None
            row["trace_agree"] = {}
        rows.append(row)
    graded = [row for row in rows if row["l1"] is not None]
    summary = {
        "n": len(graded),
        "jev_accuracy": (jev.get("identity_accuracy") or {}).get("jev"),
        "openai_accuracy": (openai.get("identity_accuracy") or {}).get("openai_decisions"),
        "top1_agree": None if not graded else round(sum(row["top1_agree"] for row in graded) / len(graded), 4),
        "mean_l1": None if not graded else round(sum(row["l1"] for row in graded) / len(graded), 4),
        "trace_agree": {
            world: None
            if not graded
            else round(sum(row["trace_agree"].get(world, 0) for row in graded) / len(graded), 4)
            for world in WORLDS
        },
    }
    return {"summary": summary, "rows": rows}


def render_markdown(payload: dict) -> str:
    summary = payload["summary"]
    lines = [
        "# Jev vs OpenAI Decisions",
        "",
        "Same identity cases, same TRACE cost worlds. Only the belief source changes.",
        "",
        f"- Jev identity accuracy: {summary['jev_accuracy']}",
        f"- OpenAI Decisions identity accuracy: {summary['openai_accuracy']}",
        f"- Top-1 agree: {summary['top1_agree']}",
        f"- Mean L1 between p_identity: {summary['mean_l1']}",
        "- TRACE action agree: "
        + ", ".join(f"{world} {value}" for world, value in (summary.get("trace_agree") or {}).items()),
        "",
        "| case | truth | Jev p | OpenAI p | Jev top | OpenAI top | L1 | cheap TRACE | expensive TRACE | both-far TRACE |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for row in payload["rows"]:
        jev = row.get("jev") or {}
        openai = row.get("openai_decisions") or {}
        lines.append(
            "| {case} | {truth} | {jev_p} | {openai_p} | {jev_top} | {openai_top} | {l1} | {cheap} | {far} | {split} |".format(
                case=row["case"],
                truth=row.get("truth"),
                jev_p=_fmt_p(jev.get("p_identity")),
                openai_p=_fmt_p(openai.get("p_identity")),
                jev_top=jev.get("top"),
                openai_top=openai.get("top"),
                l1=row.get("l1"),
                cheap=_pair_actions(row, "same_room_drawers"),
                far=_pair_actions(row, "far_fridge"),
                split=_pair_actions(row, "split_fridges"),
            )
        )
    return "\n".join(lines) + "\n"


def run(*, jev_path: Path | None = None, openai_path: Path | None = None) -> dict:
    jev = _load(jev_path or ROOT / "results" / "jev" / "live.json")
    openai = _load(openai_path or ROOT / "results" / "openai_decisions" / "live.json")
    payload = compare(jev, openai)
    out_json = ROOT / "results" / "source_compare.json"
    out_md = ROOT / "results" / "source_compare.md"
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    out_md.write_text(render_markdown(payload), encoding="utf-8")
    print(out_md.read_text(encoding="utf-8"))
    print(f"wrote {out_json}")
    print(f"wrote {out_md}")
    return payload


def main() -> None:
    run()


def _load(path: Path) -> dict:
    if not path.is_file():
        return {"cases": [], "rows": [], "identity_accuracy": {}}
    return json.loads(path.read_text(encoding="utf-8"))


def _find(report: dict, case: str, source: str) -> dict | None:
    for row in report.get("rows") or []:
        if row.get("case") == case and row.get("source") == source:
            return row
    return None


def _slice(row: dict | None) -> dict | None:
    if row is None:
        return None
    return {
        "ok": row.get("ok"),
        "skipped": row.get("skipped"),
        "error": row.get("error"),
        "p_identity": row.get("p_identity"),
        "top": row.get("top"),
        "identity_correct": row.get("identity_correct"),
        "decisions": row.get("decisions"),
    }


def _fmt_p(values) -> str:
    if not values:
        return ""
    return "[" + ", ".join(f"{float(value):.2f}" for value in values) + "]"


def _pair_actions(row: dict, world: str) -> str:
    jev = ((row.get("jev") or {}).get("decisions") or {}).get(world, {}).get("trace")
    openai = ((row.get("openai_decisions") or {}).get("decisions") or {}).get(world, {}).get("trace")
    if jev is None and openai is None:
        return ""
    if jev == openai:
        return str(jev)
    return f"{jev} vs {openai}"


if __name__ == "__main__":
    main()
