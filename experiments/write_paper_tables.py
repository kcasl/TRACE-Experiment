"""Emit FILM/CAPEAM-style TeX fragments from a closed-loop jsonl root.

    python -m experiments.write_paper_tables --root results/thor_port_v3
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from trace.config import DEFERRAL_COST
from trace.consequence.ff import TERMINAL
from trace.gate import decide, decision_utility, oracle_decision
from trace.types import Belief, CostMatrix, normalize_mode

from .paths import JW_ROOT
from .thor_jsonl import load_jsonl
from .thor_rollout import OUT, TASK_FAMILIES, task_family

SPLITS = (
    ("film", "valid_seen"),
    ("film", "valid_unseen"),
    ("capeam", "valid_seen"),
    ("capeam", "valid_unseen"),
)
SOURCES = ("host", "oracle", "jev", "openai_decisions")


def _event(row: dict, action: str) -> dict:
    items = [item for item in row.get("events") or [] if item.get("action") == action]
    return items[-1] if items else {}


def _events(row: dict, action: str) -> list[dict]:
    return [item for item in row.get("events") or [] if item.get("action") == action]


def _row_task(row: dict) -> str:
    return str(row.get("task") or task_family(row.get("rel") or ""))


def _n_rows(block: dict) -> int:
    for source in SOURCES:
        rows = (block.get("rows") or {}).get(source) or []
        if rows:
            return len(rows)
    return 0


def _load_source(directory: Path, source: str) -> list[dict]:
    hits = sorted(directory.glob(f"{source}_*.jsonl"))
    if not hits:
        return []
    by = {}
    for path in hits:
        for row in load_jsonl(path):
            if row.get("idx") is not None:
                by[int(row["idx"])] = row
    return [by[key] for key in sorted(by)]


def _sr(rows: list[dict]) -> float:
    return sum(bool(r.get("success")) for r in rows) / len(rows) if rows else 0.0


def _select_stats(rows: list[dict]) -> dict:
    n2 = sel = 0
    for row in rows:
        for item in row.get("events") or []:
            n = item.get("n")
            if n is None or int(n) < 2:
                continue
            n2 += 1
            if str(item.get("reason") or "") == "regret_below_deferral":
                sel += 1
    return {"n2": n2, "sel": sel, "ret": n2 - sel}


def _vs_host(host_rows: list[dict], other: list[dict]) -> dict:
    by = {r["idx"]: r for r in host_rows}
    put = gained = lost = 0
    gains = []
    for row in other:
        h = by.get(row["idx"])
        if h is None:
            continue
        if _event(h, "PutObject").get("id") != _event(row, "PutObject").get("id"):
            put += 1
        if row.get("success") and not h.get("success"):
            gained += 1
            pe = _event(row, "PutObject")
            gains.append(
                {
                    "idx": row["idx"],
                    "floor": row.get("floor"),
                    "target": row.get("target"),
                    "parent": row.get("parent"),
                    "task": _row_task(row),
                    "pmax": max(pe["p"][: int(pe["n"])]) if pe.get("p") and pe.get("n") else None,
                    "id": pe.get("id"),
                    "host_id": _event(h, "PutObject").get("id"),
                }
            )
        if h.get("success") and not row.get("success"):
            lost += 1
    return {"put_diff": put, "gained": gained, "lost": lost, "gains": gains}


def _matrix(event: dict) -> CostMatrix | None:
    costs = event.get("C")
    n = event.get("n")
    if not costs or n is None:
        return None
    n = int(n)
    arr = np.asarray(costs, dtype=np.float64)
    if arr.shape != (n, n + 1):
        return None
    return CostMatrix(C=arr, total=arr.copy(), base=np.zeros(n + 1), candidate_ids=tuple(str(i) for i in range(n)), action="PutObject")


def _cr(rows: list[dict], *, rule: str) -> dict | None:
    extras = []
    wrong = 0
    returned = 0
    for row in rows:
        for event in _events(row, "PutObject"):
            n = event.get("n")
            if n is None or int(n) < 2:
                continue
            matrix = _matrix(event)
            p = event.get("p")
            if matrix is None:
                continue
            n = int(n)
            truth = event.get("j_true")
            truth_j = n if truth is None else int(truth)
            reachable = np.ones_like(matrix.C)
            if event.get("reachable"):
                reachable = np.asarray(event["reachable"], dtype=np.float64)
            oracle = oracle_decision(matrix.C, reachable, truth_j, DEFERRAL_COST, TERMINAL)
            u_oracle = decision_utility(oracle, truth_j, matrix.C, reachable, DEFERRAL_COST, TERMINAL)
            if rule == "host_score":
                host_idx = event.get("host_choice")
                if host_idx is None:
                    continue
                scores = [1.0 if i == int(host_idx) else 0.0 for i in range(n)]
                belief = Belief(p_identity=np.ones(n + 1) / (n + 1), source="host")
                decision = decide(belief, matrix, mode="SAME", deferral_cost=DEFERRAL_COST, rule="host_score", scores=scores)
            elif rule == "always_return":
                extras.append(DEFERRAL_COST - u_oracle)
                continue
            elif rule == "trace":
                if not p or len(p) != n + 1:
                    continue
                belief = Belief(p_identity=np.asarray(p, dtype=np.float64), source=str(event.get("source") or "logged"))
                decision = decide(
                    belief,
                    matrix,
                    mode=normalize_mode(event.get("relation") or "SAME"),
                    deferral_cost=DEFERRAL_COST,
                )
            else:
                continue
            utility = decision_utility(decision, truth_j, matrix.C, reachable, DEFERRAL_COST, TERMINAL)
            extras.append(utility - u_oracle)
            returned += int(decision.action == "RETURN")
            if decision.action == "SELECT" and decision.idx != truth_j:
                wrong += 1
    if not extras:
        return None
    n_ev = len(extras)
    return {"n": n_ev, "CR": sum(extras) / n_ev, "wrong": wrong / n_ev, "return": returned / n_ev}


def collect(root: Path) -> dict:
    table = {}
    for host, split in SPLITS:
        directory = root / host / split
        packed = {source: _load_source(directory, source) for source in SOURCES}
        table[f"{host}/{split}"] = {
            "rows": packed,
            "sr": {source: _sr(packed[source]) for source in SOURCES if packed[source]},
            "select": {source: _select_stats(packed[source]) for source in ("jev", "openai_decisions") if packed[source]},
            "vs": {source: _vs_host(packed["host"], packed[source]) for source in ("jev", "openai_decisions") if packed[source] and packed["host"]},
            "cr": {
                "host": _cr(packed["host"], rule="host_score") if packed["host"] else None,
                "always_return": _cr(packed["jev"] or packed["openai_decisions"] or packed["host"], rule="always_return"),
                "jev": _cr(packed["jev"], rule="trace") if packed["jev"] else None,
                "openai": _cr(packed["openai_decisions"], rule="trace") if packed["openai_decisions"] else None,
            },
        }
        packed_block = table[f"{host}/{split}"]
        packed_block["n"] = _n_rows(packed_block)
        packed_block["task"] = {}
        for family in TASK_FAMILIES:
            packed_block["task"][family] = {
                source: _sr([row for row in packed[source] if _row_task(row) == family])
                for source in SOURCES
                if packed[source]
            }
            packed_block["task"][family]["n"] = sum(
                1 for row in (packed["host"] or packed["jev"] or []) if _row_task(row) == family
            )
    return table


def _fmt(sr: float) -> str:
    return f"{sr:.3f}"


def _sr_val(block: dict, source: str) -> float | None:
    return block["sr"].get(source)


def _sr_cell(block: dict, source: str, *, among: tuple[str, ...] | None = None) -> str:
    sr = _sr_val(block, source)
    if sr is None:
        sr = _sr_val(block, "host")
        if sr is None:
            return "---"
    text = _fmt(sr)
    if among:
        nums = [_sr_val(block, name) for name in among]
        nums = [x for x in nums if x is not None]
        if nums and sr >= max(nums) - 1e-12:
            return r"\textbf{" + text + "}"
    return text


def write_tex(table: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    film_s = table["film/valid_seen"]
    film_u = table["film/valid_unseen"]
    cap_s = table["capeam/valid_seen"]
    cap_u = table["capeam/valid_unseen"]

    among = ("host", "oracle", "jev", "openai_decisions")

    def cell(block, source):
        return _sr_cell(block, source, among=among)

    def sel(block, source):
        stats = block["select"].get(source) or {"sel": 0}
        return str(stats["sel"])

    ns, nu, ncs, ncu = _n_rows(film_s), _n_rows(film_u), _n_rows(cap_s), _n_rows(cap_u)
    thor = r"""\begin{table*}[t]
\centering
\caption{Closed-loop THOR~5 port on the full oct21 splits (all seven ALFRED task types; not the official 10-failure / 1000-step protocol).
Unique-min SELECT, planner extra $C$, sink-shell Put penalty, furniture kept under \texttt{SetObjectPoses}.
Off-type Puts are not marked unreachable.
Sel.: TRACE commits among $N\ge 2$ events.}
\label{tab:thor}
\scriptsize
\setlength{\tabcolsep}{3.5pt}
\begin{tabular}{lcccccccc}
\toprule
& \multicolumn{4}{c}{FILM host} & \multicolumn{4}{c}{CAPEAM host} \\
\cmidrule(lr){2-5} \cmidrule(lr){6-9}
& \multicolumn{2}{c}{Seen ($%d$)} & \multicolumn{2}{c}{Unseen ($%d$)} & \multicolumn{2}{c}{Seen ($%d$)} & \multicolumn{2}{c}{Unseen ($%d$)} \\
\cmidrule(lr){2-3} \cmidrule(lr){4-5} \cmidrule(lr){6-7} \cmidrule(lr){8-9}
Method & SR$\uparrow$ & Sel. & SR$\uparrow$ & Sel. & SR$\uparrow$ & Sel. & SR$\uparrow$ & Sel. \\
\midrule
Host (baseline) & %s & --- & %s & --- & %s & --- & %s & --- \\
GT-ID oracle & %s & --- & %s & --- & %s & --- & %s & --- \\
TRACE + Jev & %s & %s & %s & %s & %s & %s & %s & %s \\
TRACE + OpenAI & %s & %s & %s & %s & %s & %s & %s & %s \\
\bottomrule
\end{tabular}
\end{table*}
""" % (
        ns,
        nu,
        ncs,
        ncu,
        cell(film_s, "host"),
        cell(film_u, "host"),
        cell(cap_s, "host"),
        cell(cap_u, "host"),
        cell(film_s, "oracle") or cell(film_s, "host"),
        cell(film_u, "oracle") or cell(film_u, "host"),
        cell(cap_s, "oracle") or cell(cap_s, "host"),
        cell(cap_u, "oracle") or cell(cap_u, "host"),
        cell(film_s, "jev"),
        sel(film_s, "jev"),
        cell(film_u, "jev"),
        sel(film_u, "jev"),
        cell(cap_s, "jev"),
        sel(cap_s, "jev"),
        cell(cap_u, "jev"),
        sel(cap_u, "jev"),
        cell(film_s, "openai_decisions"),
        sel(film_s, "openai_decisions"),
        cell(film_u, "openai_decisions"),
        sel(film_u, "openai_decisions"),
        cell(cap_s, "openai_decisions"),
        sel(cap_s, "openai_decisions"),
        cell(cap_u, "openai_decisions"),
        sel(cap_u, "openai_decisions"),
    )
    (out_dir / "table_thor_sr.tex").write_text(thor.replace("\n", "\n"), encoding="utf-8")

    def signed(gained, lost):
        net = int(gained) - int(lost)
        return f"$+{gained}$ / $-{lost}$ ({net:+d})"

    def vs_line(block, host, split):
        j = block["vs"].get("jev") or {"gained": 0, "lost": 0}
        o = block["vs"].get("openai_decisions") or {"gained": 0, "lost": 0}
        return (
            f"{host} & {split} & {cell(block, 'host')} & {cell(block, 'jev')} & "
            f"{cell(block, 'openai_decisions')} & {signed(j.get('gained', 0), j.get('lost', 0))} & "
            f"{signed(o.get('gained', 0), o.get('lost', 0))} \\\\"
        )

    srdiff = r"""\begin{table}[t]
\centering
\caption{Closed-loop SR vs.\ the same-row host on full oct21.
$+ep$ / $-ep$ are host-fail$\to$TRACE-success and host-success$\to$TRACE-fail; parentheses are the net episode count.}
\label{tab:srdiff}
\scriptsize
\setlength{\tabcolsep}{3pt}
\begin{tabular}{llccccc}
\toprule
Host & Split & Host SR & Jev SR & OpenAI SR & Jev $+/-$ (net) & OpenAI $+/-$ (net) \\
\midrule
%s
%s
%s
%s
\bottomrule
\end{tabular}
\end{table}
""" % (
        vs_line(film_s, "FILM", "seen"),
        vs_line(film_u, "FILM", "unseen"),
        vs_line(cap_s, "CAPEAM", "seen"),
        vs_line(cap_u, "CAPEAM", "unseen"),
    )
    (out_dir / "table_sr_diff.tex").write_text(srdiff, encoding="utf-8")

    select_rows = []
    for host, split in SPLITS:
        block = table[f"{host}/{split}"]
        label = "FILM" if host == "film" else "CAPEAM"
        short = "seen" if split == "valid_seen" else "unseen"
        for source, pretty in (("jev", "Jev"), ("openai_decisions", "OpenAI")):
            stats = block["select"].get(source) or {"n2": 0, "ret": 0, "sel": 0}
            select_rows.append(
                f"{label} & {short} & {pretty} & {stats['n2']} & {stats['ret']} & {stats['sel']} & {cell(block, source)} \\\\"
            )
    select = r"""\begin{table}[t]
\centering
\caption{TRACE commits on the full oct21 THOR~5 port ($N\ge 2$, planner extra $C$, $c_{\mathrm{defer}}{=}4$).
$N{=}1$ uses the host score. A RETURN uses the host score among all listed masks.}
\label{tab:select}
\scriptsize
\setlength{\tabcolsep}{3pt}
\begin{tabular}{lllrrrr}
\toprule
Host & Split & Source & Ev. & Ret. & Sel. & SR \\
\midrule
%s
\bottomrule
\end{tabular}
\end{table}
""" % "\n".join(select_rows)
    (out_dir / "table_select.tex").write_text(select, encoding="utf-8")

    def gain_counts(block, source):
        from collections import Counter

        items = (block["vs"].get(source) or {}).get("gains") or []
        return Counter(
            item.get("task") if item.get("task") in TASK_FAMILIES else "other" for item in items
        )

    gain_lines = []
    family_order = (*TASK_FAMILIES, "other")
    labels = {
        **{
            "pick_and_place_simple": "pick-and-place",
            "pick_and_place_with_movable_recep": "movable recep",
            "pick_two_obj_and_place": "pick-two",
            "look_at_obj_in_light": "look",
            "pick_clean_then_place_in_recep": "clean",
            "pick_heat_then_place_in_recep": "heat",
            "pick_cool_then_place_in_recep": "cool",
        },
        "other": "other",
    }
    for host_name, block in (("FILM seen", film_s), ("FILM unseen", film_u), ("CAPEAM seen", cap_s), ("CAPEAM unseen", cap_u)):
        jev_c = gain_counts(block, "jev")
        oa_c = gain_counts(block, "openai_decisions")
        families = [name for name in family_order if jev_c[name] or oa_c[name]]
        if not families:
            continue
        for i, family in enumerate(families):
            left = host_name if i == 0 else ""
            gain_lines.append(
                f"{left} & {labels.get(family, family)} & {jev_c[family]} & {oa_c[family]} \\\\"
            )
    if not gain_lines:
        gain_lines = [r"\multicolumn{4}{c}{no host-fail $\to$ TRACE-success} \\"]
    gain = r"""\begin{table}[t]
\centering
\caption{Host-fail $\to$ TRACE-success recoveries by ALFRED task type (full oct21 closed loop).
Counts are episodes, not Puts.}
\label{tab:gain}
\scriptsize
\begin{tabular}{llrr}
\toprule
Split & Type & Jev $+ep$ & OpenAI $+ep$ \\
\midrule
%s
\bottomrule
\end{tabular}
\end{table}
""" % "\n".join(gain_lines)
    (out_dir / "table_gain.tex").write_text(gain, encoding="utf-8")

    def cr_cell(block, key):
        stats = block["cr"].get(key)
        if not stats:
            return "---", "---"
        return f"{stats['CR']:.2f}", f"{stats['wrong']:.3f}"

    def cr_row(name, key):
        cells = []
        for block in (film_s, film_u, cap_s, cap_u):
            cr, w = cr_cell(block, key)
            cells.extend([cr, w])
        return name + " & " + " & ".join(cells) + r" \\"

    putcr = r"""\begin{table*}[t]
\centering
\caption{Planner-extra commitment regret on THOR~5 Puts ($N\ge 2$, $c_{\mathrm{defer}}{=}4$) over the full oct21 splits.
$C$ is remaining-plan extra (sink shells unreachable; other classes keep planner cost).
Baseline (host) commits to the host mask. Always-return never commits (CR${=}4$ when the oracle extra is $0$).}
\label{tab:putcr}
\scriptsize
\setlength{\tabcolsep}{3pt}
\begin{tabular}{lcccccccc}
\toprule
& \multicolumn{2}{c}{FILM seen} & \multicolumn{2}{c}{FILM unseen} & \multicolumn{2}{c}{CAPEAM seen} & \multicolumn{2}{c}{CAPEAM unseen} \\
\cmidrule(lr){2-3} \cmidrule(lr){4-5} \cmidrule(lr){6-7} \cmidrule(lr){8-9}
Method & CR$\downarrow$ & Wrong$\downarrow$ & CR$\downarrow$ & Wrong$\downarrow$ & CR$\downarrow$ & Wrong$\downarrow$ & CR$\downarrow$ & Wrong$\downarrow$ \\
\midrule
%s
%s
%s
%s
\bottomrule
\end{tabular}
\end{table*}
""" % (
        cr_row("Baseline (host)", "host"),
        cr_row("Always-return", "always_return"),
        cr_row("TRACE + Jev", "jev"),
        cr_row("TRACE + OpenAI", "openai"),
    )
    (out_dir / "table_put_cr.tex").write_text(putcr, encoding="utf-8")

    def delta_line(block, host, split):
        j = block["vs"].get("jev") or {"gained": 0, "lost": 0}
        o = block["vs"].get("openai_decisions") or {"gained": 0, "lost": 0}
        hs = _sr_val(block, "host") or 0.0
        js = _sr_val(block, "jev") or 0.0
        os_ = _sr_val(block, "openai_decisions") or 0.0
        return (
            f"{host} & {split} & {_fmt(hs)} & {_fmt(js)} & "
            f"{js - hs:+.3f} & {int(j.get('gained', 0) - j.get('lost', 0)):+d} & "
            f"{_fmt(os_)} & {os_ - hs:+.3f} & {int(o.get('gained', 0) - o.get('lost', 0)):+d} \\\\"
        )

    v1v2 = r"""\begin{table}[t]
\centering
\caption{Net closed-loop gain vs.\ the host on the full oct21 splits (unique-min, planner extra $C$, sink shells, no off-type prior).}
\label{tab:v1v2}
\scriptsize
\setlength{\tabcolsep}{3pt}
\begin{tabular}{llccccccc}
\toprule
Host & Split & Host & Jev & $\Delta$SR & net ep & OpenAI & $\Delta$SR & net ep \\
\midrule
%s
%s
%s
%s
\bottomrule
\end{tabular}
\end{table}
""" % (
        delta_line(film_s, "FILM", "seen"),
        delta_line(film_u, "FILM", "unseen"),
        delta_line(cap_s, "CAPEAM", "seen"),
        delta_line(cap_u, "CAPEAM", "unseen"),
    )
    (out_dir / "table_v1v2.tex").write_text(v1v2, encoding="utf-8")

    def task_line(block_s, block_u, family, host_name):
        n_s = (block_s.get("task") or {}).get(family, {}).get("n", 0)
        n_u = (block_u.get("task") or {}).get(family, {}).get("n", 0)
        if n_s + n_u == 0:
            return None

        def tcell(block, source):
            stats = (block.get("task") or {}).get(family) or {}
            sr = stats.get(source)
            return "---" if sr is None else f"{sr:.3f}"

        return (
            f"{host_name} & {labels[family]} & {n_s} & {tcell(block_s, 'host')} & "
            f"{tcell(block_s, 'jev')} & {tcell(block_s, 'openai_decisions')} & "
            f"{n_u} & {tcell(block_u, 'host')} & {tcell(block_u, 'jev')} & "
            f"{tcell(block_u, 'openai_decisions')} \\\\"
        )

    task_lines = []
    for host_i, (host_name, block_s, block_u) in enumerate(
        (("FILM", film_s, film_u), ("CAPEAM", cap_s, cap_u))
    ):
        if host_i:
            task_lines.append(r"\midrule")
        for family in TASK_FAMILIES:
            line = task_line(block_s, block_u, family, host_name)
            if line:
                task_lines.append(line)
    if not task_lines:
        task_lines = [r"\multicolumn{10}{c}{no per-type rows} \\"]
    task = r"""\begin{table*}[t]
\centering
\caption{Closed-loop SR by ALFRED task type on the full oct21 splits (THOR~5 port).}
\label{tab:task}
\scriptsize
\setlength{\tabcolsep}{3pt}
\begin{tabular}{llcccccccc}
\toprule
Host & Type & $n$ seen & Host & Jev & OpenAI & $n$ unseen & Host & Jev & OpenAI \\
\midrule
%s
\bottomrule
\end{tabular}
\end{table*}
""" % "\n".join(task_lines)
    (out_dir / "table_task.tex").write_text(task, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="", help="closed-loop jsonl root")
    parser.add_argument("--tex-dir", default="", help="directory for .tex fragments")
    args = parser.parse_args(argv)
    root = Path(args.root) if args.root else JW_ROOT / "results" / "thor_port_v6"
    if not root.exists():
        root = Path(args.root) if args.root else OUT
    tex_dir = Path(args.tex_dir) if args.tex_dir else JW_ROOT / "paper" / "tables"
    table = collect(root)
    summary = {
        name: {
            "n": block.get("n", 0),
            "sr": block["sr"],
            "select": block["select"],
            "vs": {k: {kk: vv for kk, vv in v.items() if kk != "gains"} for k, v in block["vs"].items()},
            "cr": block["cr"],
            "task": block.get("task", {}),
        }
        for name, block in table.items()
    }
    out_json = root / "paper_tables.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_tex(table, tex_dir)
    print(f"wrote {tex_dir} and {out_json}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
