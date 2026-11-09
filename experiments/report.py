"""Render the result artifacts as one Markdown report."""

from __future__ import annotations


def render(
    offline: dict,
    scenes: dict,
    live: dict | None,
    *,
    tests_passed: bool | None,
    replay: dict | None = None,
    official_sr: dict | None = None,
    source_compare: dict | None = None,
    live_openai: dict | None = None,
) -> str:
    lines: list[str] = ["# TRACE results", ""]
    if tests_passed is not None:
        lines += [f"Unit tests: {'passed' if tests_passed else 'FAILED'}.", ""]

    lines += _offline_section(offline)
    lines += _scene_section(scenes)
    if live is not None:
        heading = "Jev identity experiment" if live_openai is not None else "Identity beliefs from decision models"
        lines += _live_section(live, heading=heading)
    if live_openai is not None:
        lines += _live_section(live_openai, heading="OpenAI Decisions identity experiment")
    if source_compare is not None:
        lines += _compare_section(source_compare)
    if official_sr is not None:
        lines += _sr_section(official_sr)
    if replay is not None:
        lines += _replay_section(replay)
    return "\n".join(lines) + "\n"


def _offline_section(offline: dict) -> list[str]:
    belief = ", ".join(f"{value:g}" for value in offline["belief"])
    lines = [
        "## Decision rules on the same belief",
        "",
        f"Belief p_identity = [{belief}], deferral cost = {offline['deferral_cost']:g}.",
        "Realized expected extra cost, lower is better.",
        "",
        "| rule | cheap world | expensive world | mean |",
        "|---|---|---|---|",
    ]
    by_rule: dict[str, dict[str, float]] = {}
    for row in offline["rows"]:
        by_rule.setdefault(row["rule"], {})[row["world"]] = row["realized_expected_cost"]
    for rule, mean in offline["mean_realized_cost"].items():
        cheap = by_rule[rule].get("cheap")
        expensive = by_rule[rule].get("expensive")
        lines.append(f"| {rule} | {cheap:g} | {expensive:g} | {mean:g} |")
    lines.append("")
    return lines


def _scene_section(scenes: dict) -> list[str]:
    belief = ", ".join(f"{value:g}" for value in scenes["belief"])
    lines = [
        "## Symbolic-planner regret by scene",
        "",
        f"Belief p_identity = [{belief}], deferral cost = {scenes['deferral_cost']:g}.",
        "",
        "| scene | action | regret per mask | TRACE |",
        "|---|---|---|---|",
    ]
    for scene in scenes["scenes"]:
        regret = ", ".join(f"{value:g}" for value in scene["expected_regret"])
        verdict = scene["trace_action"]
        if verdict == "SELECT":
            verdict += f" {scene['argmin_regret']}"
        lines.append(f"| {scene['scene']} | {scene['action']} | [{regret}] | {verdict} |")
    lines.append("")
    return lines


def _live_section(live: dict, heading: str = "Identity beliefs from decision models") -> list[str]:
    lines = [f"## {heading}", ""]
    accuracy = live.get("identity_accuracy", {})
    if accuracy:
        pairs = ", ".join(f"{source} {value:.2f}" for source, value in accuracy.items())
        lines += [f"Top-1 identity accuracy over graded cases: {pairs}.", ""]
    lines += [
        "| case | source | p_identity | top | correct |",
        "|---|---|---|---|---|",
    ]
    for row in live["rows"]:
        if not row.get("ok"):
            status = row.get("skipped") or row.get("error", "error")
            lines.append(f"| {row['case']} | {row['source']} | _{status}_ |  |  |")
            continue
        probabilities = ", ".join(f"{value:g}" for value in row["p_identity"])
        correct = "yes" if row["identity_correct"] else "no"
        lines.append(f"| {row['case']} | {row['source']} | [{probabilities}] | {row['top']} | {correct} |")
    lines.append("")
    return lines


def _compare_section(payload: dict) -> list[str]:
    summary = payload.get("summary") or {}
    lines = [
        "## Jev vs OpenAI Decisions",
        "",
        f"Jev accuracy {summary.get('jev_accuracy')}, OpenAI accuracy {summary.get('openai_accuracy')}, "
        f"top-1 agree {summary.get('top1_agree')}, mean L1 {summary.get('mean_l1')}.",
        "",
        "| case | Jev p | OpenAI p | L1 | cheap TRACE | expensive TRACE |",
        "|---|---|---|---|---|---|",
    ]
    for row in payload.get("rows") or []:
        jev = row.get("jev") or {}
        openai = row.get("openai_decisions") or {}
        jev_p = ", ".join(f"{value:g}" for value in jev.get("p_identity") or [])
        openai_p = ", ".join(f"{value:g}" for value in openai.get("p_identity") or [])
        cheap = ((jev.get("decisions") or {}).get("same_room_drawers") or {}).get("trace")
        far = ((jev.get("decisions") or {}).get("far_fridge") or {}).get("trace")
        cheap_o = ((openai.get("decisions") or {}).get("same_room_drawers") or {}).get("trace")
        far_o = ((openai.get("decisions") or {}).get("far_fridge") or {}).get("trace")
        lines.append(
            f"| {row['case']} | [{jev_p}] | [{openai_p}] | {row.get('l1')} | "
            f"{cheap} vs {cheap_o} | {far} vs {far_o} |"
        )
    lines.append("")
    return lines


def _sr_section(official: dict) -> list[str]:
    lines = [
        "## Official 10-failure SR (FILM / CAPEAM, 820 ep)",
        "",
        f"Protocol A: {official.get('max_fails', 10)} counted API failures, {official.get('max_steps', 1000)} steps.",
        "",
        "| host | split | selector | n | SR | GC | steps | fails |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for row in official.get("rows") or []:
        lines.append(
            f"| {row['host']} | {row['split']} | {row['selector']} | {row['n']} | "
            f"{row['sr']:.3f} | {row['gc']:.3f} | {row['steps']} | {row['fails']} |"
        )
    lines.append("")
    return lines


def _replay_section(replay: dict) -> list[str]:
    lines = [
        "## Mask-log replay (Metric-FF critic, DINOv2 / Jev / oracle)",
        "",
        f"Select-eligible events N≥{replay.get('min_n', 2)}, deferral cost {replay.get('deferral_cost', 4):g}. "
        "C(i,j) is the precomputed FF extra-cost matrix; beliefs are the logged identity distributions.",
        "",
        "| split | source | rule | n | CR | wrong | return | unrec |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for split, payload in (replay.get("splits") or {}).items():
        table = payload.get("table") or {}
        if payload.get("error"):
            lines.append(f"| {split} |  |  |  | _{payload['error']}_ |  |  |  |")
            continue
        for key, row in table.items():
            source, rule = key.split("|", 1)
            lines.append(
                f"| {split} | {source} | {rule} | {row['n']} | "
                f"{row['commitment_regret']:.3f} | {row['wrong_instance_rate']:.3f} | "
                f"{row['return_rate']:.3f} | {row['unrecoverable_rate']:.3f} |"
            )
    lines.append("")
    return lines
