"""Paper figures: mask strip and result strip. Always Baseline / Jev / OpenAI.

    python -m experiments.dump_paper_compare
"""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from experiments.capeam_mrcnn import CapeamMaskRCNN
from experiments.mask_views import PALETTE
from experiments.thor_rollout import OracleIdentity, load_split, load_traj, run_episode

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "paper_compare"
JSONL = ROOT / "results" / "thor_port_v2"

SOURCES = (
    {"key": "baseline", "label": "Baseline", "jsonl": "host"},
    {"key": "jev", "label": "Jev", "jsonl": "jev"},
    {"key": "openai", "label": "OpenAI", "jsonl": "openai_decisions"},
)

CASES = (
    {
        "stem": "01_fp422_handtowel",
        "host": "film",
        "split": "valid_seen",
        "idx": 108,
        "focus": "PutObject",
        "title": "FloorPlan422  HandTowel → SinkBasin",
    },
    {
        "stem": "02_fp7_fridge",
        "host": "film",
        "split": "valid_seen",
        "idx": 16,
        "focus": "PutObject",
        "title": "FloorPlan7  WineBottle → Fridge",
    },
    {
        "stem": "03_fp4_fridge",
        "host": "film",
        "split": "valid_seen",
        "idx": 124,
        "focus": "PutObject",
        "title": "FloorPlan4  Egg → Fridge",
    },
    {
        "stem": "04_fp406_toiletpaper",
        "host": "capeam",
        "split": "valid_seen",
        "idx": 112,
        "focus": "PickupObject",
        "title": "FloorPlan406  ToiletPaper pickup",
    },
    {
        "stem": "05_fp406_plunger",
        "host": "film",
        "split": "valid_seen",
        "idx": 128,
        "focus": "PutObject",
        "title": "FloorPlan406  Plunger → Cabinet",
    },
    {
        "stem": "06_fp10_saltshaker",
        "host": "capeam",
        "split": "valid_unseen",
        "idx": 36,
        "focus": "PutObject",
        "title": "FloorPlan10 unseen  SaltShaker → Cabinet",
    },
    {
        "stem": "07_fp219_vase",
        "host": "film",
        "split": "valid_unseen",
        "idx": 21,
        "focus": "PutObject",
        "title": "FloorPlan219 unseen  Vase → Safe",
    },
    {
        "stem": "08_fp7_capeam_fridge",
        "host": "capeam",
        "split": "valid_seen",
        "idx": 15,
        "focus": "PutObject",
        "title": "FloorPlan7  WineBottle → Fridge (CAPEAM host)",
    },
)


def jsonl_row(host: str, split: str, source: str, idx: int) -> dict | None:
    folder = JSONL / host / split
    if not folder.is_dir():
        return None
    for path in folder.glob(f"{source}_*.jsonl"):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if int(row.get("idx", -1)) == idx:
                return row
    return None


def action_event(row: dict | None, action: str) -> dict:
    if not row:
        return {}
    ranked = []
    for item in row.get("events") or []:
        if item.get("action") != action:
            continue
        ranked.append(item)
    if not ranked:
        return {}
    with_p = [item for item in ranked if item.get("p")]
    return with_p[-1] if with_p else ranked[-1]


def force_from_logged(row: dict | None, action: str) -> dict[str, str] | None:
    item = action_event(row, action)
    if not item:
        return None
    reason = str(item.get("reason") or "")
    if item.get("defer") or reason.startswith("defer") or reason in ("host_score", "oracle_match"):
        return None
    oid = str(item.get("id") or "")
    if not oid:
        return None
    if item.get("p") and not item.get("defer"):
        return {action: oid}
    return None


def letters(n: int) -> list[str]:
    return [chr(65 + i) if i < 26 else f"C{i}" for i in range(n)]


def _font(size: int):
    for name in ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", "DejaVuSans-Bold.ttf", "arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def decision_line(item: dict, ids: list[str], host_id: str, source: str) -> str:
    chosen = str(item.get("id") or "")
    letter = chr(65 + ids.index(chosen)) if chosen in ids else "?"
    if source == "baseline":
        return f"host  {letter}"
    if item.get("defer") or str(item.get("reason") or "").startswith("defer"):
        host_letter = chr(65 + ids.index(host_id)) if host_id in ids else "?"
        return f"RETURN → host {host_letter}"
    p = item.get("p") or []
    pmax = max(p) if p else 0.0
    return f"SELECT {letter}   p={pmax:.2f}"


def render_p_bars(
    p: list[float] | None,
    names: list[str],
    selected: int | None,
    width: int,
    *,
    rows: int,
    host_letter: str = "",
) -> Image.Image:
    row_h = 22
    header = 4
    height = header + row_h * max(rows, 1) + 6
    image = Image.new("RGB", (width, height), (22, 22, 22))
    draw = ImageDraw.Draw(image)
    font = _font(14)
    if not p:
        draw.text((8, 8), f"host nearest   {host_letter}".strip(), fill=(220, 220, 220), font=font)
        return image
    values = list(p)
    for i, name in enumerate(names):
        y = header + i * row_h
        color = PALETTE[i % len(PALETTE)] if name != "none" else (180, 180, 180)
        value = values[i] if i < len(values) else 0.0
        bar_w = int((width - 92) * max(0.0, min(1.0, float(value))))
        draw.rectangle([70, y + 4, 70 + max(bar_w, 1), y + 18], fill=tuple(int(v) for v in color))
        tag = f"{name}"
        if selected == i:
            tag += "*"
        draw.text((8, y + 4), tag, fill=(255, 255, 255), font=font)
        draw.text((width - 52, y + 4), f"{value:.2f}", fill=(220, 220, 220), font=font)
    return image


def mask_card(
    overlay: Image.Image,
    title: str,
    subtitle: str,
    p: list[float] | None,
    names: list[str],
    selected: int | None,
    *,
    rows: int,
    host_letter: str = "",
) -> Image.Image:
    title_h = 48
    bars = render_p_bars(p, names, selected, overlay.width, rows=rows, host_letter=host_letter)
    canvas = Image.new("RGB", (overlay.width, title_h + overlay.height + bars.height), (12, 12, 12))
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 4), title, fill=(255, 255, 255), font=_font(20))
    draw.text((8, 26), subtitle, fill=(210, 210, 210), font=_font(14))
    canvas.paste(overlay, (0, title_h))
    canvas.paste(bars, (0, title_h + overlay.height))
    return canvas


def result_card(frame: Image.Image, title: str, subtitle: str) -> Image.Image:
    title_h = 48
    canvas = Image.new("RGB", (frame.width, frame.height + title_h), (12, 12, 12))
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 4), title, fill=(255, 255, 255), font=_font(20))
    draw.text((8, 26), subtitle, fill=(210, 210, 210), font=_font(14))
    canvas.paste(frame, (0, title_h))
    return canvas


def choose_overlay(dump_dir: Path, focus: str) -> Path | None:
    near_json = dump_dir / f"{focus.lower()}_near_masks.json"
    near_img = dump_dir / f"{focus.lower()}_near_instances.png"
    choice_img = dump_dir / f"{focus.lower()}_instances.png"
    pixels = 0
    if near_json.exists():
        data = json.loads(near_json.read_text(encoding="utf-8"))
        selected = data.get("selected")
        for item in data.get("masks") or []:
            if item.get("id") == selected:
                pixels = int(item.get("pixels") or 0)
                break
    area = 600 * 600
    if near_img.exists() and 800 <= pixels <= 0.40 * area:
        return near_img
    if choice_img.exists():
        return choice_img
    if near_img.exists():
        return near_img
    fallback = dump_dir / f"{focus.lower()}_choice_overlay.png"
    return fallback if fallback.exists() else None


def strip(images: list[Image.Image], out: Path) -> None:
    height = max(im.height for im in images)
    resized = []
    for image in images:
        if image.height != height:
            width = int(image.width * height / image.height)
            image = image.resize((width, height), Image.NEAREST)
        resized.append(image)
    width = sum(im.width for im in resized) + 8 * (len(resized) - 1)
    canvas = Image.new("RGB", (width, height), (8, 8, 8))
    x = 0
    for image in resized:
        canvas.paste(image, (x, 0))
        x += image.width + 8
    out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out)


def main() -> None:
    import argparse

    from ai2thor.controller import Controller
    from ai2thor.platform import CloudRendering

    parser = argparse.ArgumentParser()
    parser.add_argument("--only", default="")
    parser.add_argument("--compose-only", action="store_true")
    args = parser.parse_args()
    want = {item.strip() for item in args.only.split(",") if item.strip()}
    cases = [
        case
        for case in CASES
        if not want or case["stem"].split("_")[0] in want or case["stem"] in want
    ]

    splits = {
        "valid_seen": load_split("valid_seen"),
        "valid_unseen": load_split("valid_unseen"),
    }
    controller = None
    oracle = OracleIdentity()
    detectors: dict[str, CapeamMaskRCNN | None] = {}
    if not args.compose_only:
        controller = Controller(
            platform=CloudRendering,
            scene="FloorPlan1",
            width=600,
            height=600,
            gridSize=0.25,
            renderInstanceSegmentation=True,
        )
    mask_dir = OUT / "masks"
    result_dir = OUT / "results"
    try:
        for case in cases:
            host = case["host"]
            traj = None if args.compose_only else load_traj(splits[case["split"]][case["idx"]]["path"])
            if host == "capeam" and host not in detectors and not args.compose_only:
                detectors[host] = CapeamMaskRCNN()
            detector = detectors.get(host)
            focus = case["focus"]
            host_row = jsonl_row(host, case["split"], "host", case["idx"])
            host_id = str(action_event(host_row, focus).get("id") or "")
            bar_rows = 1
            for spec in SOURCES:
                logged = jsonl_row(host, case["split"], spec["jsonl"], case["idx"])
                p_len = len(action_event(logged, focus).get("p") or [])
                bar_rows = max(bar_rows, p_len, 1)
            mask_cards = []
            result_cards = []
            for spec in SOURCES:
                dump_dir = OUT / "raw" / case["stem"] / spec["key"]
                dump_dir.mkdir(parents=True, exist_ok=True)
                logged = jsonl_row(host, case["split"], spec["jsonl"], case["idx"])
                force = None if spec["key"] == "baseline" else force_from_logged(logged, focus)
                if args.compose_only:
                    row = json.loads((dump_dir / "episode.json").read_text(encoding="utf-8")) if (dump_dir / "episode.json").exists() else {}
                else:
                    row = run_episode(
                        controller,
                        traj,
                        "host",
                        None,
                        oracle,
                        host=host,
                        detector=detector,
                        dump_dir=dump_dir,
                        force_ids=force,
                    )
                    row["idx"] = case["idx"]
                    row["compare"] = spec["key"]
                    (dump_dir / "episode.json").write_text(json.dumps(row, indent=2), encoding="utf-8")
                chosen = str(action_event(row, focus).get("id") or "")
                ids = list(json.loads((dump_dir / f"{focus.lower()}_masks.json").read_text(encoding="utf-8")).get("ids") or []) if (dump_dir / f"{focus.lower()}_masks.json").exists() else []
                overlay_path = choose_overlay(dump_dir, focus)
                overlay = Image.open(overlay_path).convert("RGB") if overlay_path and overlay_path.exists() else Image.new("RGB", (600, 600), (0, 0, 0))
                logged_ev = action_event(logged, focus)
                p_list = list(logged_ev.get("p") or [])
                p = p_list if spec["key"] != "baseline" and p_list else None
                n_names = max(len(ids), max(0, len(p_list) - 1))
                names = letters(n_names)
                if p_list and len(p_list) == n_names + 1:
                    names = names + ["none"]
                selected = ids.index(chosen) if chosen in ids else None
                subtitle = decision_line(logged_ev if spec["key"] != "baseline" else {"id": chosen}, ids or names, host_id, spec["key"])
                host_letter = chr(65 + ids.index(host_id)) if host_id in ids else ""
                mask_cards.append(
                    mask_card(
                        overlay,
                        spec["label"],
                        subtitle,
                        p,
                        names,
                        selected,
                        rows=bar_rows,
                        host_letter=host_letter,
                    )
                )
                after = dump_dir / ("pickup_before_act.png" if focus == "PickupObject" else "put_after.png")
                status = "success" if row.get("success") else "fail"
                frame = Image.open(after).convert("RGB") if after.exists() else overlay
                result_cards.append(result_card(frame, spec["label"], f"{subtitle}   →  {status}"))
                print(
                    f"{case['stem']} {spec['key']:8s} ok={int(bool(row.get('success')))} "
                    f"{focus}={chosen} force={force}",
                    flush=True,
                )
            strip(mask_cards, mask_dir / f"{case['stem']}.png")
            strip(result_cards, result_dir / f"{case['stem']}.png")
            print(f"wrote masks/{case['stem']}.png  results/{case['stem']}.png", flush=True)
    finally:
        if controller is not None:
            try:
                controller.stop()
            except Exception:
                pass
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
