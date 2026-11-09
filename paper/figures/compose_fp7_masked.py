"""Rebuild FloorPlan7 Put→Fridge panels with the fridge instance mask visible."""
from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
OVERLAY = ROOT / "figures" / "fp7_fridge_masked" / "overlay.png"
HANDTOWEL = ROOT / "tex_figures" / "04_fig_qual" / "01_fp422_handtowel.png"
OUT_DIR = ROOT / "figures" / "fp7_fridge_masked"

PALETTE = [
    (255, 64, 64),
    (64, 200, 64),
    (64, 128, 255),
    (255, 200, 0),
    (255, 64, 255),
    (0, 220, 220),
    (255, 140, 0),
    (160, 100, 255),
]
JEV_P = [0.96, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.04]
OAI_P = [0.97, 0.01, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.02]
NAMES = list("ABCDEFGH") + ["none"]


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    names = []
    windir = Path(r"C:\Windows\Fonts")
    if bold:
        names += [windir / "arialbd.ttf", windir / "segoeuib.ttf", windir / "calibrib.ttf"]
    names += [windir / "arial.ttf", windir / "segoeui.ttf", windir / "calibri.ttf", Path("arial.ttf")]
    for path in names:
        try:
            return ImageFont.truetype(str(path), size)
        except OSError:
            continue
    return ImageFont.load_default()


def dark_bars(p: list[float] | None, width: int, host_letter: str = "A") -> Image.Image:
    row_h = 22
    f = font(14)
    if not p:
        image = Image.new("RGB", (width, 4 + row_h + 6), (22, 22, 22))
        draw = ImageDraw.Draw(image)
        draw.text((8, 8), f"host nearest   {host_letter}", fill=(220, 220, 220), font=f)
        return image
    image = Image.new("RGB", (width, 4 + row_h * len(NAMES) + 6), (22, 22, 22))
    draw = ImageDraw.Draw(image)
    for i, name in enumerate(NAMES):
        y = 4 + i * row_h
        color = PALETTE[i % len(PALETTE)] if name != "none" else (180, 180, 180)
        value = p[i] if i < len(p) else 0.0
        bar_w = int((width - 92) * max(0.0, min(1.0, float(value))))
        draw.rectangle([70, y + 4, 70 + max(bar_w, 1), y + 18], fill=color)
        tag = f"{name}*" if i == 0 else name
        draw.text((8, y + 4), tag, fill=(255, 255, 255), font=f)
        draw.text((width - 52, y + 4), f"{value:.2f}", fill=(220, 220, 220), font=f)
    return image


def dark_card(overlay: Image.Image, title: str, subtitle: str, p: list[float] | None) -> Image.Image:
    bars = dark_bars(p, overlay.width)
    canvas = Image.new("RGB", (overlay.width, 48 + overlay.height + bars.height), (12, 12, 12))
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 4), title, fill=(255, 255, 255), font=font(20, bold=True))
    draw.text((8, 26), subtitle, fill=(210, 210, 210), font=font(14))
    canvas.paste(overlay, (0, 48))
    canvas.paste(bars, (0, 48 + overlay.height))
    return canvas


def strip(cards: list[Image.Image]) -> Image.Image:
    h = max(im.height for im in cards)
    width = sum(im.width for im in cards) + 8 * (len(cards) - 1)
    canvas = Image.new("RGB", (width, h), (8, 8, 8))
    x = 0
    for im in cards:
        canvas.paste(im, (x, 0))
        x += im.width + 8
    return canvas


def light_bars(p: list[float], width: int) -> Image.Image:
    # A, B, C–H, none  (matches the submitted panel)
    groups = [
        ("A", p[0], PALETTE[0]),
        ("B", p[1], PALETTE[1]),
        ("C–H", sum(p[2:8]), (140, 140, 150)),
        ("none", p[8], (180, 180, 180)),
    ]
    row_h = 22
    image = Image.new("RGB", (width, 8 + row_h * len(groups)), (255, 255, 255))
    draw = ImageDraw.Draw(image)
    f = font(15)
    for i, (name, value, color) in enumerate(groups):
        y = 6 + i * row_h
        bar_w = int((width - 100) * max(0.0, min(1.0, float(value))))
        draw.rectangle([56, y + 3, 56 + max(bar_w, 1), y + 16], fill=color)
        draw.text((8, y + 1), name, fill=(30, 30, 30), font=f)
        draw.text((width - 42, y + 1), f"{value:.2f}", fill=(40, 40, 40), font=f)
    return image


def rounded_rect(draw: ImageDraw.ImageDraw, box, fill, radius: int) -> None:
    draw.rounded_rectangle(box, radius=radius, fill=fill)


def light_panel(overlay: Image.Image) -> Image.Image:
    photo = overlay.resize((420, 420), Image.Resampling.LANCZOS)
    card_w, pad = 448, 14
    title_h, photo_y = 54, 62
    bars_h = 8 + 22 * 4
    body_h = 78
    card_h = title_h + photo.height + body_h + bars_h + 18
    columns = [
        {
            "title": "Host (nearest mask)",
            "decision": "executes A",
            "note": "no belief: host rule only",
            "dcolor": (25, 25, 25),
            "p": None,
        },
        {
            "title": "Jev",
            "decision": "RETURN → host A",
            "note": "",
            "dcolor": (196, 32, 32),
            "p": JEV_P,
        },
        {
            "title": "OpenAI Decisions",
            "decision": "SELECT A",
            "note": "",
            "dcolor": (32, 88, 196),
            "p": OAI_P,
        },
    ]
    banner_h = 44
    gap = 16
    width = pad + 3 * card_w + 2 * gap + pad
    height = pad + banner_h + 12 + card_h + pad
    canvas = Image.new("RGB", (width, height), (236, 238, 242))
    draw = ImageDraw.Draw(canvas)
    rounded_rect(draw, (pad, pad, width - pad, pad + banner_h), (70, 130, 210), 10)
    draw.text(
        (pad + 16, pad + 10),
        "(b) FloorPlan7, Put  →  Fridge  (8 candidates)",
        fill=(255, 255, 255),
        font=font(22, bold=True),
    )
    x = pad
    y = pad + banner_h + 12
    for col in columns:
        rounded_rect(draw, (x, y, x + card_w, y + card_h), (255, 255, 255), 12)
        draw.text((x + 14, y + 12), col["title"], fill=(20, 20, 20), font=font(20, bold=True))
        canvas.paste(photo, (x + 14, y + photo_y))
        dy = y + photo_y + photo.height + 10
        draw.text((x + 14, dy), col["decision"], fill=col["dcolor"], font=font(20, bold=True))
        if col["note"]:
            draw.text((x + 14, dy + 28), col["note"], fill=(90, 90, 90), font=font(15))
        if col["p"]:
            bars = light_bars(col["p"], card_w - 28)
            canvas.paste(bars, (x + 14, dy + 34))
        x += card_w + gap
    return canvas


def main() -> None:
    overlay = Image.open(OVERLAY).convert("RGB")
    dark = strip(
        [
            dark_card(overlay, "Baseline", "host  A", None),
            dark_card(overlay, "Jev", "RETURN → host A", JEV_P),
            dark_card(overlay, "OpenAI", "SELECT A   p=0.97", OAI_P),
        ]
    )
    light = light_panel(overlay)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    dark_path = OUT_DIR / "02_fp7_fridge_dark.png"
    light_path = OUT_DIR / "02_fp7_fridge_light.png"
    dark.save(dark_path)
    light.save(light_path)
    (ROOT / "figs" / "fp7_fridge.png").parent.mkdir(parents=True, exist_ok=True)
    dark.save(ROOT / "figs" / "fp7_fridge.png")

    targets = [
        ROOT / "tex_figures" / "04_fig_qual" / "02_fp7_fridge.png",
        ROOT / "figures" / "qual_02_fp7_fridge.png",
        ROOT / "figures" / "compare" / "masks" / "02_fp7_fridge.png",
        ROOT / "paper_figures_pack" / "paper_figures_pack" / "fig_qual" / "masks" / "02_fp7_fridge.png",
        ROOT / "paper_figures" / "paper_figures_pack" / "fig_qual" / "masks" / "02_fp7_fridge.png",
    ]
    for path in targets:
        path.parent.mkdir(parents=True, exist_ok=True)
        dark.save(path)

    figs = ROOT / "figs"
    figs.mkdir(parents=True, exist_ok=True)
    if HANDTOWEL.exists():
        top = Image.open(HANDTOWEL).convert("RGB")
        bottom = dark
        w = max(top.width, bottom.width)
        def fit(im: Image.Image) -> Image.Image:
            if im.width == w:
                return im
            h = int(im.height * w / im.width)
            return im.resize((w, h), Image.Resampling.LANCZOS)
        top, bottom = fit(top), fit(bottom)
        stacked = Image.new("RGB", (w, top.height + 8 + bottom.height), (8, 8, 8))
        stacked.paste(top, (0, 0))
        stacked.paste(bottom, (0, top.height + 8))
        stacked.save(figs / "qual.png")
        stacked.save(figs / "qual.pdf", "PDF", resolution=150.0)
        stacked.save(OUT_DIR / "qual_stacked.png")
    print("wrote", dark_path)
    print("wrote", light_path)


if __name__ == "__main__":
    main()
