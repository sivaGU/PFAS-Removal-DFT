from __future__ import annotations

import shutil
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps

import figure14_render_core as fig14
from render_figure14_raws import PANEL_C_FINAL


ROOT = fig14.ROOT
FINAL_DIR = fig14.FINAL_DIR
CURRENT_FINAL = FINAL_DIR / "Figure_14_MD_structural_snapshots_strip_manual_labels.png"
BACKUP_FINAL = FINAL_DIR / "_figure_backups" / "Figure_14_MD_structural_snapshots_strip_manual_labels.before_panel_c.png"
ASSEMBLED_BASE = FINAL_DIR / "_figure_backups" / "Figure_14_MD_structural_snapshots_strip_panel_c_base.png"


def load_font(size: int, bold: bool = True) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def add_panel_frame(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], width: int = 10) -> None:
    draw.rectangle(box, outline="black", width=width)


def add_panel_heading(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    letter: str,
    title: str,
) -> None:
    letter_font = load_font(150)
    title_font = load_font(100)
    x0, y0, x1, _ = box
    draw.text((x0 + 5, y0 - 170), letter, fill="black", font=letter_font)
    title_bbox = draw.textbbox((0, 0), title, font=title_font)
    title_w = title_bbox[2] - title_bbox[0]
    draw.text((x0 + (x1 - x0 - title_w) // 2, y0 - 128), title, fill="black", font=title_font)


def fit_into(image: Image.Image, box_size: tuple[int, int], margin: int = 75) -> Image.Image:
    return ImageOps.contain(
        image,
        (box_size[0] - 2 * margin, box_size[1] - 2 * margin),
        Image.Resampling.LANCZOS,
    )


def assemble() -> None:
    if CURRENT_FINAL.exists() and not BACKUP_FINAL.exists():
        shutil.copyfile(CURRENT_FINAL, BACKUP_FINAL)

    source_path = BACKUP_FINAL if BACKUP_FINAL.exists() else CURRENT_FINAL
    source = Image.open(source_path).convert("RGB")
    top_and_a = source.crop((0, 0, source.width, 5863))
    b_tight = source.crop((1180, 6410, 5620, 10855))
    c_tight = Image.open(PANEL_C_FINAL).convert("RGB")

    target_w = source.width
    gap = 255
    bottom_y = top_and_a.height + gap
    bottom_h = 3550
    side_margin = 70
    inter_gap = 70
    panel_w = (target_w - 2 * side_margin - inter_gap) // 2
    panel_b = (side_margin, bottom_y, side_margin + panel_w, bottom_y + bottom_h)
    panel_c = (side_margin + panel_w + inter_gap, bottom_y, side_margin + panel_w + inter_gap + panel_w, bottom_y + bottom_h)

    canvas = Image.new("RGB", (target_w, bottom_y + bottom_h + 72), "white")
    canvas.paste(top_and_a, (0, 0))
    draw = ImageDraw.Draw(canvas)

    draw.rectangle((0, 430, 1450, 648), fill="white")
    add_panel_heading(draw, (70, 650, source.width - 70, 5863), "A", "PFAS Binding Site Snapshots")

    for box, letter, title, img in (
        (panel_b, "B", "Full System Representation", b_tight),
        (panel_c, "C", "Hydrated PFOA Binding Site", c_tight),
    ):
        add_panel_frame(draw, box)
        add_panel_heading(draw, box, letter, title)
        fitted = fit_into(img, (box[2] - box[0], box[3] - box[1]), margin=105)
        x = box[0] + (box[2] - box[0] - fitted.width) // 2
        y = box[1] + (box[3] - box[1] - fitted.height) // 2 + 30
        canvas.paste(fitted, (x, y))

    ASSEMBLED_BASE.parent.mkdir(exist_ok=True)
    canvas.save(ASSEMBLED_BASE)
    canvas.save(CURRENT_FINAL)


if __name__ == "__main__":
    assemble()
    print(f"Updated {CURRENT_FINAL}")
    print(f"Panel-C base composite saved to {ASSEMBLED_BASE}")
