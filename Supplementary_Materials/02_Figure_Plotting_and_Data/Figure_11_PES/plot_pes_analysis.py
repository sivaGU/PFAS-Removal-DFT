#!/usr/bin/env python3
"""Generate Figure 11 from bundled PES data and local structure renders."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from PIL import Image, ImageDraw, ImageFont


THIS_DIR = Path(__file__).resolve().parent
INPUT_DIR = THIS_DIR / "input_data"
RENDER_DIR = THIS_DIR / "structure_renders"
OUTPUT_DIR = THIS_DIR / "outputs"
OUTPUT = OUTPUT_DIR / "Figure_11_PES_analysis_local_exchange.png"


def load_pes_data():
    grid = pd.read_csv(INPUT_DIR / "R4N+PFOA-_0.15M.grid.csv")
    mep = pd.read_csv(INPUT_DIR / "pes_mep_profile.csv")
    required_mep = {"step", "relative_energy_kcal_mol", "state"}
    if not required_mep.issubset(mep.columns):
        raise ValueError(f"pes_mep_profile.csv must contain {sorted(required_mep)}")

    r1 = grid.iloc[:, 0].to_numpy(float)
    r2 = np.asarray([float(value) for value in grid.columns[1:]], dtype=float)
    energies = grid.iloc[:, 1:].to_numpy(float)
    min_index = np.unravel_index(np.argmin(energies), energies.shape)
    minimum = {"r1": r1[min_index[0]], "r2": r2[min_index[1]]}
    return r1, r2, energies, minimum, mep


R1_VALUES, R2_VALUES, DELTA_E_KCAL_MOL, GLOBAL_MINIMUM, MEP_DATA = load_pes_data()
MEP_STEPS = MEP_DATA["step"].to_numpy(int)
MEP_RELATIVE_ENERGIES_KCAL_MOL = MEP_DATA["relative_energy_kcal_mol"].to_numpy(float)
MEP_INSETS = {
    "approach": {"title": "PFOA Approaches Ammonium Site"},
    "displacement": {"title": "PFOA Displaces Cl-"},
    "bound": {"title": "PFOA Cholestyramine Complex"},
}

DPI = 600
FONT = "DejaVu Sans"
TITLE_COLOR = "#111111"
PANEL_TITLE_SIZE = 15
DISTANCE_COLORS = {
    "3.68 Å": "#d62728",
    "4.12 Å": "#1f77b4",
    "8.52 Å": "#2ca02c",
}

STRUCTURE_RENDERS = {
    "global": RENDER_DIR / "panel_b_global_minimum.png",
    "approach": RENDER_DIR / "panel_c_approach.png",
    "displacement": RENDER_DIR / "panel_c_displacement.png",
    "bound": RENDER_DIR / "panel_c_bound.png",
}


plt.rcParams.update(
    {
        "font.family": FONT,
        "font.size": 12,
        "axes.titlesize": PANEL_TITLE_SIZE,
        "axes.titleweight": "bold",
        "figure.titlesize": 18,
        "figure.titleweight": "bold",
    }
)


def ensure_structure_renders() -> None:
    if all(path.exists() for path in STRUCTURE_RENDERS.values()):
        return
    subprocess.run([sys.executable, str(THIS_DIR / "render_pes_structures.py")], cwd=THIS_DIR, check=True)


def crop_nonwhite(path: Path, padding: int = 12) -> Image.Image:
    img = Image.open(path).convert("RGBA")
    arr = np.asarray(img)
    rgb = arr[..., :3]
    alpha = arr[..., 3] > 0
    nonwhite = np.any(rgb < 247, axis=2) & alpha
    rows, cols = np.where(nonwhite)
    if rows.size == 0:
        return img
    left = max(int(cols.min()) - padding, 0)
    upper = max(int(rows.min()) - padding, 0)
    right = min(int(cols.max()) + padding + 1, img.width)
    lower = min(int(rows.max()) + padding + 1, img.height)
    return img.crop((left, upper, right, lower))


def title_font(size: int) -> ImageFont.ImageFont:
    for candidate in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf",
    ]:
        path = Path(candidate)
        if path.exists():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


def dashed_rectangle(draw: ImageDraw.ImageDraw, xy, fill, width: int = 2, dash: int = 10, gap: int = 6) -> None:
    x0, y0, x1, y1 = map(int, xy)
    for x in range(x0, x1, dash + gap):
        draw.line([(x, y0), (min(x + dash, x1), y0)], fill=fill, width=width)
        draw.line([(x, y1), (min(x + dash, x1), y1)], fill=fill, width=width)
    for y in range(y0, y1, dash + gap):
        draw.line([(x0, y), (x0, min(y + dash, y1))], fill=fill, width=width)
        draw.line([(x1, y), (x1, min(y + dash, y1))], fill=fill, width=width)


def draw_dashed_line(draw: ImageDraw.ImageDraw, xy, fill, width: int = 4, dash: int = 12, gap: int = 8) -> None:
    x0, y0, x1, y1 = map(float, xy)
    length = float(np.hypot(x1 - x0, y1 - y0))
    if length <= 0:
        return
    ux, uy = (x1 - x0) / length, (y1 - y0) / length
    dist = 0.0
    while dist < length:
        end = min(dist + dash, length)
        draw.line(
            [(x0 + ux * dist, y0 + uy * dist), (x0 + ux * end, y0 + uy * end)],
            fill=fill,
            width=width,
        )
        dist += dash + gap


def structure_inset_image(path: Path, width: int = 760, height: int = 430) -> Image.Image:
    img = crop_nonwhite(path, padding=8).convert("RGBA")
    canvas = Image.new("RGBA", (width, height), "white")

    body_top = 18
    max_w = width - 44
    max_h = height - body_top - 22
    scale = min(max_w / img.width, max_h / img.height)
    new_size = (max(1, int(img.width * scale)), max(1, int(img.height * scale)))
    img = img.resize(new_size, Image.Resampling.LANCZOS)
    x = (width - img.width) // 2
    y = body_top + (max_h - img.height) // 2
    canvas.alpha_composite(img, (x, y))
    return canvas


def panel_b_composite(path: Path, width: int = 1320, height: int = 1040) -> Image.Image:
    img = crop_nonwhite(path, padding=24).convert("RGBA")
    canvas = Image.new("RGBA", (width, height), "white")
    max_w, max_h = 930, 720
    scale = min(max_w / img.width, max_h / img.height)
    new_size = (max(1, int(img.width * scale)), max(1, int(img.height * scale)))
    img = img.resize(new_size, Image.Resampling.LANCZOS)
    canvas.alpha_composite(img, (55, 145))

    draw = ImageDraw.Draw(canvas)
    label_font = title_font(42)
    legend_font = title_font(32)

    draw.text((64, height - 78), "PFOA", fill=(0, 0, 0, 255), font=label_font)
    draw.text((790, height - 78), "BTMA", fill=(0, 0, 0, 255), font=label_font)

    box = (1000, 88, 1255, 235)
    dashed_rectangle(draw, box, fill=(120, 120, 120, 255), width=2, dash=12, gap=7)
    for i, (label, color) in enumerate(DISTANCE_COLORS.items()):
        y = 120 + i * 42
        rgb = tuple(int(color[j : j + 2], 16) for j in (1, 3, 5)) + (255,)
        draw_dashed_line(draw, (1026, y + 12, 1084, y + 12), fill=rgb, width=4, dash=13, gap=8)
        draw.text((1105, y - 6), label, fill=(0, 0, 0, 255), font=legend_font)
    return canvas


def add_top_panel_labels(fig, ax_a, ax_b) -> None:
    pos_a = ax_a.get_position()
    pos_b = ax_b.get_position()
    y = max(pos_a.y1, pos_b.y1) + 0.032
    fig.text(pos_a.x0 - 0.047, y, "A", fontsize=19, fontweight="bold", ha="left", va="bottom")
    fig.text(pos_b.x0 - 0.047, y, "B", fontsize=19, fontweight="bold", ha="left", va="bottom")


def add_bottom_panel_label(fig, ax_a, ax_c) -> None:
    pos_a = ax_a.get_position()
    pos_c = ax_c.get_position()
    fig.text(pos_a.x0 - 0.047, pos_c.y1 + 0.032, "C", fontsize=19, fontweight="bold", ha="left", va="bottom")


def draw_contour(ax) -> None:
    r1 = np.asarray(R1_VALUES, dtype=float)
    r2 = np.asarray(R2_VALUES, dtype=float)
    z = np.asarray(DELTA_E_KCAL_MOL, dtype=float)
    mesh_r2, mesh_r1 = np.meshgrid(r2, r1)
    levels = np.arange(0, max(46, np.ceil(z.max())) + 2, 2)

    contour = ax.contourf(mesh_r1, mesh_r2, z, levels=levels, cmap="viridis")
    ax.contour(mesh_r1, mesh_r2, z, levels=levels, colors="black", linewidths=0.35, alpha=0.55)
    ax.plot(GLOBAL_MINIMUM["r1"], GLOBAL_MINIMUM["r2"], marker="*", ms=15, color="#d62728", mec="black", mew=0.5)
    ax.text(
        GLOBAL_MINIMUM["r1"] + 0.18,
        GLOBAL_MINIMUM["r2"] + 0.18,
        "minimum",
        fontsize=10,
        fontweight="bold",
        color="#d62728",
    )
    cbar = plt.colorbar(contour, ax=ax, fraction=0.046, pad=0.035)
    cbar.set_label(r"$\Delta E$ (kcal mol$^{-1}$)", fontweight="bold")
    ax.set_xlabel(r"$r_1$ (Å)", fontweight="bold")
    ax.set_ylabel(r"$r_2$ (Å)", fontweight="bold")
    ax.set_title(r"PES Contour ($\mathbf{\Delta E}$ vs. $\mathbf{r_1}$, $\mathbf{r_2}$)", pad=13)
    ax.set_xlim(min(r1), max(r1))
    ax.set_ylim(min(r2), max(r2))


def draw_global_structure(ax) -> None:
    img = panel_b_composite(STRUCTURE_RENDERS["global"])
    ax.imshow(img)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.1)
        spine.set_color("#111111")
    ax.set_title("Global Minimum Structure", pad=13)


def add_structure_inset(
    ax,
    key: str,
    xybox: tuple[float, float],
    xydata: tuple[float, float],
    zoom: float,
    width: int,
    height: int,
) -> None:
    title = MEP_INSETS[key]["title"]
    img = structure_inset_image(STRUCTURE_RENDERS[key], width=width, height=height)
    inset = OffsetImage(np.asarray(img), zoom=zoom)
    box = AnnotationBbox(
        inset,
        xybox,
        xycoords="data",
        frameon=True,
        bboxprops={
            "edgecolor": "#1f77b4",
            "linewidth": 1.5,
            "facecolor": "white",
            "boxstyle": "square,pad=0.0",
        },
        zorder=5,
    )
    ax.add_artist(box)
    ax.plot([xydata[0], xybox[0]], [xydata[1], xybox[1]], color="#1f77b4", lw=1.5, zorder=3)
    ax.text(
        xybox[0],
        xybox[1] + 7.5,
        title,
        ha="center",
        va="bottom",
        fontsize=13.5,
        fontweight="bold",
        zorder=6,
        clip_on=False,
    )


def draw_energy_state_diagram(ax) -> None:
    x = np.asarray(MEP_STEPS, dtype=float)
    y = np.asarray(MEP_RELATIVE_ENERGIES_KCAL_MOL, dtype=float)
    ax.plot(x, y, "-o", color="#1f77b4", lw=2.3, ms=8.5, mec="#1f77b4", mfc="#1f77b4", zorder=2)

    ax.grid(True, color="#c7c7c7", alpha=0.45, linewidth=1.1)
    ax.set_xlim(0.4, 16.7)
    ax.set_ylim(-1.8, 35.8)
    ax.set_xlabel("Step Index (PES Scan)", fontsize=17)
    ax.set_ylabel("E (kcal/mol)", fontsize=15)
    ax.set_title("Energy State Diagram of Anion Exchange", pad=8)
    ax.tick_params(axis="both", labelsize=13, width=1.3, length=6)
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)

    inset_size = {"zoom": 0.255, "width": 850, "height": 500}
    add_structure_inset(ax, "approach", xybox=(3.20, 17.4), xydata=(1, y[0]), **inset_size)
    add_structure_inset(ax, "displacement", xybox=(7.75, 18.0), xydata=(8, y[7]), **inset_size)
    add_structure_inset(ax, "bound", xybox=(12.95, 12.4), xydata=(12, y[11]), **inset_size)


def make_figure() -> None:
    ensure_structure_renders()
    OUTPUT_DIR.mkdir(exist_ok=True)

    fig = plt.figure(figsize=(14.0, 13.35), dpi=DPI)
    fig.suptitle("PES Analysis of Local Anion Exchange", y=0.992)

    grid = fig.add_gridspec(
        2,
        2,
        left=0.065,
        right=0.975,
        top=0.892,
        bottom=0.066,
        height_ratios=[1.0, 1.25],
        hspace=0.25,
        wspace=0.30,
    )

    ax_a = fig.add_subplot(grid[0, 0])
    ax_b = fig.add_subplot(grid[0, 1])
    ax_c = fig.add_subplot(grid[1, :])

    draw_contour(ax_a)
    draw_global_structure(ax_b)
    draw_energy_state_diagram(ax_c)

    add_top_panel_labels(fig, ax_a, ax_b)
    add_bottom_panel_label(fig, ax_a, ax_c)

    fig.savefig(OUTPUT, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {OUTPUT}")


if __name__ == "__main__":
    make_figure()
