from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# =========================
# USER SETTINGS
# =========================
ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
OUTDIR = ROOT / "plots"
OUTDIR.mkdir(exist_ok=True)
FINAL_OUTDIR = PROJECT_ROOT / "final_plots"

DPI = 600
FONT_FAMILY = "DejaVu Sans"

PANEL_IMAGES = [
    {
        "letter": "A",
        "title": "",
        "path": ROOT / "plots" / "R4N+PFOA-_0.15M.contour.png",
        "crop_border": True,
    },
    {
        "letter": "B",
        "title": "Global Minimum Structure",
        "path": FINAL_OUTDIR / "Figure 11 Panel C.png",
        "crop_border": True,
    },
    {
        "letter": "C",
        "title": "",
        "path": FINAL_OUTDIR / "Figure 11 Panel B.png",
        "crop_border": True,
    },
]

plt.rcParams.update(
    {
        "font.family": FONT_FAMILY,
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.titleweight": "normal",
        "figure.titlesize": 17,
        "figure.titleweight": "bold",
    }
)


def crop_white_border(img, padding=8, threshold=248):
    """Trim source-image whitespace so embedded plots can fill their panel."""
    rgb = img[..., :3]

    if img.shape[-1] == 4:
        mask = img[..., 3] > 0.01
    else:
        mask = np.ones(rgb.shape[:2], dtype=bool)

    nonwhite = np.any(rgb < threshold / 255.0, axis=2) & mask
    rows = np.where(nonwhite.any(axis=1))[0]
    cols = np.where(nonwhite.any(axis=0))[0]

    if rows.size == 0 or cols.size == 0:
        return img

    y0 = max(rows[0] - padding, 0)
    y1 = min(rows[-1] + padding + 1, img.shape[0])
    x0 = max(cols[0] - padding, 0)
    x1 = min(cols[-1] + padding + 1, img.shape[1])

    return img[y0:y1, x0:x1]


def load_panel_image(panel):
    image_path = panel["path"]

    if not image_path.exists():
        return None

    img = plt.imread(image_path)

    if panel.get("crop_border", False):
        img = crop_white_border(img)

    return img


def add_aspect_preserved_axes(
    fig,
    img,
    max_box,
    h_align="center",
    v_align="center",
    x_nudge=0.0,
    y_nudge=0.0,
):
    """
    Add an axes inside max_box while preserving the image aspect ratio.

    max_box is [x0, y0, width, height] in figure coordinates.
    x_nudge and y_nudge are small figure-coordinate adjustments applied
    after aspect-preserved placement.
    """
    if img is None:
        x0, y0, w, h = max_box
        return fig.add_axes([x0 + x_nudge, y0 + y_nudge, w, h])

    fig_w, fig_h = fig.get_size_inches()
    x0, y0, max_w, max_h = max_box

    img_h, img_w = img.shape[:2]
    img_aspect = img_h / img_w

    max_w_in = max_w * fig_w
    max_h_in = max_h * fig_h
    box_aspect = max_h_in / max_w_in

    if img_aspect > box_aspect:
        ax_h = max_h
        ax_w = (max_h_in / img_aspect) / fig_w
    else:
        ax_w = max_w
        ax_h = (max_w_in * img_aspect) / fig_h

    if h_align == "left":
        ax_x = x0
    elif h_align == "right":
        ax_x = x0 + max_w - ax_w
    else:
        ax_x = x0 + (max_w - ax_w) / 2

    if v_align == "bottom":
        ax_y = y0
    elif v_align == "top":
        ax_y = y0 + max_h - ax_h
    else:
        ax_y = y0 + (max_h - ax_h) / 2

    return fig.add_axes([ax_x + x_nudge, ax_y + y_nudge, ax_w, ax_h])


def draw_image_panel(ax, panel, img):
    if img is None:
        ax.text(
            0.5,
            0.5,
            f"Missing image:\n{panel['path']}",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=10,
        )
        ax.set_axis_off()
        return

    # No aspect="auto"; this preserves native source-image aspect ratio.
    ax.imshow(img)

    if panel["title"]:
        ax.set_title(panel["title"], pad=5, fontweight="normal")

    ax.set_axis_off()


def add_panel_letters(fig, axes, panels, top_row_boxes):
    """
    Keep A and B labels on the same horizontal guide line.
    C is placed from the actual C axes position.
    """
    x_offset = 0.014
    y_offset = 0.004

    # A/B: same row-level y anchor, regardless of slight axes-height differences
    for i in [0, 1]:
        x0, y0, w, h = top_row_boxes[i]
        fig.text(
            x0 - x_offset,
            y0 + h - y_offset,
            panels[i]["letter"],
            fontsize=18,
            fontweight="bold",
            va="top",
            ha="right",
        )

    # C: anchor to the actual C axes box
    pos = axes[2].get_position()
    fig.text(
        pos.x0 - x_offset,
        pos.y1 - y_offset,
        panels[2]["letter"],
        fontsize=18,
        fontweight="bold",
        va="top",
        ha="right",
    )


def make_pes_analysis_figure():
    loaded_images = [load_panel_image(panel) for panel in PANEL_IMAGES]

    fig = plt.figure(figsize=(12.0, 12.9), dpi=DPI)
    fig.suptitle("PES Analysis of Local Exchange", y=0.982)

    # A/B share the same row box. Center alignment avoids the previous
    # too-high / too-low behavior.
    top_row_boxes = [
        [0.075, 0.595, 0.405, 0.300],  # A
        [0.545, 0.595, 0.405, 0.300],  # B
    ]

    # C is moved upward to reduce whitespace between rows.
    c_box = [0.125, 0.105, 0.750, 0.455]

    ax_a = add_aspect_preserved_axes(
        fig,
        loaded_images[0],
        top_row_boxes[0],
        h_align="center",
        v_align="center",
    )

    ax_b = add_aspect_preserved_axes(
        fig,
        loaded_images[1],
        top_row_boxes[1],
        h_align="center",
        v_align="center",
        y_nudge=0.006,  # small lift: between prior top-align and bottom-align versions
    )

    ax_c = add_aspect_preserved_axes(
        fig,
        loaded_images[2],
        c_box,
        h_align="center",
        v_align="top",
    )

    axes = [ax_a, ax_b, ax_c]

    for ax, panel, img in zip(axes, PANEL_IMAGES, loaded_images):
        draw_image_panel(ax, panel, img)

    add_panel_letters(fig, axes, PANEL_IMAGES, top_row_boxes)

    out_path = OUTDIR / "Figure_11_PES_analysis_local_exchange.png"
    final_path = FINAL_OUTDIR / "Figure_11_PES_analysis_local_exchange.png"

    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    fig.savefig(final_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved figure to: {out_path}")
    print(f"Saved figure to: {final_path}")


if __name__ == "__main__":
    make_pes_analysis_figure()