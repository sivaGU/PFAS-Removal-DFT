"""Figure 4 exchange plots"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))
from shared_helpers.figure_style import BLUE, ORANGE, GREY, FONT, EXPORT_DPI
from shared_helpers.annotation_style import VALUE_SIZE_PT, VALUE_BOX, energy_grid
from shared_helpers.letter_alignment import check_letters
from shared_helpers.thermochemical_panels import data_table, PFAS, METHODS, value

OUTPUT = ROOT / "figure_exports/Figure_04_BTMA_exchange_energy_panels.png"


def draw_panel(ax, data, category, letter, title, limits):
    xs = np.arange(4)
    for i, pfas in enumerate(PFAS):
        first, second = [value(data, "exchange", category, pfas, method)
                         for method in METHODS]
        ax.plot([i, i], [first, second], color="#444444", lw=.85, zorder=2)
        for result, marker, color in ((first, "o", BLUE), (second, "s", ORANGE)):
            ax.scatter(i, result, marker=marker, color=color, s=28, zorder=3)
            is_upper = result >= (second if marker == "o" else first)
            ax.annotate(f"{result:+.2f}" if category == "DeltaE" else f"{result:.2f}",
                        (i, result), xytext=(0, 8 if is_upper else -9),
                        textcoords="offset points", ha="center",
                        va="bottom" if is_upper else "top", color=color,
                        fontsize=VALUE_SIZE_PT, fontweight="bold", bbox=VALUE_BOX,
                        zorder=5)
    ax.set(xlim=(-.4, 3.4), ylim=limits)
    ax.set_xticks(xs, ["6:2\nFTCA", "PFHxA", "PFOA", "PFOS"])
    ax.tick_params(labelsize=9, width=.7, length=2.5)
    ax.tick_params(axis="y", pad=5)
    ax.set_title(title, fontsize=10.5, pad=8)
    ax.set_ylabel("Energy (kcal/mol)", fontsize=9.5, labelpad=12)
    energy_grid(ax)
    ax.text(-.09, 1.14, letter, transform=ax.transAxes, fontsize=11.5,
            fontweight="bold", ha="right", va="top", clip_on=False)


def render(output=OUTPUT, dpi=EXPORT_DPI):
    data = data_table()
    plt.rcParams.update({"font.family": FONT, "font.size": 9.5,
                         "axes.linewidth": .75, "savefig.facecolor": "white"})
    fig, axes = plt.subplots(2, 1, figsize=(5.0, 7.67))
    fig.subplots_adjust(left=.19, right=.95, top=.8403, bottom=.09, hspace=.4575)
    draw_panel(axes[0], data, "DeltaE", "A", r"$\Delta E_{\mathrm{exchange}}$", (-4.4, 1.8))
    draw_panel(axes[1], data, "DeltaG", "B", r"$\Delta G_{\mathrm{exchange}}$", (1.3, 9.7))
    legend = [Line2D([], [], color=c, marker=m, linestyle="None", markersize=6)
              for c, m in ((BLUE, "o"), (ORANGE, "s"))]
    fig.legend(legend, (r"r$^2$SCAN-3c", r"$\omega$B97X-D3"),
               loc="upper center", bbox_to_anchor=(.57, .965), ncol=2,
               fontsize=10, frameon=False)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    px_per_pt = fig.dpi / 72
    legend_box = fig.legends[0].get_window_extent(renderer)
    for ax in axes:
        boxes = [t.get_window_extent(renderer) for t in ax.get_xticklabels()]
        minimum = min((right.x0-left.x1)/px_per_pt
                      for left,right in zip(boxes, boxes[1:]))
        if minimum < 8:
            raise ValueError(f"PFAS labels too close: {minimum:.1f} pt")
        headroom = (ax.bbox.y1-max(t.get_window_extent(renderer).y1
                    for t in ax.texts[:-1]))/px_per_pt
        ylabel = ax.yaxis.label.get_window_extent(renderer)
        tick_boxes = [t.label1.get_window_extent(renderer)
                      for t in ax.yaxis.get_major_ticks() if t.label1.get_text()]
        ylabel_gap = (min(b.x0 for b in tick_boxes)-ylabel.x1)/px_per_pt
        if headroom < 8 or ylabel_gap < 5:
            raise ValueError(f"Panel {ax.texts[-1].get_text()} clearance: "
                             f"top {headroom:.1f} pt; y label {ylabel_gap:.1f} pt")
        print(f"Panel {ax.texts[-1].get_text()}: label gap {minimum:.1f} pt, "
              f"top clearance {headroom:.1f} pt, y-label clearance {ylabel_gap:.1f} pt")
    if abs(axes[0].bbox.height-axes[1].bbox.height) > .01:
        raise ValueError("Panel boxes have unequal height")
    upper = max(axes[0].title.get_window_extent(renderer).y1,
                axes[0].texts[-1].get_window_extent(renderer).y1)
    legend_gap = (legend_box.y0-upper)/px_per_pt
    lower = max(axes[1].title.get_window_extent(renderer).y1,
                axes[1].texts[-1].get_window_extent(renderer).y1)
    first_ticks_bottom = min(t.get_window_extent(renderer).y0
                             for t in axes[0].get_xticklabels())
    panel_gap = (first_ticks_bottom-lower)/px_per_pt
    if legend_gap < 12:
        raise ValueError(f"Legend-to-panel gap too small: {legend_gap:.1f} pt")
    if panel_gap < 12:
        raise ValueError(f"Between-panel gap too small: {panel_gap:.1f} pt")
    print(f"Equal panel heights: {axes[0].bbox.height/px_per_pt:.1f} pt; "
          f"legend gap: {legend_gap:.1f} pt; A-to-B gap: {panel_gap:.1f} pt")
    output.parent.mkdir(parents=True, exist_ok=True)
    check_letters(fig, axes, columns=((0,1),), rows=())
    fig.savefig(output, dpi=dpi, bbox_inches="tight", pad_inches=.07)
    plt.close(fig)
    print(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--dpi", type=int, default=EXPORT_DPI)
    opts = parser.parse_args()
    render(opts.output, opts.dpi)
