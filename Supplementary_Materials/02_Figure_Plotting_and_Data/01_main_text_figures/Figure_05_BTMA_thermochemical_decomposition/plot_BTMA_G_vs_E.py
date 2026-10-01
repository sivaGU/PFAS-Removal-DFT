"""Figure 5 thermochemistry"""
import argparse
import sys
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent.parent))
from shared_helpers.figure_style import EXPORT_DPI, FONT
from shared_helpers.letter_alignment import check_letters
from shared_helpers.thermochemical_panels import (data_table, energy_panel, METHOD_COLORS, MARKERS, METHOD_LABELS)
DEFAULT_OUTPUT_DIR = ROOT.parent.parent / "figure_exports"


def verify_layout(fig, axes):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    px_per_pt = fig.dpi / 72
    min_tick_gap_pt = float("inf")
    min_row_gap_pt = float("inf")
    for ax in axes:
        labels = [label.get_window_extent(renderer) for label in ax.get_xticklabels()]
        for left, right in zip(labels, labels[1:]):
            gap = (right.x0 - left.x1) / px_per_pt
            min_tick_gap_pt = min(min_tick_gap_pt, gap)
            if gap < 8:
                raise ValueError(f"Figure 5 PFAS label gap is only {gap:.1f} pt")
        title = ax.title.get_window_extent(renderer)
        letter = ax.texts[-1].get_window_extent(renderer)
        if title.overlaps(letter):
            raise ValueError("Figure 5 panel letter overlaps a panel title")
        bounds = ax.get_window_extent(renderer)
        for annotation in ax.texts[:-1]:
            box = annotation.get_window_extent(renderer)
            clearance = min(box.y0 - bounds.y0, bounds.y1 - box.y1) / px_per_pt
            if clearance < 2:
                raise ValueError(f"Figure 5 value {annotation.get_text()} in {ax.get_title()} has only {clearance:.1f} pt of vertical clearance")
    for upper, lower in zip(axes, axes[1:]):
        upper_labels = [label.get_window_extent(renderer) for label in upper.get_xticklabels()]
        lower_title = lower.title.get_window_extent(renderer)
        lower_letter = lower.texts[-1].get_window_extent(renderer)
        gap = (min(box.y0 for box in upper_labels) -
               max(lower_title.y1, lower_letter.y1)) / px_per_pt
        min_row_gap_pt = min(min_row_gap_pt, gap)
        if gap < 10:
            raise ValueError(f"Figure 5 inter-panel label gap is only {gap:.1f} pt")
    print(f"Layout check: minimum PFAS-label gap {min_tick_gap_pt:.1f} pt; "
          f"minimum inter-panel label gap {min_row_gap_pt:.1f} pt")


def make_figure(output_dir, dpi=EXPORT_DPI):
    data = data_table()
    plt.rcParams.update({"font.family": FONT, "font.size": 9.5,
                         "axes.linewidth": .75, "savefig.facecolor": "white"})
    fig, axes = plt.subplots(3, 1, figsize=(4.4, 8.0))
    fig.subplots_adjust(left=.17, right=.95, bottom=.09, top=.89, hspace=.67)
    titles = (r"$\Delta E_{\rm exchange}$", r"$\Delta(G-E_{\rm el})_{\rm exchange}$", r"$\Delta G_{\rm exchange}$")
    for ax, category, title, letter, bounds in zip(axes,
            ("DeltaE", "DeltaCorr", "DeltaG"), titles, "ABC",
            ((-4.6, 2.5), (3.8, 9.7), (0, 10.4))):
        energy_panel(ax, data, category, title, letter, bounds)
        if letter == 'A':
            current_limits = ax.get_ylim()
            ax.set_yticks([tick for tick in ax.get_yticks() if abs(tick - 1) > 1e-8])
            ax.set_ylim(current_limits)
    legend = [Line2D([], [], color=color, marker=mark, linestyle="None", markersize=6)
              for color, mark in zip(METHOD_COLORS, MARKERS)]
    fig.legend(legend, METHOD_LABELS, loc="upper center", bbox_to_anchor=(.54, .975),
               ncol=2, fontsize=10, frameon=False)
    verify_layout(fig, axes)
    output_dir.mkdir(parents=True, exist_ok=True)
    out = output_dir / "Figure_05_exchange_thermochemistry.png"
    check_letters(fig, axes, columns=((0,1),(1,2)), rows=())
    fig.savefig(out, dpi=dpi, bbox_inches="tight", pad_inches=.055)
    plt.close(fig)
    print(f"Wrote {out}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dpi", type=int, default=EXPORT_DPI)
    args = parser.parse_args()
    make_figure(args.output_dir, args.dpi)
