"""Figure 6 species comparison"""
import argparse
import sys
from pathlib import Path
import matplotlib.pyplot as plt
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent.parent))
from shared_helpers.figure_style import EXPORT_DPI, FONT
from shared_helpers.thermochemical_panels import data_table, species_panel, component_panel
DEFAULT_OUTPUT_DIR = ROOT.parent.parent / "figure_exports"

def make_figure(output_dir, dpi=EXPORT_DPI):
    data = data_table()
    plt.rcParams.update({"font.family": FONT, "font.size": 9.5,
                         "axes.linewidth": .75, "savefig.facecolor": "white"})
    fig, (ax_a, ax_b) = plt.subplots(2, 1, figsize=(6.5, 7.25),
                                    gridspec_kw={"height_ratios": (1, 1.34)})
    fig.subplots_adjust(left=.19, right=.89, bottom=.135, top=.875, hspace=.40)
    img = species_panel(ax_a, data)
    component_panel(ax_b, data)
    cbar = fig.colorbar(img, ax=ax_a, fraction=.045, pad=.04)
    cbar.ax.tick_params(labelsize=8.5, length=2)
    cbar.set_label("Species contribution (kcal/mol)", fontsize=8.5, labelpad=8)
    fig.canvas.draw()
    x = min(ax_a.get_position().x0, ax_b.get_position().x0) - .065
    for ax, letter in ((ax_a, 'A'), (ax_b, 'B')):
        label = next(t for t in ax.texts if t.get_text() == letter)
        label.set_transform(fig.transFigure)
        label.set_position((x, ax.get_position().y1 + .025))
    fig.canvas.draw()
    extents = [next(t for t in ax.texts if t.get_text()==letter).get_window_extent(
        fig.canvas.get_renderer()) for ax,letter in ((ax_a,'A'),(ax_b,'B'))]
    if abs(extents[0].x1-extents[1].x1)*72/fig.dpi > .25:
        raise ValueError('Figure 6 panel letters do not align')
    output_dir.mkdir(parents=True, exist_ok=True)
    out = output_dir / "Figure_06_species_and_components.png"
    fig.savefig(out, dpi=dpi, bbox_inches="tight", pad_inches=.055)
    plt.close(fig)
    print(f"Wrote {out}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dpi", type=int, default=EXPORT_DPI)
    args = parser.parse_args()
    make_figure(args.output_dir, args.dpi)
