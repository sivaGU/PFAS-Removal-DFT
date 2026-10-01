
from pathlib import Path
import csv
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from shared_helpers.md_replica_style import REPLICA_COLORS, REPLICA_MARKERS
from shared_helpers.annotation_style import GRID_COLOR, GRID_WIDTH_PT
from shared_helpers.letter_alignment import check_letters

HERE = Path(__file__).resolve().parent
COLORS = REPLICA_COLORS
TICKS = tuple(range(0, 11, 2))

def load(name):
    with (HERE / 'input_data' / name).open() as f:
        return list(csv.DictReader(f))

def plot_windows(ax, rows, metric, scale=1.0):
    for rep, color in COLORS.items():
        subset = sorted([r for r in rows if r['replica'] == rep and r['metric'] == metric],
                        key=lambda r: int(r['bin_index']))
        if [int(r['bin_index']) for r in subset] != list(range(1, 41)):
            raise ValueError(f'{rep}/{metric}: expected 40 display windows')
        if any(int(r['n_frames']) != 125 for r in subset):
            raise ValueError(f'{rep}/{metric}: expected 125 frames per window')
        if any(abs(float(r['time_mid_ns']) - (i+.5)*.25) > 1e-10
               for i, r in enumerate(subset)):
            raise ValueError(f'{rep}/{metric}: unexpected midpoint time')
        ax.plot([float(r['time_mid_ns']) for r in subset],
                [scale * float(r['mean']) for r in subset],
                marker=REPLICA_MARKERS[rep], markersize=2.8, linewidth=1.1, color=color,
                label=rep.replace('replica_', 'Replica '))

def main(dpi=600):
    windows = load('figure15_display_250ps.csv')
    if len(windows) != 720:
        raise ValueError('Expected 3 replicas x 40 windows x 6 metrics')
    fig = plt.figure(figsize=(7.25, 8.75))
    grid = fig.add_gridspec(3, 2, left=.115, right=.975, top=.91,
                            bottom=.075, hspace=.50, wspace=.39)
    axes = [fig.add_subplot(grid[r, c]) for r in range(3) for c in range(2)]
    specs = [
        ('rmsd', 'Oligomer RMSD (Å)'),
        ('rg', r'Oligomer $R_g$ (Å)'),
        ('head_any', 'Head–nearest N (Å)'),
        ('head_contact_fraction', 'Head–N contact (<5 Å, %)'),
        ('tail_contacts', 'Tail–oligomer atoms (<4 Å)'),
        ('tail_waters', 'Tail waters within 5 Å'),
    ]
    for i, (ax, (metric, ylabel)) in enumerate(zip(axes, specs)):
        plot_windows(ax, windows, metric, 100 if metric == 'head_contact_fraction' else 1)
        if metric == 'head_contact_fraction':
            ax.set_ylim(0, 100)
        if metric == 'head_any':
            ax.axhline(5, color='#000000', ls='--', lw=.8)
        ax.set(xlim=(0, 10), xticks=TICKS, ylabel=ylabel,
               xlabel='Production time (ns)')
        ax.tick_params(axis='y', labelsize=8)
        ax.tick_params(axis='x', labelsize=7.5)
        ax.grid(axis='y', color=GRID_COLOR, linewidth=GRID_WIDTH_PT, zorder=0)
        ax.text(-.17, 1.10, 'ABCDEF'[i], transform=ax.transAxes,
                fontweight='bold', fontsize=11)
    fig.canvas.draw()
    for index, value in ((0, 5.5), (2, 5.0), (4, 25.0)):
        for tick in axes[index].yaxis.get_major_ticks():
            if abs(tick.get_loc() - value) < 1e-8:
                tick.gridline.set_visible(False)
    fig.legend(*axes[0].get_legend_handles_labels(), loc='upper center',
               bbox_to_anchor=(.5, 1.0), ncol=3, frameon=False)
    output = ROOT / 'figure_exports/Figure_15_MD_blocks_and_diagnostics.png'
    output.parent.mkdir(parents=True, exist_ok=True)
    check_letters(fig, axes, columns=((0, 2), (2, 4), (1, 3), (3, 5)),
                  rows=((0, 1), (2, 3), (4, 5)))
    fig.savefig(output, dpi=dpi, bbox_inches='tight', pad_inches=.055)
    plt.close(fig)
    print(output)

if __name__ == '__main__':
    main()
