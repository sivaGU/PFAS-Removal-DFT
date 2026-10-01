
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from shared_helpers.figure_style import EXPORT_DPI
from shared_helpers.md_replica_style import REPLICA_COLORS
from shared_helpers.annotation_style import GRID_COLOR, GRID_WIDTH_PT
from shared_helpers.letter_alignment import check_letters

HERE = Path(__file__).resolve().parent
TICKS = tuple(range(0, 11, 2))
FILES = {
    'rmsd': 'polymer_rmsd.dat',
    'rg': 'polymer_rg.dat',
    'head_any': 'pfoa_head_anyN.dat',
    'tail_contacts': 'pfoa_tail_polymer_heavy_contacts_4A.dat',
    'tail_waters': 'pfoa_tail_waters_within_5A.dat',
}
SPECS = [
    ('rmsd', 'Oligomer RMSD (Å)'),
    ('rg', r'Oligomer $R_g$ (Å)'),
    ('head_any', 'Head–nearest N (Å)'),
    ('tail_contacts', 'Tail–oligomer atoms (<4 Å)'),
    ('tail_waters', 'Tail waters within 5 Å'),
]


def load_series(replica, metric):
    path = HERE / 'input_data' / f'{replica}_{FILES[metric]}'
    values = np.loadtxt(path, comments='#')
    if values.shape[0] != 5000 or values.shape[1] < 2:
        raise ValueError(f'{path}: expected 5000 rows and at least two columns')
    if not np.array_equal(values[:, 0], np.arange(1, 5001)):
        raise ValueError(f'{path}: frame indices are not consecutive')
    if metric == 'tail_waters':
        if values.shape[1] < 3 or not np.array_equal(values[:, 1], values[:, 2]):
            raise ValueError(f'{path}: watershell columns disagree')
    return values[:, 0] * 0.002, values[:, 1]


def main(dpi=EXPORT_DPI):
    fig = plt.figure(figsize=(7.25, 8.75))
    grid = fig.add_gridspec(3, 2, left=.115, right=.975, top=.91,
                            bottom=.075, hspace=.50, wspace=.39)
    axes = [fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1]),
            fig.add_subplot(grid[1, 0]), fig.add_subplot(grid[1, 1]),
            fig.add_subplot(grid[2, :])]
    standard = axes[0].get_position()
    bottom = axes[4].get_position()
    axes[4].set_position([.5 - standard.width / 2, bottom.y0,
                          standard.width, bottom.height])

    for index, (ax, (metric, ylabel)) in enumerate(zip(axes, SPECS)):
        for replica, color in REPLICA_COLORS.items():
            time_ns, values = load_series(replica, metric)
            ax.plot(time_ns, values, color=color, linewidth=.45, alpha=.68,
                    rasterized=True, label=replica.replace('replica_', 'Replica '))
        if metric == 'head_any':
            ax.axhline(5, color='#000000', linestyle='--', linewidth=.8)
        ax.set(xlim=(0, 10), xticks=TICKS, ylabel=ylabel,
               xlabel='Production time (ns)')
        ax.tick_params(axis='y', labelsize=8)
        ax.tick_params(axis='x', labelsize=7.5)
        ax.grid(axis='y', color=GRID_COLOR, linewidth=GRID_WIDTH_PT, zorder=0)
        ax.text(-.17, 1.10, 'ABCDE'[index], transform=ax.transAxes,
                fontweight='bold', fontsize=11)

    fig.canvas.draw()
    for index, value in ((0, 5.5), (2, 5.0), (3, 25.0)):
        for tick in axes[index].yaxis.get_major_ticks():
            if abs(tick.get_loc() - value) < 1e-8:
                tick.gridline.set_visible(False)

    legend = fig.legend(*axes[0].get_legend_handles_labels(), loc='upper center',
                        bbox_to_anchor=(.5, 1.0), ncol=3, frameon=False,
                        handlelength=3.1)
    for handle in legend.get_lines():
        handle.set_linewidth(1.5)
        handle.set_alpha(1.0)

    output = ROOT / 'figure_exports/Figure_S1_MD_raw_RMSD_and_proximity.png'
    output.parent.mkdir(parents=True, exist_ok=True)
    check_letters(fig, axes, columns=((0, 2), (1, 3)),
                  rows=((0, 1), (2, 3)))
    fig.savefig(output, dpi=dpi, bbox_inches='tight', pad_inches=.055)
    plt.close(fig)
    print(output)


if __name__ == '__main__':
    main()
