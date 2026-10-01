"""Figure 12 octanol EDA"""
from pathlib import Path
import csv
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from shared_helpers.figure_style import GREEN, PURPLE, EXPORT_DPI
from shared_helpers.annotation_style import VALUE_SIZE_PT, VALUE_BOX, energy_grid

STEPS = [('Pauli Energy', 'Pauli'), ('Electrostatic Energy', 'Elstat'),
         ('Orbital Energy', 'Orb'), ('Delta Dispersion', 'Disp'),
         ('Delta E^0(XC)', 'XC'), ('Delta CPCM Dielectric', 'CPCM')]


def main(dpi=EXPORT_DPI):
    here = Path(__file__).resolve().parent
    rows = {r['solvent']: r for r in csv.DictReader(
        (here / 'input_data/eda_components.csv').open())}
    assert set(rows) == {'water', '1-octanol'}
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    fig.subplots_adjust(left=.115, right=.94, top=.79, bottom=.19)
    labels = {}
    for index, (solvent, color, marker) in enumerate(
            [('water', GREEN, 'o'), ('1-octanol', PURPLE, 's')]):
        row = rows[solvent]
        values = np.cumsum([0.] + [float(row[k]) for k, _ in STEPS])
        bond = float(row['Bond Energy'])
        shift = -.045 if index == 0 else .045
        assert abs(values[-1] - float(row['Sum of listed terms'])) < 1e-7
        for j in range(len(STEPS)):
            ax.plot([j+shift, j+1+shift], [values[j]]*2, color=color, lw=1.6)
            ax.plot([j+1+shift]*2, [values[j], values[j+1]], color=color, lw=1.6)
        ax.plot([6+shift, 7+shift], [values[-1]]*2,
                color=color, lw=1.6, ls=(0, (3, 2)))
        ax.plot([7+shift]*2, [values[-1], bond],
                color=color, lw=1.6, ls=(0, (3, 2)))
        ax.scatter(7+shift, bond, c=color, marker=marker, zorder=4)
        label_y = -7.4 if solvent == 'water' else -20.0
        labels[solvent] = ax.text(7.29, label_y, f'{bond:+.2f}', ha='left',
            va='center', color=color, fontsize=VALUE_SIZE_PT,
            fontweight='bold', bbox=VALUE_BOX, zorder=6)
    assert float(rows['water']['Bond Energy']) > float(rows['1-octanol']['Bond Energy'])
    ax.set_xlim(-.15, 8.20)
    ax.set_ylim(-105, 48)
    ax.set_xticks(np.arange(1, 8), [name for _, name in STEPS] + [r'$E_{\rm bond}$'])
    ax.set_ylabel('Energy (kcal/mol)', labelpad=7)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    energy_grid(ax)
    handles = [plt.Line2D([], [], color=c, marker=m, lw=1.5)
               for c, m in [(GREEN, 'o'), (PURPLE, 's')]]
    fig.legend(handles, ['Water', '1-octanol'], loc='upper center',
               bbox_to_anchor=(.52, .96), ncol=2, frameon=False)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    water = labels['water'].get_window_extent(renderer)
    octanol = labels['1-octanol'].get_window_extent(renderer)
    if water.y0 - octanol.y1 < 3*fig.dpi/72:
        raise ValueError('Figure 12 endpoint labels have insufficient vertical clearance')
    if abs(water.x0 - octanol.x0) > .25*fig.dpi/72:
        raise ValueError('Figure 12 endpoint values do not share an aligned column')
    if max(water.x1, octanol.x1) >= ax.bbox.x1 - 6*fig.dpi/72:
        raise ValueError('Figure 12 endpoint labels crowd the right plot margin')
    output = ROOT / 'figure_exports/Figure_12_PFOA_water_octanol_EDA.png'
    fig.savefig(output, dpi=dpi, bbox_inches='tight', pad_inches=.07)
    plt.close(fig)
    print(output)


if __name__ == '__main__':
    main()
