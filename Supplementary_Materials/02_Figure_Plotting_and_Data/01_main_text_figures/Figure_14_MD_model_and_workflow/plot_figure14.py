"""Figure 14 panel composition"""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
VIEWS = (('A', r'PFOA$^{-}$–pVBTMA$_{12}^{12+}$ Simulation Box', 'figure14_full_box.png'),
         ('B', r'PFOA$^{-}$-Associated Site', 'figure14_hydrated_site.png'))
ATOM_COLORS = (('C', '#4d4d4d'), ('N', '#354f9c'), ('F', '#47b8a6'),
               (r'Na$^{+}$', '#d68f38'), (r'Cl$^{-}$', '#63a854'))

def main():
    original_height = 4.2
    height = 4.45
    title_fontsize = 11.5
    legend_scale = title_fontsize/9.2
    fig, axes = plt.subplots(1, 2, figsize=(7.4, height))
    fig.subplots_adjust(left=.065, right=.98, top=.81*original_height/height,
                        bottom=.07*original_height/height, wspace=.19)
    handles = [Line2D([], [], marker='o', linestyle='none', color='none',
                      markerfacecolor=color, markeredgecolor=color,
                      markeredgewidth=.5*legend_scale,
                      markersize=7*legend_scale, label=name)
               for name, color in ATOM_COLORS]
    legend = fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.52, .969373),
                        ncol=len(handles), frameon=False, fontsize=title_fontsize,
                        handlelength=.5, handletextpad=.32, columnspacing=1.15)
    for ax, (letter, title, name) in zip(axes, VIEWS):
        ax.set_facecolor('white')
        ax.set_aspect('equal', adjustable='box')
        ax.set_anchor('C')
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color('#4c4c4c'); spine.set_linewidth(.75)
        ax.set_title(title, fontsize=title_fontsize, fontweight='normal', pad=9)
        ax.text(-.05, 1.077, letter, transform=ax.transAxes, fontsize=14,
                fontweight='bold', ha='right', va='bottom', clip_on=False)
        path = HERE / 'structure_renders' / name
        if not path.exists():
            raise FileNotFoundError(f'{path} requires render_figure14_pymol.py in WSL')
        with Image.open(path) as image:
            source = image.convert('RGBA')
            white = Image.new('RGBA', source.size, (255, 255, 255, 255))
            flattened = Image.alpha_composite(white, source).convert('RGB')
            ax.imshow(np.asarray(flattened), aspect='equal', extent=(0, 1, 0, 1),
                      interpolation='hanning')
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    if abs(axes[0].bbox.y1 - axes[1].bbox.y1) * 72 / fig.dpi > .25:
        raise ValueError('Figure 14 panels are not vertically aligned')
    label_boxes = [ax.texts[0].get_window_extent(renderer) for ax in axes]
    letter_offset_pt = abs(label_boxes[0].y0 - label_boxes[1].y0)*72/fig.dpi
    if letter_offset_pt > .25:
        raise ValueError('Figure 14 panel letters are not horizontally aligned')
    legend_box = legend.get_window_extent(renderer)
    gap_pt = (legend_box.y0 - max(box.y1 for box in label_boxes))*72/fig.dpi
    if abs(gap_pt - 8.0) > .1:
        raise ValueError('Figure 14 atom key-to-letter gap changed')
    if any(legend_box.overlaps(ax.title.get_window_extent(renderer)) for ax in axes):
        raise ValueError('Figure 14 atom key overlaps a panel title')
    if any(legend_box.overlaps(letter_box) for letter_box in label_boxes):
        raise ValueError('Figure 14 atom key overlaps a panel letter')
    if any(letter_box.y0 <= ax.title.get_window_extent(renderer).y0 + 3*fig.dpi/72
           for ax, letter_box in zip(axes, label_boxes)):
        raise ValueError('Figure 14 panel letters need to sit above the titles')
    print(f'Legend-to-letter gap: {gap_pt:.3f} pt; A/B letter offset: {letter_offset_pt:.3f} pt')
    out = ROOT / 'figure_exports/Figure_14_MD_observed_frame.png'
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=600, facecolor='white'); plt.close(fig)
    print(out)

if __name__ == '__main__':
    main()
