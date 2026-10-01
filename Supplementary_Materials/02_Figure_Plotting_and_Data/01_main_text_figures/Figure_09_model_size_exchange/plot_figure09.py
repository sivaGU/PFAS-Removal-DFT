"""Figure 9 model comparison"""
from pathlib import Path
import csv
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from shared_helpers.figure_style import ORANGE, GREEN, EXPORT_DPI
from shared_helpers.annotation_style import VALUE_SIZE_PT, VALUE_BOX, energy_grid
from shared_helpers.letter_alignment import check_letters

def main(dpi=EXPORT_DPI):
    data=list(csv.DictReader((Path(__file__).parent/'input_data/exchange.csv').open()))
    pfas=['6:2 FTCA','PFHxA','PFOA','PFOS']
    assert len(data)==8 and {r['PFAS'] for r in data}==set(pfas)
    fig,axs=plt.subplots(2,1,figsize=(6.3,7.3),sharex=True)
    fig.subplots_adjust(left=.18,right=.94,top=.89,bottom=.115,hspace=.32)
    for ax,(key,label,lims) in zip(axs,[('DeltaE_kcal_mol',r'$\Delta E_{\mathrm{exchange}}$',(-11.7,3)),('DeltaG_kcal_mol',r'$\Delta G_{\mathrm{exchange}}$',(-2,9.5))]):
        for i,p in enumerate(pfas):
            a,b=[float(next(r[key] for r in data if r['PFAS']==p and r['model']==m)) for m in ['BTMA','DVB-BTMA']]
            ax.plot([i,i],[a,b],color='#888888',lw=.9,zorder=1)
            ax.scatter([i,i],[a,b],c=[ORANGE,GREEN],s=37,zorder=3)
            shift = 5 if i == 0 else -5 if i == 3 else 0
            ax.annotate(f'{a:+.2f}',(i,a),xytext=(shift,9),textcoords='offset points',ha='center',fontsize=VALUE_SIZE_PT,fontweight='bold',color=ORANGE,bbox=VALUE_BOX,zorder=5)
            ax.annotate(f'{b:+.2f}',(i,b),xytext=(shift,-14),textcoords='offset points',ha='center',fontsize=VALUE_SIZE_PT,fontweight='bold',color=GREEN,bbox=VALUE_BOX,zorder=5)
        ax.set_xlim(-.35, 3.35)
        ax.set_ylim(*lims);energy_grid(ax)
        ax.set_ylabel('Energy (kcal/mol)',labelpad=8);ax.set_title(label,pad=8)
        ax.text(-.11,1.13,'A' if key.startswith('DeltaE') else 'B',transform=ax.transAxes,fontweight='bold',fontsize=12)
    axs[-1].set_xticks(range(4),pfas)
    fig.legend([plt.Line2D([],[],marker='o',ls='',color=c) for c in (ORANGE,GREEN)],['BTMA⁺','DVB-BTMA⁺'],loc='upper center',bbox_to_anchor=(.55,.975),ncol=2,frameon=False)
    output=ROOT/'figure_exports/Figure_09_model_size_exchange.png'
    check_letters(fig, axs, columns=((0,1),), rows=())
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    margin = 3 * fig.dpi / 72
    for ax in axs:
        for label in (t for t in ax.texts if t.get_bbox_patch() is not None):
            box = label.get_bbox_patch().get_window_extent(renderer)
            frame = ax.bbox
            if (box.x0 < frame.x0 + margin or box.x1 > frame.x1 - margin or
                    box.y0 < frame.y0 + margin or box.y1 > frame.y1 - margin):
                raise ValueError(f'Figure 9 value {label.get_text()} covers a plot border')
    fig.savefig(output,dpi=dpi,bbox_inches='tight',pad_inches=.08);plt.close(fig);print(output)

if __name__=='__main__': main()
