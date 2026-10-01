"""Figure 10 solvent comparison"""
from pathlib import Path
import csv,sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from shared_helpers.figure_style import GREEN,PURPLE,EXPORT_DPI
from shared_helpers.annotation_style import VALUE_SIZE_PT,VALUE_BOX,energy_grid
from shared_helpers.letter_alignment import check_letters

def main(dpi=EXPORT_DPI):
    rows={r['bound_state_solvent']:r for r in csv.DictReader((Path(__file__).parent/'input_data/exchange.csv').open())}
    assert set(rows)=={'water','octanol'}
    fig,axes=plt.subplots(2,1,figsize=(4.7,6.4));fig.subplots_adjust(left=.26,right=.91,top=.89,bottom=.11,hspace=.43)
    for ax,(col,title,ylim,letter) in zip(axes,[('DeltaE_exchange_kcal_mol',r'$\Delta E_{\mathrm{exchange}}$',(-10.2,-8.8),'A'),('DeltaG_exchange_kcal_mol',r'$\Delta G_{\mathrm{exchange}}$',(-.8,.25),'B')]):
        vals=[float(rows[x][col]) for x in ['water','octanol']]
        ax.plot([0,1],vals,color='#777777',lw=1)
        for i,(value,color) in enumerate(zip(vals,[GREEN,PURPLE])):
            ax.scatter(i,value,s=60,c=color,zorder=4)
            ax.annotate(f'{value:+.2f}',(i,value),xytext=(0,10),textcoords='offset points',ha='center',color=color,fontweight='bold',fontsize=VALUE_SIZE_PT,bbox=VALUE_BOX,zorder=5)
        ax.set(ylim=ylim,xlim=(-.35,1.35));ax.set_xticks([0,1],['Water','1-octanol'])
        energy_grid(ax)
        ax.set_title(title,pad=8);ax.set_ylabel('Energy (kcal/mol)',labelpad=9)
        ax.text(-.16,1.13,letter,transform=ax.transAxes,fontweight='bold',fontsize=12)
    output=ROOT/'figure_exports/Figure_10_PFOA_water_octanol_exchange.png'
    check_letters(fig, axes, columns=((0,1),), rows=())
    fig.savefig(output,dpi=dpi,bbox_inches='tight',pad_inches=.06);plt.close(fig);print(output)

if __name__=='__main__':main()
