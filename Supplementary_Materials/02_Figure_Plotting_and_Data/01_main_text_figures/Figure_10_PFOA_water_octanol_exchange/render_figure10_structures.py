"""Figure 10 molecular assets"""
from pathlib import Path
import importlib.util
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE=Path(__file__).resolve().parent
SOURCE_SCRIPT=HERE.parent/'Figure_04_BTMA_exchange_and_structures/render_figure04_structures.py'
spec=importlib.util.spec_from_file_location('figure03_structure_renderer',SOURCE_SCRIPT)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
module.SOURCE=HERE/'input_data'

def render_xyz(source, path, dpi=1200):
    elements,xyz=module.read_xyz(source)
    plane,_,_=module.project(elements,xyz)
    fig,ax=plt.subplots(figsize=(5.2,3.8));fig.patch.set_alpha(0);ax.patch.set_alpha(0)
    for i,j in module.bonds(elements,xyz):
        middle=(plane[i,:2]+plane[j,:2])/2
        for k,pt in [(i,plane[i,:2]),(j,plane[j,:2])]:
            ax.plot([pt[0],middle[0]],[pt[1],middle[1]],color=module.COLORS[elements[k]],lw=1.15 if elements[k]=='H' else 2.1,zorder=1)
    for k in np.argsort(plane[:,2]):
        el=elements[k];ax.scatter(*plane[k,:2],s=12 if el=='H' else 42,c=module.COLORS[el],edgecolor='#555555' if el!='H' else 'none',lw=.25,zorder=2)
    low=plane[:,:2].min(axis=0);high=plane[:,:2].max(axis=0);span=high-low
    ax.set(xlim=(low[0]-.07*span[0],high[0]+.07*span[0]),ylim=(low[1]-.09*span[1],high[1]+.09*span[1]))
    ax.set_aspect('equal');ax.axis('off')
    path.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(path,dpi=dpi,transparent=True,bbox_inches='tight',pad_inches=.04);plt.close(fig)
    print(path)

if __name__=='__main__':
    for solvent in ('water','octanol'):
        render_xyz(HERE/'input_data'/f'{solvent}_PFOA_complex.xyz',HERE/'structure_renders'/f'{solvent}_PFOA_complex.png')
