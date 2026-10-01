"""Figure 3 atlas composition"""
from __future__ import annotations

import csv
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image
from atlas_records import records

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
W,H=7.2,9.25
RED='#d62728';BLUE='#1f77b4'

def main():
    items=records()
    with (HERE/'verified_contact_distances.csv').open() as f:
        contacts={(r['group'],r['species']):r for r in csv.DictReader(f)}
    images=[]
    for item in items:
        image=HERE/'pymol_structure_renders'/f"{item['group']}_{item['xyz'].stem}.png"
        if not image.exists():
            raise FileNotFoundError(f'Missing {image}; run render_atlas_pymol.py in WSL')
        images.append(image)

    fig=plt.figure(figsize=(W,H),facecolor='white')
    def border(x,y,w,h,lw=.55):
        fig.patches.append(Rectangle((x/W,y/H),w/W,h/H,transform=fig.transFigure,
            fill=False,edgecolor='#4c4c4c',linewidth=lw))
    def pane(x,y,w,h,item,img):
        border(x,y,w,h,.45)
        fig.text((x+.06)/W,(y+h-.09)/H,item['species'],fontsize=7.5,
                 fontweight='bold',color='#222',va='center',ha='left')
        line=contacts[item['group'],item['species']]
        bx=x+w-.44;by=y+h-.25
        border(bx,by,.44,.25,.35)
        fig.text((bx+.22)/W,(by+.17)/H,f"{float(line['red_A']):.2f} Å",
                 fontsize=6.4,ha='center',va='center',color=RED)
        fig.text((bx+.22)/W,(by+.065)/H,f"{float(line['blue_A']):.2f} Å",
                 fontsize=6.4,ha='center',va='center',color=BLUE)
        ax=fig.add_axes(((x+.035)/W,(y+.04)/H,(w-.07)/W,(h-.345)/H))
        with Image.open(img) as raw:
            im=raw.convert('RGBA');bbox=im.getchannel('A').getbbox()
            if bbox:im=im.crop(bbox)
            white=Image.new('RGBA',im.size,(255,255,255,255))
            im=Image.alpha_composite(white,im).convert('RGB')
        ax.imshow(im,aspect='equal',interpolation='hanning');ax.axis('off')

    group_w=(W-.36-.12)/2
    e_w=2.48
    groups=[('A','BTMA⁺ (r²SCAN-3c)',.18,6.30,group_w,2.70),
            ('B','BTMA⁺ (ωB97X-D3)',.18+group_w+.12,6.30,group_w,2.70),
            ('C','DVB-BTMA⁺ (r²SCAN-3c)',.18,2.51,group_w,3.58),
            ('D','DVB-BTMA⁺ (ωB97X-D3)',.18+group_w+.12,2.51,group_w,3.58),
            ('E','DVB-BTMA⁺ (ωB97X-D3, 1-octanol)',
             (W-e_w)/2,.18,e_w,2.12)]
    for letter,title,x,y,w,h in groups:
        fig.text((x-.07)/W,(y+h-.015)/H,letter,fontsize=10,
                 ha='left',va='center',fontweight='bold',color='#222')
        heading=fig.text((x+w/2)/W,(y+h-.105)/H,title,fontsize=8.1,
                         ha='center',va='center',fontweight='normal',color='#222')
        if letter=='E':
            fig.canvas.draw()
            box=heading.get_window_extent(fig.canvas.get_renderer())
            if box.x0 < x*fig.dpi-2 or box.x1 > (x+w)*fig.dpi+2:
                raise ValueError('Figure 3 E title extends beyond its pane')
        if letter=='E':
            pane(x,y,w,h-.25,items[16],images[16])
        else:
            for j in range(4):
                inner_w=w/2;inner_h=(h-.25)/2
                col=j%2;row=j//2
                xx=x+col*inner_w;yy=y+(1-row)*inner_h
                index='ABCD'.index(letter)*4+j
                pane(xx,yy,inner_w,inner_h,items[index],images[index])
    output=ROOT/'figure_exports'/'Figure_03_DFT_structure_atlas.png'
    output.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(output,dpi=600,facecolor='white')
    plt.close(fig)
    print(output)

if __name__=='__main__':main()
