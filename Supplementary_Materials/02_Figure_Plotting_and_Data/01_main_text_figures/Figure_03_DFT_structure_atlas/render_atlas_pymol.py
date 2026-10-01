"""Figure 3 PyMOL atlas structures"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/'shared_helpers'))
import render_structure_3d as mol
from atlas_records import records

COLORS=('#d62728','#1f77b4')


def contacts_for(path):
    structure=mol.read_xyz(path,-1)
    elements,coords=structure.elements,structure.coords
    bonds=mol.infer_bonds(elements,coords,scale=1.10)
    neighbors=[set() for _ in elements]
    for i,j in bonds:
        neighbors[i].add(j);neighbors[j].add(i)
    nitrogen=[i for i,el in enumerate(elements) if el=='N']
    if len(nitrogen)!=1:raise ValueError(f'Expected one cation nitrogen in {path}')
    cation={nitrogen[0]};queue=[nitrogen[0]]
    while queue:
        for j in neighbors[queue.pop()]-cation:
            cation.add(j);queue.append(j)
    candidates=sorted((float(np.linalg.norm(coords[o]-coords[h])),o,h)
        for o,e in enumerate(elements) if e=='O' and o not in cation
        for h in cation if elements[h]=='H')
    chosen=[];used_h=set()
    for dist,o,h in candidates:
        if h in used_h:continue
        if dist>3.5:break
        chosen.append((dist,mol.Contact(i=o,j=h,color=COLORS[len(chosen)],label='')))
        used_h.add(h)
        if len(chosen)==2:break
    if len(chosen)!=2:raise ValueError(f'No two O···H separations below 3.5 Å in {path}')
    return structure,chosen


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--mamba',type=Path,default=mol.PYMOL_RUNNER)
    ap.add_argument('--env',default=mol.PYMOL_ENV)
    ap.add_argument('--contacts-only',action='store_true')
    args=ap.parse_args()
    mol.PYMOL_RUNNER=args.mamba
    mol.PYMOL_ENV=args.env
    folder=HERE/'pymol_structure_renders'
    folder.mkdir(exist_ok=True)
    rows=[]
    for item in records():
        structure,pairs=contacts_for(item['xyz'])
        name=f"{item['group']}_{item['xyz'].stem}.png"
        rows.append(dict(group=item['group'],species=item['species'],solvent=item['solvent'],
            xyz=str(item['xyz'].relative_to(ROOT)),render=str((folder/name).relative_to(ROOT)),
            red_O_atom=pairs[0][1].i+1,red_H_atom=pairs[0][1].j+1,
            red_A=f'{pairs[0][0]:.3f}',blue_O_atom=pairs[1][1].i+1,
            blue_H_atom=pairs[1][1].j+1,blue_A=f'{pairs[1][0]:.3f}'))
        if not args.contacts_only:
            mol.draw_structure_pymol(
                input_path=item['xyz'],structure=structure,out_path=folder/name,
                contacts=[pair[1] for pair in pairs],hide_h=False,show_indices=False,
                title=None,frame=-1,dpi=650,width=3400,
                height=2350 if item['group'] in ('A','B') else 3100,
                no_bonds=False,sphere_scale=.25,stick_radius=.13,ray=True,
                pymol_projection='orthoscopic',no_crop=False,keep_script=None)
            print('PyMOL:',name,flush=True)
    with (HERE/'verified_contact_distances.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=rows[0].keys())
        writer.writeheader();writer.writerows(rows)
    print('Verified contact pairs for',len(rows),'optimized XYZ structures')

if __name__=='__main__':main()
