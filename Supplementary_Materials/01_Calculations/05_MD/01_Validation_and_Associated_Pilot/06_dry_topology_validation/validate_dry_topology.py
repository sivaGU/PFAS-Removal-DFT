#!/usr/bin/env python3
from pathlib import Path
import parmed as pmd

def mol2_bonds(path):
    lines=Path(path).read_text(errors='replace').splitlines(); inside=False; pairs=set(); nat=0
    for line in lines:
        if line.startswith('@<TRIPOS>ATOM'): inside='atom'; continue
        if line.startswith('@<TRIPOS>BOND'): inside='bond'; continue
        if line.startswith('@<TRIPOS>') and not line.startswith('@<TRIPOS>ATOM') and not line.startswith('@<TRIPOS>BOND'): inside=False; continue
        if inside=='atom' and line.strip(): nat+=1
        if inside=='bond' and line.strip():
            f=line.split(); pairs.add(tuple(sorted((int(f[1])-1,int(f[2])-1))))
    return nat,pairs
p=pmd.load_file('pVBTMA12_dry.prmtop','pVBTMA12_dry.inpcrd'); q=sum(a.charge for a in p.atoms); nat,b0=mol2_bonds('../04_charge_parameterization/pVBTMA12_gaff2_rct.mol2'); b1={tuple(sorted((b.atom1.idx,b.atom2.idx))) for b in p.bonds}
print(f'atoms_prmtop: {len(p.atoms)}\natoms_mol2: {nat}\nbonds_prmtop: {len(b1)}\nbonds_mol2: {len(b0)}\npartial_charge_sum: {q:.8f}\nbond_index_set_match: {b0==b1}\nbox: {p.box}')
ok=(len(p.atoms)==nat and b0==b1 and abs(q-12)<1e-5 and p.box is None)
raise SystemExit(0 if ok else 2)
