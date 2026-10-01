#!/usr/bin/env python3

from pathlib import Path
import argparse, hashlib, json
from rdkit import Chem, rdBase
from rdkit.Chem import Descriptors, rdMolDescriptors

REPEAT_SMILES = "[*:1]CC(c1ccc(C[N+](C)(C)C)cc1)[*:2]"


def dummy(mol, mapno):
    hits=[a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum()==0 and a.GetAtomMapNum()==mapno]
    if len(hits)!=1:
        raise ValueError(f"Expected exactly one dummy atom with map {mapno}; got {hits}")
    idx=hits[0]
    nbr=[n.GetIdx() for n in mol.GetAtomWithIdx(idx).GetNeighbors()]
    if len(nbr)!=1:
        raise ValueError(f"Dummy atom {mapno} must have exactly one neighbor")
    return idx,nbr[0]


def remove(rw, indices):
    for idx in sorted(indices, reverse=True):
        rw.RemoveAtom(idx)


def connect(a,b):
    da,na=dummy(a,2); db,nb=dummy(b,1)
    off=a.GetNumAtoms()
    rw=Chem.RWMol(Chem.CombineMols(a,b))
    rw.AddBond(na,off+nb,Chem.BondType.SINGLE)
    remove(rw,[da,off+db])
    m=rw.GetMol(); Chem.SanitizeMol(m)
    return m


def build(n):
    if n<1: raise ValueError("n must be >=1")
    unit=Chem.MolFromSmiles(REPEAT_SMILES)
    if unit is None: raise RuntimeError("Could not parse repeat SMILES")
    chain=Chem.Mol(unit)
    for _ in range(1,n): chain=connect(chain,unit)
    rw=Chem.RWMol(chain)
    remove(rw,[a.GetIdx() for a in chain.GetAtoms() if a.GetAtomicNum()==0])
    chain=rw.GetMol(); Chem.SanitizeMol(chain)
    return chain


def aromatic_backbone_centers(m):
    out=[]
    for a in m.GetAtoms():
        if a.GetAtomicNum()!=6 or a.GetIsAromatic(): continue
        if not any(n.GetIsAromatic() for n in a.GetNeighbors()): continue

        if any(n.GetAtomicNum()==7 for n in a.GetNeighbors()): continue
        out.append(a.GetIdx())
    return out


def validation(m,n):
    qn=[a for a in m.GetAtoms() if a.GetAtomicNum()==7 and a.GetFormalCharge()==1 and a.GetDegree()==4]
    rings=[r for r in m.GetRingInfo().AtomRings() if len(r)==6 and all(m.GetAtomWithIdx(i).GetIsAromatic() for i in r)]
    centers=aromatic_backbone_centers(m)
    dmat=Chem.GetDistanceMatrix(m)
    nearest=[]
    for c in centers:
        vals=sorted(int(dmat[c,d]) for d in centers if d!=c)
        if vals: nearest.append(vals[0])
    para=[]
    for ring_tuple in rings:
        ring=set(ring_tuple); subs=[]
        for i in ring:
            for x in m.GetAtomWithIdx(i).GetNeighbors():
                if x.GetIdx() in ring: continue
                if x.GetAtomicNum()!=6 or x.GetIsAromatic(): continue
                typ='benzyl' if any(y.GetAtomicNum()==7 and y.GetFormalCharge()==1 for y in x.GetNeighbors()) else 'backbone'
                subs.append((i,typ))
        b=[i for i,t in subs if t=='backbone']; z=[i for i,t in subs if t=='benzyl']
        para.append(len(b)==1 and len(z)==1 and int(dmat[b[0],z[0]])==3)
    return {
        'rdkit_version': rdBase.rdkitVersion,
        'n_repeats': n,
        'repeat_template': REPEAT_SMILES,
        'canonical_smiles': Chem.MolToSmiles(m,canonical=True),
        'formula': rdMolDescriptors.CalcMolFormula(m),
        'molecular_weight_g_mol': Descriptors.MolWt(m),
        'heavy_atoms': m.GetNumHeavyAtoms(),
        'connected_components': len(Chem.GetMolFrags(m)),
        'dummy_atoms_remaining': sum(a.GetAtomicNum()==0 for a in m.GetAtoms()),
        'formal_charge': sum(a.GetFormalCharge() for a in m.GetAtoms()),
        'quaternary_ammonium_sites': len(qn),
        'aromatic_six_membered_rings': len(rings),
        'aromatic_bearing_backbone_centers': len(centers),
        'nearest_aromatic_bearing_center_graph_distance_bonds': sorted(set(nearest)),
        'all_aromatic_substitutions_para': all(para),
        'sanitized': True,
        'expected_checks_pass': (
            len(Chem.GetMolFrags(m))==1 and
            sum(a.GetAtomicNum()==0 for a in m.GetAtoms())==0 and
            sum(a.GetFormalCharge() for a in m.GetAtoms())==n and
            len(qn)==n and len(rings)==n and len(centers)==n and
            sorted(set(nearest))==([2] if n>1 else []) and all(para)
        )
    }


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('-n','--repeats',type=int,default=12); ap.add_argument('-o','--outdir',default='.')
    args=ap.parse_args(); out=Path(args.outdir); out.mkdir(parents=True,exist_ok=True)
    m=build(args.repeats); v=validation(m,args.repeats)
    stem=f'pVBTMA{args.repeats}'
    smi=out/f'{stem}_topology.smi'; mol=out/f'{stem}_topology.mol'; js=out/f'{stem}_topology_validation.json'
    smi.write_text(v['canonical_smiles']+'\n'); Chem.MolToMolFile(m,str(mol)); js.write_text(json.dumps(v,indent=2)+'\n')
    hashes={}
    for p in [smi,mol,js]: hashes[p.name]=hashlib.sha256(p.read_bytes()).hexdigest()
    (out/f'{stem}_SHA256.json').write_text(json.dumps(hashes,indent=2)+'\n')
    print(json.dumps(v,indent=2))
    if not v['expected_checks_pass']: raise SystemExit(2)

if __name__=='__main__': main()
