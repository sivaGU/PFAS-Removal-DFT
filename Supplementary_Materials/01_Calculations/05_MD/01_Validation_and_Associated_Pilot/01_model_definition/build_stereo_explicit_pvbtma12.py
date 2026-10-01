#!/usr/bin/env python3

from pathlib import Path
import argparse, hashlib, json, random
from rdkit import Chem, rdBase
from rdkit.Chem import rdMolDescriptors, rdDepictor

SEED = 20260914
N_STEREO = 11

ALLOWED_PLUS_COUNTS = {5, 6}
SAME_DIAD_RANGE = (4, 6)
MAX_RUN = 2


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def constitutional_smiles(m):
    q = Chem.Mol(m)
    for a in q.GetAtoms():
        a.SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)
        if a.HasProp('_CIPCode'):
            a.ClearProp('_CIPCode')
    Chem.RemoveStereochemistry(q)
    return Chem.MolToSmiles(q, canonical=True, isomericSmiles=False)


def aromatic_backbone_centers(m):
    centers=[]
    for a in m.GetAtoms():
        if a.GetAtomicNum()!=6 or a.GetIsAromatic():
            continue
        if not any(n.GetIsAromatic() for n in a.GetNeighbors()):
            continue
        if any(n.GetAtomicNum()==7 for n in a.GetNeighbors()):
            continue  # benzyl CH2
        centers.append(a.GetIdx())
    return centers


def ordered_centers(m):

    centers=aromatic_backbone_centers(m)
    if len(centers)!=12:
        raise ValueError(f'Expected 12 aryl-bearing backbone centers, found {len(centers)}')
    d=Chem.GetDistanceMatrix(m)
    adj={c:[] for c in centers}
    for i,c in enumerate(centers):
        for x in centers[i+1:]:
            if int(d[c,x])==2:
                adj[c].append(x); adj[x].append(c)
    ends=[c for c in centers if len(adj[c])==1]
    if len(ends)!=2:
        raise ValueError(f'Expected two chain endpoints among centers, got {ends}')

    start_candidates=[]
    for c in ends:
        a=m.GetAtomWithIdx(c)
        if a.GetTotalNumHs()==1:
            start_candidates.append(c)
    if len(start_candidates)!=1:
        raise ValueError(f'Could not uniquely identify CH3-end stereogenic endpoint: {start_candidates}')
    start=start_candidates[0]
    order=[start]; prev=None; cur=start
    while True:
        nxt=[x for x in adj[cur] if x!=prev]
        if not nxt: break
        if len(nxt)!=1: raise ValueError('Backbone center graph is not a simple path')
        prev,cur=cur,nxt[0]; order.append(cur)
    if len(order)!=12:
        raise ValueError(f'Expected 12 ordered centers, got {len(order)}')
    if m.GetAtomWithIdx(order[-1]).GetTotalNumHs()!=2:
        raise ValueError('Final aryl-bearing center is not terminal CH2(aryl)')
    return order


def max_run(seq):
    best=cur=1
    for i in range(1,len(seq)):
        if seq[i]==seq[i-1]:
            cur+=1; best=max(best,cur)
        else:
            cur=1
    return best


def choose_sequence():
    rng=random.Random(SEED)
    attempt=0
    while True:
        attempt+=1
        seq=[rng.choice([1,-1]) for _ in range(N_STEREO)]
        plus=sum(x==1 for x in seq)
        same=sum(seq[i]==seq[i+1] for i in range(len(seq)-1))
        if (plus in ALLOWED_PLUS_COUNTS and
            SAME_DIAD_RANGE[0] <= same <= SAME_DIAD_RANGE[1] and
            max_run(seq) <= MAX_RUN):
            return seq,attempt


def neighbor_role_order(m,c,prev_center,next_center):


    a=m.GetAtomWithIdx(c)
    nbrs=[n.GetIdx() for n in a.GetNeighbors()]
    aromatic=[n.GetIdx() for n in a.GetNeighbors() if n.GetIsAromatic()]
    aliph=[n.GetIdx() for n in a.GetNeighbors() if n.GetAtomicNum()==6 and not n.GetIsAromatic()]
    if len(aromatic)!=1 or len(aliph)!=2 or a.GetTotalNumHs()!=1:
        raise ValueError(f'Unexpected local environment at center {c}: nbrs={nbrs}')

    path_prev=Chem.GetShortestPath(m,c,prev_center)
    path_next=Chem.GetShortestPath(m,c,next_center)
    role=[path_prev[1], path_next[1], aromatic[0]]
    if nbrs != role:
        raise ValueError(
            f'RDKit neighbor order changed at center {c}. Expected prev,next,aryl={role}; got {nbrs}. '
            'Do not silently continue because local parity mapping depends on this order.'
        )
    return role


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--constitutional-smi', default='pVBTMA12_constitutional_reference_smiles.txt')
    ap.add_argument('--out-prefix', default='pVBTMA12_atactic_seed20260914')
    args=ap.parse_args()
    src=Path(args.constitutional_smi)
    m=Chem.MolFromSmiles(src.read_text().strip())
    if m is None: raise SystemExit('Could not parse constitutional reference SMILES')
    order=ordered_centers(m)
    stereo_centers=order[:-1]
    seq,attempt=choose_sequence()


    for i,(c,sign) in enumerate(zip(stereo_centers,seq)):
        prev_center = order[i-1] if i>0 else None
        next_center = order[i+1]
        if i==0:

            a=m.GetAtomWithIdx(c)
            nbrs=[n.GetIdx() for n in a.GetNeighbors()]
            aromatic=[n.GetIdx() for n in a.GetNeighbors() if n.GetIsAromatic()]
            aliph=[n.GetIdx() for n in a.GetNeighbors() if n.GetAtomicNum()==6 and not n.GetIsAromatic()]
            path_next=Chem.GetShortestPath(m,c,next_center)
            next_atom=path_next[1]
            prev_atoms=[x for x in aliph if x!=next_atom]
            role=[prev_atoms[0], next_atom, aromatic[0]]
            if nbrs!=role:
                raise ValueError(f'Unexpected repeat-1 neighbor order; expected {role}, got {nbrs}')
        else:
            neighbor_role_order(m,c,prev_center,next_center)
        tag = Chem.ChiralType.CHI_TETRAHEDRAL_CW if sign==1 else Chem.ChiralType.CHI_TETRAHEDRAL_CCW
        m.GetAtomWithIdx(c).SetChiralTag(tag)

    Chem.AssignStereochemistry(m, cleanIt=True, force=True)
    iso=Chem.MolToSmiles(m, canonical=True, isomericSmiles=True)

    rt=Chem.MolFromSmiles(iso)
    Chem.AssignStereochemistry(rt,cleanIt=True,force=True)
    found=Chem.FindMolChiralCenters(rt,includeUnassigned=True,includeCIP=True)
    assigned=[x for x in found if x[1] != '?']
    if len(assigned)!=11:
        raise ValueError(f'Expected 11 assigned stereocenters after SMILES round trip, got {found}')
    if constitutional_smiles(rt) != constitutional_smiles(Chem.MolFromSmiles(src.read_text().strip())):
        raise ValueError('Constitutional graph changed after stereochemical serialization')

    out=Path(args.out_prefix)
    smi=out.with_suffix('.smi')
    mol=out.with_suffix('.mol')
    js=out.with_name(out.name+'_stereo_spec.json')
    smi.write_text(iso+'\n')

    ref=Chem.Mol(rt)
    rdDepictor.Compute2DCoords(ref)
    Chem.WedgeMolBonds(ref, ref.GetConformer())
    Chem.MolToMolFile(ref,str(mol),includeStereo=True)
    same=sum(seq[i]==seq[i+1] for i in range(10))
    spec={
        'purpose':'stereo-explicit atactic-like finite pVBTMA12 topology; no 3D coordinates selected here',
        'rdkit_version':rdBase.rdkitVersion,
        'selection_seed':SEED,
        'accepted_random_draw_number':attempt,
        'selection_criteria':{
            'n_stereogenic_backbone_centers':N_STEREO,
            'plus_count_allowed':sorted(ALLOWED_PLUS_COUNTS),
            'same_handed_diads_allowed_inclusive':list(SAME_DIAD_RANGE),
            'maximum_same_handed_run':MAX_RUN,
            'selection_uses_geometry_or_energy':False,
        },
        'local_parity_sequence_repeat_1_to_11':['+' if x==1 else '-' for x in seq],
        'plus_count':sum(x==1 for x in seq),
        'minus_count':sum(x==-1 for x in seq),
        'same_handed_diads':same,
        'opposite_handed_diads':10-same,
        'maximum_run':max_run(seq),
        'classification':'mixed/atactic-like finite realization',
        'repeat_12':'terminal CH2(aryl); not stereogenic',
        'canonical_constitutional_smiles':constitutional_smiles(rt),
        'canonical_isomeric_smiles':iso,
        'roundtrip_rdkit_chiral_centers':found,
        'formula':rdMolDescriptors.CalcMolFormula(rt),
        'formal_charge':sum(a.GetFormalCharge() for a in rt.GetAtoms()),
        'heavy_atoms':rt.GetNumHeavyAtoms(),
        'quaternary_ammonium_sites':sum(a.GetAtomicNum()==7 and a.GetFormalCharge()==1 and a.GetDegree()==4 for a in rt.GetAtoms()),
    }
    js.write_text(json.dumps(spec,indent=2)+'\n')
    hashes={p.name:sha256(p) for p in [smi,mol,js]}
    out.with_name(out.name+'_SHA256.json').write_text(json.dumps(hashes,indent=2)+'\n')
    print(json.dumps(spec,indent=2))

if __name__=='__main__': main()
