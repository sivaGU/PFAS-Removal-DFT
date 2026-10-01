#!/usr/bin/env python3
from pathlib import Path
import csv,json,math
from rdkit import Chem,rdBase
from rdkit.Chem import rdMolDescriptors
ROOT=Path(__file__).resolve().parents[1]
REF=ROOT/'01_model_definition'/'pVBTMA12_model_definition.smi'
FILES=[ROOT/'02_3d_structure_generation'/x for x in ('pVBTMA12_gen3d.sdf','pVBTMA12_gen3d_uffmin.sdf','pVBTMA12_gen3d_mmffmin.sdf')]
def from_smi(p):
    m=Chem.MolFromSmiles(Path(p).read_text().strip()); Chem.AssignStereochemistry(m,cleanIt=True,force=True); return m
def from_sdf(p):
    m=Chem.MolFromMolFile(str(p),removeHs=False,sanitize=True)
    if m is None: raise ValueError(f'Cannot parse {p}')
    Chem.AssignAtomChiralTagsFromStructure(m,confId=0,replaceExistingTags=True); Chem.AssignStereochemistry(m,cleanIt=True,force=True); return m
def norm(m):
    q=Chem.RemoveHs(Chem.Mol(m),sanitize=True); Chem.AssignStereochemistry(q,cleanIt=True,force=True); return q
def smiles(m,iso):
    q=norm(m)
    if not iso: Chem.RemoveStereochemistry(q)
    return Chem.MolToSmiles(q,canonical=True,isomericSmiles=iso)
def stats(m):
    q=norm(m)
    return {'formula':rdMolDescriptors.CalcMolFormula(q),'formal_charge':sum(a.GetFormalCharge() for a in q.GetAtoms()),'heavy_atoms':q.GetNumHeavyAtoms(),'components':len(Chem.GetMolFrags(q)),'dummy_atoms':sum(a.GetAtomicNum()==0 for a in q.GetAtoms()),'quaternary_ammonium_sites':sum(a.GetAtomicNum()==7 and a.GetFormalCharge()==1 and a.GetDegree()==4 for a in q.GetAtoms()),'aromatic_six_membered_rings':sum(len(r)==6 and all(q.GetAtomWithIdx(i).GetIsAromatic() for i in r) for r in q.GetRingInfo().AtomRings())}
def mindist(m):
    conf=m.GetConformer(); bonds={tuple(sorted((b.GetBeginAtomIdx(),b.GetEndAtomIdx()))) for b in m.GetBonds()}; heavy=[a.GetIdx() for a in m.GetAtoms() if a.GetAtomicNum()>1]; best=None
    for ii,i in enumerate(heavy):
        pi=conf.GetAtomPosition(i)
        for j in heavy[ii+1:]:
            if tuple(sorted((i,j))) in bonds: continue
            pj=conf.GetAtomPosition(j); d=math.dist((pi.x,pi.y,pi.z),(pj.x,pj.y,pj.z)); best=d if best is None or d<best else best
    return best
ref=from_smi(REF); rc=smiles(ref,False); ri=smiles(ref,True); rs=stats(ref); rows=[]; ok=True
for p in FILES:
    m=from_sdf(p); st=stats(m); centers=Chem.FindMolChiralCenters(norm(m),includeUnassigned=True,includeCIP=True)
    cp=(smiles(m,False)==rc and st==rs); sp=(smiles(m,True)==ri and len([x for x in centers if x[1]!='?'])==11); ok &= cp and sp
    rows.append({'file':p.name,'constitutional_pass':cp,'exact_stereoisomer_pass':sp,'minimum_nonbonded_heavy_distance_A':mindist(m),**st})
report={'rdkit_version':rdBase.rdkitVersion,'reference_model_definition':REF.name,'files':rows,'overall_pass':ok}
Path('structure_validation.json').write_text(json.dumps(report,indent=2)+'\n')
with open('structure_validation.csv','w',newline='') as fh:
    w=csv.DictWriter(fh,fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
print(json.dumps(report,indent=2)); raise SystemExit(0 if ok else 2)
