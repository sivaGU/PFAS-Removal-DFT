#!/usr/bin/env python3
from pathlib import Path
from collections import defaultdict
import json, math
import parmed as pmd
import numpy as np
ROOT=Path(__file__).resolve().parent
prm=ROOT/'build/pVBTMA12_PFOA_assoc.prmtop'; crd=ROOT/'build/pVBTMA12_PFOA_assoc.inpcrd'
rep=json.loads((ROOT/'build/construction_report.json').read_text())
s=pmd.load_file(str(prm),str(crd))
labels=[r.name for r in s.residues]
counts={}
for x in labels: counts[x]=counts.get(x,0)+1
na=sum(v for k,v in counts.items() if k.upper() in {'NA+','NA','SOD'})
cl=sum(v for k,v in counts.items() if k.upper() in {'CL-','CL','CLA'})
pfo=counts.get('PFO',0); pvb=counts.get('PVB',0)
water=sum(v for k,v in counts.items() if k.upper() in {'WAT','HOH','TIP3','TP3'})
q=sum(a.charge for a in s.atoms)
issues=[]
if pvb!=1: issues.append(f'PVB residue count {pvb} != 1')
if pfo!=1: issues.append(f'PFO residue count {pfo} != 1')
allowed={'PVB','PFO','WAT','HOH','TIP3','TP3','NA+','NA','SOD','CL-','CL','CLA'}
unknown=[k for k in counts if k.upper() not in allowed]
if unknown: issues.append(f'unexpected residue names: {unknown}')
for r in s.residues:
    if r.name=='PVB' and len(r.atoms)!=374: issues.append(f'PVB atom count {len(r.atoms)} != 374')
    if r.name=='PFO' and len(r.atoms)!=25: issues.append(f'PFO atom count {len(r.atoms)} != 25')
if na!=19: issues.append(f'Na count {na} != 19')
if cl!=30: issues.append(f'Cl count {cl} != 30')
if abs(q)>1e-4: issues.append(f'total charge {q:.8f} not neutral')
if water!=rep['kept_water_residue_count']: issues.append(f'water count {water} != expected {rep["kept_water_residue_count"]}')
for r in s.residues:
    if r.name=='PFO':
        qr=sum(a.charge for a in r.atoms)
        if abs(qr+1)>1e-4: issues.append(f'PFO charge {qr:.8f} != -1')
    if r.name=='PVB':
        qr=sum(a.charge for a in r.atoms)
        if abs(qr-12)>1e-4: issues.append(f'PVB charge {qr:.8f} != +12')
if s.box is None: issues.append('missing periodic box')
else:
    for got,want in zip(s.box[:3],rep['box_A']):
        if abs(float(got)-float(want))>0.02: issues.append(f'box length {got:.6f} differs from accepted Stage08 {want:.6f}')
    if any(abs(float(x)-90.0)>0.02 for x in s.box[3:6]): issues.append(f'non-orthorhombic output box angles: {s.box[3:6]}')


def residue_class(name):
    upper=name.upper()
    if upper in {'WAT','HOH','TIP3','TP3'}: return 'WAT'
    if upper in {'NA+','NA','SOD'}: return 'NA'
    if upper in {'CL-','CL','CLA'}: return 'CL'
    return upper

def pdb_residues(path):
    residues=[]; current=None
    for ln in path.read_text().splitlines():
        if not ln.startswith(('ATOM','HETATM')): continue
        fields=ln.split()
        key=(fields[3],fields[4])
        atom={'name':fields[2], 'coord':np.asarray(list(map(float,fields[5:8])),float)}
        if key!=current:
            residues.append({'name':fields[3], 'atoms':[]}); current=key
        residues[-1]['atoms'].append(atom)
    return residues

start_res=pdb_residues(ROOT/'build/stage09_start.pdb')
leap_res=pdb_residues(ROOT/'build/pVBTMA12_PFOA_assoc.pdb')
top_res=[{'name':r.name,'atoms':[{'name':a.name,'coord':None} for a in r.atoms]} for r in s.residues]
coords=np.asarray(s.coordinates,float)
for residue,record in zip(s.residues,top_res):
    for atom,atom_record in zip(residue.atoms,record['atoms']): atom_record['coord']=coords[atom.idx]

def grouped(residues):
    result=defaultdict(list)
    for residue in residues: result[residue_class(residue['name'])].append(residue)
    return result

def class_runs(residues):
    runs=[]
    for residue in residues:
        cls=residue_class(residue['name'])
        if runs and runs[-1]['class']==cls: runs[-1]['residue_count']+=1
        else: runs.append({'class':cls,'residue_count':1})
    return runs

start_groups=grouped(start_res); leap_groups=grouped(leap_res); top_groups=grouped(top_res)
classes=('PVB','PFO','WAT','NA','CL')
identity_issues=[]; mapped_start=[]; mapped_top=[]; mapped_class=[]
for cls in classes:
    if not (len(start_groups[cls])==len(leap_groups[cls])==len(top_groups[cls])):
        identity_issues.append(f'{cls} residue counts differ across constructed PDB, LEaP PDB, and topology')
        continue
    for ordinal,(start_r,leap_r,top_r) in enumerate(zip(start_groups[cls],leap_groups[cls],top_groups[cls]),1):
        start_names=[a['name'] for a in start_r['atoms']]
        leap_names=[a['name'] for a in leap_r['atoms']]
        top_names=[a['name'] for a in top_r['atoms']]
        if not (start_names==leap_names==top_names):
            identity_issues.append(f'{cls} residue {ordinal} atom-name sequence differs')
            continue
        for start_a,top_a in zip(start_r['atoms'],top_r['atoms']):
            mapped_start.append(start_a['coord']); mapped_top.append(top_a['coord']); mapped_class.append(cls)
if identity_issues: issues.extend(identity_issues)

start_flat=np.asarray([a['coord'] for r in start_res for a in r['atoms']],float)
direct_raw_max=None; mapped_raw_max=None; mapped_mic_max=None; mapped_mic_rms=None; mapped_mic_gt_002=None
origin_shift=None; aligned_mic_max=None; aligned_mic_rms=None; aligned_mic_gt_002=None; class_validation={}
if len(start_flat)!=len(s.atoms):
    issues.append(f'start PDB atom count {len(start_flat)} != topology atom count {len(s.atoms)}')
elif not identity_issues and len(mapped_start)==len(s.atoms):
    direct_raw_max=float(np.linalg.norm(coords-start_flat,axis=1).max())
    mapped_start=np.asarray(mapped_start,float); mapped_top=np.asarray(mapped_top,float)
    box=np.asarray(s.box[:3],float)
    delta=mapped_top-mapped_start
    mapped_raw=np.linalg.norm(delta,axis=1); mapped_raw_max=float(mapped_raw.max())
    mapped_mic_delta=delta-box*np.round(delta/box)
    mapped_mic=np.linalg.norm(mapped_mic_delta,axis=1)
    mapped_mic_max=float(mapped_mic.max()); mapped_mic_rms=float(np.sqrt(np.mean(mapped_mic**2)))
    mapped_mic_gt_002=int(np.sum(mapped_mic>0.02))
    origin_shift=np.median(delta,axis=0)
    expected_shift=box/2.0
    if np.max(np.abs(origin_shift-expected_shift))>0.02:
        issues.append(f'unexpected LEaP global origin shift: {origin_shift.tolist()} vs half-box {expected_shift.tolist()} A')
    residual=delta-origin_shift
    residual-=box*np.round(residual/box)
    aligned=np.linalg.norm(residual,axis=1)
    aligned_mic_max=float(aligned.max()); aligned_mic_rms=float(np.sqrt(np.mean(aligned**2)))
    aligned_mic_gt_002=int(np.sum(aligned>0.02))
    labels=np.asarray(mapped_class)
    for cls in classes:
        values=aligned[labels==cls]
        class_validation[cls]={'atom_count':int(len(values)),'max_residual_A':float(values.max()),'rms_residual_A':float(np.sqrt(np.mean(values**2))),'atoms_over_0.02_A':int(np.sum(values>0.02))}
    if aligned_mic_max>0.02:
        issues.append(f'LEaP mapped coordinate residual too large after origin translation: max {aligned_mic_max:.4f} A')
else:
    issues.append(f'mapped atom count {len(mapped_start)} != topology atom count {len(s.atoms)}')

post_on=None
try:
    pvb_res=next(r for r in s.residues if r.name=='PVB'); pfo_res=next(r for r in s.residues if r.name=='PFO')
    nidx=int(rep['selected_polymer_atom_index_1based'])-1
    N=coords[nidx]
    oxy=[coords[a.idx] for a in pfo_res.atoms if a.name in {'O','O1'}]
    box=np.asarray(rep['box_A'],float)
    def mic(v): return v-box*np.round(v/box)
    post_on=min(float(np.linalg.norm(mic(o-N))) for o in oxy)
    if abs(post_on-float(rep['initial_nearest_carboxylate_O_N_A']))>0.03:
        issues.append(f'prepared O-N distance changed across LEaP: {post_on:.4f} vs {rep["initial_nearest_carboxylate_O_N_A"]:.4f} A')
except Exception as e:
    issues.append(f'could not verify prepared O-N geometry: {e}')

max_bond=None
if s.box is not None:
    box=np.asarray(s.box[:3],float); vals=[]
    for b in s.bonds:
        v=coords[b.atom1.idx]-coords[b.atom2.idx]; v=v-box*np.round(v/box); vals.append(float(np.linalg.norm(v)))
    if vals:
        max_bond=max(vals)
        if max_bond>2.2: issues.append(f'bond longer than 2.2 A after build: {max_bond:.4f} A')
out={'status':'PASS' if not issues else 'FAIL','natom':len(s.atoms),'residue_counts':counts,'Na_count':na,'Cl_count':cl,'water_count':water,'total_charge':q,'box':None if s.box is None else [float(x) for x in s.box[:6]],'constructed_residue_class_order':class_runs(start_res),'leap_residue_class_order':class_runs(leap_res),'topology_residue_class_order':class_runs(top_res),'atom_identity_mapping_status':'PASS' if not identity_issues else 'FAIL','direct_index_raw_max_A':direct_raw_max,'mapped_raw_max_A':mapped_raw_max,'mapped_minimum_image_max_before_origin_alignment_A':mapped_mic_max,'mapped_minimum_image_rms_before_origin_alignment_A':mapped_mic_rms,'mapped_minimum_image_atoms_over_0.02_A_before_origin_alignment':mapped_mic_gt_002,'leap_global_origin_shift_A':None if origin_shift is None else origin_shift.tolist(),'mapped_aligned_minimum_image_max_A':aligned_mic_max,'mapped_aligned_minimum_image_rms_A':aligned_mic_rms,'mapped_aligned_minimum_image_atoms_over_0.02_A':aligned_mic_gt_002,'coordinate_validation_by_class':class_validation,'postbuild_nearest_carboxylate_O_N_A':post_on,'max_bond_minimum_image_A':max_bond,'issues':issues}
(ROOT/'build/build_validation.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
raise SystemExit(0 if not issues else 1)
