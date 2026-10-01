#!/usr/bin/env python3
from pathlib import Path
import json, math, sys
import numpy as np

ROOT=Path(__file__).resolve().parent
PDB=ROOT/'build/stage08_endpoint.pdb'
XSC=ROOT/'../08_md_pilot/05_acceptance_npt_1ns/acceptance1ns.restart.xsc'
MOL2_PFO=ROOT/'inputs/pfoa/pfoa_gaff2.mol2'
MOL2_PVB=ROOT/'inputs/polymer/pVBTMA12_gaff2_rct.mol2'
OUT=ROOT/'build'; OUT.mkdir(exist_ok=True)

def parse_xsc(path):
    rows=[]
    for ln in path.read_text().splitlines():
        if not ln.strip() or ln.lstrip().startswith('#'): continue
        p=ln.split()
        if len(p)>=13: rows.append([float(x) for x in p[:13]])
    if not rows: raise SystemExit('No XSC record found')
    r=rows[-1]; a=np.array(r[1:4]); b=np.array(r[4:7]); c=np.array(r[7:10]); o=np.array(r[10:13])
    M=np.vstack([a,b,c])
    if np.max(np.abs(M-np.diag(np.diag(M))))>1e-4: raise SystemExit('Stage09 builder currently requires orthorhombic accepted cell')
    return np.diag(M),o

def mic(d,box): return d-box*np.round(d/box)
def dist(a,b,box): return float(np.linalg.norm(mic(a-b,box)))
def min_dist(A,B,box):
    best=1e9
    for a in A:
        ds=np.linalg.norm((a-B)-box*np.round((a-B)/box),axis=1)
        best=min(best,float(ds.min()))
    return best

def parse_pdb(path):
    atoms=[]
    for ln in path.read_text().splitlines():
        if ln.startswith(('ATOM','HETATM')):
            elem=ln[76:78].strip().upper() or ''.join(c for c in ln[12:16] if c.isalpha())[:2].upper()
            atoms.append(dict(record=ln[:6].strip() or 'ATOM',serial=int(ln[6:11]),name=ln[12:16].strip(),resname=ln[17:20].strip(),chain=ln[21:22],resid=int(ln[22:26]),coord=np.array([float(ln[30:38]),float(ln[38:46]),float(ln[46:54])]),elem=elem))
    return atoms

def parse_mol2(path):
    lines=path.read_text().splitlines(); ai=bi=None; atoms=[]; bonds=[]
    for i,l in enumerate(lines):
        if l.startswith('@<TRIPOS>ATOM'): ai=i+1
        elif l.startswith('@<TRIPOS>BOND'): bi=i; bj=i+1; break
    for l in lines[ai:bi]:
        p=l.split(); typ=p[5].lower(); name=p[1]
        if typ.startswith('f'): e='F'
        elif typ.startswith('o'): e='O'
        elif typ.startswith('n'): e='N'
        elif typ.startswith('c'): e='C'
        elif typ.startswith('h'): e='H'
        else: e='X'
        atoms.append(dict(id=int(p[0]),name=name,coord=np.array(list(map(float,p[2:5]))),type=p[5],charge=float(p[8]),elem=e))
    for l in lines[bj:]:
        if l.startswith('@<TRIPOS>'): break
        p=l.split()
        if len(p)>=4: bonds.append((int(p[1])-1,int(p[2])-1))
    nbr={i:[] for i in range(len(atoms))}
    for i,j in bonds: nbr[i].append(j); nbr[j].append(i)
    return atoms,nbr

def rot_axis(axis,t):
    axis=axis/np.linalg.norm(axis); x,y,z=axis; c=math.cos(t); s=math.sin(t); C=1-c
    return np.array([[c+x*x*C,x*y*C-z*s,x*z*C+y*s],[y*x*C+z*s,c+y*y*C,y*z*C-x*s],[z*x*C-y*s,z*y*C+x*s,c+z*z*C]])
def rot_from_to(a,b):
    a=a/np.linalg.norm(a); b=b/np.linalg.norm(b); v=np.cross(a,b); c=float(np.dot(a,b))
    if c>0.999999:return np.eye(3)
    if c<-0.999999:
        ax=np.cross(a,[1,0,0]);
        if np.linalg.norm(ax)<1e-8: ax=np.cross(a,[0,1,0])
        return rot_axis(ax,math.pi)
    vx=np.array([[0,-v[2],v[1]],[v[2],0,-v[0]],[-v[1],v[0],0]])
    return np.eye(3)+vx+vx@vx/(1+c)

def pdbline(serial,name,resname,resid,xyz,elem):
    return f"ATOM  {serial:5d} {name:<4.4s} {resname:>3.3s} {resid:4d}    {xyz[0]:8.3f}{xyz[1]:8.3f}{xyz[2]:8.3f}  1.00  0.00          {elem:>2.2s}"

box,origin=parse_xsc(XSC); atoms=parse_pdb(PDB)
if len(atoms)!=15973: raise SystemExit(f'Expected 15973 Stage08 atoms, found {len(atoms)}')

pvb=atoms[:374]
if len(pvb)!=374 or any(a['resname'].upper()!='PVB' for a in pvb): raise SystemExit('PVB atom count/residue mapping mismatch in exported Stage08 endpoint')

water_names={'WAT','HOH','TIP3','TP3'}
waters={}; chlor=[]; sodium=[]; other=[]
for a in atoms[374:]:
    rn=a['resname'].upper()
    if rn in water_names: waters.setdefault((a['chain'],a['resid'],a['resname']),[]).append(a)
    elif rn in {'CL-','CL','CLA'} or a['elem']=='CL': chlor.append(a)
    elif rn in {'NA+','NA','SOD'} or a['elem']=='NA': sodium.append(a)
    else: other.append(a)
if len(chlor)!=31 or len(sodium)!=19: raise SystemExit(f'Expected 31 Cl and 19 Na, found Cl={len(chlor)} Na={len(sodium)}; inspect exported residue names')
if other: raise SystemExit(f'Unexpected non-water/non-ion residues after PVB: {sorted(set(a["resname"] for a in other))}')

pfo,nbr=parse_mol2(MOL2_PFO)
pvb_m,nbrp=parse_mol2(MOL2_PVB)
if len(pfo)!=25: raise SystemExit(f'Authoritative PFOA MOL2 should contain 25 atoms, found {len(pfo)}')
if abs(sum(a['charge'] for a in pfo)+1.0)>1e-6: raise SystemExit('Authoritative PFOA MOL2 charge does not sum to -1.000000')
if len(pvb_m)!=374: raise SystemExit(f'Authoritative pVBTMA12 MOL2 should contain 374 atoms, found {len(pvb_m)}')
if abs(sum(a['charge'] for a in pvb_m)-12.0)>1e-4: raise SystemExit('Authoritative pVBTMA12 MOL2 charge does not sum to +12')
if [a['name'] for a in pvb] != [a['name'] for a in pvb_m]: raise SystemExit('Stage08 PVB atom order/names do not match authoritative pVBTMA12 MOL2')
Ns=[i for i,a in enumerate(pvb_m) if a['type'].lower()=='n4']
if len(Ns)!=12: raise SystemExit(f'Expected 12 n4 atoms in PVB mol2, got {len(Ns)}')

Os=[i for i,a in enumerate(pfo) if a['elem']=='O']; Cs=[i for i,a in enumerate(pfo) if a['elem']=='C']
headC=next(i for i in Cs if sum(pfo[j]['elem']=='O' for j in nbr[i])>=2)
headOs=[j for j in nbr[headC] if pfo[j]['elem']=='O']; head_center=np.mean([pfo[i]['coord'] for i in headOs],axis=0)
tail=max(Cs,key=lambda i: np.linalg.norm(pfo[i]['coord']-head_center)); tailvec=pfo[tail]['coord']-head_center
pfo_coords=np.array([a['coord'] for a in pfo]); centered=pfo_coords-head_center
polyheavy=np.array([a['coord'] for a in pvb if a['elem']!='H'])
ioncoords=np.array([a['coord'] for a in chlor+sodium])


trials=[]
for rep in range(2,12):
    ni=Ns[rep-1]; N=pvb[ni]['coord']


    local_delta=np.array([mic(p-N,box) for p in polyheavy])
    local_r=np.linalg.norm(local_delta,axis=1)
    local=local_delta[(local_r>0.5) & (local_r<8.0)]
    if len(local)<4: continue
    inward=np.mean(local,axis=0)
    if np.linalg.norm(inward)<1e-8: continue
    direction=-inward/np.linalg.norm(inward)
    R=rot_from_to(tailvec,direction); aligned=centered@R.T
    for hd in (4.0,4.2,4.4,4.6,4.8):
      target=N+direction*hd
      for deg in range(0,360,15):
        placed=aligned@rot_axis(direction,math.radians(deg)).T+target

        dpoly=min_dist(placed,polyheavy,box); dion=min_dist(placed,ioncoords,box)
        on=min(dist(placed[o],N,box) for o in headOs)
        if dpoly<2.80 or dion<2.80 or not (2.8<=on<=5.0): continue

        score=min(dpoly,dion)-0.05*abs(on-4.0)
        trials.append((score,rep,ni,hd,deg,dpoly,dion,on,placed,N))
if not trials: raise SystemExit('No clash-free deterministic interior PFOA placement found; do not loosen criteria silently')
trials.sort(key=lambda x:(x[0],-x[1],-x[3],-x[4]),reverse=True)
best=trials[0]; score,rep,ni,hd,deg,dpoly,dion,on,placed,N=best


clrank=[]
for c in chlor:
    dp=min(dist(c['coord'],p,box) for p in polyheavy)
    dn=dist(c['coord'],N,box)
    clrank.append((dp,dn,c))
clrank.sort(key=lambda x:(x[0],x[1]),reverse=True)
removed_dp,removed_dn,removed_cl=clrank[0]
if removed_dp < 6.0:
    raise SystemExit(f'No sufficiently remote chloride available for bookkeeping: farthest polymer-heavy distance is {removed_dp:.3f} A (<6.0 A)')
remaining_cl=[c for c in chlor if c is not removed_cl]


remove_waters=set()
for key,wa in waters.items():
    coords=np.array([a['coord'] for a in wa]); oxy=[a['coord'] for a in wa if a['elem']=='O']
    close_any=min_dist(placed,coords,box)<2.0
    close_o=bool(oxy) and min_dist(placed,np.array(oxy),box)<2.4
    if close_any or close_o: remove_waters.add(key)
if len(remove_waters)>25: raise SystemExit(f'Placement would require deleting {len(remove_waters)} waters (>25); inspect rather than continuing')


lines=[]; serial=1; resid=1
for a in pvb: lines.append(pdbline(serial,a['name'],'PVB',resid,a['coord'],a['elem'])); serial+=1
resid+=1
for a,xyz in zip(pfo,placed): lines.append(pdbline(serial,a['name'],'PFO',resid,xyz,a['elem'])); serial+=1
resid+=1
kept_water_atoms=0
for key in sorted(waters,key=lambda x:(x[0],x[1],x[2])):
    if key in remove_waters: continue
    for a in waters[key]: lines.append(pdbline(serial,a['name'],'WAT',resid,a['coord'],a['elem'])); serial+=1; kept_water_atoms+=1
    resid+=1
for a in sodium: lines.append(pdbline(serial,'Na+','Na+',resid,a['coord'],'NA')); serial+=1; resid+=1
for a in remaining_cl: lines.append(pdbline(serial,'Cl-','Cl-',resid,a['coord'],'CL')); serial+=1; resid+=1
lines.append('END')
(OUT/'stage09_start.pdb').write_text('\n'.join(lines)+'\n')

report={
 'source_stage08_atoms':len(atoms),'box_A':box.tolist(),'origin_A':origin.tolist(),
 'selected_repeat':rep,'selected_polymer_atom_index_1based':ni+1,'selected_N_name':pvb[ni]['name'],
 'head_center_target_distance_A':hd,'axial_rotation_deg':deg,'initial_nearest_carboxylate_O_N_A':on,
 'initial_min_PFOA_polymer_heavy_A':dpoly,'initial_min_PFOA_ion_A':dion,
 'removed_chloride_source_serial':removed_cl['serial'],'removed_chloride_distance_to_selected_N_A':removed_dn,
 'removed_chloride_min_distance_to_polymer_heavy_A':removed_dp,'removed_chloride_remote_minimum_required_A':6.0,
 'removed_water_residue_count':len(remove_waters),'kept_water_residue_count':len(waters)-len(remove_waters),
 'expected_Na_count':19,'expected_Cl_count':30,'expected_PFO_count':1,
 'construction_interpretation':'preassociated endpoint; chloride removal is stoichiometric bookkeeping only',
 'candidate_pose_count':len(trials),'placement_minimum_heavy_clearance_required_A':2.80,'placement_nearest_O_N_allowed_A':[2.8,5.0]
}
(OUT/'construction_report.json').write_text(json.dumps(report,indent=2)+'\n')

leap=f'''source leaprc.gaff2\nsource leaprc.water.tip3p\nloadAmberParams frcmod.ionsjc_tip3p\nloadAmberParams inputs/polymer/pVBTMA12.frcmod\nloadAmberParams inputs/pfoa/pfoa.frcmod\nPVB = loadMol2 inputs/polymer/pVBTMA12_gaff2_rct.mol2\nPFO = loadMol2 inputs/pfoa/pfoa_gaff2.mol2\nSYS = loadPdb build/stage09_start.pdb\nset SYS box {{ {box[0]:.6f} {box[1]:.6f} {box[2]:.6f} }}\ncheck SYS\ncharge SYS\nsaveAmberParm SYS build/pVBTMA12_PFOA_assoc.prmtop build/pVBTMA12_PFOA_assoc.inpcrd\nsavePdb SYS build/pVBTMA12_PFOA_assoc.pdb\nquit\n'''
(OUT/'build_stage09.generated.leap').write_text(leap)
(ROOT/'stage09_system.generated.conf').write_text(f'''set stage09_prmtop "build/pVBTMA12_PFOA_assoc.prmtop"\nset stage09_inpcrd "build/pVBTMA12_PFOA_assoc.inpcrd"\nset boxX {box[0]:.6f}\nset boxY {box[1]:.6f}\nset boxZ {box[2]:.6f}\n''')
print(json.dumps(report,indent=2))
