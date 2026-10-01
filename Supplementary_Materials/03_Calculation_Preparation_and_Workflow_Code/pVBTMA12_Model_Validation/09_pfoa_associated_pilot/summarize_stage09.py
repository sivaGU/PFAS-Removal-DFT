#!/usr/bin/env python3
from pathlib import Path
import json, statistics, math
R=Path(__file__).resolve().parent; A=R/'analysis'; rep=json.loads((R/'build/construction_report.json').read_text())
DT_PS=2.0

def series(name):
    vals=[]
    for l in (A/name).read_text(errors='replace').splitlines():
        if not l.strip() or l.lstrip().startswith('#'): continue
        nums=[]
        for x in l.split():
            try: nums.append(float(x))
            except ValueError: pass
        if len(nums)>=2: vals.append(nums[1])  # first numeric column after frame; important for minimage A1/A2 outputs
    return vals

def stats(v): return {'n':len(v),'mean':statistics.fmean(v),'sd':statistics.pstdev(v) if len(v)>1 else 0.0,'min':min(v),'max':max(v)}
def parse_namd_energy(path):
    header=None; rows=[]
    for line in Path(path).read_text(errors='replace').splitlines():
        if line.startswith('ETITLE:'): header=line.split()[1:]
        elif line.startswith('ENERGY:') and header:
            parts=line.split()[1:]
            if len(parts)>=len(header):
                d={}
                for k,x in zip(header,parts):
                    try:d[k]=float(x)
                    except ValueError: pass
                rows.append(d)
    return rows

def col(rows,*names):
    for n in names:
        v=[r[n] for r in rows if n in r]
        if v:return v
    return []
def window(v,ps):
    n=max(1,round(ps/DT_PS)); return v[-n:]
def assoc_block(v,ps):
    w=window(v,ps); return {'distance_A':stats(w),'fraction_below_5A':sum(x<5.0 for x in w)/len(w)}

rg=series('polymer_rg.dat'); rms=series('polymer_rmsd.dat'); hs=series('pfoa_head_selectedN.dat'); ha=series('pfoa_head_anyN.dat'); pm=series('pfoa_polymer_mindist.dat'); pi=series('polymer_self_image.dat'); fi=series('pfoa_self_image.dat')
errs=[]
for n,v in [('rg',rg),('rms',rms),('head_selected',hs),('head_any',ha),('pfoa_polymer',pm),('polymer_image',pi),('pfoa_image',fi)]:
    if len(v)!=500: errs.append(f'{n}: expected 500 frames, found {len(v)}')
for n,v in [('polymer_image',pi),('pfoa_image',fi)]:
    if v and min(v)<9.0: errs.append(f'{n}: periodic image distance below 9 A ({min(v):.3f})')
energy=parse_namd_energy(R/'03_acceptance_npt_1ns/acceptance1ns.log'); temp=col(energy,'TEMP','TEMPAVG'); vol=col(energy,'VOLUME')
if temp and abs(statistics.fmean(temp)-310.15)>20: errs.append(f'acceptance mean temperature {statistics.fmean(temp):.2f} K differs from target by >20 K')
occ5=sum(x<5.0 for x in ha)/len(ha) if ha else None
out={
 'technical_status':'FAIL' if errs else 'PASS',
 'scientific_interpretation':'prepared preassociated-state persistence only; no spontaneous binding/displacement claim',
 'selected_repeat':rep['selected_repeat'],'selected_N_atom_index_1based':rep['selected_polymer_atom_index_1based'],'initial_nearest_O_N_A':rep['initial_nearest_carboxylate_O_N_A'],
 'polymer_RMSD_A':stats(rms) if rms else None,'polymer_Rg_A':stats(rg) if rg else None,
 'PFOA_head_selectedN_A':stats(hs) if hs else None,'PFOA_head_any_ammonium_A':stats(ha) if ha else None,'PFOA_polymer_heavy_mindist_A':stats(pm) if pm else None,
 'fraction_head_anyN_below_5A':occ5,'final_head_anyN_A':ha[-1] if ha else None,
 'head_anyN_final_200ps':assoc_block(ha,200) if ha else None,'head_anyN_final_100ps':assoc_block(ha,100) if ha else None,
 'polymer_self_image_A':stats(pi) if pi else None,'PFOA_self_image_A':stats(fi) if fi else None,
 'namd_temperature_K':stats(temp) if temp else None,'namd_volume_A3':stats(vol) if vol else None,
 'association_is_acceptance_gate':False,'errors':errs}
(A/'stage09_summary.json').write_text(json.dumps(out,indent=2)+'\n')
lines=['Stage 09 prepared-PFOA acceptance summary',f"technical status: {out['technical_status']}",'PFOA association is an observation, not a technical pass/fail gate.',f"selected repeat: {out['selected_repeat']}",f"initial nearest carboxylate O-N: {out['initial_nearest_O_N_A']:.3f} A"]
if ha:
    lines += [f"mean nearest PFOA head-to-any-ammonium distance: {statistics.fmean(ha):.3f} A",f"fraction <5 A: {occ5:.3f}",f"final nearest head-to-any-ammonium distance: {ha[-1]:.3f} A",f"final 200 ps fraction <5 A: {out['head_anyN_final_200ps']['fraction_below_5A']:.3f}",f"final 100 ps fraction <5 A: {out['head_anyN_final_100ps']['fraction_below_5A']:.3f}"]
if pm: lines.append(f"mean PFOA-polymer heavy-atom minimum distance: {statistics.fmean(pm):.3f} A")
if pi: lines.append(f"minimum polymer-image distance: {min(pi):.3f} A")
if fi: lines.append(f"minimum PFOA-image distance: {min(fi):.3f} A")
if temp: lines.append(f"mean acceptance temperature: {statistics.fmean(temp):.3f} K")
lines += ['errors:']+([f'  - {e}' for e in errs] or ['  - none'])
(A/'stage09_summary.txt').write_text('\n'.join(lines)+'\n'); print('\n'.join(lines)); raise SystemExit(1 if errs else 0)
