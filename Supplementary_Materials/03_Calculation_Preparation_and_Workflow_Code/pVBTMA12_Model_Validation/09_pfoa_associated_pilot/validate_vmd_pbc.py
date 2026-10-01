#!/usr/bin/env python3
from pathlib import Path
import math,json,re
R=Path(__file__).resolve().parent; A=R/'analysis'
def norm(v): return math.sqrt(sum(x*x for x in v))
def ang(a,b):
 c=sum(x*y for x,y in zip(a,b))/(norm(a)*norm(b)); c=max(-1,min(1,c)); return math.degrees(math.acos(c))
if 'VMD_PBC_STATUS PASS' not in (A/'vmd_pbc.log').read_text(errors='replace'): raise SystemExit('VMD/PBCTools did not PASS')
v={}
for l in (A/'vmd_box.dat').read_text().splitlines():
 if l.strip() and not l.startswith('#'):
  p=l.split(); v[int(p[0])]=list(map(float,p[1:7]))
if len(v)!=500: raise SystemExit(f'Expected 500 VMD frames, found {len(v)}')
x=[]
for l in (R/'03_acceptance_npt_1ns/acceptance1ns.xst').read_text().splitlines():
 if l.strip() and not l.lstrip().startswith('#'):
  p=list(map(float,l.split()));
  if len(p)>=10:
   a,b,c=p[1:4],p[4:7],p[7:10]; x.append((int(round(p[0])),[norm(a),norm(b),norm(c),ang(b,c),ang(a,c),ang(a,b)]))
ld=[];ad=[]
for step,xb in x:
 if step<=0 or step%1000: continue
 f=step//1000-1
 if f in v: ld.append(max(abs(v[f][i]-xb[i]) for i in range(3))); ad.append(max(abs(v[f][i]-xb[i]) for i in range(3,6)))
errs=[]
if not ld: errs.append('no matching XST/DCD records')
if ld and max(ld)>0.05: errs.append(f'max length delta {max(ld):.6f} A >0.05')
if ad and max(ad)>0.05: errs.append(f'max angle delta {max(ad):.6f} deg >0.05')
for f,b in v.items():
 if any(abs(z-90)>0.05 for z in b[3:]): errs.append(f'frame {f} nonorthorhombic'); break
out={'status':'FAIL' if errs else 'PASS','vmd_frames':len(v),'xst_records_compared':len(ld),'max_length_delta_A':max(ld) if ld else None,'max_angle_delta_deg':max(ad) if ad else None,'errors':errs}
(A/'vmd_pbc_validation.json').write_text(json.dumps(out,indent=2)+'\n'); (A/'vmd_pbc_validation.txt').write_text('\n'.join([f"status: {out['status']}",f"VMD frames: {len(v)}",f"XST records compared: {len(ld)}",f"max length delta A: {out['max_length_delta_A']}",f"max angle delta deg: {out['max_angle_delta_deg']}"]+['errors:']+([f'  - {e}' for e in errs] or ['  - none']))+'\n'); print((A/'vmd_pbc_validation.txt').read_text()); raise SystemExit(1 if errs else 0)
