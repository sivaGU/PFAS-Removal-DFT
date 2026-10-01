
#!/usr/bin/env python3
from pathlib import Path
import json, math
R=Path(__file__).resolve().parent; S10=R.parent/"10_pfoa_associated_production"; A=R/"analysis"
manifest=json.loads((A/"analysis_manifest.json").read_text()); chunks=manifest["chunks_per_replica"]; expected=chunks*500
def norm(v): return math.sqrt(sum(x*x for x in v))
def ang(a,b):
    c=sum(x*y for x,y in zip(a,b))/(norm(a)*norm(b)); c=max(-1,min(1,c)); return math.degrees(math.acos(c))
allout={}; global_issues=[]
for r in range(1,4):
    rr=f"{r:02d}"; v={}
    for line in (A/f"replica_{rr}/vmd_box.dat").read_text().splitlines():
        if line.strip() and not line.startswith("#"):
            p=line.split(); v[int(p[0])]=list(map(float,p[1:7]))
    issues=[]; ld=[]; ad=[]
    if len(v)!=expected: issues.append(f"expected {expected} VMD frames, found {len(v)}")
    for c in range(1,chunks+1):
        start=250000+(c-1)*500000; offset=(c-1)*500
        xst=S10/f"replica_{rr}/{c:02d}_prod_1ns/prod_{c:02d}.xst"
        for line in xst.read_text().splitlines():
            if not line.strip() or line.lstrip().startswith("#"): continue
            p=list(map(float,line.split()))
            if len(p)<10: continue
            step=int(round(p[0])); delta=step-start
            if delta<=0 or delta%1000: continue
            frame=offset+delta//1000-1
            if frame not in v: continue
            a,b,cvec=p[1:4],p[4:7],p[7:10]; xb=[norm(a),norm(b),norm(cvec),ang(b,cvec),ang(a,cvec),ang(a,b)]
            ld.append(max(abs(v[frame][i]-xb[i]) for i in range(3))); ad.append(max(abs(v[frame][i]-xb[i]) for i in range(3,6)))
    if not ld: issues.append("no matching XST/DCD records")
    if ld and max(ld)>0.05: issues.append(f"max length delta {max(ld):.6f} A > 0.05")
    if ad and max(ad)>0.05: issues.append(f"max angle delta {max(ad):.6f} deg > 0.05")
    if any(any(abs(z-90)>0.05 for z in box[3:]) for box in v.values()): issues.append("nonorthorhombic DCD cell detected")
    out={"status":"FAIL" if issues else "PASS","vmd_frames":len(v),"xst_records_compared":len(ld),"max_length_delta_A":max(ld) if ld else None,"max_angle_delta_deg":max(ad) if ad else None,"issues":issues}; allout[f"replica_{rr}"]=out
    (A/f"replica_{rr}/pbc_validation.json").write_text(json.dumps(out,indent=2)+"\n"); global_issues += [f"replica_{rr}: {x}" for x in issues]
summary={"status":"FAIL" if global_issues else "PASS","replicas":allout,"issues":global_issues}; (A/"pbc_validation.json").write_text(json.dumps(summary,indent=2)+"\n")
print("Stage 11 PBC validation: "+summary["status"]); [print(" - "+x) for x in global_issues]; raise SystemExit(1 if global_issues else 0)
