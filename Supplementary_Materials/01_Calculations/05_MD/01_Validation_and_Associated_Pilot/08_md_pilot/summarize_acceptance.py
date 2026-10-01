#!/usr/bin/env python3
from pathlib import Path
import json, math, statistics, re
HERE=Path(__file__).resolve().parent
ANA=HERE/"analysis"
DT_PS=2.0

def read_series(path):
    vals=[]
    for line in Path(path).read_text(errors="replace").splitlines():
        if not line.strip() or line.lstrip().startswith("#"): continue
        p=line.split()
        nums=[]
        for x in p:
            try: nums.append(float(x))
            except ValueError: pass
        if len(nums)>=2: vals.append(nums[1])
    if not vals: raise RuntimeError(f"No numeric data parsed from {path}")
    return vals

def stats(v):
    return {"n":len(v),"mean":statistics.fmean(v),"min":min(v),"max":max(v),"sd":statistics.pstdev(v) if len(v)>1 else 0.0}

def longest_true(seq):
    best=cur=0
    for x in seq:
        if x: cur+=1; best=max(best,cur)
        else: cur=0
    return best

def parse_namd_energy(path):
    header=None; rows=[]
    for line in Path(path).read_text(errors="replace").splitlines():
        if line.startswith("ETITLE:"):
            header=line.split()[1:]
        elif line.startswith("ENERGY:") and header:
            parts=line.split()[1:]
            if len(parts)>=len(header):
                d={}
                for k,x in zip(header,parts):
                    try:d[k]=float(x)
                    except ValueError:pass
                rows.append(d)
    return rows

rms=read_series(ANA/"polymer_rmsd.dat")
rg=read_series(ANA/"polymer_rg.dat")
e2e=read_series(ANA/"end_to_end.dat")
t1=read_series(ANA/"terminal1_central_mindist.dat")
t12=read_series(ANA/"terminal12_central_mindist.dat")
img=read_series(ANA/"polymer_self_image.dat")
contact_cut=4.5
c1=[x<contact_cut for x in t1]; c12=[x<contact_cut for x in t12]
block=max(1,round(200.0/DT_PS))
rg_first=statistics.fmean(rg[:block]); rg_last=statistics.fmean(rg[-block:])
rg_rel=abs(rg_last-rg_first)/rg_first if rg_first else math.inf

def parse_namd_xst(path):
    cells=[]
    for line in Path(path).read_text(errors="replace").splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        try: p=[float(x) for x in line.split()]
        except ValueError: continue
        if len(p)<10: continue
        a,b,c=p[1:4],p[4:7],p[7:10]
        def norm(v): return math.sqrt(sum(x*x for x in v))
        def ang(u,v):
            co=sum(x*y for x,y in zip(u,v))/(norm(u)*norm(v))
            co=max(-1.0,min(1.0,co))
            return math.degrees(math.acos(co))
        vol=abs(a[0]*(b[1]*c[2]-b[2]*c[1])-a[1]*(b[0]*c[2]-b[2]*c[0])+a[2]*(b[0]*c[1]-b[1]*c[0]))
        cells.append({"lengths":[norm(a),norm(b),norm(c)],"angles":[ang(b,c),ang(a,c),ang(a,b)],"volume":vol})
    return cells

energy=parse_namd_energy(HERE/"05_acceptance_npt_1ns/acceptance1ns.log")
xst=parse_namd_xst(HERE/"05_acceptance_npt_1ns/acceptance1ns.xst")
def col(*names):
    for n in names:
        v=[r[n] for r in energy if n in r]
        if v:return v
    return []
temp=col("TEMP","TEMPAVG")
press=col("PRESSURE","GPRESSURE","PRESSAVG")
vol=col("VOLUME")

flags=[]; hard=[]
occ1=sum(c1)/len(c1); occ12=sum(c12)/len(c12)
run1=longest_true(c1)*DT_PS; run12=longest_true(c12)*DT_PS
if occ1>0.25 or run1>100: flags.append(f"repeat 1 central-contact persistence: occupancy={occ1:.3f}, longest={run1:.1f} ps")
if occ12>0.25 or run12>100: flags.append(f"repeat 12 central-contact persistence: occupancy={occ12:.3f}, longest={run12:.1f} ps")
minimg=min(img)
if minimg<9.0: hard.append(f"polymer periodic-image minimum {minimg:.3f} A is below 9 A force cutoff")
elif minimg<11.0: flags.append(f"polymer periodic-image minimum {minimg:.3f} A is below 11 A pair-list distance")
if rg_rel>0.10: flags.append(f"Rg first/last 200 ps means differ by {100*rg_rel:.1f}%")
if temp:
    tm=statistics.fmean(temp)
    if abs(tm-310.15)>5: flags.append(f"acceptance mean temperature {tm:.2f} K differs from 310.15 K by >5 K")
    if abs(tm-310.15)>20: hard.append(f"acceptance mean temperature {tm:.2f} K differs from target by >20 K")
if vol and len(vol)>=2:
    n=max(1,len(vol)//5)
    v0=statistics.fmean(vol[:n]); v1=statistics.fmean(vol[-n:]); vd=abs(v1-v0)/v0
    if vd>0.10: flags.append(f"volume first/last 20% means differ by {100*vd:.1f}%")

result={
  "purpose":"chloride-form pVBTMA12 model/protocol acceptance; not a production replica",
  "trajectory_frames":len(rms),"trajectory_interval_ps":DT_PS,
  "polymer_rmsd_A":stats(rms),"radius_of_gyration_A":stats(rg),"end_to_end_A":stats(e2e),
  "Rg_first_200ps_mean_A":rg_first,"Rg_last_200ps_mean_A":rg_last,"Rg_relative_change":rg_rel,
  "terminal_repeat_1_to_central":{"mindist_A":stats(t1),"contact_cutoff_A":contact_cut,"contact_occupancy":occ1,"longest_contiguous_contact_ps":run1},
  "terminal_repeat_12_to_central":{"mindist_A":stats(t12),"contact_cutoff_A":contact_cut,"contact_occupancy":occ12,"longest_contiguous_contact_ps":run12},
  "polymer_periodic_image_mindist_A":stats(img),
  "namd_temperature_K":stats(temp) if temp else None,
  "namd_pressure_bar":stats(press) if press else None,
  "namd_volume_A3":stats(vol) if vol else None,
  "namd_xst_cell": {
      "records": len(xst),
      "mean_lengths_A": [statistics.fmean(c["lengths"][i] for c in xst) for i in range(3)],
      "mean_angles_deg": [statistics.fmean(c["angles"][i] for c in xst) for i in range(3)],
      "volume_A3": stats([c["volume"] for c in xst]),
  } if xst else None,
  "periodic_cell_validation_file":"analysis/box_validation.txt",
  "review_flags":flags,"hard_failures":hard,
  "automated_status":"FAIL" if hard else ("REVIEW" if flags else "NO_AUTOMATED_FLAGS"),
  "interpretation":"Terminal contacts are review flags, not automatic failures. Final Stage 08 acceptance requires trajectory/metric review before production model freeze."
}
(ANA/"acceptance_summary.json").write_text(json.dumps(result,indent=2)+"\n")
lines=["Stage 08 acceptance summary",f"automated_status: {result['automated_status']}",f"frames: {len(rms)} at {DT_PS:.1f} ps",f"Rg mean: {statistics.fmean(rg):.4f} A",f"RMSD mean: {statistics.fmean(rms):.4f} A",f"end-to-end mean: {statistics.fmean(e2e):.4f} A",f"terminal1-central occupancy <4.5 A: {occ1:.4f}; longest {run1:.1f} ps",f"terminal12-central occupancy <4.5 A: {occ12:.4f}; longest {run12:.1f} ps",f"minimum polymer-image distance: {minimg:.4f} A"]
if temp: lines.append(f"mean temperature: {statistics.fmean(temp):.3f} K")
if vol: lines.append(f"mean volume from NAMD ENERGY records: {statistics.fmean(vol):.3f} A^3")
if xst:
    xl=[statistics.fmean(c["lengths"][i] for c in xst) for i in range(3)]
    xa=[statistics.fmean(c["angles"][i] for c in xst) for i in range(3)]
    xv=statistics.fmean(c["volume"] for c in xst)
    lines.append("mean XST cell lengths: " + " x ".join(f"{v:.4f}" for v in xl) + " A")
    lines.append("mean XST cell angles: " + ", ".join(f"{v:.4f}" for v in xa) + " deg")
    lines.append(f"mean XST cell volume: {xv:.3f} A^3")
lines.append("periodic-cell validation: see analysis/box_validation.txt")
lines.append("review_flags:")
lines += ["  - "+x for x in flags] or ["  - none"]
lines.append("hard_failures:")
lines += ["  - "+x for x in hard] or ["  - none"]
lines.append("Final acceptance is not assigned automatically; review metrics/trajectory before model freeze.")
(ANA/"acceptance_summary.txt").write_text("\n".join(lines)+"\n")
print("\n".join(lines))
raise SystemExit(2 if hard else 0)
