#!/usr/bin/env python3
from pathlib import Path
import argparse, re, sys, json
ap=argparse.ArgumentParser()
ap.add_argument("log")
ap.add_argument("--expected-last-step",type=int)
ap.add_argument("--minimize",action="store_true")
a=ap.parse_args()
p=Path(a.log)
if not p.is_file() or p.stat().st_size==0:
    raise SystemExit(f"ERROR: missing/empty NAMD log: {p}")
text=p.read_text(errors="replace")
low=text.lower()
fatal_patterns=["fatal error","segmentation fault","cuda error","atoms moving too fast","nan detected","constraint failure"]
hits=[x for x in fatal_patterns if x in low]
energy_steps=[]
for line in text.splitlines():
    if line.startswith("ENERGY:"):
        parts=line.split()
        if len(parts)>1:
            try: energy_steps.append(int(parts[1]))
            except ValueError: pass

normal_marker=("timing:" in low) or ("wallclock:" in low)
issues=[]
if hits: issues.append("fatal markers: "+", ".join(hits))
if not normal_marker: issues.append("no TIMING/WallClock normal-completion marker")
if a.expected_last_step is not None and not a.minimize:
    if not energy_steps: issues.append("no ENERGY records")
    elif max(energy_steps) < a.expected_last_step:
        issues.append(f"last ENERGY step {max(energy_steps)} < expected {a.expected_last_step}")
result={"log":str(p),"normal_completion_marker":normal_marker,"max_energy_step":max(energy_steps) if energy_steps else None,"issues":issues}
print(json.dumps(result,indent=2))
raise SystemExit(2 if issues else 0)
