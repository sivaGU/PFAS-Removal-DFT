
#!/usr/bin/env python3
from pathlib import Path
import argparse, json
ap=argparse.ArgumentParser()
ap.add_argument("log")
ap.add_argument("--expected-last-step",type=int,required=True)
a=ap.parse_args(); p=Path(a.log)
if not p.is_file() or p.stat().st_size == 0: raise SystemExit(f"ERROR: missing/empty NAMD log: {p}")
text=p.read_text(errors="replace"); low=text.lower()
fatal=[x for x in ["fatal error","segmentation fault","cuda error","atoms moving too fast","nan detected","constraint failure"] if x in low]
steps=[]
for line in text.splitlines():
    if line.startswith("ENERGY:"):
        try: steps.append(int(line.split()[1]))
        except (ValueError,IndexError): pass
issues=[]
if fatal: issues.append("fatal markers: "+", ".join(fatal))
if "timing:" not in low and "wallclock:" not in low: issues.append("no TIMING/WallClock normal-completion marker")
if not steps: issues.append("no ENERGY records")
elif max(steps) < a.expected_last_step: issues.append(f"last ENERGY step {max(steps)} < expected {a.expected_last_step}")
print(json.dumps({"log":str(p),"max_energy_step":max(steps) if steps else None,"issues":issues},indent=2))
raise SystemExit(2 if issues else 0)
