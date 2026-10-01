
#!/usr/bin/env python3
from pathlib import Path
import hashlib, re
R=Path(__file__).resolve().parent; issues=[]
if "FINAL STATUS: ACCEPTED" not in (R/"inputs/stage09_completion.txt").read_text(errors="replace"):
    issues.append("Stage 09 handoff is not marked ACCEPTED")
expected={}
for line in (R/"INPUT_SHA256SUMS.txt").read_text().splitlines():
    if line.strip(): h,n=line.split(None,1); expected[n.strip()]=h
for name,h in expected.items():
    p=R/name
    if not p.is_file(): issues.append(f"missing input: {name}")
    elif hashlib.sha256(p.read_bytes()).hexdigest()!=h: issues.append(f"input hash mismatch: {name}")
seeds=[]
for rep in range(1,4):
    rp=f"replica_{rep:02d}"
    files=[R/rp/"00_equil_npt_500ps.conf"]+[R/rp/f"{c:02d}_prod_1ns.conf" for c in range(1,6)]
    for p in files:
        if not p.is_file(): issues.append(f"missing config: {p.relative_to(R)}"); continue
        t=p.read_text()
        m=re.search(r"^seed\s+(\d+)",t,re.M)
        if not m: issues.append(f"missing seed: {p.relative_to(R)}")
        else: seeds.append(int(m.group(1)))
        if "DCDfreq 1000" not in t: issues.append(f"unexpected DCD frequency: {p.relative_to(R)}")
    if "run 250000" not in files[0].read_text(): issues.append(f"wrong equilibration length: {rp}")
    for p in files[1:]:
        if "run 500000" not in p.read_text(): issues.append(f"wrong production chunk length: {p.relative_to(R)}")
if len(seeds)!=len(set(seeds)): issues.append("duplicate NAMD seeds")
print("Stage 10 static package validation: "+("FAIL" if issues else "PASS"))
for x in issues: print(" - "+x)
raise SystemExit(1 if issues else 0)
