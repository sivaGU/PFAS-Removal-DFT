
#!/usr/bin/env python3
from pathlib import Path
import argparse, json, subprocess
ap=argparse.ArgumentParser(); ap.add_argument("--target-ns",type=int,choices=[5,10],default=5); a=ap.parse_args()
R=Path(__file__).resolve().parent; issues=[]; records=[]
for r in range(1,4):
    rp=f"replica_{r:02d}"
    segments=[("00_equil_npt_500ps/equil",250000)]
    segments += [(f"{c:02d}_prod_1ns/prod_{c:02d}",250000+c*500000) for c in range(1,a.target_ns+1)]
    for rel,expected in segments:
        prefix=R/rp/rel; log=prefix.with_suffix(".log")
        cp=subprocess.run(["python3",str(R/"validate_namd_log.py"),str(log),"--expected-last-step",str(expected)],capture_output=True,text=True)
        records.append({"replica":rp,"segment":rel,"expected_last_step":expected,"validator_exit":cp.returncode})
        if cp.returncode: issues.append(f"{rp}/{rel}: NAMD log validation failed")
        for ext in ["coor","vel","xsc","dcd","xst"]:
            p=prefix.with_suffix("."+ext)
            if not p.is_file() or p.stat().st_size==0: issues.append(f"missing/empty {p.relative_to(R)}")
out={"status":"FAIL" if issues else "PASS","production_ns_per_replica":a.target_ns,"replicas":3,"records":records,"issues":issues}
(R/"stage10_validation.json").write_text(json.dumps(out,indent=2)+"\n")
if issues:
    print("Stage 10 output validation: FAIL"); [print(" - "+x) for x in issues]; raise SystemExit(1)
(R/"stage10_completion.txt").write_text(f"FINAL STATUS: PRODUCTION COMPLETE\nREPLICAS: 3\nPRODUCTION LENGTH PER REPLICA: {a.target_ns} ns\nStage 11 trajectory, PBC, cross-replica, and convergence analysis is required.\nScientific interpretation remains prepared associated-state persistence/rearrangement only.\n")
print(f"Stage 10 output validation: PASS (3 x {a.target_ns} ns)")
