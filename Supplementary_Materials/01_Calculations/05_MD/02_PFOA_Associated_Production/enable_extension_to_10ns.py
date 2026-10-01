
#!/usr/bin/env python3
from pathlib import Path
import argparse
ap=argparse.ArgumentParser(description="Create continuation chunks 06-10 without changing chunks 01-05.")
ap.add_argument("--confirm-extend",action="store_true"); a=ap.parse_args()
if not a.confirm_extend: raise SystemExit("Refusing: rerun with --confirm-extend only after Stage 11 reports EXTEND_TO_10NS")
R=Path(__file__).resolve().parent
if "PRODUCTION LENGTH PER REPLICA: 5 ns" not in (R/"stage10_completion.txt").read_text(errors="replace"):
    raise SystemExit("Stage 10 does not have a validated 3 x 5 ns completion marker")
for replica in range(1,4):
    rep=f"replica_{replica:02d}"
    previous="05_prod_1ns/prod_05"
    for chunk in range(6,11):
        p=R/rep/f"{chunk:02d}_prod_1ns.conf"
        if p.exists(): raise SystemExit(f"Refusing to overwrite {p.relative_to(R)}")
        seed=202609300+replica*20+chunk; start=250000+(chunk-1)*500000; current=f"{chunk:02d}_prod_1ns/prod_{chunk:02d}"
        p.write_text(f'''source ../stage10_common.conf
parmfile ../inputs/pVBTMA12_PFOA_assoc.prmtop
ambercoor ../inputs/pVBTMA12_PFOA_assoc.inpcrd
bincoordinates {previous}.coor
binvelocities {previous}.vel
extendedSystem {previous}.xsc
firsttimestep {start}
seed {seed}
langevin on
langevinTemp 310.15
langevinPiston on
langevinPistonTarget 1.01325
langevinPistonPeriod 100.0
langevinPistonDecay 50.0
langevinPistonTemp 310.15
useGroupPressure yes
DCDfile {current}.dcd
DCDfreq 1000
outputName {current}
run 500000
''')
        previous=current
print("Created continuation configs 06-10 for all replicas; existing 5 ns files were not changed.")
