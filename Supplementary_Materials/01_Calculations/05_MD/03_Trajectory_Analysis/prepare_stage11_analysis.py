
#!/usr/bin/env python3
from pathlib import Path
import json
R=Path(__file__).resolve().parent; S10=R.parent/"10_pfoa_associated_production"; A=R/"analysis"
if not (S10/"stage10_completion.txt").is_file(): raise SystemExit("ERROR: Stage 10 completion marker missing")
base=[]; ext=[]
for r in range(1,4):
    base += [S10/f"replica_{r:02d}/{c:02d}_prod_1ns/prod_{c:02d}.dcd" for c in range(1,6)]
    ext += [S10/f"replica_{r:02d}/{c:02d}_prod_1ns/prod_{c:02d}.dcd" for c in range(6,11)]
if not all(p.is_file() and p.stat().st_size for p in base): raise SystemExit("ERROR: incomplete 3 x 5 ns Stage 10 trajectory set")
present=[p.is_file() and p.stat().st_size for p in ext]
if any(present) and not all(present): raise SystemExit("ERROR: partial 5-10 ns extension detected; do not mix replica lengths")
chunks=10 if all(present) else 5; A.mkdir(parents=True,exist_ok=True)
for r in range(1,4):
    d=A/f"replica_{r:02d}"; d.mkdir(parents=True,exist_ok=True)
    lines=["# PBC-sensitive actions precede RMS fitting.",f"parm {S10/'inputs/pVBTMA12_PFOA_assoc.prmtop'}"]
    for c in range(1,chunks+1): lines.append(f"trajin {S10/f'replica_{r:02d}/{c:02d}_prod_1ns/prod_{c:02d}.dcd'} shape")
    lines += ["autoimage anchor :PVB",f"radgyr :PVB out {d/'polymer_rg.dat'} mass",f"distance EndToEnd @1,2,3,4,5,6,7,8,9,10,11,12,13,14 @145,146,147,148,149,150,151,152,153,154,155,156 geom out {d/'end_to_end.dat'}",f"mindist mask1 :PFO@O,O1 mask2 @112 byatom name HeadSelectedN out {d/'pfoa_head_selectedN.dat'}",f"mindist mask1 :PFO@O,O1 mask2 :PVB@N1,N2,N3,N4,N5,N6,N7,N8,N9,N10,N11,N12 byatom name HeadAnyN out {d/'pfoa_head_anyN.dat'}",f"mindist mask1 :PFO&!@H= mask2 :PVB&!@H= byatom name PFOAPolymer out {d/'pfoa_polymer_mindist.dat'}"]


    tail=":PFO@F*,C,C1,C2,C3,C4,C5,C6"


    lines += [
        f'mask "(({tail}<@4.0)&(:PVB&!@H=))" name TailPolymerHeavy4A nselectedout {d/"pfoa_tail_polymer_heavy_contacts_4A.dat"}',
        f'watershell {tail} lower 5.0 upper 5.0 out {d/"pfoa_tail_waters_within_5A.dat"}',
    ]
    lines += [f"minimage PolymerSelfImage :PVB&!@H= :PVB&!@H= out {d/'polymer_self_image.dat'}",f"minimage PFOASelfImage :PFO&!@H= :PFO&!@H= out {d/'pfoa_self_image.dat'}",f"avgbox AverageBox out {d/'average_box.dat'}","check :PVB,PFO",f"rms PolymerRMSD first :PVB&!@H= mass out {d/'polymer_rmsd.dat'}","run"]
    (A/f"replica_{r:02d}.cpptraj").write_text("\n".join(lines)+"\n")
(A/"analysis_manifest.json").write_text(json.dumps({"replicas":3,"chunks_per_replica":chunks,"production_ns_per_replica":chunks,"frame_spacing_ps":2,"expected_frames_per_replica":chunks*500},indent=2)+"\n")
print(f"Prepared Stage 11 analysis for 3 x {chunks} ns")
