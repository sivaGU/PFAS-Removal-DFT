#!/usr/bin/env python3
from pathlib import Path
import json

HERE = Path(__file__).resolve().parent
j = json.loads((HERE / "repeat_groups.json").read_text())
heavy = {int(k): v for k, v in j["repeat_heavy_groups_1based"].items()}


def mask(ids):
    return "@" + ",".join(map(str, ids))


term1 = mask(heavy[1])
term12 = mask(heavy[12])
central = mask(sum((heavy[i] for i in [5, 6, 7, 8]), []))

text = f'''# Generated from repeat_groups.json; do not hand-edit atom assignments.
parm ../07_solvation_ion_validation/pVBTMA12_12Cl_015MNaCl_pilot.prmtop
# NAMD/VMD DCD unit-cell records from this trajectory are read with explicit
# CHARMM shape-matrix decoding. Keep all PBC-sensitive actions BEFORE the RMS
# best-fit action: CPPTRAJ RMS fitting modifies coordinates and can rotate the
# unit-cell vectors seen by later actions.
trajin 05_acceptance_npt_1ns/acceptance1ns.dcd shape

autoimage
# PBC-sensitive / box-sensitive measurements on the unrotated cell.
radgyr :1 out analysis/polymer_rg.dat mass
distance EndToEnd {term1} {term12} geom out analysis/end_to_end.dat
mindist mask1 {term1} mask2 {central} byatom name Term1Central out analysis/terminal1_central_mindist.dat
mindist mask1 {term12} mask2 {central} byatom name Term12Central out analysis/terminal12_central_mindist.dat
minimage PolymerSelfImage :1&!@H= :1&!@H= out analysis/polymer_self_image.dat
avgbox AverageBox out analysis/average_box.dat
check :1
# RMS fitting is deliberately LAST so its coordinate/cell rotation cannot
# contaminate PBC-sensitive analysis above.
rms PolymerRMSD first :1&!@H= mass out analysis/polymer_rmsd.dat
run
'''
(HERE / "acceptance_analysis.generated.cpptraj").write_text(text)
print("Wrote acceptance_analysis.generated.cpptraj with PBC-sensitive actions before RMS fitting")
