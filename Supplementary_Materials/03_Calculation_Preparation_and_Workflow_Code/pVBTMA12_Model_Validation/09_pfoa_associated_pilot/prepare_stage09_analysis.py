#!/usr/bin/env python3
from pathlib import Path
import json
R=Path(__file__).resolve().parent
rep=json.loads((R/'build/construction_report.json').read_text())
nidx=rep['selected_polymer_atom_index_1based']
text=f'''# Generated Stage09 analysis. PBC-sensitive actions precede RMS fitting.
parm build/pVBTMA12_PFOA_assoc.prmtop
trajin 03_acceptance_npt_1ns/acceptance1ns.dcd shape
autoimage anchor :PVB
radgyr :PVB out analysis/polymer_rg.dat mass
mindist mask1 :PFO@O,O1 mask2 @{nidx} byatom name HeadSelectedN out analysis/pfoa_head_selectedN.dat
mindist mask1 :PFO@O,O1 mask2 :PVB@N1,N2,N3,N4,N5,N6,N7,N8,N9,N10,N11,N12 byatom name HeadAnyN out analysis/pfoa_head_anyN.dat
mindist mask1 :PFO&!@H= mask2 :PVB&!@H= byatom name PFOAPolymer out analysis/pfoa_polymer_mindist.dat
minimage PolymerSelfImage :PVB&!@H= :PVB&!@H= out analysis/polymer_self_image.dat
minimage PFOASelfImage :PFO&!@H= :PFO&!@H= out analysis/pfoa_self_image.dat
avgbox AverageBox out analysis/average_box.dat
check :PVB,PFO
# RMS fitting LAST to avoid rotating the cell for PBC-sensitive actions.
rms PolymerRMSD first :PVB&!@H= mass out analysis/polymer_rmsd.dat
run
'''
(R/'stage09_analysis.generated.cpptraj').write_text(text)
print(f'selected_N_atom_index_1based={nidx}')
