VMD loading guide for PFOA exchange inspection

Directory:
work_amber/16_vmd_pfoa_exchange_inspection

Main mechanism/movie view:
1. Load solvated_resin_pfoa_unbound_reference.prmtop as an Amber7 Parm file.
2. Add pfoa_guided_approach.dcd as a DCD trajectory.
3. Add pfoa_guided_relax_250ps.dcd as a DCD trajectory to the same molecule.

This shows:
- PFOA starting in the bulk-like reference state.
- Guided approach toward the selected ammonium-rich site.
- Unrestrained relaxation after steering is turned off.

Bound endpoint comparison:
1. Load solvated_resin_pfoa_exchange.prmtop as an Amber7 Parm file.
2. Add resin_pfoa_exchange_npt_1ns.dcd as a DCD trajectory.

Unbound endpoint comparison:
1. Load solvated_resin_pfoa_unbound_reference.prmtop as an Amber7 Parm file.
2. Add resin_pfoa_unbound_reference_npt_0p5ns.dcd as a DCD trajectory.

Representative static snapshots:
- resin_pfoa_exchange_npt_1ns_final_imaged.pdb
- resin_pfoa_unbound_reference_npt_0p5ns_final_imaged.pdb
- pfoa_guided_approach_final_imaged.pdb
- pfoa_guided_relax_250ps_final_imaged.pdb

Suggested VMD representations:
- Resin: resname R48, NewCartoon or Licorice, colored tan/gray.
- PFOA: resname PFO, VDW or Licorice, colored by element.
- Chloride: resname Cl- or name Cl*, VDW, colored green.
- Water: resname WAT, Lines, transparent or hidden for clarity.
- Selected ammonium site: serial 176 177 178 179 180, VDW/Licorice.

For the cleanest mechanism movie, hide most water first and focus on:
resname R48 or resname PFO or resname Cl-
