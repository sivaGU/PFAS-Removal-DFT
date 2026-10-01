# Calculation Folder Map

All paths are relative to `Supplementary_Materials/01_Calculations`.

| Folder | Contents |
|---|---|
| `01_BTMA` | Aqueous minimal-site BTMA optimization, frequency, and EDA calculations. |
| `02_Extended_Monomer/water` | Aqueous DVB-BTMA optimization, frequency, and EDA calculations, including the r2SCAN-3c ion-pair and isolated-ion records used in Table S11. |
| `02_Extended_Monomer/octanol_corrected` | SMD 1-octanol PFOA and chloride optimization/frequency records and the PFOA EDA input-output set. |
| `03_NBO/DVB-BTMA+PFOA-` | DVB-BTMA/PFOA NBO and NAO calculation. `PFOA-Ch_wb97xd3_opt.out.gz` contains the complete compressed ORCA output. |
| `04_PES/R4N+PFOA-` | PFOA/BTMA scan input, scan geometries, and complete compressed ORCA output. The gzip is stored as three binary parts under `out/`; run `python reconstruct_pes_output.py` in that directory to reconstruct and verify it. |
| `05_MD/01_Validation_and_Associated_Pilot` | pVBTMA12 model definition, parameterization, dry and solvated builds, oligomer pilot, and PFOA-associated system construction and pilot. |
| `05_MD/02_PFOA_Associated_Production` | Three independently seeded NPT replicas, each stored as ten consecutive 1 ns production chunks. |
| `05_MD/03_Trajectory_Analysis` | CPPTRAJ and VMD/PBCTools inputs and outputs, per-replica time series, block summaries, and contact analyses. |
| `06_GOAT_Conformer_Sensitivity` | Six selected DVB-BTMA, PFOA, and chloride alternative conformers with r2SCAN-3c optimization and frequency records. |

Raw octanol and conformer-sensitivity filenames retain their original `R4N`
stems. Replica directories are independent; numbered production chunks within
one replica are consecutive restarts.
