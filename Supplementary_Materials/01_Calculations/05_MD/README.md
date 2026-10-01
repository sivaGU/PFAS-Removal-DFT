# pVBTMA12 Molecular Dynamics Source Files

The system uses the 12-site pVBTMA12 model. All production replicas use the
same Stage 09 starting coordinates, separate equilibration, and independent
velocity seeds. Within a replica, numbered production chunks are consecutive
restarts; replica directories are independent.

| Folder | Contents |
|---|---|
| `01_Validation_and_Associated_Pilot/01_model_definition` through `09_pfoa_associated_pilot` | Stereo-defined graph, UFF structure, RCT charge model, GAFF2/frcmod, dry and solvated LEaP builds, oligomer pilot, and PFOA-associated construction/pilot. |
| `02_PFOA_Associated_Production` | Common topology/restart inputs and three replica directories. Each replica contains NPT equilibration followed by `01_prod_1ns` through `10_prod_1ns`, with NAMD configurations, DCDs, XSTs, logs, and restart provenance. |
| `03_Trajectory_Analysis` | CPPTRAJ and VMD/PBCTools checks, time-series data, block summaries, and cross-replica analysis outputs used in the MD figures and tables. |

`01_Validation_and_Associated_Pilot/04_charge_parameterization/pVBTMA12_gaff2_rct.mol2`
is the production full-oligomer charge/type input. The RCT 5-mer AM1-BCC
files document the charge source; direct whole-12-mer AM1-BCC was a
comparison, not the production model. Stage 09's
`build/pVBTMA12_PFOA_assoc.prmtop` is identical to the topology in the Stage
10 `inputs/` directory.

The complete raw production DCD chunks are retained individually so that their
time and restart relationships remain inspectable. `PRODUCTION_SHA256SUMS.txt`
lists the production files. The movies in `../../05_MD_Animations` are derived
from these trajectories.
