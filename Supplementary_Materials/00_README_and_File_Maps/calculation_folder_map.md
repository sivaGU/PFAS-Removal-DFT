# Calculation Folder Map

All calculation folders live under `Supplementary_Materials/01_Calculations`.
These folders are copied and pruned versions of the working calculation
directories. They contain main input/output files and input/output geometries
where available.

## 01_BTMA

Path: `01_Calculations/01_BTMA`

Contents:
- `01_Optimization/r2scan-3c`
- `01_Optimization/wb97x-d3`
- `02_Frequency/r2scan-3c`
- `02_Frequency/wb97x-d3`
- `03_EDA/r2scan-3c`
- `03_EDA/wb97x-d3`

Purpose:
BTMA model optimization, frequency, and EDA/NOCV calculations supporting BTMA
exchange energetics, thermochemistry, and BTMA EDA analyses.

## 02_Extended_Monomer

Path: `01_Calculations/02_Extended_Monomer`

Contents:
- `water/01_Optimization/xtb-goat`
- `water/01_Optimization/r2scan-3c`
- `water/01_Optimization/wb97x-d3`
- `water/02_Frequency/r2scan-3c`
- `water/02_Frequency/wb97x-d3`
- `water/03_EDA/wb97x-d3`
- `octanol_72.5/01_Optimization/r2scan-3c`
- `octanol_72.5/01_Optimization/wb97x-d3`
- `octanol_72.5/02_Frequency/wb97x-d3`
- `octanol_72.5/03_EDA/wb97x-d3`

Purpose:
Extended-monomer model calculations supporting model-size comparison,
implicit-solvent comparison, and extended-monomer EDA analyses. The octanol
branch uses the project convention `octanol_72.5` for the epsilon-72.5 implicit
solvent setup and contains fewer runs than the water branch by design.

Note:
The larger GOAT `finalensemble.xyz` files are gzip-compressed as
`finalensemble.xyz.gz` to keep individual supplementary files below upload
limits. They can be decompressed with `gzip -d` if the full ensemble text files
are needed.

## 03_NBO

Path: `01_Calculations/03_NBO/DVB-BTMA+PFOA-`

Purpose:
NBO/NAO analysis for the DVB-BTMA+PFOA- extended-monomer complex. The
calculation is an ORCA 6.1.0 single point at the
ωB97X-D3/def2-TZVPD level with TightSCF and SMD water, using the supplied
optimized geometry.

Note:
The original `PFOA-Ch_wb97xd3_opt` job stem is retained for provenance. The
complete ORCA/NBO output is gzip-compressed as `.out.gz` to keep individual
supplementary files below upload limits. It can be decompressed with `gzip -d`.

## 04_PES

Path: `01_Calculations/04_PES/R4N+PFOA-`

Purpose:
PFOA/BTMA local exchange potential-energy surface scan and associated scan
outputs.

Note:
The full PES ORCA `.out` file is not included in the supplementary archive due
to file size. The scan input, extracted grid/path data, and output geometries
are retained; additional intermediate files and large outputs can be provided
upon request.

## 05_MD

Path: `01_Calculations/05_MD`

Contents:
- `01_System_Builds/R48_48Cl`
- `01_System_Builds/R48_PFOA_47Cl`
- `02_MD_Endpoint_Runs/R48_48Cl`
- `02_MD_Endpoint_Runs/R48_PFOA_47Cl`
- `03_MD_GBSA_Exchange_Cycle`
- `04_Structural_Inspection`

Purpose:
R48 resin model MD setup, endpoint trajectories/outputs, MD/GBSA exchange-cycle
analysis, and structural-inspection files.
