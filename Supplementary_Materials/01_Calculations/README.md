# Calculation Inputs and Outputs

This section carries primary inputs, software outputs, geometries, Amber
topologies and MD trajectories. Figure CSVs are under
`../02_Figure_Plotting_and_Data`, not substitutes for these source files.

## Electronic-structure calculations

| Folder | Contents |
|---|---|
| `01_BTMA` | Aqueous minimal-site BTMA optimization, frequency, and EDA input-output records. |
| `02_Extended_Monomer/water` | Aqueous DVB-BTMA optimization, frequency, and EDA input-output records. |
| `02_Extended_Monomer/octanol_corrected` | SMD 1-octanol PFOA and chloride optimization/frequency records and PFOA EDA complex/fragment records. |
| `03_NBO/DVB-BTMA+PFOA-` | DVB-BTMA/PFOA NBO and NAO input, geometry, and compressed ORCA output. |
| `04_PES/R4N+PFOA-` | PFOA/BTMA scan input, geometries, and complete compressed ORCA output. The gzip is supplied as three binary parts with an adjacent reconstruction and SHA-256 verification script. |
| `06_GOAT_Conformer_Sensitivity` | Selected starting geometries and six r2SCAN-3c optimization/frequency input-output-geometry sets. |

The octanol and conformer-sensitivity source stems use `R4N` but describe the
extended DVB-BTMA model. Solvent and electronic-structure settings are recorded
in the supplied inputs and outputs.

## Molecular dynamics

See `05_MD/README.md` for the model-validation, production, and trajectory
analysis directory map. NAMD output files retain their software-generated
settings and restart relationships.
