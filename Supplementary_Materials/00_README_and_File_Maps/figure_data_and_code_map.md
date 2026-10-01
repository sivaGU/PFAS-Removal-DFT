# Figure Data and Code Map

Plotting code, adjacent curated CSV/structure inputs, and PNG exports are
under `Supplementary_Materials/02_Figure_Plotting_and_Data`. Main-text folders
are under `01_main_text_figures`; Figure S1 is under
`02_supplementary_figures`. Raw source families below are relative to
`Supplementary_Materials/01_Calculations`. Consult each figure's script and
the plotting-root README for exact adjacent filenames.

| Figure | Active plotting folder | Raw source family |
|---|---|---|
| 01 | `Figure_01_workflow_structure` | A reusable polymer structure asset is rendered here; the complete workflow is assembled externally in BioRender. |
| 02 | No plotting folder | BioRender assembly using reused renders/screenshots; no separate calculation. |
| 03 | `Figure_03_DFT_structure_atlas` | Minimal BTMA and extended DVB-BTMA optimized structures in `01_BTMA` and `02_Extended_Monomer`, with solvent identified in the panel labels. |
| 04 | `Figure_04_BTMA_exchange_and_structures` | `01_BTMA` aqueous optimization/frequency outputs. |
| 05 | `Figure_05_BTMA_thermochemical_decomposition` | `01_BTMA` aqueous frequency outputs; plotted values are in its adjacent CSV. |
| 06 | `Figure_06_BTMA_species_and_components` | `01_BTMA` aqueous calculations and adjacent curated CSV. |
| 07 | `Figure_07_BTMA_energy_decomposition_analysis` | `01_BTMA/03_EDA`. |
| 08 | `Figure_08_NBO` | `03_NBO/DVB-BTMA+PFOA-/PFOA-Ch_wb97xd3_opt.out.gz`, with adjacent curated plotting CSV. |
| 09 | `Figure_09_model_size_exchange` | `01_BTMA` and `02_Extended_Monomer/water`; source structures remain external to numerical CSVs. |
| 10 | `Figure_10_PFOA_water_octanol_exchange` | `02_Extended_Monomer/water` and `02_Extended_Monomer/octanol_corrected` frequency outputs. |
| 11 | `Figure_11_DVB_BTMA_energy_decomposition_analysis` | `02_Extended_Monomer/water/03_EDA`. |
| 12 | `Figure_12_PFOA_water_octanol_EDA` | Aqueous and 1-octanol extended-monomer EDA records under `02_Extended_Monomer`. |
| 13 | `Figure_13_PES` | `04_PES/R4N+PFOA-` input, complete compressed ORCA output and scan structures. |
| 14 | `Figure_14_MD_model_and_workflow` | pVBTMA12/PFOA Stage 09 topology and a documented Stage 10 replica-01 frame. Its selection record is adjacent to the renderer. |
| 15 | `Figure_15_MD_blocks_and_diagnostics` | `05_MD/03_Trajectory_Analysis` from three Stage 10 production replicas; plotting CSVs are adjacent. |
| S1 | `Figure_S1_MD_raw_RMSD_and_proximity` | `05_MD/03_Trajectory_Analysis` per-frame traces for five continuous/count observables displayed as temporal means in Figure 15. |

The supplementary table workbook is
`06_Table_Workbooks/Supplementary_Tables_S1-S11.xlsx`.
