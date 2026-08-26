# Figure Plotting and Data Map

Figure plotting/data folders live under
`Supplementary_Materials/02_Figure_Plotting_and_Data`. Main figure PNG/PDF files
are intentionally not included in this archive section.

## Figure_03_BTMA_exchange_energetics

Script:
- `plot_BTMA_data.py`

Input data:
- `input_data/btma_exchange_summary.csv`
- manuscript structure panels in `input_structures/`

The numerical CSV is curated from the BTMA optimization and frequency outputs
archived under `01_Calculations/01_BTMA`.

## Figure_04_BTMA_thermochemical_decomposition

Script:
- `plot_BTMA_G_vs_E.py`

Input data:
- `input_data/thermochemical_decomposition.csv`

The curated values are derived from the BTMA frequency calculations archived
under `01_Calculations/01_BTMA/02_Frequency`.

## Figures_05_09_10_EDA

Script:
- `plot_eda_data.py`

Input data:
- `input_data/eda_component_summary.csv`

This single script/data bundle supports the BTMA EDA, extended-monomer EDA, and
solvent-dependent extended-monomer EDA figure families.

## Figure_06_NBO

Script:
- `plot_nao_nbo_interaction.py`

Input data:
- `input_data/figure_06_nbo_data.csv`

The curated CSV contains the NAO/NBO levels and selected donor-acceptor
interaction read by the plotting script for the DVB-BTMA+PFOA- complex. The
complete source output is archived at
`01_Calculations/03_NBO/DVB-BTMA+PFOA-/PFOA-Ch_wb97xd3_opt.out.gz`.

## Figures_07_08_extended_monomer_exchange

Script:
- `plot_ext_monomer_data.py`

Input data:
- `input_data/ext_monomer_exchange_summary.csv`
- manuscript structure panels in `input_structures/`

The numerical CSV is curated from calculations archived under
`01_Calculations/01_BTMA` and `01_Calculations/02_Extended_Monomer`.

## Figure_11_PES

Script:
- `plot_pes_analysis.py`

Input data:
- `input_data/R4N+PFOA-_0.15M.grid.csv`
- `input_data/pes_mep_profile.csv`
- molecular structures in `input_structures/`
- curated structure panels in `structure_renders/`

The PES grid, trajectory, and ORCA source files are archived under
`01_Calculations/04_PES`.

## Figure_12_MD_GBSA_exchange_cycle

Script:
- `plot_MD_exchange_cycle.py`

Input data:
- `input_data/exchange_cycle_proxy_summary.csv`
- `input_data/exchange_cycle_proxy_per_frame.csv`

The four endpoint MM/GBSA output tables used to curate these CSVs are archived
under `01_Calculations/05_MD/03_MD_GBSA_Exchange_Cycle/04_gbsa`.

## Figure_13_MD_structural_metrics

Script:
- `plot_MD_structural_metrics.py`

Input data:
- `input_data/pfoa_association_mechanism_metrics.csv`
- `input_data/pfoa_tail_waters.csv`

The trajectory-derived source tables and structural frames are archived under
`01_Calculations/05_MD/03_MD_GBSA_Exchange_Cycle/05_metrics`.

## Figure_14_MD_structural_rendering

Script:
- `figure14_plot.py`

Input data:
- `input_render/Figure_14_MD_structural_snapshots.png`

The curated raster preserves the approved camera placement, transparency, and
labels. Its source PDB files, Amber topologies, and MD endpoint structures are
archived under `01_Calculations/05_MD`.
