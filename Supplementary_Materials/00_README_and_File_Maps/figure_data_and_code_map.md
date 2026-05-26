# Figure Plotting and Data Map

Figure plotting/data folders live under
`Supplementary_Materials/02_Figure_Plotting_and_Data`. Main figure PNG/PDF files
are intentionally not included in this archive section.

## Figure_03_BTMA_exchange_energetics

Script:
- `plot_BTMA_data.py`

Input data:
- `input_data/btma_exchange_summary.csv`
- `input_data/parsed_raw_energies.csv`
- `input_data/selectivity_energy_summary.txt`

## Figure_04_BTMA_thermochemical_decomposition

Script:
- `plot_BTMA_G_vs_E.py`

Input data:
- `input_data/full_thermo_BTMA.txt`
- `input_data/btma_exchange_summary.csv`

## Figures_05_09_10_EDA

Script:
- `plot_eda_data.py`

Input data:
- `input_data/eda_sections_main_out.txt`
- `input_data/nocv_sections_main_out.txt`
- `input_data/eda_component_summary.csv`

This single script/data bundle supports the BTMA EDA, extended-monomer EDA, and
solvent-dependent extended-monomer EDA figure families.

## Figure_06_NBO

Script:
- `plot_nao_nbo_interaction.py`

Input data:
- `input_data/NBO_source_output_R4N+PFOA-_EDA.out.gz`

The plotting script contains the extracted NBO/NAO values used in the figure.
The copied output file provides the source run context and is gzip-compressed
to keep individual supplementary files below upload limits.

## Figures_07_08_extended_monomer_exchange

Script:
- `plot_ext_monomer_data.py`

Input data:
- `input_data/ext_monomer_wb_energy_summary_merged.txt`
- `input_data/ext_monomer_exchange_summary.csv`
- `input_data/parsed_ext_monomer_raw_energies.csv`

## Figure_11_PES

Scripts:
- `analyze_pes.py`
- `plot_pes_analysis.py`

Input data:
- `input_data/R4N+PFOA-_0.15M.relaxscanact.dat`
- `input_data/R4N+PFOA-_0.15M.relaxscanscf.dat`
- `input_data/R4N+PFOA-_0.15M.grid.csv`

## Figure_12_MD_GBSA_exchange_cycle

Scripts:
- `plot_MD_exchange_cycle.py`
- `combine_exchange_gbsa.py`

Input data:
- `input_data/exchange_cycle_proxy_summary.csv`
- `input_data/exchange_cycle_proxy_per_frame.csv`
- `input_data/FINAL_CL_AQ_GBSA.dat`
- `input_data/FINAL_PFOA_AQ_GBSA.dat`
- `input_data/FINAL_R48_48CL_GBSA.dat`
- `input_data/FINAL_R48_PFOA_47CL_GBSA.dat`
- `input_data/README_exchange_cycle_proxy_results.txt`

## Figure_13_MD_structural_metrics

Scripts:
- `plot_MD_structural_metrics.py`
- `pfoa_association_metrics.py`
- `waters_within_5A_tail.cpptraj`

Input data:
- `input_data/pfoa_association_mechanism_metrics.csv`
- `input_data/pfoa_association_mechanism_metrics_summary.txt`
- `input_data/pfoa_tail_waters_within_5A.dat`

## Figures_14_15_MD_structural_rendering

Script:
- `plot_MD_site_hopping_figure.py`

Input data:
- representative PDB files in `input_data/`

Shared helper:
- `shared_helpers/render_structure_3d.py`
