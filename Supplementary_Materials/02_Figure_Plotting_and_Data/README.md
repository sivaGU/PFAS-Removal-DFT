# Figure Plotting and Data

This directory contains plotting scripts, adjacent numerical and structural
inputs, molecular rendering scripts, reporting tables, shared helpers, and PNG
exports for main Figures 1 and 3–15 and supplementary Figure S1. Figure 2 is
assembled externally and has no plotting script here. Scripts write PNG files
to `figure_exports/`.

Run commands sequentially from this directory. Python dependencies include
NumPy, pandas, Matplotlib, Pillow, and SciPy where imported by a script.
Molecular renders for Figures 3, 13, and 14 additionally require PyMOL. The
renderers accept their documented `--mamba` and `--env` options for selecting a
PyMOL environment.

## Molecular rendering

Run the input checks followed by the renderers:

```bash
python 01_main_text_figures/Figure_03_DFT_structure_atlas/render_atlas_pymol.py --contacts-only
python 01_main_text_figures/Figure_13_PES/render_pes_structures.py --audit-only
python 01_main_text_figures/Figure_14_MD_model_and_workflow/render_figure14_pymol.py --prepare-only

python 01_main_text_figures/Figure_03_DFT_structure_atlas/render_atlas_pymol.py
python 01_main_text_figures/Figure_13_PES/render_pes_structures.py
python 01_main_text_figures/Figure_14_MD_model_and_workflow/render_figure14_pymol.py
```

Figure 3 uses 17 transparent molecular tiles. Panels A/B are BTMA⁺ water
models at r²SCAN-3c and ωB97X-D3, panels C/D are DVB-BTMA⁺ water models at the
same respective methods, and panel E is DVB-BTMA⁺/PFOA⁻ at ωB97X-D3 in
1-octanol.

Figure 13 uses the four supplied optimized XYZ files. Its standalone selected
configuration renders use red and blue O–N guides and a green N–Cl guide. The
PES input scans the carboxyl C–N and ammonium N–Cl coordinates.

Figure 14 uses replica 01 at 9.856 ns. Panel A displays the polymer, PFOA,
ions, and periodic box without water particles. Panel B displays complete
waters selected by the uniform camera-volume sampling rule recorded in
`input_data/frame_selection.md` and
`structure_renders/render_inputs/selection_audit.csv`.

## Figure assembly

The numerical source tables are in each figure's `input_data/` directory.
Run Figures 7 and 11 sequentially because they use the same EDA helper.

```bash
python 01_main_text_figures/Figure_01_workflow_structure/render_figure01_polymer.py
python 01_main_text_figures/Figure_03_DFT_structure_atlas/plot_window_panes.py
python 01_main_text_figures/Figure_04_BTMA_exchange_and_structures/plot_figure04_exchange.py
python 01_main_text_figures/Figure_05_BTMA_thermochemical_decomposition/plot_BTMA_G_vs_E.py
python 01_main_text_figures/Figure_06_BTMA_species_and_components/plot_BTMA_species_components.py
python 01_main_text_figures/Figure_07_BTMA_energy_decomposition_analysis/plot_figure07_eda.py
python 01_main_text_figures/Figure_08_NBO/plot_nao_nbo_interaction.py
python 01_main_text_figures/Figure_09_model_size_exchange/plot_figure09.py
python 01_main_text_figures/Figure_10_PFOA_water_octanol_exchange/plot_figure10.py
python 01_main_text_figures/Figure_11_DVB_BTMA_energy_decomposition_analysis/plot_figure11_eda.py
python 01_main_text_figures/Figure_12_PFOA_water_octanol_EDA/plot_figure12.py
python 01_main_text_figures/Figure_13_PES/plot_figure13_pes.py
python 01_main_text_figures/Figure_14_MD_model_and_workflow/plot_figure14.py
python 01_main_text_figures/Figure_15_MD_blocks_and_diagnostics/plot_figure15.py
python 02_supplementary_figures/Figure_S1_MD_raw_RMSD_and_proximity/plot_figure_s1.py
```

Figure 15 reads `figure15_display_250ps.csv`, which contains 250 ps
nonoverlapping display windows for six observables and three replicas. Figure
S1 displays the corresponding per-frame series for oligomer RMSD, radius of
gyration, head-to-nearest-N distance, tail-oligomer contacts, and tail
hydration. EDA reporting tables are generated from the records under
`03_reporting_tables/EDA` by `shared_helpers/build_eda_reporting_tables.py`.

All figure scripts produce PNG output only.
