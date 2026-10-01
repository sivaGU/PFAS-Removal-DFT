# Figure 3: DFT structure atlas

Run `render_atlas_pymol.py` in the configured WSL PyMOL environment to refresh the 17 high-resolution PNG molecular tiles and verified distance CSV. Run `plot_window_panes.py` to assemble `figure_exports/Figure_03_DFT_structure_atlas.png`. Existing tiles allow an assembly preview without PyMOL. The compositor places each tile on its white figure background before a single resampling pass, preventing dark dotted atom edges.

All 17 optimized geometries used by the atlas are stored locally under
`input_structures/`, grouped as aqueous BTMA, aqueous DVB-BTMA, and
1-octanol DVB-BTMA inputs.

A/B are BTMA⁺ water models at r²SCAN-3c/ωB97X-D3. C/D are DVB-BTMA⁺ water models at the same respective methods. E is DVB-BTMA⁺/PFOA⁻ at ωB97X-D3 in 1-octanol. Each distance box displays the red and blue geometrical oxygen-to-cation-hydrogen distances. PFAS labels are bold; panel titles are regular weight and centered, with larger bold panel letters. E is wider than an A–D sub-box to accommodate its heading.

From `02_Figure_Plotting_and_Data/`:

```bash
python 01_main_text_figures/Figure_03_DFT_structure_atlas/render_atlas_pymol.py
python 01_main_text_figures/Figure_03_DFT_structure_atlas/plot_window_panes.py
```

The renderer accepts `--contacts-only` when PyMOL is unavailable, and `--mamba PATH --env NAME` for a different environment. Output is PNG only.
