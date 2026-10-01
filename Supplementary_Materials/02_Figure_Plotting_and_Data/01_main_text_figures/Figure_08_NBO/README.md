# Figure 8 NAO and NBO diagram

Run from `02_Figure_Plotting_and_Data`:

```bash
python 01_main_text_figures/Figure_08_NBO/plot_nao_nbo_interaction.py
```

The script reads the existing `input_data/figure_07_nbo_data.csv`. Its historical filename is intentional. Orbital diagonal energies are converted from hartree to eV for the vertical axis. The interaction record independently supplies the NBO second-order stabilization value, 0.20 kcal/mol.

The blue arrow identifies donor-to-acceptor delocalization. Its height does not encode the stabilization magnitude or a photon excitation energy. The donor and acceptor orbital levels differ by approximately 22.17 eV. Arrow endpoints contain visual offsets. Keep eV for orbital levels and kcal/mol for stabilization.

