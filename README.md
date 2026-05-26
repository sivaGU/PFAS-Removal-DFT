# PFAS-Removal-DFT

Streamlit GUI for generating ORCA input files used in PFAS/cholestyramine DFT workflows.

## What It Does

- Generates ORCA input decks without hand-editing templates.
- Provides built-in XYZ examples for PFOA, PFOS, PFHxA, and FHEA.
- Provides built-in BTMA and Extended Monomer PFAS-complex XYZ examples.
- Generates inputs for GOAT/GFN2-xTB, r2SCAN-3c optimization, wB97X-D3 optimization, frequency calculations, EDA-NOCV, and NBO.
- Generates exchange-energy frequency inputs for `R4N+X-`, `R4N+Cl-`, `X-`, and `Cl-`.
- Allows optional upload of custom PFAS, complex, and `R4N+Cl-` XYZ structures.

The app generates input files only. ORCA calculations should be run on suitable local or HPC resources.

## Run Locally

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Repository Layout

- `streamlit_app.py`: Streamlit entry point.
- `app/`: template, workflow, theme, and XYZ helper code.
- `examples/`: lightweight XYZ files bundled with the deployed app.
- `Supplementary_Materials/`: manuscript-supporting calculation and table files.
