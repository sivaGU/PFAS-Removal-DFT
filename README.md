# PFAS-Removal-DFT

Streamlit GUI for generating ORCA input files used in PFAS/cholestyramine DFT workflows.

## What It Does

- Generates ORCA input decks without hand-editing templates.
- Provides built-in XYZ examples for PFOA, PFOS, PFHxA, and FHEA.
- Provides built-in BTMA and Extended Monomer PFAS-complex XYZ examples.
- Generates exchange energetics inputs for `R4N+X-`, `R4N+Cl-`, `X-`, and `Cl-`, including configurable geometry optimization, configurable frequency inputs, and optional GOAT/GFN2-xTB inputs.
- Generates interaction-analysis inputs for EDA-NOCV and/or NBO.
- Allows optional upload of custom PFAS, complex, and `R4N+Cl-` XYZ structures.
- Writes each generated ORCA input next to the XYZ file it references, so `* xyzfile` lines use local filenames only.

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
