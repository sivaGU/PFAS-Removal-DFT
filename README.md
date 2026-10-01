# PFAS Removal

This repository contains the supplementary calculation records, molecular
structures, molecular dynamics data, analysis and plotting materials, workflow
code, and Streamlit interface associated with *Anion Exchange Mechanism of
PFAS with Cholestyramine: Implications for PFAS Removal from the Human Body*.

The Streamlit application generates ORCA input files used in the
PFAS/cholestyramine DFT workflows.

## What It Does

- Generates ORCA input decks without manually editing templates.
- Provides Demo XYZ examples for PFOA, PFOS, PFHxA, and FHEA.
- Provides Demo BTMA and Extended Monomer XYZ examples.
- Generates exchange energetics inputs for `R4N+X-`, `R4N+Cl-`, `X-`, and `Cl-`, including configurable geometry optimization, configurable frequency inputs, and optional GOAT/GFN2-xTB inputs.
- Generates interaction analysis inputs for EDA-NOCV and/or NBO.
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
- `Supplementary_Materials/`: supporting calculation and table files.

## Citation

The first archival release is version `1.0.0`. Citation metadata are provided
in `CITATION.cff`. The version-specific Zenodo DOI will be added after the
dataset record is published.

## Licensing

Project-authored software is licensed under the MIT License in `LICENSE`.
Project-authored research data, documentation, tables, and figures are
licensed under CC BY 4.0 as described in `LICENSE-DATA`. See `LICENSES.md` for
the path-based scope, required BioRender attributions, and third-party terms.
