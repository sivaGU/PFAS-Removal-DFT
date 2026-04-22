# Final Ronan GUI

This Streamlit app focuses on manuscript-aligned PFAS interaction visualization for cholestyramine complexes.

## What it does

- Loads bundled PFAS structures from `PFAS Files/` (included inside this folder)
- Supports user-uploaded `.xyz` (required) and `.cube` (optional) files
- Renders 3D molecular interaction maps
- Highlights likely ionic headgroup contacts and local tail proximity
- Displays manuscript-aligned EDA, NOCV, and NBO summaries for PFOS/PFOA/PFHxA/FHEA
- Shows cube volumetric grid metadata for orbital context checks

## Run locally

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Notes

- Interaction detection is heuristic and designed for quick visual interpretation.
- Manuscript metrics are preloaded for bundled PFAS names; custom uploads still receive full geometric visualization.
