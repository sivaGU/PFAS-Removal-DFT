# PFAS Removal DFT - Binding Affinity Analysis GUI

A production-ready Streamlit GUI for analyzing DFT calculation results for PFAS-cholestyramine binding affinities and energy decomposition.

## Overview

This application provides a platform to analyze and visualize Density Functional Theory (DFT) calculation results for the binding of per- and polyfluoroalkyl substances (PFAS) with cholestyramine. The tool supports binding energy analysis, energy decomposition analysis (EDA), Natural Bond Orbital (NBO) analysis, and comparison of multiple PFAS molecules.

## Features

- **Single PFAS analysis** — Enter SMILES string or upload structure file to analyze binding affinities
- **Batch CSV processing** — Upload CSV with SMILES and calculation results for batch analysis
- **Energy Decomposition Analysis** — Visualize electrostatic, orbital, dispersion, and solvation contributions
- **NOCV Pairs** — Display charge transfer channels and orbital contributions
- **NBO Analysis** — Show donor-acceptor interactions and stabilization energies
- **Comparison View** — Compare binding affinities across multiple PFAS molecules

## Quick Start

### 1. Create and activate a virtual environment

```bash
python -m venv venv
```

- **Windows:** `venv\Scripts\activate`
- **Mac/Linux:** `source venv/bin/activate`

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

If RDKit fails, try:
```bash
pip install rdkit
pip install pandas numpy streamlit
```

### 3. Run the Streamlit GUI

```bash
streamlit run streamlit_app.py
```

The app will open at `http://localhost:8501`.

## Project Structure

```
.
├── streamlit_app.py       # PFAS DFT GUI
├── requirements.txt       # Dependencies
├── src/
│   └── pfasdft/           # Prediction module
│       ├── predict.py     # predict_single, predict_batch, load_predictor
│       └── cli.py         # Command-line interface
├── artifacts/             # Model artifacts and reference data
│   ├── reference_data.json
│   └── model_config.json
├── example_inputs.csv
└── README.md
```

## Usage

### GUI

- **Single PFAS:** Enter SMILES (e.g., `C(=O)([O-])C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)` for PFOA) or upload structure file, click **Analyze**
- **Batch CSV:** Upload a CSV with columns `smiles` (or `SMILES`) and optionally `pfas_name`, click **Analyze batch**, then **Download CSV**

### CLI

From the project root:

```bash
# Analyze a single PFAS
python -m src.pfasdft.cli --smiles "C(=O)([O-])C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)" --pfas-name PFOA --output out.csv

# Analyze from CSV
python -m src.pfasdft.cli --input example_inputs.csv --output out.csv
```

## Input Data Format

The application expects DFT calculation results in JSON format. Upload your results to `artifacts/reference_data.json`:

```json
{
  "PFOS": {
    "delta_e_exchange": -5.62,
    "delta_g_exchange": null,
    "e_electrostatic": -67.2,
    "e_pauli": 7.9,
    "e_orbital": -6.2,
    "e_dispersion": -2.7,
    "e_solvation": -45.8,
    "e_preparation": 15.5,
    "e_binding_total": -101.2,
    "nocv_pairs": [-0.76, -0.69, -0.48, -0.28, -0.23],
    "nbo_donor_acceptor": "O lone pair → σ* C-H",
    "nbo_stabilization_energy": null
  }
}
```

## Model Details

- **DFT Functionals:** ωB97X-D3 (range-separated hybrid), r2SCAN-3c (composite)
- **Basis Sets:** def2-TZVPD (with diffuse functions for anions)
- **Solvation:** SMD/C-PCM (water, ε = 72.5 for physiological conditions)
- **Temperature:** 310.15 K (physiological)
- **Models:** BTMA (minimal benzyltrimethylammonium) and Extended cholestyramine monomer

## Supported PFAS Molecules

- **PFOS** — Perfluorooctanesulfonic acid
- **PFOA** — Perfluorooctanoic acid
- **PFHxA** — Perfluorohexanoic acid
- **FHEA** — Perfluorohexyl ethanoic acid

## Requirements

- Python 3.9 or 3.10 (3.11/3.12 usually work)
- Dependencies in `requirements.txt`

## Notes

- This GUI is designed to work with DFT calculation results from ORCA or similar quantum chemistry software
- Default reference data includes values from the PFAS Removal DFT paper (see manuscript)
- Upload your own calculation results to `artifacts/reference_data.json` for custom analysis
- The application currently uses reference data from the paper; integrate your own calculation pipeline as needed

## Contact

[Your Contact Information]
