"""
PFAS Removal DFT Streamlit GUI. Binding affinity and energy decomposition analysis for PFAS-cholestyramine complexes.

Run from this folder (project root):
  streamlit run streamlit_app.py
"""
import os
import sys
from pathlib import Path
from typing import Optional

# Ensure project root (this folder) is on path for src.pfasdft
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

HANDOFF_DIR = PROJECT_ROOT

import io
import urllib.request
import urllib.parse
import streamlit as st
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem

from src.pfasdft.predict import predict_single, predict_batch, load_predictor
from src.pfasdft.orca_input import generate_workflow_inputs, generate_exchange_calculation_inputs, ORCAConfig
from src.pfasdft.orca_parser import parse_orca_output, calculate_exchange_energy, ORCAResults
from src.pfasdft.structure_utils import prepare_pfas_structure, prepare_cholestyramine_structure, prepare_complex_structure


def extract_smiles_from_file(file_content: bytes, file_extension: str) -> Optional[str]:
    """
    Extract SMILES string from various molecular file formats.
    Supported formats: SDF, PDB, PDBQT, MOL, MOL2, CSV (first row only).
    """
    try:
        ext = file_extension.lower()
        if ext == ".sdf":
            from io import StringIO
            sdf_data = StringIO(file_content.decode("utf-8"))
            supplier = Chem.SDMolSupplier(sdf_data)
            for m in supplier:
                if m is not None:
                    return Chem.MolToSmiles(m, canonical=True)
        elif ext == ".mol":
            mol = Chem.MolFromMolBlock(file_content.decode("utf-8"))
            if mol:
                return Chem.MolToSmiles(mol, canonical=True)
        elif ext == ".pdb":
            mol = Chem.MolFromPDBBlock(file_content.decode("utf-8"))
            if mol:
                return Chem.MolToSmiles(mol, canonical=True)
        elif ext == ".mol2":
            try:
                mol = Chem.MolFromMol2Block(file_content.decode("utf-8"))
                if mol:
                    return Chem.MolToSmiles(mol, canonical=True)
            except Exception:
                pass
        elif ext == ".csv":
            from io import BytesIO
            df = pd.read_csv(BytesIO(file_content))
            col = next((c for c in df.columns if c.lower() in ("smiles", "smi") or c == "SMILES"), None)
            if col and len(df) > 0:
                return str(df[col].iloc[0]).strip()
    except Exception:
        pass
    return None


def get_mol_for_drawing(smiles: Optional[str] = None, file_content: Optional[bytes] = None, file_extension: Optional[str] = None):
    """
    Get an RDKit mol for 2D structure drawing (no 3D embedding required).
    Uses uploaded file if present, else SMILES. Returns Chem.Mol or None.
    """
    mol = None
    if file_content is not None and file_extension is not None:
        ext = file_extension.lower()
        try:
            text = file_content.decode("utf-8")
            if ext == ".sdf":
                from io import StringIO
                supplier = Chem.SDMolSupplier(StringIO(text))
                mols = [m for m in supplier if m is not None]
                mol = mols[0] if mols else None
            elif ext == ".mol":
                mol = Chem.MolFromMolBlock(text)
            elif ext in (".pdb", ".pdbqt"):
                mol = Chem.MolFromPDBBlock(text)
            elif ext == ".mol2":
                mol = Chem.MolFromMol2Block(text)
        except Exception:
            mol = None
    if mol is None and smiles is not None:
        smiles_str = str(smiles).strip()
        if smiles_str:
            mol = Chem.MolFromSmiles(smiles_str)
    return mol


def render_ligand_structure(mol, size: int = 400) -> Optional[bytes]:
    """
    Draw the ligand as a 2D chemical structure (atoms and bonds) using RDKit.
    Returns PNG image bytes or None on failure.
    """
    if mol is None:
        return None
    try:
        from rdkit.Chem import Draw
        try:
            AllChem.Compute2DCoords(mol)
        except Exception:
            pass
        img = Draw.MolToImage(mol, size=(size, size))
        if img is None:
            return None
        buf = io.BytesIO()
        if hasattr(img, "mode") and img.mode not in ("RGB", "L"):
            img = img.convert("RGB")
        img.save(buf, format="PNG")
        buf.seek(0)
        return buf.getvalue()
    except Exception:
        return None


# ============================================================================
# CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="PFAS Removal DFT - Binding Affinity Studio",
    page_icon=None,
    layout="wide",
    menu_items={
        "Report a bug": "https://github.com/your-org/pfas-dft-gui/issues",
        "About": "DFT-based binding affinity and energy decomposition analysis for PFAS-cholestyramine complexes.",
    },
)

# Inject custom CSS for color palette - oak green solid colors
st.markdown("""
<style>
    /* Oak green palette: solid colors */
    :root {
        --oak-dark: #2F4F2F;
        --oak-medium-dark: #556B2F;
        --oak-medium: #6B8E23;
        --oak-light: #8FBC8F;
        --oak-pale: #B8D4B8;
        --oak-very-pale: #D4E4D4;
    }
    
    .stApp {
        background-color: #ffffff;
    }
    
    section.main,
    .main,
    [data-testid="stAppViewContainer"] > div:not([data-testid="stSidebar"]) {
        background-color: #ffffff !important;
    }
    
    div[data-testid="stAppViewContainer"] > div > div:not([data-testid="stSidebar"]) {
        background-color: #ffffff !important;
    }
    
    .main .block-container,
    section.main .block-container {
        background-color: #ffffff !important;
        padding: 2rem 3rem;
        margin: 2rem auto;
        max-width: 1400px;
        border-radius: 8px;
        box-shadow: 0 2px 12px rgba(47, 79, 47, 0.08);
    }
    
    [data-testid="stSidebar"] {
        background: #2F4F2F;
        color: #ffffff;
        min-width: 200px !important;
        max-width: 280px !important;
        width: 280px !important;
    }
    
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
        width: 280px !important;
        min-width: 200px !important;
        max-width: 280px !important;
    }
    
    [data-testid="stSidebar"] .css-1d391kg {
        background-color: #2F4F2F;
    }
    
    .stButton > button {
        background: #6B8E23;
        color: white;
        border: none;
        border-radius: 6px;
        font-weight: 600;
        box-shadow: 0 2px 4px rgba(107, 142, 35, 0.35);
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        background: #556B2F;
        box-shadow: 0 4px 8px rgba(85, 107, 47, 0.4);
        transform: translateY(-1px);
    }
    
    .stButton > button:focus {
        background: #2F4F2F;
        box-shadow: 0 0 0 0.3rem rgba(107, 142, 35, 0.35);
    }
    
    .stDownloadButton > button {
        background: #556B2F;
        color: white;
        border-radius: 6px;
        font-weight: 500;
    }
    
    .stDownloadButton > button:hover {
        background: #2F4F2F;
    }
    
    h1, h2, h3 {
        color: #2F4F2F;
        font-weight: 700;
    }
    
    a {
        color: #556B2F;
        text-decoration: none;
    }
    
    a:hover {
        color: #6B8E23;
        text-decoration: underline;
    }
    
    [data-testid="stMetricValue"] {
        color: #2F4F2F;
        font-weight: 600;
    }
    
    .stSuccess {
        background: #D4E4D4;
        border-left: 4px solid #6B8E23;
        color: #1a1a1a;
        border-radius: 4px;
    }
    
    .stInfo {
        background: #D4E4D4;
        border-left: 4px solid #6B8E23;
        color: #1a1a1a;
        border-radius: 4px;
    }
    
    .stWarning {
        background: #B8D4B8;
        border-left: 4px solid #556B2F;
        color: #1a1a1a;
        border-radius: 4px;
    }
    
    .stError {
        background: #B8D4B8;
        border-left: 4px solid #2F4F2F;
        color: #1a1a1a;
        border-radius: 4px;
    }
    
    .stRadio > label,
    .stSelectbox > label,
    .stTextInput > label,
    .stSlider > label,
    .stFileUploader > label {
        color: #2F4F2F;
        font-weight: 500;
    }
    
    .streamlit-expanderHeader {
        background: #D4E4D4;
        color: #2F4F2F;
        border-radius: 4px;
        font-weight: 500;
    }
    
    .streamlit-expanderHeader:hover {
        background: #B8D4B8;
    }
    
    .stDataFrame {
        border: 2px solid #6B8E23;
        border-radius: 4px;
    }
    
    hr {
        border-color: #6B8E23;
        border-width: 2px;
    }
    
    .stSlider .stSlider > div > div {
        background-color: #6B8E23;
    }
    
    [data-testid="stSidebar"] .stButton {
        margin-bottom: 0.5rem;
    }
    
    [data-testid="stSidebar"] .stButton > button {
        width: 100%;
        padding: 0.75rem 1rem;
        font-size: 1rem;
        text-align: center;
        margin-bottom: 0.5rem;
        background: #556B2F !important;
        color: white !important;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
    }
    
    [data-testid="stSidebar"] .stButton > button:hover {
        background: #2F4F2F !important;
    }
    
    [data-testid="stSidebar"] h3 {
        color: #ffffff;
        font-weight: 600;
        margin-top: 0;
        margin-bottom: 0.5rem;
    }
    
    [data-testid="stSidebar"] p, [data-testid="stSidebar"] li {
        color: rgba(255, 255, 255, 0.9);
    }
    
    [data-testid="stSidebar"] .stSuccess {
        background: #8FBC8F;
        border-left: 4px solid #6B8E23;
        color: #ffffff;
    }
    
    [data-testid="stSidebar"] .stInfo {
        background: #8FBC8F;
        border-left: 4px solid #6B8E23;
        color: #ffffff;
    }
    
    [data-testid="stSidebar"] hr {
        margin: 1rem 0;
        border-color: rgba(255, 255, 255, 0.25);
    }
    
    .main .block-container > div {
        background-color: #ffffff;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# PREDICTOR (cached)
# ============================================================================

@st.cache_resource
def get_predictor():
    return load_predictor(HANDOFF_DIR)

# ============================================================================
# PAGES
# ============================================================================

def render_home_page():
    """Render the home/dashboard page."""
    st.title("PFAS Removal DFT - Binding Affinity Studio")
    st.caption(
        "DFT-based binding affinity and energy decomposition analysis for PFAS-cholestyramine complexes."
    )

    st.sidebar.markdown("### Project Snapshot")
    st.sidebar.markdown(
        """
        - **Focus:** PFAS-cholestyramine binding mechanism
        - **Method:** Density Functional Theory (DFT)
        - **Functionals:** ωB97X-D3, r2SCAN-3c
        - **Analysis:** EDA, NBO, NOCV
        - **Status:** Ready for calculation results
        """
    )
    st.sidebar.success("Upload your DFT calculation results to get started!")

    st.markdown(
        """
        ## Why this app exists
        Per- and polyfluoroalkyl substances (PFAS) are persistent environmental contaminants 
        associated with endocrine and oncogenic risks. Clinical studies have shown that oral 
        administration of cholestyramine reduces serum concentrations of these molecules by 
        interrupting enterohepatic recirculation. However, the molecular mechanism of anion 
        exchange remains unclear. This application provides a platform to analyze and visualize 
        DFT calculation results for PFAS-cholestyramine binding affinities.
        """
    )

    st.markdown(
        """
        ### Analysis capabilities
        - **Binding Energies:** ΔEexchange and ΔGexchange for anion exchange mechanism
        - **Energy Decomposition Analysis (EDA):** Electrostatic, Pauli repulsion, orbital interactions, dispersion, solvation
        - **Natural Bond Orbital (NBO) Analysis:** Donor-acceptor interactions and stabilization energies
        - **NOCV Pairs:** Charge transfer channels and orbital contributions
        - **Selectivity Screening:** Compare PFAS binding vs. endogenous ligands
        """
    )

    st.divider()

    st.markdown("## Quick start")

    st.info(
        "**Ready to analyze!** Use the **PFAS Binding Prediction** page in the sidebar to enter SMILES strings or upload a CSV file with your DFT calculation results."
    )

    st.markdown(
        """
        ---
        ### Navigation
        - **Home:** This overview
        - **Documentation:** Setup, model details, and usage
        - **PFAS Binding Prediction:** Analyze binding affinities and energy decomposition
        - **Comparison View:** Compare multiple PFAS molecules side-by-side
        - **DFT Calculations:** Generate ORCA input files and parse output files
        """
    )


def render_documentation_page():
    """Render the documentation page."""
    st.title("Documentation & Runbook")
    st.caption("Reference material for the PFAS Removal DFT analysis tool.")

    st.markdown(
        """
        ## Purpose
        This application provides a Streamlit interface for analyzing DFT calculation results 
        for PFAS-cholestyramine binding affinities. It supports single SMILES input, structure file 
        upload (SDF, MOL, PDB, MOL2), and batch CSV processing with calculation results.
        """
    )

    st.markdown(
        """
        ## Repository structure
        ```
        .
        ├── streamlit_app.py       # Main application
        ├── requirements.txt      # Dependencies
        ├── src/pfasdft/          # Prediction module
        │   ├── predict.py        # predict_single, predict_batch, load_predictor
        │   └── cli.py            # Command-line interface
        └── artifacts/            # Model artifacts and reference data
            ├── reference_data.json
            └── model_config.json
        ```
        """
    )

    st.markdown(
        """
        ## Local setup
        1. Create and activate a virtual environment (conda, venv, or poetry).
        2. Install dependencies: `pip install -r requirements.txt`.
        3. Launch the app: `streamlit run streamlit_app.py`.
        4. Streamlit will open at `http://localhost:8501`. Use the sidebar to switch between pages.
        """
    )

    st.markdown(
        """
        ## Model overview
        - **DFT Functionals:** ωB97X-D3 (range-separated hybrid), r2SCAN-3c (composite)
        - **Basis Sets:** def2-TZVPD (with diffuse functions for anions)
        - **Solvation:** SMD/C-PCM (water, ε = 72.5 for physiological conditions)
        - **Temperature:** 310.15 K (physiological)
        - **Models:** BTMA (minimal) and Extended cholestyramine monomer
        """
    )

    st.markdown(
        """
        ## Input data format
        The application expects DFT calculation results in JSON format. Reference data includes:
        - Binding energies (ΔEexchange, ΔGexchange)
        - EDA components (electrostatic, Pauli, orbital, dispersion, solvation, preparation)
        - NOCV pair stabilization energies
        - NBO analysis results
        
        Upload your calculation results to `artifacts/reference_data.json` or use the CSV upload feature.
        """
    )

    st.markdown(
        """
        ## CLI usage
        From the project folder:
        ```bash
        python -m src.pfasdft.cli --smiles "C(=O)([O-])C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)" --pfas-name PFOA --output out.csv
        python -m src.pfasdft.cli --input example_inputs.csv --output out.csv
        ```
        Output columns: pfas_name, smiles, canonical_smiles, delta_e_exchange, delta_g_exchange, 
        e_electrostatic, e_pauli, e_orbital, e_dispersion, e_solvation, e_preparation, e_binding_total, 
        model_type, functional, error.
        """
    )

    st.success("Questions? Contact: [Your Contact Information]")


def render_prediction_page():
    """Render the PFAS binding prediction page."""
    st.title("PFAS Binding Affinity Analysis")
    st.markdown(
        """
        Analyze PFAS-cholestyramine binding affinities using DFT calculation results. 
        Enter a SMILES string, upload a structure file (SDF, MOL, PDB, MOL2), or upload a CSV file 
        with calculation results for batch processing.
        
        **Input modes:** Single SMILES/structure file | Batch (CSV with SMILES and calculation results)
        """
    )

    try:
        predictor = get_predictor()
    except Exception as e:
        st.error(f"Could not load predictor: {e}")
        st.info(
            "Ensure the **artifacts/** folder contains:\n"
            "- reference_data.json (with DFT calculation results)\n"
            "- model_config.json (optional, for configuration)"
        )
        return

    st.sidebar.markdown("### Settings")
    model_type = st.sidebar.selectbox(
        "Model type",
        ["Extended", "BTMA"],
        help="Extended: cholestyramine monomer with backbone. BTMA: minimal benzyltrimethylammonium model."
    )
    st.sidebar.info(
        "**Default:** Extended model with ωB97X-D3/def2-TZVPD. "
        "Upload your calculation results to artifacts/reference_data.json."
    )

    st.divider()

    input_mode = st.radio(
        "Input mode",
        ["Single SMILES or structure file", "Batch (CSV)"],
        horizontal=True,
        key="input_mode",
    )

    if input_mode == "Single SMILES or structure file":
        pfas_name = st.selectbox(
            "PFAS molecule",
            ["Auto-detect", "PFOS", "PFOA", "PFHxA", "FHEA"],
            help="Select PFAS molecule or let the system auto-detect from SMILES"
        )
        
        smiles_input = st.text_input(
            "SMILES (or upload a structure file below)",
            placeholder="e.g. C(=O)([O-])C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F) for PFOA",
            key="smiles_input",
        )
        st.markdown("**Or upload a structure file:**")
        structure_file = st.file_uploader(
            "Upload structure file",
            type=["sdf", "mol", "pdb", "mol2"],
            key="structure_upload",
            help="Supported: SDF, MOL, PDB, MOL2. First molecule will be used.",
        )
        
        smiles_to_use = None
        pfas_name_to_use = None if pfas_name == "Auto-detect" else pfas_name
        
        if structure_file:
            content = structure_file.read()
            ext = os.path.splitext(structure_file.name)[1]
            extracted = extract_smiles_from_file(content, ext)
            if extracted:
                smiles_to_use = extracted
                st.session_state.structure_file_content = content
                st.session_state.structure_file_ext = ext
                st.success(f"Extracted SMILES from {structure_file.name}")
            else:
                st.session_state.structure_file_content = None
                st.session_state.structure_file_ext = None
                st.error(f"Could not extract SMILES from {ext.upper()} file. Try SMILES input instead.")
        elif smiles_input and smiles_input.strip():
            smiles_to_use = smiles_input.strip()
            st.session_state.structure_file_content = None
            st.session_state.structure_file_ext = None
        
        if st.button("Analyze", type="primary", key="btn_single"):
            if smiles_to_use:
                result = predict_single(
                    smiles_to_use,
                    pfas_name=pfas_name_to_use,
                    model_type=model_type,
                    predictor=predictor,
                )
                if result.is_valid:
                    st.success("Valid SMILES")
                    
                    # Display molecule structure
                    st.subheader("PFAS Structure")
                    file_content = st.session_state.get("structure_file_content")
                    file_ext = st.session_state.get("structure_file_ext")
                    mol = get_mol_for_drawing(
                        result.canonical_smiles if result.canonical_smiles else None,
                        file_content=file_content,
                        file_extension=file_ext,
                    )
                    if mol:
                        img_bytes = render_ligand_structure(mol)
                        if img_bytes:
                            st.image(io.BytesIO(img_bytes), use_container_width=False, width=400)
                            st.caption(f"2D structure · {result.pfas_name}")
                    
                    # Binding energies
                    st.subheader("Binding Energies")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        if result.delta_e_exchange is not None:
                            st.metric("ΔEexchange", f"{result.delta_e_exchange:.4f} kcal/mol")
                        else:
                            st.metric("ΔEexchange", "N/A")
                    with col2:
                        if result.delta_g_exchange is not None:
                            st.metric("ΔGexchange", f"{result.delta_g_exchange:.4f} kcal/mol")
                        else:
                            st.metric("ΔGexchange", "N/A")
                    with col3:
                        st.metric("Model", result.model_type or model_type)
                    
                    # EDA components
                    if result.e_binding_total is not None:
                        st.subheader("Energy Decomposition Analysis (EDA)")
                        eda_col1, eda_col2 = st.columns(2)
                        with eda_col1:
                            st.metric("Total Binding Energy", f"{result.e_binding_total:.4f} kcal/mol")
                            if result.e_electrostatic is not None:
                                st.metric("Electrostatic", f"{result.e_electrostatic:.4f} kcal/mol")
                            if result.e_solvation is not None:
                                st.metric("Solvation", f"{result.e_solvation:.4f} kcal/mol")
                            if result.e_orbital is not None:
                                st.metric("Orbital Interaction", f"{result.e_orbital:.4f} kcal/mol")
                        with eda_col2:
                            if result.e_dispersion is not None:
                                st.metric("Dispersion", f"{result.e_dispersion:.4f} kcal/mol")
                            if result.e_pauli is not None:
                                st.metric("Pauli Repulsion", f"{result.e_pauli:.4f} kcal/mol")
                            if result.e_preparation is not None:
                                st.metric("Preparation Energy", f"{result.e_preparation:.4f} kcal/mol")
                        
                        # EDA visualization
                        if result.e_electrostatic is not None:
                            eda_data = {
                                "Component": ["Electrostatic", "Solvation", "Orbital", "Dispersion", "Pauli", "Preparation"],
                                "Energy (kcal/mol)": [
                                    result.e_electrostatic or 0,
                                    result.e_solvation or 0,
                                    result.e_orbital or 0,
                                    result.e_dispersion or 0,
                                    result.e_pauli or 0,
                                    result.e_preparation or 0,
                                ]
                            }
                            eda_df = pd.DataFrame(eda_data)
                            st.bar_chart(eda_df.set_index("Component"))
                    
                    # NOCV pairs
                    if result.nocv_pairs and len(result.nocv_pairs) > 0:
                        st.subheader("NOCV Pairs (Top 5)")
                        nocv_data = {
                            "Pair Index": [f"Pair {i+1}" for i in range(len(result.nocv_pairs))],
                            "Stabilization Energy (kcal/mol)": result.nocv_pairs
                        }
                        nocv_df = pd.DataFrame(nocv_data)
                        st.bar_chart(nocv_df.set_index("Pair Index"))
                    
                    # NBO analysis
                    if result.nbo_donor_acceptor:
                        st.subheader("Natural Bond Orbital (NBO) Analysis")
                        st.info(f"**Primary Interaction:** {result.nbo_donor_acceptor}")
                        if result.nbo_stabilization_energy is not None:
                            st.metric("Stabilization Energy", f"{result.nbo_stabilization_energy:.4f} kcal/mol")
                    
                    # Model info
                    st.subheader("Calculation Details")
                    info_col1, info_col2 = st.columns(2)
                    with info_col1:
                        st.text(f"Functional: {result.functional or 'N/A'}")
                        st.text(f"Model Type: {result.model_type or model_type}")
                    with info_col2:
                        st.text(f"PFAS: {result.pfas_name}")
                        st.text(f"Canonical SMILES: {result.canonical_smiles}")
                else:
                    st.error(result.error)
            else:
                st.warning("Please enter a SMILES string or upload a structure file (SDF, MOL, PDB, MOL2).")

    else:
        uploaded_file = st.file_uploader(
            "Upload CSV",
            type=["csv"],
            key="csv_upload",
        )
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
            col = next(
                (
                    c
                    for c in df.columns
                    if c.lower() in ("smiles", "canonical_smiles", "smi") or c == "SMILES"
                ),
                None,
            )
            if col is None:
                st.error("CSV must have a SMILES column (smiles, SMILES, canonical_smiles, or smi).")
                st.info(f"Available columns: {', '.join(df.columns)}")
            else:
                pfas_col = next((c for c in df.columns if c.lower() in ("pfas_name", "pfas", "name")), None)
                if st.button("Analyze batch", type="primary", key="btn_batch"):
                    smiles_list = df[col].astype(str).tolist()
                    pfas_names = df[pfas_col].tolist() if pfas_col else [None] * len(df)
                    results = predict_batch(
                        smiles_list,
                        pfas_names=pfas_names,
                        model_type=model_type,
                        predictor=predictor,
                    )
                    df_out = df.copy()
                    df_out["pfas_name"] = [r.pfas_name for r in results]
                    df_out["canonical_smiles"] = [r.canonical_smiles for r in results]
                    df_out["delta_e_exchange"] = [r.delta_e_exchange if r.delta_e_exchange is not None else "" for r in results]
                    df_out["delta_g_exchange"] = [r.delta_g_exchange if r.delta_g_exchange is not None else "" for r in results]
                    df_out["e_binding_total"] = [r.e_binding_total if r.e_binding_total is not None else "" for r in results]
                    df_out["e_electrostatic"] = [r.e_electrostatic if r.e_electrostatic is not None else "" for r in results]
                    df_out["e_orbital"] = [r.e_orbital if r.e_orbital is not None else "" for r in results]
                    df_out["e_dispersion"] = [r.e_dispersion if r.e_dispersion is not None else "" for r in results]
                    df_out["error"] = [r.error for r in results]

                    st.subheader("Results")
                    st.dataframe(df_out, use_container_width=True)

                    st.subheader("Download results")
                    st.download_button(
                        "Download CSV",
                        df_out.to_csv(index=False),
                        "pfas_dft_predictions.csv",
                        "text/csv",
                        key="download_csv",
                    )
        else:
            st.info("Upload a CSV file with a SMILES column to run batch predictions.")

    st.divider()
    st.caption(
        "PFAS Removal DFT. Binding affinity analysis using ωB97X-D3/def2-TZVPD with SMD/C-PCM solvation."
    )


def render_comparison_page():
    """Render the comparison page."""
    st.title("PFAS Comparison View")
    st.markdown("Compare binding affinities and energy decomposition across multiple PFAS molecules.")
    
    try:
        predictor = get_predictor()
    except Exception as e:
        st.error(f"Could not load predictor: {e}")
        return
    
    # Get reference data for known PFAS
    reference_data = predictor.reference_data
    
    if reference_data:
        st.subheader("Binding Energy Comparison")
        comparison_data = []
        for pfas_name, data in reference_data.items():
            comparison_data.append({
                "PFAS": pfas_name,
                "ΔEexchange (kcal/mol)": data.get("delta_e_exchange"),
                "Total Binding (kcal/mol)": data.get("e_binding_total"),
                "Electrostatic (kcal/mol)": data.get("e_electrostatic"),
                "Orbital (kcal/mol)": data.get("e_orbital"),
            })
        
        comp_df = pd.DataFrame(comparison_data)
        comp_df = comp_df.sort_values("ΔEexchange (kcal/mol)", na_position='last')
        st.dataframe(comp_df, use_container_width=True)
        
        # Visualization
        if len(comp_df) > 0:
            st.subheader("Binding Energy Trends")
            chart_data = comp_df[["PFAS", "ΔEexchange (kcal/mol)"]].set_index("PFAS")
            st.bar_chart(chart_data)
    else:
        st.info("No reference data available. Upload calculation results to artifacts/reference_data.json.")


def render_dft_calculations_page():
    """Render the DFT calculations page."""
    st.title("DFT Calculation Setup")
    st.markdown(
        """
        Generate ORCA input files for PFAS-cholestyramine DFT calculations following the methodology 
        from the PFAS Removal DFT paper. This tool helps prepare calculation workflows for:
        - Geometry optimization (r2SCAN-3c and ωB97X-D3)
        - Frequency calculations
        - Energy Decomposition Analysis (EDA-NOCV)
        - Natural Bond Orbital (NBO) analysis
        - Global optimization (GOAT)
        - Anion exchange energy calculations
        """
    )
    
    st.divider()
    
    # Calculation type selection
    calc_type = st.radio(
        "Calculation Type",
        ["Complete Workflow", "Anion Exchange Energy", "Single Calculation"],
        horizontal=True,
    )
    
    if calc_type == "Complete Workflow":
        st.subheader("Complete Calculation Workflow")
        st.info(
            "This generates input files for the complete workflow:\n"
            "1. GOAT global optimization (GFN2-xTB)\n"
            "2. r2SCAN-3c optimization\n"
            "3. ωB97X-D3 optimization\n"
            "4. Frequency calculation\n"
            "5. EDA-NOCV analysis\n"
            "6. NBO analysis"
        )
        
        col1, col2 = st.columns(2)
        with col1:
            pfas_name = st.selectbox("PFAS Molecule", ["PFOS", "PFOA", "PFHxA", "FHEA", "Custom"])
            if pfas_name == "Custom":
                pfas_smiles = st.text_input("PFAS SMILES", placeholder="C(=O)([O-])C(F)(F)...")
            else:
                pfas_smiles_map = {
                    "PFOS": "C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)S(=O)(=O)[O-]",
                    "PFOA": "C(=O)([O-])C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)",
                    "PFHxA": "C(=O)([O-])C(F)(F)C(F)(F)C(F)(F)C(F)(F)",
                    "FHEA": "C(=O)([O-])CC(F)(F)C(F)(F)C(F)(F)C(F)(F)",
                }
                pfas_smiles = pfas_smiles_map.get(pfas_name, "")
        
        with col2:
            model_type = st.selectbox("Cholestyramine Model", ["BTMA", "Extended"])
            output_dir = st.text_input("Output Directory", value="orca_calculations")
        
        # Structure upload options
        st.subheader("Structure Files")
        structure_mode = st.radio(
            "Structure Input",
            ["Generate from SMILES", "Upload XYZ files"],
            horizontal=True,
        )
        
        pfas_xyz = None
        chol_xyz = None
        complex_xyz = None
        
        if structure_mode == "Generate from SMILES":
            if pfas_smiles and pfas_name != "Custom":
                if st.button("Generate Structures", type="primary"):
                    with st.spinner("Generating structures..."):
                        import tempfile
                        temp_dir = Path(tempfile.mkdtemp())
                        
                        pfas_xyz = prepare_pfas_structure(pfas_smiles, pfas_name, temp_dir)
                        chol_xyz = prepare_cholestyramine_structure(model_type, temp_dir)
                        
                        if pfas_xyz and chol_xyz:
                            complex_xyz = temp_dir / f"{pfas_name}_complex.xyz"
                            if prepare_complex_structure(pfas_xyz, chol_xyz, complex_xyz):
                                st.success("Structures generated successfully!")
                                st.session_state.pfas_xyz = pfas_xyz
                                st.session_state.chol_xyz = chol_xyz
                                st.session_state.complex_xyz = complex_xyz
                            else:
                                st.error("Failed to generate complex structure")
                        else:
                            st.error("Failed to generate structures")
        else:
            pfas_xyz_file = st.file_uploader("PFAS XYZ file", type=["xyz"])
            chol_xyz_file = st.file_uploader("Cholestyramine XYZ file", type=["xyz"])
            complex_xyz_file = st.file_uploader("Complex XYZ file (optional)", type=["xyz"])
            
            if pfas_xyz_file and chol_xyz_file:
                import tempfile
                temp_dir = Path(tempfile.mkdtemp())
                pfas_xyz = temp_dir / pfas_xyz_file.name
                pfas_xyz.write_bytes(pfas_xyz_file.read())
                chol_xyz = temp_dir / chol_xyz_file.name
                chol_xyz.write_bytes(chol_xyz_file.read())
                if complex_xyz_file:
                    complex_xyz = temp_dir / complex_xyz_file.name
                    complex_xyz.write_bytes(complex_xyz_file.read())
        
        # Configuration
        st.subheader("Calculation Configuration")
        col1, col2 = st.columns(2)
        with col1:
            functional = st.selectbox("Functional", ["r2SCAN-3c", "wB97X-D3"], index=1)
            basis_set = st.selectbox("Basis Set", ["def2-TZVPD", "def2-TZVP", "def2-SVP"])
            dielectric = st.number_input("Dielectric Constant", value=72.5, min_value=1.0)
        with col2:
            temperature = st.number_input("Temperature (K)", value=310.15)
            nprocs = st.number_input("Number of Processors", value=4, min_value=1)
            memory = st.number_input("Memory (MB)", value=4000)
        
        # Generate inputs
        if st.button("Generate ORCA Input Files", type="primary"):
            if pfas_xyz and chol_xyz:
                try:
                    config = ORCAConfig(
                        functional=functional,
                        basis_set=basis_set,
                        dielectric=dielectric,
                        temperature=temperature,
                        nprocs=nprocs,
                        mem=memory,
                    )
                    
                    output_path = Path(output_dir)
                    inputs = generate_workflow_inputs(
                        pfas_name,
                        pfas_xyz,
                        chol_xyz,
                        complex_xyz,
                        output_path,
                        config,
                    )
                    
                    st.success(f"Generated {len(inputs)} input files in {output_dir}")
                    st.subheader("Generated Input Files")
                    for calc_type_name, inp_path in inputs.items():
                        st.text(f"{calc_type_name}: {inp_path}")
                        with open(inp_path, 'r') as f:
                            st.download_button(
                                f"Download {calc_type_name}",
                                f.read(),
                                inp_path.name,
                                "text/plain",
                                key=f"download_{calc_type_name}",
                            )
                except Exception as e:
                    st.error(f"Error generating input files: {e}")
            else:
                st.warning("Please generate or upload structure files first")
    
    elif calc_type == "Anion Exchange Energy":
        st.subheader("Anion Exchange Energy Calculation")
        st.info(
            "Calculate ΔGexchange = G(R4N+X-) - G(R4N+Cl-) - G(X-) + G(Cl-)\n"
            "Requires frequency calculations for all four components."
        )
        
        st.warning("This feature requires uploading XYZ files for all components.")
        st.info("Upload structure files and configure calculation settings.")
    
    else:  # Single Calculation
        st.subheader("Single Calculation")
        st.info("Generate input file for a single calculation type.")
        
        single_calc_type = st.selectbox(
            "Calculation Type",
            ["Geometry Optimization", "Frequency", "Single Point", "EDA-NOCV", "NBO", "GOAT"],
        )
        
        xyz_file = st.file_uploader("Upload XYZ file", type=["xyz"])
        if xyz_file:
            import tempfile
            temp_dir = Path(tempfile.mkdtemp())
            temp_xyz = temp_dir / xyz_file.name
            temp_xyz.write_bytes(xyz_file.read())
            
            calc_type_map = {
                "Geometry Optimization": "opt",
                "Frequency": "freq",
                "Single Point": "sp",
                "EDA-NOCV": "eda",
                "NBO": "nbo",
                "GOAT": "goat",
            }
            
            config = ORCAConfig()
            if st.button("Generate Input File"):
                from src.pfasdft.orca_input import generate_orca_input
                inp_content = generate_orca_input(
                    temp_xyz,
                    Path("output.inp"),
                    config,
                    calculation_type=calc_type_map[single_calc_type],
                )
                st.download_button(
                    "Download ORCA Input",
                    inp_content,
                    "orca_input.inp",
                    "text/plain",
                )
    
    st.divider()
    
    # ORCA Output Parser Section
    st.subheader("Parse ORCA Output Files")
    st.markdown("Upload ORCA output files to extract calculation results.")
    
    output_files = st.file_uploader(
        "Upload ORCA Output Files (.out)",
        type=["out"],
        accept_multiple=True,
    )
    
    if output_files:
        results_data = []
        for out_file in output_files:
            import tempfile
            temp_dir = Path(tempfile.mkdtemp())
            temp_out = temp_dir / out_file.name
            temp_out.write_bytes(out_file.read())
            
            results = parse_orca_output(temp_out)
            results_data.append({
                "File": out_file.name,
                "SCF Energy (Ha)": results.scf_energy,
                "Gibbs Free Energy (Ha)": results.gibbs_free_energy,
                "Optimization Converged": results.optimization_converged,
                "Imaginary Frequencies": results.n_imaginary_frequencies,
                "Total Binding (kcal/mol)": results.e_binding_total,
            })
        
        if results_data:
            st.dataframe(pd.DataFrame(results_data), use_container_width=True)



# ============================================================================
# MAIN - NAVIGATION
# ============================================================================

def main():
    """Main app entry point with navigation."""
    if "current_page" not in st.session_state:
        st.session_state.current_page = "Home"

    st.sidebar.markdown("### Navigation")
    st.sidebar.markdown("")

    if st.sidebar.button("Home", use_container_width=True, key="nav_home"):
        st.session_state.current_page = "Home"

    if st.sidebar.button("Documentation", use_container_width=True, key="nav_docs"):
        st.session_state.current_page = "Documentation"

    if st.sidebar.button("PFAS Binding Prediction", use_container_width=True, key="nav_prediction"):
        st.session_state.current_page = "PFAS Binding Prediction"

    if st.sidebar.button("Comparison View", use_container_width=True, key="nav_comparison"):
        st.session_state.current_page = "Comparison View"

    if st.sidebar.button("DFT Calculations", use_container_width=True, key="nav_dft"):
        st.session_state.current_page = "DFT Calculations"

    st.sidebar.markdown("---")

    if st.session_state.current_page == "Home":
        render_home_page()
    elif st.session_state.current_page == "Documentation":
        render_documentation_page()
    elif st.session_state.current_page == "PFAS Binding Prediction":
        render_prediction_page()
    elif st.session_state.current_page == "Comparison View":
        render_comparison_page()
    elif st.session_state.current_page == "DFT Calculations":
        render_dft_calculations_page()


if __name__ == "__main__":
    main()
