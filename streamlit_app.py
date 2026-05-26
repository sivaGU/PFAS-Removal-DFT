from __future__ import annotations

from pathlib import Path

import streamlit as st

from app.orca_templates import CALC_TYPES, OrcaSettings
from app.presets import (
    MODEL_NAMES,
    PFAS_NAMES,
    load_chloride,
    load_chloride_complex,
    load_complex,
    load_pfas,
    validate_examples,
)
from app.theme import apply_theme
from app.workflow_builder import exchange_bundle, full_workflow_bundle, single_calculation_bundle
from app.xyz_utils import XyzStructure, render_xyz_preview, xyz_from_upload


st.set_page_config(page_title="PFAS-Removal-DFT", layout="wide")
apply_theme()


def settings_panel(prefix: str = "") -> OrcaSettings:
    st.subheader("Computational Settings")
    c1, c2, c3 = st.columns(3)
    functional = c1.selectbox(
        "Functional",
        ["wB97X-D3", "r2SCAN-3c"],
        index=0,
        key=f"{prefix}_functional",
    )
    basis = c2.selectbox(
        "Basis set",
        ["def2-TZVPD", "def2-TZVP", "ma-def2-TZVP"],
        index=0,
        key=f"{prefix}_basis",
        disabled=functional == "r2SCAN-3c",
    )
    solvent = c3.text_input("SMD solvent", value="Water", key=f"{prefix}_solvent")
    c4, c5, c6 = st.columns(3)
    epsilon = c4.number_input("Dielectric constant", min_value=1.0, value=72.5, step=0.5, key=f"{prefix}_epsilon")
    temperature = c5.number_input("Temperature (K)", min_value=1.0, value=310.15, step=1.0, key=f"{prefix}_temp")
    maxiter = c6.number_input("Geometry MaxIter", min_value=1, value=350, step=25, key=f"{prefix}_maxiter")
    c7, c8 = st.columns(2)
    maxcore = c7.number_input("Max core per process (MB)", min_value=500, value=4000, step=500, key=f"{prefix}_maxcore")
    nprocs = c8.number_input("Number of processes", min_value=1, value=32, step=1, key=f"{prefix}_nprocs")
    return OrcaSettings(
        functional=functional,
        basis=basis,
        solvent=solvent,
        epsilon=float(epsilon),
        temperature=float(temperature),
        maxcore=int(maxcore),
        nprocs=int(nprocs),
        maxiter=int(maxiter),
    )


def download_zip(bundle, filename: str, label: str) -> None:
    st.download_button(
        label,
        data=bundle.as_zip_bytes(),
        file_name=filename,
        mime="application/zip",
        use_container_width=True,
    )


def source_selector(prefix: str, *, require_complex: bool) -> tuple[str, str, XyzStructure, XyzStructure | None]:
    pfas_mode = st.radio(
        "PFAS coordinate source",
        ["Built-in manuscript PFAS", "Upload custom PFAS XYZ"],
        horizontal=True,
        key=f"{prefix}_pfas_mode",
    )
    model = st.selectbox("Cholestyramine model", MODEL_NAMES, key=f"{prefix}_model")
    if pfas_mode == "Built-in manuscript PFAS":
        pfas_name = st.selectbox("PFAS", PFAS_NAMES, key=f"{prefix}_pfas")
        pfas = load_pfas(pfas_name)
        complex_structure = load_complex(model, pfas_name)
        st.success(f"Using bundled {pfas_name} and bundled {model} {pfas_name} complex XYZ files.")
        return pfas_name, model, pfas, complex_structure

    uploaded_pfas = st.file_uploader("Upload PFAS XYZ", type=["xyz"], key=f"{prefix}_custom_pfas")
    if uploaded_pfas is None:
        st.info("Upload a PFAS XYZ file to continue.")
        st.stop()
    pfas_name = st.text_input("Custom PFAS label", value=Path(uploaded_pfas.name).stem, key=f"{prefix}_custom_label")
    pfas = xyz_from_upload(uploaded_pfas, pfas_name)
    complex_structure = None
    if require_complex:
        uploaded_complex = st.file_uploader(
            "Upload matching R4N+X- complex XYZ",
            type=["xyz"],
            key=f"{prefix}_custom_complex",
        )
        if uploaded_complex is None:
            st.warning("Full complex calculations require a matching R4N+X- complex XYZ for custom PFAS inputs.")
            st.stop()
        complex_structure = xyz_from_upload(uploaded_complex, f"R4N+{pfas_name}-")
    return pfas_name, model, pfas, complex_structure


def render_home() -> None:
    st.title("PFAS-Removal-DFT ORCA Input Generator")
    st.write(
        "Generate manuscript-style ORCA input files for PFAS/cholestyramine DFT workflows without editing input decks by hand."
    )
    st.markdown(
        """
        **Supported workflows**

        - Full workflow inputs for GOAT/GFN2-xTB, r2SCAN-3c optimization, wB97X-D3 optimization, frequency, EDA-NOCV, and NBO calculations.
        - Single ORCA input generation from one uploaded XYZ file.
        - Exchange-energy component frequency inputs using R4N+X-, R4N+Cl-, X-, and Cl- structures.

        Built-in examples are provided for PFOA, PFOS, PFHxA, and FHEA with BTMA and Extended Monomer complexes.
        """
    )
    st.info("This Streamlit app generates input files only. ORCA, GOAT, and NBO calculations should be run on your own computing resources.")
    missing = validate_examples()
    if missing:
        st.error("Some built-in example structures are missing.")
        st.code("\n".join(missing))
    else:
        st.success("Built-in PFAS, complex, and chloride-complex examples are available.")


def render_full_workflow() -> None:
    st.title("Full Workflow Generator")
    pfas_name, model, pfas, complex_structure = source_selector("full", require_complex=True)
    assert complex_structure is not None

    with st.expander("Preview structures", expanded=False):
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"**{pfas_name} anion** ({pfas.atom_count} atoms)")
            render_xyz_preview(pfas, height=360)
        with c2:
            st.markdown(f"**{model} {pfas_name} complex** ({complex_structure.atom_count} atoms)")
            render_xyz_preview(complex_structure, height=360)

    settings = settings_panel("full")
    st.subheader("Calculations to Generate")
    c1, c2, c3 = st.columns(3)
    include_goat = c1.checkbox("GOAT global optimization (GFN2-xTB)", value=True)
    include_r2 = c1.checkbox("r2SCAN-3c geometry optimization", value=True)
    include_wb = c2.checkbox("wB97X-D3 geometry optimization", value=True)
    include_freq = c2.checkbox("Frequency calculation", value=True)
    include_eda = c3.checkbox("EDA-NOCV analysis", value=True)
    include_nbo = c3.checkbox("NBO analysis", value=True)

    bundle = full_workflow_bundle(
        pfas_name=pfas_name,
        model_name=model,
        pfas=pfas,
        complex_structure=complex_structure,
        settings=settings,
        include_goat=include_goat,
        include_r2scan_opt=include_r2,
        include_wb97xd3_opt=include_wb,
        include_frequency=include_freq,
        include_eda=include_eda,
        include_nbo=include_nbo,
    )
    st.metric("Files in ZIP", len(bundle.files))
    st.dataframe({"Generated file": sorted(bundle.files)}, hide_index=True, use_container_width=True)
    download_zip(bundle, f"{model.replace(' ', '_')}_{pfas_name}_ORCA_workflow.zip", "Download full workflow ZIP")


def render_single_calculation() -> None:
    st.title("Single Calculation Generator")
    uploaded = st.file_uploader("Upload one XYZ file", type=["xyz"], key="single_xyz")
    if uploaded is None:
        st.info("Upload an XYZ file to generate one ORCA input.")
        return
    structure = xyz_from_upload(uploaded, Path(uploaded.name).stem)
    c1, c2, c3 = st.columns(3)
    calc_type = c1.selectbox("Calculation type", CALC_TYPES, key="single_calc")
    charge = c2.number_input("Total charge", value=0, step=1, key="single_charge")
    multiplicity = c3.number_input("Multiplicity", min_value=1, value=1, step=1, key="single_mult")
    pfas_atom_count = None
    if calc_type == "EDA-NOCV analysis":
        pfas_atom_count = st.number_input(
            "PFAS fragment atom count",
            min_value=1,
            max_value=max(1, structure.atom_count - 1),
            value=max(1, min(25, structure.atom_count - 1)),
            step=1,
            help="The template assumes the PFAS fragment is the final block of atoms in the complex XYZ.",
        )
    settings = settings_panel("single")
    with st.expander("Preview uploaded structure", expanded=False):
        render_xyz_preview(structure)
    try:
        bundle = single_calculation_bundle(
            structure=structure,
            calc_type=calc_type,
            charge=int(charge),
            multiplicity=int(multiplicity),
            settings=settings,
            pfas_atom_count=int(pfas_atom_count) if pfas_atom_count is not None else None,
        )
    except ValueError as exc:
        st.error(str(exc))
        return
    st.dataframe({"Generated file": sorted(bundle.files)}, hide_index=True, use_container_width=True)
    download_zip(bundle, f"{structure.name}_{calc_type.split()[0]}_ORCA_input.zip", "Download ORCA input ZIP")


def render_exchange_generator() -> None:
    st.title("Exchange-Energy Component Generator")
    pfas_name, model, pfas, complex_structure = source_selector("exchange", require_complex=True)
    assert complex_structure is not None
    st.subheader("R4N+Cl- coordinate source")
    uploaded_cl_complex = st.file_uploader(
        "Optional: upload custom R4N+Cl- XYZ",
        type=["xyz"],
        key="exchange_custom_r4ncl",
    )
    chloride_complex = (
        xyz_from_upload(uploaded_cl_complex, "R4N+Cl-")
        if uploaded_cl_complex is not None
        else load_chloride_complex(model)
    )
    chloride = load_chloride()
    settings = settings_panel("exchange")
    bundle = exchange_bundle(
        pfas_name=pfas_name,
        model_name=model,
        pfas=pfas,
        complex_structure=complex_structure,
        chloride_complex=chloride_complex,
        chloride=chloride,
        settings=settings,
    )
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("R4N+X- atoms", complex_structure.atom_count)
    c2.metric("R4N+Cl- atoms", chloride_complex.atom_count)
    c3.metric("X- atoms", pfas.atom_count)
    c4.metric("Cl- atoms", chloride.atom_count)
    st.dataframe({"Generated file": sorted(bundle.files)}, hide_index=True, use_container_width=True)
    download_zip(bundle, f"{model.replace(' ', '_')}_{pfas_name}_exchange_frequency_inputs.zip", "Download exchange input ZIP")


def render_documentation() -> None:
    st.title("Documentation")
    st.markdown(
        """
        This GUI generates ORCA input files that follow the DFT workflow described in the manuscript methods section.

        **Coordinate assumptions**

        - Built-in PFAS examples include standalone PFAS anions and matching BTMA/Extended Monomer complexes.
        - Custom PFAS uploads are accepted as XYZ files.
        - Full complex workflows for custom PFAS require a matching `R4N+X-` complex XYZ upload.
        - EDA-NOCV fragment definitions assume the cholestyramine model atoms come first and the PFAS atoms are the final block in the complex XYZ.

        **Default settings**

        - SMD/CPCM solvent: Water
        - Dielectric constant: 72.5
        - Temperature: 310.15 K
        - Geometry optimization MaxIter: 350

        **Outputs**

        All files are generated in memory and downloaded as ZIP archives. The app does not run ORCA calculations.
        """
    )


PAGES = {
    "Home": render_home,
    "Full Workflow Generator": render_full_workflow,
    "Exchange Energy Inputs": render_exchange_generator,
    "Single Calculation Generator": render_single_calculation,
    "Documentation": render_documentation,
}


if "current_page" not in st.session_state:
    st.session_state.current_page = "Home"

with st.sidebar:
    st.markdown("### Navigation")
    for idx, page_name in enumerate(PAGES):
        button_type = "primary" if st.session_state.current_page == page_name else "secondary"
        if st.button(page_name, key=f"nav_{idx}", type=button_type, use_container_width=True):
            st.session_state.current_page = page_name
            st.rerun()
    st.divider()
    st.caption("PFAS-Removal-DFT")
    st.caption("ORCA input generation for manuscript-style workflows.")

PAGES[st.session_state.current_page]()
