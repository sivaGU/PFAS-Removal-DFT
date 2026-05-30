from __future__ import annotations

from pathlib import Path

import streamlit as st

from app.orca_templates import CALC_TYPES, GEOMETRY_CALC_TYPES, OrcaSettings
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
from app.workflow_builder import exchange_bundle, interaction_bundle, single_calculation_bundle
from app.xyz_utils import XyzStructure, render_xyz_preview, xyz_from_upload


st.set_page_config(page_title="PFAS Removal", layout="wide")
apply_theme()


def settings_panel(prefix: str = "", heading: str = "Computational Settings", default_functional: str = "wB97X-D3") -> OrcaSettings:
    st.subheader(heading)
    c1, c2, c3 = st.columns(3)
    functional_options = ["wB97X-D3", "r2SCAN-3c"]
    functional = c1.selectbox(
        "Functional",
        functional_options,
        index=functional_options.index(default_functional) if default_functional in functional_options else 0,
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
        ["Demo PFAS", "Upload custom PFAS XYZ"],
        horizontal=True,
        key=f"{prefix}_pfas_mode",
    )
    model = st.selectbox("Cholestyramine model", MODEL_NAMES, key=f"{prefix}_model")
    if pfas_mode == "Demo PFAS":
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
    st.title("PFAS Removal ORCA Input Generator")
    st.write(
        "Generate ORCA input files for PFAS/cholestyramine DFT workflows without editing input decks by hand."
    )
    st.markdown(
        """
        **Supported workflows**

        - Anion exchange energetics inputs with configurable geometry optimization, frequency, and optional GOAT/GFN2-xTB stages.
        - Interaction analysis inputs for EDA-NOCV and/or NBO calculations.
        - Single ORCA input generation from one uploaded XYZ file.

        Demo examples are provided for PFOA, PFOS, PFHxA, and FHEA with BTMA and Extended Monomer complexes.
        """
    )
    workflow_image = Path(__file__).resolve().parent / "assets" / "workflow_diagram.png"
    if workflow_image.exists():
        st.image(str(workflow_image), use_container_width=True)
        st.markdown(
            """
            Workflow diagram of study methods. Step 1 summarizes the gathering of starting geometries for PFAS, BTMA, and cholestyramine model structures from PubChem. Polymer construction is described in Sections 2.1.1 and 2.1.2. Step 2 covers structure cleaning, assignment of protonation states at physiological pH, hydrogen addition, and basic optimization using force fields, as described in Section 2.1.1. Step 3a covers conformer searching with the GOAT global optimizer and GFN2-xTB for the extended monomer systems, as described in Section 2.3.1. Step 3b covers DFT geometry optimization and vibrational frequency calculations using r²SCAN-3c and ωB97X-D3 as described in Sections 2.2.1, 2.2.2, and 2.3.1. Step 3c covers calculation of electronic exchange energies and Gibbs free energies of exchange using the thermodynamic cycles defined in the Methods, as described in Sections 2.2.1, 2.2.3, and 2.3.1. Step 4a covers EDA-NOCV analysis of PFAS resin complexes to separate electrostatic, Pauli, orbital, dispersion, exchange-correlation, solvation, and preparation energy terms, as described in Section 2.4. Step 4b covers NBO analysis of donor-acceptor orbital interactions in selected PFAS-BTMA complexes, as described in Section 2.4. Step 5 presents the relaxed potential energy surface scan used to evaluate the local exchange coordinate for PFOA, chloride, and BTMA, as described in Section 2.5. Step 6 presents MD simulations of solvated 48-unit cholestyramine oligomers and the associated MD exchange free energy workflow, as described in Section 2.6. Step 7 presents the graphical user interface developed to generate ORCA input files, parse ORCA outputs, and support reproducible application of the workflow, as described in Section 2.7. Implicit solvation with water was used for the BTMA and extended monomer calculations unless otherwise stated. The extended monomer PFOA and PFOS bound states were also evaluated with 1-octanol implicit solvation to approximate the local resin microenvironment.
            """
        )
    st.info("This Streamlit app generates input files only. ORCA, GOAT, and NBO calculations should be run on your own computing resources.")
    missing = validate_examples()
    if missing:
        st.error("Some Demo example structures are missing.")
        st.code("\n".join(missing))
    else:
        st.success("Demo PFAS, complex, and chloride complex examples are available.")


def render_exchange_analysis() -> None:
    st.title("Anion Exchange Energetics Analysis")
    st.write(
        "Generate the ORCA inputs needed for anion exchange energetics using R4N+X-, R4N+Cl-, X-, and Cl- structures."
    )
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

    with st.expander("Structure previews", expanded=False):
        tabs = st.tabs(["R4N+X-", "R4N+Cl-", "X-", "Cl-"])
        with tabs[0]:
            render_xyz_preview(complex_structure, height=360)
        with tabs[1]:
            render_xyz_preview(chloride_complex, height=360)
        with tabs[2]:
            render_xyz_preview(pfas, height=360)
        with tabs[3]:
            render_xyz_preview(chloride, height=300)

    with st.expander("1. Geometry optimization configuration", expanded=True):
        geometry_calc_type = st.selectbox(
            "Geometry optimization template",
            GEOMETRY_CALC_TYPES,
            index=1,
            key="exchange_geometry_calc_type",
        )
        default_geom_functional = "r2SCAN-3c" if geometry_calc_type.startswith("r2SCAN") else "wB97X-D3"
        geometry_settings = settings_panel("exchange_geom", "Geometry Optimization Settings", default_geom_functional)

    with st.expander("2. Frequency calculation configuration", expanded=True):
        frequency_settings = settings_panel("exchange_freq", "Frequency Settings", "wB97X-D3")

    with st.expander("Optional GOAT global optimization", expanded=False):
        include_goat = st.checkbox("Generate GOAT/GFN2-xTB inputs before optimization", value=False)
        goat_settings = settings_panel("exchange_goat", "GOAT Settings", "wB97X-D3") if include_goat else None

    bundle = exchange_bundle(
        pfas_name=pfas_name,
        model_name=model,
        pfas=pfas,
        complex_structure=complex_structure,
        chloride_complex=chloride_complex,
        chloride=chloride,
        geometry_calc_type=geometry_calc_type,
        geometry_settings=geometry_settings,
        frequency_settings=frequency_settings,
        include_goat=include_goat,
        goat_settings=goat_settings,
    )
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("R4N+X- atoms", complex_structure.atom_count)
    c2.metric("R4N+Cl- atoms", chloride_complex.atom_count)
    c3.metric("X- atoms", pfas.atom_count)
    c4.metric("Cl- atoms", chloride.atom_count)
    st.metric("Files in ZIP", len(bundle.files))
    st.dataframe({"Generated file": sorted(bundle.files)}, hide_index=True, use_container_width=True)
    download_zip(bundle, f"{model.replace(' ', '_')}_{pfas_name}_anion_exchange_energetics_inputs.zip", "Download anion exchange energetics ZIP")


def render_interaction_analysis() -> None:
    st.title("Interaction Analysis")
    st.write("Generate EDA-NOCV and/or NBO ORCA inputs for PFAS-cholestyramine interaction analysis.")
    pfas_name, model, pfas, complex_structure = source_selector("interaction", require_complex=True)
    assert complex_structure is not None

    with st.expander("Preview complex and fragment assignment", expanded=False):
        c1, c2 = st.columns([2, 1])
        with c1:
            render_xyz_preview(complex_structure, height=390)
        with c2:
            frag2_start = complex_structure.atom_count - pfas.atom_count
            st.metric("Complex atoms", complex_structure.atom_count)
            st.metric("PFAS fragment atoms", pfas.atom_count)
            st.markdown("**Generated EDA fragment blocks**")
            st.code(
                f"1 {{0:{frag2_start - 1}}} end\n2 {{{frag2_start}:{complex_structure.atom_count - 1}}} end",
                language="text",
            )
            st.caption("The bundled complexes place the PFAS atoms as the final block in the XYZ file.")

    settings = settings_panel("interaction", "Interaction Calculation Settings", "wB97X-D3")
    st.subheader("Analyses to Generate")
    c1, c2 = st.columns(2)
    include_eda = c1.checkbox("EDA-NOCV analysis", value=True)
    include_nbo = c2.checkbox("NBO analysis", value=True)
    if not include_eda and not include_nbo:
        st.warning("Select at least one interaction analysis.")
        return
    bundle = interaction_bundle(
        pfas_name=pfas_name,
        model_name=model,
        pfas=pfas,
        complex_structure=complex_structure,
        settings=settings,
        include_eda=include_eda,
        include_nbo=include_nbo,
    )
    st.metric("Files in ZIP", len(bundle.files))
    st.dataframe({"Generated file": sorted(bundle.files)}, hide_index=True, use_container_width=True)
    download_zip(bundle, f"{model.replace(' ', '_')}_{pfas_name}_interaction_analysis_inputs.zip", "Download interaction analysis ZIP")


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


def render_documentation() -> None:
    st.title("Documentation")
    st.markdown(
        """
        This GUI generates ORCA input files that follow the DFT workflow described in the methods section.

        **Coordinate assumptions**

        - Demo PFAS examples include standalone PFAS anions and matching BTMA/Extended Monomer complexes.
        - Custom PFAS uploads are accepted as XYZ files.
        - Anion exchange energetics inputs use `R4N+X-`, `R4N+Cl-`, `X-`, and `Cl-` components.
        - Custom `R4N+Cl-` uploads are optional; otherwise the bundled BTMA/Extended Monomer chloride complex is used.
        - Interaction analysis inputs for custom PFAS require a matching `R4N+X-` complex XYZ upload.
        - EDA-NOCV fragment definitions assume the cholestyramine model atoms come first and the PFAS atoms are the final block in the complex XYZ.
        - Every generated ORCA input references a local XYZ filename only, with the corresponding XYZ copied into the same ZIP folder.
        - EDA-NOCV ZIP folders include `frag1method.txt` and `frag2method.txt` for fragment methods and CPCM settings.

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
    "Anion Exchange Energetics": render_exchange_analysis,
    "Interaction Analysis": render_interaction_analysis,
    "Single Calculation Generator": render_single_calculation,
    "Documentation": render_documentation,
}


if "current_page" not in st.session_state or st.session_state.current_page not in PAGES:
    st.session_state.current_page = "Home"

with st.sidebar:
    st.markdown("### Navigation")
    for idx, page_name in enumerate(PAGES):
        button_type = "primary" if st.session_state.current_page == page_name else "secondary"
        if st.button(page_name, key=f"nav_{idx}", type=button_type, use_container_width=True):
            st.session_state.current_page = page_name
            st.rerun()
    st.divider()
    st.caption("PFAS Removal")
    st.caption("ORCA input generation for anion exchange energetics and interaction analysis workflows.")

PAGES[st.session_state.current_page]()
