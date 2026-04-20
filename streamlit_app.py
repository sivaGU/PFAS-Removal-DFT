"""
PFAS Removal DFT - Interaction Visualization GUI

Focused on manuscript-aligned visual exploration of PFAS/cholestyramine interactions.
"""
from __future__ import annotations

import io
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

try:
    import py3Dmol
    PY3DMOL_AVAILABLE = True
except ImportError:
    PY3DMOL_AVAILABLE = False


# ---------------------------------------------------------------------------
# Manuscript-aligned reference metrics for PFAS complexes
# ---------------------------------------------------------------------------
REFERENCE_METRICS: Dict[str, Dict[str, object]] = {
    "PFOA": {
        "delta_e_exchange": -5.65,
        "delta_g_exchange": -1.7904,
        "e_binding_total": -106.2,
        "e_electrostatic": -68.0,
        "e_pauli": 9.1,
        "e_orbital": -7.0,
        "e_dispersion": -3.0,
        "e_solvation": -50.4,
        "e_preparation": 16.0,
        "nocv_pairs": [-0.78, -0.74, -0.48, -0.46, -0.38],
        "nbo_donor_acceptor": "O lone pair -> sigma* C-H",
        "nbo_note": "Primary donation from PFAS headgroup oxygen lone pairs to methyl antibonding orbitals near quaternary ammonium.",
    },
    "PFOS": {
        "delta_e_exchange": -5.62,
        "delta_g_exchange": -0.96795,
        "e_binding_total": -101.2,
        "e_electrostatic": -67.2,
        "e_pauli": 7.9,
        "e_orbital": -6.2,
        "e_dispersion": -2.7,
        "e_solvation": -45.8,
        "e_preparation": 15.5,
        "nocv_pairs": [-0.76, -0.69, -0.48, -0.28, -0.23],
        "nbo_donor_acceptor": "O lone pair -> sigma* C-H",
        "nbo_note": "Sulfonate headgroup shows strong electrostatic interaction; orbital channels are present but secondary.",
    },
    "PFHXA": {
        "delta_e_exchange": -4.13,
        "delta_g_exchange": 2.02,
        "e_binding_total": -109.2,
        "e_electrostatic": -72.0,
        "e_pauli": 12.2,
        "e_orbital": -7.0,
        "e_dispersion": -3.5,
        "e_solvation": -48.8,
        "e_preparation": 13.7,
        "nocv_pairs": [-0.78, -0.82, -0.48, -0.46, -0.37],
        "nbo_donor_acceptor": "O lone pair -> sigma* C-H",
        "nbo_note": "Carboxylate-driven interaction pattern consistent with manuscript EDA/NBO trends.",
    },
    "FHEA": {
        "delta_e_exchange": -6.99,
        "delta_g_exchange": 2.99,
        "e_binding_total": -113.7,
        "e_electrostatic": -75.5,
        "e_pauli": 9.8,
        "e_orbital": -9.1,
        "e_dispersion": -1.8,
        "e_solvation": -50.1,
        "e_preparation": 16.2,
        "nocv_pairs": [-1.79, -1.14, -0.78, -0.67, -0.59],
        "nbo_donor_acceptor": "O lone pair -> sigma* C-H",
        "nbo_note": "Largest orbital stabilization among studied PFAS, matching manuscript NOCV discussion.",
    },
}

ELEMENT_COLORS = {
    "C": "#777777",
    "H": "#CCCCCC",
    "N": "#2C7FB8",
    "O": "#E34A33",
    "F": "#31A354",
    "S": "#756BB1",
}

OAK_DARK = "#2F4F2F"
OAK_MEDIUM_DARK = "#556B2F"
OAK_MEDIUM = "#6B8E23"
OAK_LIGHT = "#8FBC8F"
OAK_PALE = "#D4E4D4"

COVALENT_RADII = {
    "H": 0.31,
    "C": 0.76,
    "N": 0.71,
    "O": 0.66,
    "F": 0.57,
    "S": 1.05,
}


@dataclass
class Atom:
    idx: int
    symbol: str
    x: float
    y: float
    z: float


def apply_oak_theme() -> None:
    st.markdown(
        f"""
        <style>
            :root {{
                --oak-dark: {OAK_DARK};
                --oak-medium-dark: {OAK_MEDIUM_DARK};
                --oak-medium: {OAK_MEDIUM};
                --oak-light: {OAK_LIGHT};
                --oak-pale: {OAK_PALE};
            }}
            .stApp {{
                background-color: #ffffff;
            }}
            [data-testid="stSidebar"] {{
                background: var(--oak-dark);
                color: white;
            }}
            [data-testid="stSidebar"] * {{
                color: white !important;
            }}
            [data-testid="stSidebar"] [data-baseweb="select"] > div {{
                background: #f5f7f2 !important;
                color: #1f1f1f !important;
                border-radius: 8px !important;
                border: 1px solid #d7e3d1 !important;
            }}
            [data-testid="stSidebar"] [data-baseweb="select"] span {{
                color: #1f1f1f !important;
                opacity: 1 !important;
            }}
            [data-testid="stSidebar"] [data-baseweb="select"] div {{
                color: #1f1f1f !important;
                opacity: 1 !important;
            }}
            [data-testid="stSidebar"] [data-baseweb="select"] input {{
                color: #1f1f1f !important;
                -webkit-text-fill-color: #1f1f1f !important;
                opacity: 1 !important;
            }}
            [data-testid="stSidebar"] [data-baseweb="select"] svg {{
                fill: #1f1f1f !important;
                color: #1f1f1f !important;
            }}
            [data-testid="stSidebar"] [role="listbox"] * {{
                color: #1f1f1f !important;
            }}
            h1, h2, h3, h4 {{
                color: var(--oak-dark);
            }}
            .stButton > button, .stDownloadButton > button {{
                background: var(--oak-medium) !important;
                color: white !important;
                border: none;
                border-radius: 8px;
                font-weight: 600;
            }}
            .stButton > button:hover, .stDownloadButton > button:hover {{
                background: var(--oak-medium-dark) !important;
                color: white !important;
            }}
            [data-testid="stMetricValue"] {{
                color: var(--oak-dark);
                font-weight: 700;
            }}
            .stAlert {{
                border-left: 4px solid var(--oak-medium);
                border-radius: 6px;
            }}
            .block-container {{
                padding-top: 1.2rem;
                padding-bottom: 2rem;
            }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def _distance(a: Atom, b: Atom) -> float:
    return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2)


def parse_xyz_bytes(content: bytes) -> List[Atom]:
    lines = content.decode("utf-8", errors="ignore").splitlines()
    if len(lines) < 3:
        return []
    try:
        natoms = int(lines[0].strip())
    except ValueError:
        return []
    atoms: List[Atom] = []
    for i, line in enumerate(lines[2 : 2 + natoms]):
        parts = line.split()
        if len(parts) < 4:
            continue
        atoms.append(
            Atom(
                idx=i + 1,
                symbol=parts[0],
                x=float(parts[1]),
                y=float(parts[2]),
                z=float(parts[3]),
            )
        )
    return atoms


def parse_cube_header(content: bytes) -> Dict[str, float]:
    lines = content.decode("utf-8", errors="ignore").splitlines()
    if len(lines) < 6:
        return {}
    header = {}
    try:
        atom_line = lines[2].split()
        nx_line = lines[3].split()
        ny_line = lines[4].split()
        nz_line = lines[5].split()
        natoms_signed = int(atom_line[0])
        header["natoms"] = abs(natoms_signed)
        header["origin_x"] = float(atom_line[1])
        header["origin_y"] = float(atom_line[2])
        header["origin_z"] = float(atom_line[3])
        header["nx"] = int(nx_line[0])
        header["ny"] = int(ny_line[0])
        header["nz"] = int(nz_line[0])
        header["dx"] = math.sqrt(sum(float(v) ** 2 for v in nx_line[1:4]))
        header["dy"] = math.sqrt(sum(float(v) ** 2 for v in ny_line[1:4]))
        header["dz"] = math.sqrt(sum(float(v) ** 2 for v in nz_line[1:4]))
    except Exception:
        return {}
    return header


def infer_quaternary_n(atoms: List[Atom]) -> Optional[Atom]:
    nitrogens = [a for a in atoms if a.symbol == "N"]
    if not nitrogens:
        return None
    best_n = None
    best_score = -1
    for n in nitrogens:
        carbon_neighbors = sum(1 for a in atoms if a.symbol == "C" and _distance(a, n) <= 1.75)
        if carbon_neighbors > best_score:
            best_n = n
            best_score = carbon_neighbors
    return best_n


def infer_interactions(atoms: List[Atom]) -> pd.DataFrame:
    qn = infer_quaternary_n(atoms)
    if qn is None:
        return pd.DataFrame(columns=["type", "atom", "distance_A"])
    records: List[Dict[str, object]] = []
    headgroup_atoms = [a for a in atoms if a.symbol in {"O", "S"}]
    for a in headgroup_atoms:
        d = _distance(qn, a)
        if d <= 6.0:
            records.append(
                {
                    "type": "headgroup-ion_pair",
                    "atom": f"{a.symbol}{a.idx}",
                    "distance_A": round(d, 3),
                    "atom_idx": a.idx,
                }
            )
    fluorines = [a for a in atoms if a.symbol == "F"]
    for a in fluorines:
        d = _distance(qn, a)
        if d <= 8.0:
            records.append(
                {
                    "type": "tail-proximity",
                    "atom": f"{a.symbol}{a.idx}",
                    "distance_A": round(d, 3),
                    "atom_idx": a.idx,
                }
            )
    df = pd.DataFrame(records)
    if df.empty:
        return pd.DataFrame(columns=["type", "atom", "distance_A", "atom_idx"])
    return df.sort_values("distance_A").reset_index(drop=True)


def build_3d_figure(atoms: List[Atom], interactions: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    by_element: Dict[str, List[Atom]] = {}
    for a in atoms:
        by_element.setdefault(a.symbol, []).append(a)
    for element, group in sorted(by_element.items()):
        fig.add_trace(
            go.Scatter3d(
                x=[a.x for a in group],
                y=[a.y for a in group],
                z=[a.z for a in group],
                mode="markers",
                name=element,
                text=[f"{a.symbol}{a.idx}" for a in group],
                marker={
                    "size": 5 if element != "H" else 3,
                    "color": ELEMENT_COLORS.get(element, "#444444"),
                    "opacity": 0.9,
                },
            )
        )

    qn = infer_quaternary_n(atoms)
    if qn is not None and not interactions.empty:
        index = {a.idx: a for a in atoms}
        for _, row in interactions.iterrows():
            target = index.get(int(row["atom_idx"]))
            if target is None:
                continue
            line_color = "#D95F02" if row["type"] == "headgroup-ion_pair" else "#1B9E77"
            fig.add_trace(
                go.Scatter3d(
                    x=[qn.x, target.x],
                    y=[qn.y, target.y],
                    z=[qn.z, target.z],
                    mode="lines",
                    showlegend=False,
                    line={"width": 5, "color": line_color},
                    hovertext=f"{row['type']}: {row['distance_A']} A",
                    hoverinfo="text",
                )
            )

    fig.update_layout(
        margin={"l": 0, "r": 0, "b": 0, "t": 30},
        height=600,
        paper_bgcolor="white",
        plot_bgcolor="white",
        scene={
            "bgcolor": "white",
            "xaxis_title": "X (A)",
            "yaxis_title": "Y (A)",
            "zaxis_title": "Z (A)",
        },
        legend={"orientation": "h"},
    )
    return fig


def _bond_cutoff(a: Atom, b: Atom) -> float:
    ra = COVALENT_RADII.get(a.symbol, 0.77)
    rb = COVALENT_RADII.get(b.symbol, 0.77)
    return 1.25 * (ra + rb)


def infer_ligand_indices(atoms: List[Atom]) -> Set[int]:
    # PFAS is the fluorinated component in these complexes.
    fluorine_indices = [a.idx for a in atoms if a.symbol == "F"]
    if not fluorine_indices:
        return set()
    idx_map = {a.idx: a for a in atoms}
    adj: Dict[int, Set[int]] = {a.idx: set() for a in atoms}
    n = len(atoms)
    for i in range(n):
        for j in range(i + 1, n):
            ai = atoms[i]
            aj = atoms[j]
            if _distance(ai, aj) <= _bond_cutoff(ai, aj):
                adj[ai.idx].add(aj.idx)
                adj[aj.idx].add(ai.idx)

    seen: Set[int] = set()
    stack: List[int] = [fluorine_indices[0]]
    while stack:
        cur = stack.pop()
        if cur in seen:
            continue
        seen.add(cur)
        stack.extend(adj[cur] - seen)
    return seen


def build_ligand_figure(atoms: List[Atom], ligand_indices: Set[int]) -> go.Figure:
    ligand_atoms = [a for a in atoms if a.idx in ligand_indices]
    fig = go.Figure()
    if not ligand_atoms:
        fig.update_layout(
            height=520,
            paper_bgcolor="white",
            plot_bgcolor="white",
            margin={"l": 0, "r": 0, "b": 0, "t": 10},
        )
        return fig

    by_element: Dict[str, List[Atom]] = {}
    for a in ligand_atoms:
        by_element.setdefault(a.symbol, []).append(a)

    for element, group in sorted(by_element.items()):
        fig.add_trace(
            go.Scatter3d(
                x=[a.x for a in group],
                y=[a.y for a in group],
                z=[a.z for a in group],
                mode="markers+text" if element in {"O", "S"} else "markers",
                text=[f"{a.symbol}{a.idx}" for a in group] if element in {"O", "S"} else None,
                textposition="top center",
                name=element,
                marker={
                    "size": 7 if element != "H" else 4,
                    "color": ELEMENT_COLORS.get(element, "#444444"),
                    "opacity": 0.95,
                },
            )
        )
    fig.update_layout(
        height=520,
        margin={"l": 0, "r": 0, "b": 0, "t": 25},
        paper_bgcolor="white",
        plot_bgcolor="white",
        title="Ligand-Only 3D View (PFAS Component)",
        scene={
            "bgcolor": "white",
            "xaxis_title": "X (A)",
            "yaxis_title": "Y (A)",
            "zaxis_title": "Z (A)",
        },
        legend={"orientation": "h"},
    )
    return fig


def ligand_atoms_to_xyz_block(atoms: List[Atom], ligand_indices: Set[int]) -> str:
    ligand_atoms = [a for a in atoms if a.idx in ligand_indices]
    lines = [str(len(ligand_atoms)), "PFAS ligand fragment"]
    for a in ligand_atoms:
        lines.append(f"{a.symbol:2s} {a.x: .8f} {a.y: .8f} {a.z: .8f}")
    return "\n".join(lines)


def atoms_to_xyz_block(atoms: List[Atom]) -> str:
    lines = [str(len(atoms)), "PFAS-cholestyramine complex"]
    for a in atoms:
        lines.append(f"{a.symbol:2s} {a.x: .8f} {a.y: .8f} {a.z: .8f}")
    return "\n".join(lines)


def render_interaction_3d_view(atoms: List[Atom], interactions: pd.DataFrame) -> None:
    if not PY3DMOL_AVAILABLE:
        st.info("Install `py3Dmol` for bond-level molecular interaction rendering. Showing Plotly fallback.")
        st.plotly_chart(build_3d_figure(atoms, interactions), use_container_width=True)
        return

    xyz_block = atoms_to_xyz_block(atoms)
    viewer = py3Dmol.view(width=980, height=640)
    viewer.addModel(xyz_block, "xyz")

    # Base molecular rendering for full complex (shows actual bonds/sticks).
    viewer.setStyle({}, {"stick": {"radius": 0.14, "colorscheme": "default"}})
    viewer.addStyle({"elem": "H"}, {"stick": {"radius": 0.08, "colorscheme": "default"}})
    viewer.addStyle({"elem": "N"}, {"sphere": {"radius": 0.55, "color": "#2C7FB8"}})
    viewer.addStyle({"elem": "O"}, {"sphere": {"radius": 0.48, "color": "#E34A33"}})
    viewer.addStyle({"elem": "S"}, {"sphere": {"radius": 0.58, "color": "#756BB1"}})

    # Highlight interaction lines on top of bonded molecular structure.
    qn = infer_quaternary_n(atoms)
    if qn is not None and not interactions.empty:
        index = {a.idx: a for a in atoms}
        for _, row in interactions.iterrows():
            target = index.get(int(row["atom_idx"]))
            if target is None:
                continue
            color = "#D95F02" if row["type"] == "headgroup-ion_pair" else "#1B9E77"
            viewer.addCylinder(
                {
                    "start": {"x": qn.x, "y": qn.y, "z": qn.z},
                    "end": {"x": target.x, "y": target.y, "z": target.z},
                    "radius": 0.08 if row["type"] == "headgroup-ion_pair" else 0.06,
                    "color": color,
                    "fromCap": 1,
                    "toCap": 1,
                }
            )
            viewer.addLabel(
                f"{row['distance_A']} A",
                {
                    "position": {
                        "x": (qn.x + target.x) / 2.0,
                        "y": (qn.y + target.y) / 2.0,
                        "z": (qn.z + target.z) / 2.0,
                    },
                    "fontColor": "#1f1f1f",
                    "backgroundColor": "#f3f7f1",
                    "fontSize": 10,
                    "showBackground": True,
                    "backgroundOpacity": 0.8,
                },
            )

    viewer.setBackgroundColor("0xf7fbf7")
    viewer.zoomTo()
    viewer.rotate(18, "y")
    components.html(viewer._make_html(), height=660, scrolling=False)


def render_ligand_3d_view(atoms: List[Atom], ligand_indices: Set[int]) -> None:
    if not PY3DMOL_AVAILABLE:
        st.info("Install `py3Dmol` for stick/bond molecule rendering. Showing Plotly fallback.")
        # Fallback: display full complex points if molecular renderer unavailable.
        st.plotly_chart(build_3d_figure(atoms, pd.DataFrame()), use_container_width=True)
        return

    xyz_block = atoms_to_xyz_block(atoms)
    viewer = py3Dmol.view(width=980, height=620)
    viewer.addModel(xyz_block, "xyz")
    # Full molecular view: PFAS + cholestyramine.
    viewer.setStyle({}, {"stick": {"radius": 0.16, "colorscheme": "default"}})
    viewer.addStyle({"elem": "H"}, {"stick": {"radius": 0.09, "colorscheme": "default"}})
    viewer.addStyle({"elem": "F"}, {"sphere": {"radius": 0.36, "color": "#31A354"}})
    viewer.addStyle({"elem": "N"}, {"sphere": {"radius": 0.55, "color": "#2C7FB8"}})
    viewer.addStyle({"elem": "O"}, {"sphere": {"radius": 0.48, "color": "#E34A33"}})
    viewer.addStyle({"elem": "S"}, {"sphere": {"radius": 0.58, "color": "#756BB1"}})
    viewer.setBackgroundColor("0xf7fbf7")
    viewer.zoomTo()
    viewer.rotate(22, "y")
    components.html(viewer._make_html(), height=640, scrolling=False)


def detect_pfas_name(path_name: str) -> str:
    upper = path_name.upper()
    for key in ["PFOS", "PFOA", "PFHXA", "FHEA"]:
        if key in upper:
            return key
    return "CUSTOM"


def get_default_complexes(pfas_root: Path) -> Dict[str, Dict[str, Path]]:
    complexes: Dict[str, Dict[str, Path]] = {}
    for sub in sorted(pfas_root.glob("*")):
        if not sub.is_dir():
            continue
        xyz_files = list(sub.glob("*.xyz"))
        cube_files = list(sub.glob("*.cube"))
        if not xyz_files:
            continue
        key = detect_pfas_name(sub.name)
        complexes[key] = {
            "label": sub.name,
            "xyz": xyz_files[0],
            "cube": cube_files[0] if cube_files else None,
        }
    return complexes


def format_complex_label(key: str, data: Dict[str, Path]) -> str:
    label = str(data.get("label", key))
    if label:
        return f"{key} - {label}"
    return key


def render_metrics_panel(pfas_key: str) -> None:
    m = REFERENCE_METRICS.get(pfas_key)
    st.subheader("Manuscript-Aligned Energetics")
    if m is None:
        st.info("No default manuscript metrics are linked to this uploaded complex.")
        return
    c1, c2, c3 = st.columns(3)
    c1.metric("Delta E_exchange (kcal/mol)", f"{m['delta_e_exchange']:.3f}")
    c2.metric("Delta G_exchange (kcal/mol)", f"{m['delta_g_exchange']:.3f}")
    c3.metric("E_binding_total (kcal/mol)", f"{m['e_binding_total']:.1f}")

    eda_df = pd.DataFrame(
        {
            "Component": ["Electrostatic", "Pauli", "Orbital", "Dispersion", "Solvation", "Preparation"],
            "Energy (kcal/mol)": [
                m["e_electrostatic"],
                m["e_pauli"],
                m["e_orbital"],
                m["e_dispersion"],
                m["e_solvation"],
                m["e_preparation"],
            ],
        }
    ).set_index("Component")
    st.bar_chart(eda_df)

    st.subheader("NOCV + NBO")
    nocv = m["nocv_pairs"]
    nocv_df = pd.DataFrame(
        {
            "NOCV Pair": [f"Pair {i+1}" for i in range(len(nocv))],
            "Stabilization (kcal/mol)": nocv,
        }
    ).set_index("NOCV Pair")
    st.bar_chart(nocv_df)
    st.info(f"Primary donor-acceptor pattern: {m['nbo_donor_acceptor']}")
    st.caption(str(m["nbo_note"]))


def main() -> None:
    st.set_page_config(page_title="PFAS Interaction Visualizer", layout="wide")
    apply_oak_theme()
    st.title("PFAS Interaction Visualizer (Manuscript-Aligned)")
    st.caption(
        "Visualize PFAS-cholestyramine complexes with interaction overlays, EDA/NOCV/NBO summaries, and support for user-uploaded ORCA outputs."
    )

    app_dir = Path(__file__).resolve().parent
    pfas_root = app_dir / "PFAS Files"
    defaults = get_default_complexes(pfas_root)

    st.sidebar.header("Data Source")
    source_mode = st.sidebar.radio("Choose structure input", ["Bundled PFAS Files", "Upload your own files"])
    uploaded_xyz = None
    uploaded_cube = None
    active_label = ""
    pfas_key = "CUSTOM"

    if source_mode == "Bundled PFAS Files":
        if not defaults:
            st.error("No default complexes found in PFAS Files.")
            return
        keys = sorted(defaults.keys())
        selected_key = st.sidebar.selectbox(
            "Select bundled complex",
            keys,
            format_func=lambda k: format_complex_label(k, defaults[k]),
        )
        selected = defaults[selected_key]
        active_label = selected["label"]  # type: ignore[index]
        pfas_key = selected_key
        xyz_content = Path(selected["xyz"]).read_bytes()  # type: ignore[index]
        cube_content = Path(selected["cube"]).read_bytes() if selected.get("cube") else None  # type: ignore[arg-type]
    else:
        uploaded_xyz = st.sidebar.file_uploader("Upload complex XYZ", type=["xyz"])
        uploaded_cube = st.sidebar.file_uploader("Upload cube (optional)", type=["cube"])
        if uploaded_xyz is None:
            st.warning("Upload an XYZ file to start.")
            return
        xyz_content = uploaded_xyz.read()
        cube_content = uploaded_cube.read() if uploaded_cube else None
        active_label = uploaded_xyz.name
        pfas_key = detect_pfas_name(uploaded_xyz.name)

    atoms = parse_xyz_bytes(xyz_content)
    if not atoms:
        st.error("Could not parse XYZ file.")
        return

    st.markdown(f"### Active Complex: `{active_label}`")
    interactions = infer_interactions(atoms)
    ligand_indices = infer_ligand_indices(atoms)

    interaction_tab, ligand_tab = st.tabs(["Interaction View", "Ligand View"])

    with interaction_tab:
        st.subheader("3D Interaction Map")
        render_interaction_3d_view(atoms, interactions)
        st.caption(
            "Full bonded structure is shown. Orange lines = likely headgroup ion-pair contacts (N+ with O/S). Green lines = nearby fluorinated-tail contacts."
        )

        st.divider()
        st.subheader("Detected Interactions")
        if interactions.empty:
            st.info("No interactions detected with current geometric heuristics.")
        else:
            table_df = interactions[["type", "atom", "distance_A"]].copy()
            table_df.columns = ["Interaction type", "Atom", "Distance (A)"]
            st.dataframe(table_df, use_container_width=True, hide_index=True)

        st.divider()
        render_metrics_panel(pfas_key)

    with ligand_tab:
        st.subheader("Molecular View (Full Complex)")
        render_ligand_3d_view(atoms, ligand_indices)
        if ligand_indices:
            st.success(
                f"Full complex shown (PFAS + cholestyramine). PFAS fragment detected with {len(ligand_indices)} fluorinated-ligand atoms."
            )

    st.divider()
    st.subheader("Cube Grid / Orbital Context")
    if cube_content:
        cube_info = parse_cube_header(cube_content)
        if cube_info:
            c1, c2, c3 = st.columns(3)
            c1.metric("Grid Size", f"{int(cube_info['nx'])} x {int(cube_info['ny'])} x {int(cube_info['nz'])}")
            c2.metric("Voxel Spacing (A)", f"{cube_info['dx']:.3f}, {cube_info['dy']:.3f}, {cube_info['dz']:.3f}")
            c3.metric("Atoms in Cube Header", f"{int(cube_info['natoms'])}")
            st.caption("Cube parsing currently summarizes volumetric metadata and supports manuscript-style orbital context checks.")
        else:
            st.warning("Cube file uploaded but header parsing failed.")
    else:
        st.info("Upload/select a `.cube` file to show orbital-grid metadata.")

    st.divider()
    st.markdown(
        """
        #### Manuscript Logic Implemented
        - Structures are centered on PFAS-cholestyramine complexes and quaternary-ammonium interaction geometry.
        - Interaction visualization emphasizes headgroup electrostatics and local tail proximity effects.
        - EDA, NOCV, and NBO panels follow manuscript interpretation: electrostatics/solvation dominate, orbital terms are secondary but meaningful.
        - Upload mode lets users test their own complexes while keeping the same interpretation framework.
        """
    )


if __name__ == "__main__":
    main()
