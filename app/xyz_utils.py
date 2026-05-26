from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

try:
    import py3Dmol

    PY3DMOL_AVAILABLE = True
except ImportError:
    py3Dmol = None
    PY3DMOL_AVAILABLE = False


@dataclass(frozen=True)
class XyzStructure:
    name: str
    text: str
    atom_count: int


def load_xyz(path: Path, name: str | None = None) -> XyzStructure:
    return parse_xyz(path.read_text(errors="replace"), name or path.stem)


def parse_xyz(text: str, name: str) -> XyzStructure:
    lines = [line.rstrip() for line in text.replace("\r\n", "\n").replace("\r", "\n").splitlines() if line.strip()]
    if len(lines) < 2:
        raise ValueError("XYZ file must contain an atom count line, comment line, and coordinates.")
    try:
        atom_count = int(lines[0].strip())
    except ValueError as exc:
        raise ValueError("The first XYZ line must be an integer atom count.") from exc
    if len(lines) < atom_count + 2:
        raise ValueError(f"XYZ declares {atom_count} atoms but only {max(0, len(lines) - 2)} coordinate lines were found.")
    coord_lines = lines[2 : atom_count + 2]
    cleaned = [str(atom_count), lines[1] if len(lines) > 1 else name]
    for line in coord_lines:
        parts = line.split()
        if len(parts) < 4:
            raise ValueError(f"Invalid XYZ coordinate line: {line}")
        symbol = parts[0]
        x, y, z = (float(parts[1]), float(parts[2]), float(parts[3]))
        cleaned.append(f"{symbol:<2} {x:16.8f} {y:16.8f} {z:16.8f}")
    return XyzStructure(name=name, text="\n".join(cleaned) + "\n", atom_count=atom_count)


def xyz_from_upload(uploaded_file, fallback_name: str) -> XyzStructure:
    text = uploaded_file.read().decode("utf-8", errors="replace")
    return parse_xyz(text, Path(uploaded_file.name).stem or fallback_name)


def render_xyz_preview(structure: XyzStructure, *, height: int = 430) -> None:
    import streamlit as st
    import streamlit.components.v1 as components

    if not PY3DMOL_AVAILABLE:
        st.info("Install py3Dmol to enable the optional structure preview.")
        return
    viewer = py3Dmol.view(width="100%", height=height)
    viewer.addModel(structure.text, "xyz")
    viewer.setStyle({"stick": {"radius": 0.16}, "sphere": {"scale": 0.22}})
    viewer.setBackgroundColor("white")
    viewer.zoomTo()
    components.html(viewer._make_html(), height=height + 20)
