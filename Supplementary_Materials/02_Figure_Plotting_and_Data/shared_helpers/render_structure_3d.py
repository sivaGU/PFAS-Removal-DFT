#!/usr/bin/env python3
"""
Render simple 3D molecular structure figures from XYZ or PDB files.

Examples
--------
python3 tools/render_structure_3d.py molecule.xyz --out molecule.png
python3 tools/render_structure_3d.py frame.pdb --hide-h --contact 12:48 --contact 15:101:#d62728:O...N
"""

from __future__ import annotations

import argparse
import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


ELEMENT_COLORS = {
    "H": "#f2f2f2",
    "C": "#4d4d4d",
    "N": "#2f65d9",
    "O": "#d62728",
    "F": "#47b8a6",
    "P": "#e0702f",
    "S": "#e0b33f",
    "Cl": "#2ca02c",
    "Br": "#8c564b",
    "I": "#9467bd",
    "Na": "#1f77b4",
    "K": "#9467bd",
}

VDW_RADII = {
    "H": 0.31,
    "C": 0.76,
    "N": 0.71,
    "O": 0.66,
    "F": 0.57,
    "P": 1.07,
    "S": 1.05,
    "Cl": 1.02,
    "Br": 1.20,
    "I": 1.39,
    "Na": 1.66,
    "K": 2.03,
}

PYMOL_ENV = "pymol-render"
PYMOL_RUNNER = Path("/home/zwa/miniforge3/bin/mamba")


@dataclass
class Structure:
    elements: list[str]
    coords: np.ndarray
    names: list[str]
    residues: list[str]


@dataclass
class Contact:
    i: int
    j: int
    color: str
    label: str | None
    label_position: tuple[float, float, float] | None = None


def crop_transparent_png(path: Path, pad: int = 24) -> None:
    try:
        from PIL import Image
    except ImportError:
        return

    img = Image.open(path).convert("RGBA")
    alpha = np.asarray(img)[..., 3]
    rows, cols = np.where(alpha > 0)
    if rows.size == 0 or cols.size == 0:
        return
    left = max(int(cols.min()) - pad, 0)
    upper = max(int(rows.min()) - pad, 0)
    right = min(int(cols.max()) + pad + 1, img.width)
    lower = min(int(rows.max()) + pad + 1, img.height)
    img.crop((left, upper, right, lower)).save(path)


def quote_pymol(value: str | Path) -> str:
    text = str(value).replace("\\", "\\\\").replace('"', '\\"')
    return f'"{text}"'


def pymol_path(value: str | Path) -> str:
    return str(value).replace("\\", "\\\\").replace(" ", "\\ ")


def normalize_element(raw: str) -> str:
    text = "".join(ch for ch in raw.strip() if ch.isalpha())
    if not text:
        return "C"
    if len(text) >= 2 and text[:2].capitalize() in ELEMENT_COLORS:
        return text[:2].capitalize()
    return text[0].upper()


def read_xyz(path: Path, frame: int) -> Structure:
    lines = path.read_text().splitlines()
    offset = 0
    frames: list[Structure] = []
    while offset < len(lines):
        if not lines[offset].strip():
            offset += 1
            continue
        try:
            natoms = int(lines[offset].strip())
        except ValueError as exc:
            raise ValueError(f"Could not parse XYZ atom count near line {offset + 1}") from exc
        start = offset + 2
        stop = start + natoms
        if stop > len(lines):
            raise ValueError(f"XYZ frame beginning line {offset + 1} is incomplete")
        elements, coords, names, residues = [], [], [], []
        for atom_idx, line in enumerate(lines[start:stop], start=1):
            parts = line.split()
            if len(parts) < 4:
                raise ValueError(f"Could not parse XYZ atom line {start + atom_idx}: {line}")
            element = normalize_element(parts[0])
            elements.append(element)
            coords.append([float(parts[1]), float(parts[2]), float(parts[3])])
            names.append(f"{element}{atom_idx}")
            residues.append("")
        frames.append(Structure(elements, np.asarray(coords, dtype=float), names, residues))
        offset = stop
    if not frames:
        raise ValueError(f"No XYZ frames found in {path}")
    if frame < 0:
        frame = len(frames) + frame
    if frame < 0 or frame >= len(frames):
        raise IndexError(f"Frame {frame} requested, but {path} contains {len(frames)} frame(s)")
    return frames[frame]


def read_pdb(path: Path) -> Structure:
    elements, coords, names, residues = [], [], [], []
    for line in path.read_text().splitlines():
        if not line.startswith(("ATOM  ", "HETATM")):
            continue
        name = line[12:16].strip()
        residue = line[17:20].strip()
        element = normalize_element(line[76:78].strip() or name)
        elements.append(element)
        coords.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
        names.append(name or f"{element}{len(elements)}")
        residues.append(residue)
    if not elements:
        raise ValueError(f"No ATOM/HETATM records found in {path}")
    return Structure(elements, np.asarray(coords, dtype=float), names, residues)


def read_structure(path: Path, frame: int) -> Structure:
    suffix = path.suffix.lower()
    if suffix == ".xyz":
        return read_xyz(path, frame)
    if suffix == ".pdb":
        return read_pdb(path)
    raise ValueError(f"Unsupported input format {suffix}; use .xyz or .pdb")


def write_xyz_frame(path: Path, structure: Structure, title: str = "selected frame") -> None:
    with path.open("w") as handle:
        handle.write(f"{len(structure.elements)}\n{title}\n")
        for element, xyz in zip(structure.elements, structure.coords):
            handle.write(f"{element:<2s} {xyz[0]:16.8f} {xyz[1]:16.8f} {xyz[2]:16.8f}\n")


def infer_bonds(elements: list[str], coords: np.ndarray, scale: float) -> list[tuple[int, int]]:
    bonds: list[tuple[int, int]] = []
    natoms = len(elements)
    for i in range(natoms - 1):
        ri = VDW_RADII.get(elements[i], 0.77)
        for j in range(i + 1, natoms):
            rj = VDW_RADII.get(elements[j], 0.77)
            max_dist = scale * (ri + rj)
            dist = np.linalg.norm(coords[i] - coords[j])
            if 0.35 < dist <= max_dist:
                bonds.append((i, j))
    return bonds


def parse_contact(spec: str) -> Contact:
    parts = spec.split(":")
    if len(parts) < 2:
        raise argparse.ArgumentTypeError("Contacts must look like i:j, i:j:color, or i:j:color:label")
    try:
        i = int(parts[0]) - 1
        j = int(parts[1]) - 1
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Contact atom indices must be integers") from exc
    color = parts[2] if len(parts) >= 3 and parts[2] else "#d62728"
    label = parts[3] if len(parts) >= 4 and parts[3] else None
    return Contact(i=i, j=j, color=color, label=label)


def set_equal_axes(ax, coords: np.ndarray, pad: float = 1.8) -> None:
    center = coords.mean(axis=0)
    span = float(np.max(np.ptp(coords, axis=0)))
    radius = max(span / 2.0 + pad, 2.0)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def draw_structure(
    structure: Structure,
    out_path: Path,
    contacts: list[Contact],
    hide_h: bool,
    show_indices: bool,
    title: str | None,
    elev: float,
    azim: float,
    dpi: int,
    bond_scale: float,
    atom_scale: float,
    no_bonds: bool,
    legend: bool,
    no_crop: bool,
) -> None:
    coords = structure.coords
    elements = structure.elements
    atom_mask = np.ones(len(elements), dtype=bool)
    if hide_h:
        atom_mask &= np.asarray([element != "H" for element in elements])

    fig = plt.figure(figsize=(7.0, 6.2), dpi=dpi)
    ax = fig.add_subplot(111, projection="3d")
    ax.view_init(elev=elev, azim=azim)
    ax.set_proj_type("persp")
    ax.set_box_aspect((1, 1, 1))
    ax.set_axis_off()
    if title:
        ax.set_title(title, pad=2)

    if not no_bonds:
        for i, j in infer_bonds(elements, coords, bond_scale):
            if not (atom_mask[i] and atom_mask[j]):
                continue
            segment = coords[[i, j]]
            ax.plot(segment[:, 0], segment[:, 1], segment[:, 2], color="#9a9a9a", lw=1.4, alpha=0.85)

    for element in sorted(set(elements)):
        indices = [i for i, item in enumerate(elements) if item == element and atom_mask[i]]
        if not indices:
            continue
        xyz = coords[indices]
        radius = VDW_RADII.get(element, 0.8)
        ax.scatter(
            xyz[:, 0],
            xyz[:, 1],
            xyz[:, 2],
            s=(radius * atom_scale) ** 2,
            c=ELEMENT_COLORS.get(element, "#8c8c8c"),
            edgecolors="#1a1a1a",
            linewidths=0.35,
            depthshade=True,
            label=element,
        )

    for contact in contacts:
        if contact.i < 0 or contact.j < 0 or contact.i >= len(elements) or contact.j >= len(elements):
            raise IndexError(f"Contact index out of range: {contact.i + 1}:{contact.j + 1}")
        segment = coords[[contact.i, contact.j]]
        dist = float(np.linalg.norm(segment[0] - segment[1]))
        ax.plot(
            segment[:, 0],
            segment[:, 1],
            segment[:, 2],
            color=contact.color,
            lw=2.0,
            linestyle="--",
            alpha=0.95,
        )
        midpoint = segment.mean(axis=0)
        label = contact.label or f"{dist:.2f} A"
        ax.text(midpoint[0], midpoint[1], midpoint[2], label, color=contact.color, fontsize=9, ha="center")

    if show_indices:
        for i, (element, xyz) in enumerate(zip(elements, coords), start=1):
            if hide_h and element == "H":
                continue
            ax.text(xyz[0], xyz[1], xyz[2], str(i), fontsize=7, color="#111111")

    set_equal_axes(ax, coords[atom_mask])
    if legend and len(set(elements)) <= 10:
        handles = [
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                label=element,
                markerfacecolor=ELEMENT_COLORS.get(element, "#8c8c8c"),
                markeredgecolor="#1a1a1a",
                markersize=6,
            )
            for element in sorted(set(elements))
            if any(item == element and atom_mask[i] for i, item in enumerate(elements))
        ]
        ax.legend(handles=handles, loc="upper right", frameon=False, fontsize=8)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=0.94 if title else 1)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, transparent=True, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    if not no_crop and out_path.suffix.lower() == ".png":
        crop_transparent_png(out_path)


def validate_contacts(contacts: list[Contact], natoms: int) -> None:
    for contact in contacts:
        if contact.i < 0 or contact.j < 0 or contact.i >= natoms or contact.j >= natoms:
            raise IndexError(f"Contact index out of range: {contact.i + 1}:{contact.j + 1}")


def pymol_rgb_tuple(hex_color: str) -> tuple[float, float, float]:
    color = hex_color.strip()
    named = {
        "red": "#d62728",
        "blue": "#2f65d9",
        "green": "#2ca02c",
        "orange": "#e0702f",
        "yellow": "#e0b33f",
        "cyan": "#47b8a6",
        "gray": "#8c8c8c",
        "grey": "#8c8c8c",
        "black": "#000000",
        "white": "#ffffff",
    }
    color = named.get(color.lower(), color)
    if not color.startswith("#") or len(color) != 7:
        return (0.839, 0.153, 0.157)
    return tuple(int(color[i : i + 2], 16) / 255 for i in (1, 3, 5))


def build_pymol_script(
    input_path: Path,
    out_path: Path,
    structure: Structure,
    contacts: list[Contact],
    hide_h: bool,
    show_indices: bool,
    title: str | None,
    dpi: int,
    width: int,
    height: int,
    no_bonds: bool,
    sphere_scale: float,
    stick_radius: float,
    ray: bool,
    pymol_projection: str,
) -> str:
    validate_contacts(contacts, len(structure.elements))
    lines = [
        "reinitialize",
        "set suspend_updates, on",
        f"load {pymol_path(input_path)}, mol",
        "hide everything, mol",
        "set ray_opaque_background, off",
        "set antialias, 2",
        "set depth_cue, 0",
        "set spec_reflect, 0.25",
        "set ambient, 0.35",
        "set direct, 0.65",
        f"set orthoscopic, {'on' if pymol_projection == 'orthoscopic' else 'off'}",
        "bg_color white",
        f"set sphere_scale, {sphere_scale:.4f}, mol",
        f"set stick_radius, {stick_radius:.4f}, mol",
    ]
    selection = "mol and not hydro" if hide_h else "mol"
    if not no_bonds:
        lines.append(f"show sticks, {selection}")
    lines.append(f"show spheres, {selection}")

    color_names: dict[str, str] = {}
    for element, color in ELEMENT_COLORS.items():
        color_name = f"elem_{element.lower()}"
        color_names[element] = color_name
        rgb = pymol_rgb_tuple(color)
        lines.append(f"set_color {color_name}, [{rgb[0]:.4f}, {rgb[1]:.4f}, {rgb[2]:.4f}]")
        lines.append(f"color {color_name}, mol and elem {element}")

    lines.extend(
        [
            "set dash_gap, 0.25",
            "set dash_radius, 0.055",
            "set dash_round_ends, on",
            "set label_size, 11",
            "set label_color, black",
        ]
    )

    for n, contact in enumerate(contacts, start=1):
        color_name = f"contact_{n}"
        rgb = pymol_rgb_tuple(contact.color)
        lines.append(f"set_color {color_name}, [{rgb[0]:.4f}, {rgb[1]:.4f}, {rgb[2]:.4f}]")
        atom_a = f"(mol and index {contact.i + 1})"
        atom_b = f"(mol and index {contact.j + 1})"
        dist_name = f"dist_{n}"
        lines.append(f"distance {dist_name}, {atom_a}, {atom_b}")
        lines.append(f"color {color_name}, {dist_name}")
        lines.append(f"set dash_color, {color_name}, {dist_name}")
        if contact.label == "":
            lines.append(f"hide labels, {dist_name}")
        elif contact.label:
            lines.append(f"hide labels, {dist_name}")
            midpoint = (
                np.asarray(contact.label_position, dtype=float)
                if contact.label_position is not None
                else structure.coords[[contact.i, contact.j]].mean(axis=0)
            )
            pseudo = f"label_{n}"
            lines.append(f"pseudoatom {pseudo}, pos=[{midpoint[0]:.6f}, {midpoint[1]:.6f}, {midpoint[2]:.6f}]")
            lines.append(f"label {pseudo}, {quote_pymol(contact.label)}")
            lines.append(f"color {color_name}, {pseudo}")
            lines.append(f"set label_color, {color_name}, {pseudo}")
        else:
            lines.append(f"set label_color, {color_name}, {dist_name}")

    if show_indices:
        index_selection = "mol and not hydro" if hide_h else "mol"
        lines.append(f"label {index_selection}, index")

    lines.extend(
        [
            "orient mol",
            "zoom visible, 2.0",
            "set suspend_updates, off",
        ]
    )
    if title:
        lines.append(f"pseudoatom title_label, pos=[0, 0, 0]")
        lines.append(f"label title_label, {quote_pymol(title)}")
        lines.append("hide everything, title_label")
    ray_flag = "1" if ray else "0"
    lines.extend(
        [
            f"png {pymol_path(out_path)}, width={width}, height={height}, dpi={dpi}, ray={ray_flag}",
            "quit",
        ]
    )
    return "\n".join(lines) + "\n"


def draw_structure_pymol(
    input_path: Path,
    structure: Structure,
    out_path: Path,
    contacts: list[Contact],
    hide_h: bool,
    show_indices: bool,
    title: str | None,
    frame: int,
    dpi: int,
    width: int,
    height: int,
    no_bonds: bool,
    sphere_scale: float,
    stick_radius: float,
    ray: bool,
    pymol_projection: str,
    no_crop: bool,
    keep_script: Path | None,
) -> None:
    if not PYMOL_RUNNER.exists():
        raise FileNotFoundError(f"Could not find mamba runner at {PYMOL_RUNNER}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="pymol_render_", dir="/tmp") as tmpdir_raw:
        tmpdir = Path(tmpdir_raw)
        pymol_input = input_path
        if input_path.suffix.lower() == ".xyz":
            pymol_input = tmpdir / "frame.xyz"
            write_xyz_frame(pymol_input, structure, title=f"{input_path.name} frame {frame}")

        script = build_pymol_script(
            input_path=pymol_input,
            out_path=out_path,
            structure=structure,
            contacts=contacts,
            hide_h=hide_h,
            show_indices=show_indices,
            title=title,
            dpi=dpi,
            width=width,
            height=height,
            no_bonds=no_bonds,
            sphere_scale=sphere_scale,
            stick_radius=stick_radius,
            ray=ray,
            pymol_projection=pymol_projection,
        )
        script_path = keep_script or (tmpdir / "render.pml")
        script_path.parent.mkdir(parents=True, exist_ok=True)
        script_path.write_text(script)
        env = os.environ.copy()
        env.setdefault("PYMOL_GIT_MOD", "0")
        cmd = [str(PYMOL_RUNNER), "run", "-n", PYMOL_ENV, "pymol", "-cq", str(script_path)]
        result = subprocess.run(cmd, cwd=input_path.parent, text=True, capture_output=True, env=env, check=False)
        if result.returncode != 0:
            raise RuntimeError(
                "PyMOL rendering failed.\n"
                f"Command: {' '.join(cmd)}\n"
                f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
            )
    if not out_path.exists():
        raise FileNotFoundError(
            f"PyMOL did not create expected output {out_path}\n"
            f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )
    if not no_crop and out_path.suffix.lower() == ".png":
        crop_transparent_png(out_path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("structure", type=Path, help="Input .xyz or .pdb file")
    parser.add_argument("--backend", choices=["matplotlib", "pymol"], default="matplotlib", help="Rendering backend")
    parser.add_argument("--out", type=Path, default=None, help="Output image path; default is input name + .png")
    parser.add_argument("--frame", type=int, default=-1, help="XYZ frame index, 0-based; negative values count from end")
    parser.add_argument("--contact", action="append", type=parse_contact, default=[], help="Distance/contact line: i:j[:color[:label]]")
    parser.add_argument("--hide-h", action="store_true", help="Hide hydrogen atoms")
    parser.add_argument("--indices", action="store_true", help="Draw 1-based atom indices")
    parser.add_argument("--title", default=None, help="Optional plot title")
    parser.add_argument("--elev", type=float, default=18.0, help="Camera elevation")
    parser.add_argument("--azim", type=float, default=-62.0, help="Camera azimuth")
    parser.add_argument("--dpi", type=int, default=450, help="Output DPI")
    parser.add_argument("--bond-scale", type=float, default=1.25, help="Covalent-radius scale for inferred bonds")
    parser.add_argument("--atom-scale", type=float, default=20.0, help="Atom marker scale")
    parser.add_argument("--no-bonds", action="store_true", help="Do not draw inferred bonds")
    parser.add_argument("--legend", action="store_true", help="Draw a compact element legend")
    parser.add_argument("--no-crop", action="store_true", help="Keep Matplotlib's original output bounds")
    parser.add_argument("--width", type=int, default=1800, help="PyMOL output width in pixels")
    parser.add_argument("--height", type=int, default=1400, help="PyMOL output height in pixels")
    parser.add_argument("--pymol-sphere-scale", type=float, default=0.25, help="PyMOL sphere scale")
    parser.add_argument("--pymol-stick-radius", type=float, default=0.13, help="PyMOL stick radius")
    parser.add_argument(
        "--pymol-projection",
        choices=["orthoscopic", "perspective"],
        default="orthoscopic",
        help="PyMOL camera projection; orthoscopic is PyMOL's orthographic mode",
    )
    parser.add_argument("--no-ray", action="store_true", help="Disable PyMOL ray tracing")
    parser.add_argument("--keep-pymol-script", type=Path, default=None, help="Write the generated PyMOL script to this path")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    structure_path = args.structure.resolve()
    out_path = (args.out or args.structure.with_suffix(".png")).resolve()
    keep_script = args.keep_pymol_script.resolve() if args.keep_pymol_script else None
    structure = read_structure(structure_path, args.frame)
    if args.backend == "pymol":
        draw_structure_pymol(
            input_path=structure_path,
            structure=structure,
            out_path=out_path,
            contacts=args.contact,
            hide_h=args.hide_h,
            show_indices=args.indices,
            title=args.title,
            frame=args.frame,
            dpi=args.dpi,
            width=args.width,
            height=args.height,
            no_bonds=args.no_bonds,
            sphere_scale=args.pymol_sphere_scale,
            stick_radius=args.pymol_stick_radius,
            ray=not args.no_ray,
            pymol_projection=args.pymol_projection,
            no_crop=args.no_crop,
            keep_script=keep_script,
        )
    else:
        draw_structure(
            structure=structure,
            out_path=out_path,
            contacts=args.contact,
            hide_h=args.hide_h,
            show_indices=args.indices,
            title=args.title,
            elev=args.elev,
            azim=args.azim,
            dpi=args.dpi,
            bond_scale=args.bond_scale,
            atom_scale=args.atom_scale,
            no_bonds=args.no_bonds,
            legend=args.legend,
            no_crop=args.no_crop,
        )
    print(f"Rendered {len(structure.elements)} atoms from {args.structure}")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
