from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from math import dist
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps

import figure14_render_core as fig14


ROOT = fig14.ROOT
FINAL_DIR = fig14.FINAL_DIR
THIS_DIR = Path(__file__).resolve().parent

CURRENT_FINAL = FINAL_DIR / "Figure_14_MD_structural_snapshots_strip_manual_labels.png"
BACKUP_FINAL = FINAL_DIR / "_figure_backups" / "Figure_14_MD_structural_snapshots_strip_manual_labels.before_panel_c.png"
PANEL_C_PDB = THIS_DIR / "figure14_panel_c_hydrated_binding_site.pdb"
PANEL_C_BONDS = THIS_DIR / "figure14_panel_c_hydrated_binding_site_bonds.json"
PANEL_C_RAW = THIS_DIR / "figure14_panel_c_hydrated_binding_site_raw.png"
PANEL_C_FINAL = FINAL_DIR / "_figure_backups" / "Figure_14_panel_C_hydrated_binding_site.png"
ASSEMBLED_BASE = FINAL_DIR / "_figure_backups" / "Figure_14_MD_structural_snapshots_strip_panel_c_base.png"


def load_font(size: int, bold: bool = True) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def pdb_xyz(line: str) -> tuple[float, float, float]:
    return float(line[30:38]), float(line[38:46]), float(line[46:54])


def write_hydrated_binding_site_pdb(
    source: Path,
    dest: Path,
    prmtop: Path,
    *,
    water_cutoff: float = 5.0,
    max_waters: int = 650,
    chloride_cutoff: float = 8.0,
) -> None:
    """Write R48, PFO, nearby waters, and nearby chlorides with PRMTOP CONECT records."""
    other_lines, atoms = fig14.parse_pdb_atom_records(source)
    atom_meta = []
    pfo_atoms = []
    resin_pfo_atoms = []
    waters: list[list[tuple[int, str]]] = []
    cl_atoms = []
    for ordinal, line in atoms:
        resn = line[17:21].strip()
        chain = line[21:22]
        resid = line[22:26].strip()
        icode = line[26:27]
        atom_meta.append((ordinal, line, resn, chain, resid, icode))
        if resn == "PFO":
            pfo_atoms.append((ordinal, line))
            resin_pfo_atoms.append((ordinal, line))
        elif resn == "R48":
            resin_pfo_atoms.append((ordinal, line))
        elif resn == "WAT":
            if (
                not waters
                or atoms[ordinal - 2][1][17:21].strip() != "WAT"
                or atoms[ordinal - 2][1][21:27] != line[21:27]
            ):
                waters.append([])
            waters[-1].append((ordinal, line))
        elif resn == "Cl-":
            cl_atoms.append((ordinal, line))

    if not pfo_atoms:
        raise ValueError(f"No PFO residue found in {source}")

    pfo_xyz = [pdb_xyz(line) for _, line in pfo_atoms]
    resin_pfo_xyz = [pdb_xyz(line) for _, line in resin_pfo_atoms]
    selected_ordinals: set[int] = set()
    for ordinal, line, resn, *_ in atom_meta:
        if resn in {"R48", "PFO"}:
            selected_ordinals.add(ordinal)

    selected_water_residues: list[list[tuple[int, str]]] = []
    for residue_atoms in waters:
        oxygen_lines = [line for _, line in residue_atoms if (line[76:78].strip() or line[12:16].strip()[0]).upper().startswith("O")]
        test_lines = oxygen_lines or [line for _, line in residue_atoms]
        min_distance = min(dist(pdb_xyz(line), xyz) for line in test_lines for xyz in resin_pfo_xyz)
        if min_distance <= water_cutoff:
            selected_water_residues.append(residue_atoms)

    if len(selected_water_residues) > max_waters:
        near_pfo: list[list[tuple[int, str]]] = []
        remaining: list[list[tuple[int, str]]] = []
        for residue_atoms in selected_water_residues:
            oxygen_lines = [line for _, line in residue_atoms if (line[76:78].strip() or line[12:16].strip()[0]).upper().startswith("O")]
            test_lines = oxygen_lines or [line for _, line in residue_atoms]
            min_pfo_distance = min(dist(pdb_xyz(line), xyz) for line in test_lines for xyz in pfo_xyz)
            if min_pfo_distance <= water_cutoff:
                near_pfo.append(residue_atoms)
            else:
                remaining.append(residue_atoms)

        def water_center(residue_atoms: list[tuple[int, str]]) -> tuple[float, float, float]:
            oxygen = [line for _, line in residue_atoms if (line[76:78].strip() or line[12:16].strip()[0]).upper().startswith("O")]
            coords = [pdb_xyz(line) for line in (oxygen or [line for _, line in residue_atoms])]
            return tuple(sum(coord[i] for coord in coords) / len(coords) for i in range(3))

        chosen = list(near_pfo[:max_waters])
        if len(chosen) < max_waters and remaining:
            remaining_centers = [water_center(residue_atoms) for residue_atoms in remaining]
            chosen_centers = [water_center(residue_atoms) for residue_atoms in chosen]
            if not chosen_centers:
                first = min(range(len(remaining_centers)), key=lambda i: remaining_centers[i][0])
                chosen.append(remaining.pop(first))
                chosen_centers.append(remaining_centers.pop(first))
            while len(chosen) < max_waters and remaining:
                best_i = max(
                    range(len(remaining_centers)),
                    key=lambda i: min(dist(remaining_centers[i], center) for center in chosen_centers),
                )
                chosen.append(remaining.pop(best_i))
                chosen_centers.append(remaining_centers.pop(best_i))
        selected_water_residues = chosen

    for residue_atoms in selected_water_residues:
        selected_ordinals.update(ordinal for ordinal, _ in residue_atoms)

    for ordinal, line in cl_atoms:
        if min(dist(pdb_xyz(line), xyz) for xyz in pfo_xyz) <= chloride_cutoff:
            selected_ordinals.add(ordinal)

    selected = []
    ordinal_to_serial: dict[int, int] = {}
    serial_atom_info: dict[int, tuple[str, str]] = {}
    for ordinal, line in atoms:
        if ordinal not in selected_ordinals:
            continue
        serial = len(selected) + 1
        if serial > 99999:
            raise ValueError("Selected atom count exceeds PDB serial limit.")
        selected.append((ordinal, serial, fig14.renumber_pdb_atom_line(line, serial)))
        ordinal_to_serial[ordinal] = serial
        elem = (line[76:78].strip() or line[12:16].strip()[0]).upper()
        serial_atom_info[serial] = (line[17:21].strip(), elem)

    adjacency: dict[int, set[int]] = {serial: set() for _, serial, _ in selected}
    explicit_bonds: list[tuple[int, int]] = []
    for ai, aj in fig14.read_prmtop_bonds(prmtop):
        if ai in ordinal_to_serial and aj in ordinal_to_serial:
            si = ordinal_to_serial[ai]
            sj = ordinal_to_serial[aj]
            res_i, elem_i = serial_atom_info[si]
            res_j, elem_j = serial_atom_info[sj]
            if res_i == "WAT" and res_j == "WAT" and elem_i.startswith("H") and elem_j.startswith("H"):
                continue
            adjacency[si].add(sj)
            adjacency[sj].add(si)
            explicit_bonds.append((si, sj))

    lines = [line for line in other_lines if not line.startswith("CRYST1")]
    lines.extend(line for _, _, line in selected)
    lines.append("TER")
    for serial in sorted(adjacency):
        bonded = sorted(adjacency[serial])
        for start in range(0, len(bonded), 4):
            chunk = bonded[start : start + 4]
            if chunk:
                lines.append("CONECT" + f"{serial:5d}" + "".join(f"{j:5d}" for j in chunk))
    lines.append("END")
    dest.write_text("\n".join(lines) + "\n")
    PANEL_C_BONDS.write_text(str(sorted(explicit_bonds)) + "\n")


def render_panel_c() -> None:
    write_hydrated_binding_site_pdb(
        fig14.WHOLE_SYSTEM_PDB,
        PANEL_C_PDB,
        fig14.WHOLE_SYSTEM_PRMTOP,
    )
    with tempfile.TemporaryDirectory(prefix="fig14_panel_c_", dir="/tmp") as td:
        pml = Path(td) / "panel_c.pml"
        pml.write_text(
            f"""
reinitialize
set connect_mode, 0
load {fig14.render.pymol_path(PANEL_C_PDB)}, mol
python
from pymol import cmd
import ast
with open(r"{PANEL_C_BONDS.as_posix()}") as fh:
    panel_c_bonds = ast.literal_eval(fh.read())
cmd.unbond("mol", "mol")
for ai, aj in panel_c_bonds:
    cmd.bond(f"mol and id {{ai}}", f"mol and id {{aj}}")
python end
hide everything, mol
bg_color white
set ray_opaque_background, off
set orthoscopic, off
set field_of_view, 20
set antialias, 2
set depth_cue, 0
set ray_shadow, on
set ray_trace_mode, 1
set ray_trace_color, black
set ambient, 0.23
set direct, 0.84
set spec_reflect, 0.30
set stick_quality, 24
set sphere_quality, 2
set stick_radius, 0.105
set sphere_scale, 0.18

select resin_sel, mol and resn R48 and not hydro
select pfo_sel, mol and resn PFO and not hydro
select water_sel, mol and resn WAT
select cl_sel, mol and resn Cl- and not hydro

set_color resin_c, [0.620, 0.620, 0.620]
set_color elem_c, [0.302, 0.302, 0.302]
set_color elem_h, [1.000, 1.000, 1.000]
set_color elem_n, [0.184, 0.396, 0.851]
set_color elem_o, [0.839, 0.153, 0.157]
set_color elem_f, [0.278, 0.722, 0.651]
set_color elem_cl, [0.173, 0.627, 0.173]

show sticks, resin_sel
show spheres, resin_sel
set sphere_scale, 0.145, resin_sel
set stick_radius, 0.090, resin_sel
set stick_transparency, 0.12, resin_sel
set sphere_transparency, 0.12, resin_sel
color resin_c, resin_sel and elem C
color elem_n, resin_sel and elem N
color elem_o, resin_sel and elem O

show sticks, pfo_sel
show spheres, pfo_sel
set sphere_scale, 0.27, pfo_sel
set stick_radius, 0.145, pfo_sel
color elem_c, pfo_sel and elem C
color elem_o, pfo_sel and elem O
color elem_f, pfo_sel and elem F

show sticks, water_sel
show spheres, water_sel
set sphere_scale, 0.080, water_sel
set sphere_scale, 0.055, water_sel and elem H
set stick_radius, 0.060, water_sel
set stick_transparency, 0.38, water_sel
set sphere_transparency, 0.52, water_sel
color elem_o, water_sel and elem O
color elem_h, water_sel and elem H

show spheres, cl_sel
set sphere_scale, 0.34, cl_sel
color elem_cl, cl_sel

orient (resin_sel or pfo_sel)
turn y, 35
turn x, -25
zoom (resin_sel or pfo_sel or water_sel or cl_sel), 7
png {fig14.render.pymol_path(PANEL_C_RAW)}, width=3600, height=3000, dpi=650, ray=1
quit
"""
        )
        cmd = [str(fig14.render.PYMOL_RUNNER), "run", "-n", fig14.render.PYMOL_ENV, "pymol", "-cq", str(pml)]
        env = os.environ.copy()
        env.setdefault("PYMOL_GIT_MOD", "0")
        res = subprocess.run(cmd, cwd=fig14.WHOLE_SYSTEM_PDB.parent, text=True, capture_output=True, env=env, check=False)
        if res.returncode != 0:
            raise RuntimeError(f"PyMOL panel C rendering failed.\nSTDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}")

    raw = Image.open(PANEL_C_RAW).convert("RGB")
    crop = fig14.white_border_crop_box(raw, padding=80)
    panel = raw.crop(crop)
    PANEL_C_FINAL.parent.mkdir(exist_ok=True)
    panel.save(PANEL_C_FINAL)



def render_all_raws() -> None:
    # Panel A and panel B raw renders are generated by the existing MD rendering module.
    fig14.make_snapshot_strip()
    fig14.make_whole_system_figure()
    render_panel_c()


if __name__ == "__main__":
    render_all_raws()
    print("Figure 14 raw panels generated.")
    print(f"Panel A raw: {fig14.FINAL_STRIP}")
    print(f"Panel B raw: {fig14.FINAL_WHOLE}")
    print(f"Panel C raw: {PANEL_C_FINAL}")
