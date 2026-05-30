from __future__ import annotations

import os
import json
import shutil
import subprocess
import tempfile
from pathlib import Path
from math import dist

import matplotlib.pyplot as plt
from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageFont, ImageOps

import sys


def find_project_root(start: Path) -> Path:
    for candidate in (start.parent, *start.parents):
        if (candidate / "work_amber").exists() and (candidate / "final_plots").exists():
            return candidate
    return start.parents[2]


ROOT = Path(__file__).resolve().parent
TOOLS_DIR = ROOT.parent / "shared_helpers"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))
import render_structure_3d as render  # noqa: E402


THIS_DIR = Path(__file__).resolve().parent
FINAL_DIR = THIS_DIR / "outputs"
METRICS_DIR = THIS_DIR / "input_data"
VMD_DIR = THIS_DIR / "input_data"
GBSA_DIR = THIS_DIR / "input_data"

SNAPSHOT_PDBS = [
    METRICS_DIR / "pfoa_assoc_frame_001_resin_pfo_cl.pdb",
    METRICS_DIR / "pfoa_assoc_frame_025_resin_pfo_cl.pdb",
    METRICS_DIR / "pfoa_assoc_frame_050_resin_pfo_cl.pdb",
    METRICS_DIR / "pfoa_assoc_frame_100_resin_pfo_cl.pdb",
]
SNAPSHOT_LABELS = ["Frame 1", "Frame 25", "Frame 50", "Frame 100"]
PANEL_SCALE_FACTORS = [1.18, 1.16, 1.20, 1.00]
WHOLE_SYSTEM_PDB = VMD_DIR / "resin_pfoa_exchange_npt_1ns_final_imaged.pdb"
SNAPSHOT_PRMTOP = GBSA_DIR / "r48_pfoa_47cl.prmtop"
WHOLE_SYSTEM_PRMTOP = VMD_DIR / "solvated_resin_pfoa_exchange.prmtop"

OUT_STRIP = FINAL_DIR / "figure_MD_structural_snapshots_strip.png"
OUT_WHOLE = FINAL_DIR / "figure_MD_whole_system_water_transparent.png"
FINAL_STRIP = FINAL_DIR / OUT_STRIP.name
FINAL_WHOLE = FINAL_DIR / OUT_WHOLE.name
FINAL_STRIP_NUMBERED = FINAL_DIR / "Figure_14_MD_structural_snapshots_strip.png"
FINAL_WHOLE_NUMBERED = FINAL_DIR / "Figure_14_MD_whole_system_water_transparent.png"
FINAL_LABEL_ANCHORS_JSON = FINAL_DIR / "Figure_14_label_anchors.json"
FINAL_LABEL_ANCHORS_JS = FINAL_DIR / "Figure_14_label_anchors.js"
FINAL_MANUAL_LABEL_LAYOUT = FINAL_DIR / "Figure_14_manual_label_layout.json"
SHOW_STRUCTURE_LABELS = os.environ.get("MD_FIG14_STRUCTURE_LABELS", "0").lower() not in {"0", "false", "no"}
HIDE_LABELS_PML = """
python
from pymol import cmd
for obj in cmd.get_names("objects"):
    if obj.startswith("lbl_") or obj.startswith("lead_") or obj.startswith("anchor_"):
        cmd.disable(obj)
cmd.hide("labels", "all")
python end
"""


def require_inputs(paths: list[Path]) -> None:
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing required inputs:\n" + "\n".join(f"  - {m}" for m in missing))


def read_prmtop_bonds(prmtop: Path) -> set[tuple[int, int]]:
    """Return 1-based atom-index bonds from an Amber prmtop."""
    flags: dict[str, list[str]] = {}
    lines = prmtop.read_text(errors="replace").splitlines()
    i = 0
    while i < len(lines):
        if not lines[i].startswith("%FLAG"):
            i += 1
            continue
        flag = lines[i].split()[1]
        i += 1
        if i < len(lines) and lines[i].startswith("%FORMAT"):
            i += 1
        values: list[str] = []
        while i < len(lines) and not lines[i].startswith("%FLAG"):
            values.extend(lines[i].split())
            i += 1
        flags[flag] = values

    bonds: set[tuple[int, int]] = set()
    for flag in ("BONDS_INC_HYDROGEN", "BONDS_WITHOUT_HYDROGEN"):
        values = list(map(int, flags.get(flag, [])))
        for start in range(0, len(values), 3):
            if start + 1 >= len(values):
                continue
            ai = values[start] // 3 + 1
            aj = values[start + 1] // 3 + 1
            if ai != aj:
                bonds.add(tuple(sorted((ai, aj))))
    return bonds


def parse_pdb_atom_records(source: Path) -> tuple[list[str], list[tuple[int, str]]]:
    other_lines: list[str] = []
    atoms: list[tuple[int, str]] = []
    atom_ordinal = 0
    for line in source.read_text().splitlines():
        if line.startswith(("ATOM", "HETATM")):
            atom_ordinal += 1
            atoms.append((atom_ordinal, line))
        elif not line.startswith(("CONECT", "END", "TER")):
            other_lines.append(line)
    return other_lines, atoms


def renumber_pdb_atom_line(line: str, serial: int) -> str:
    return f"{line[:6]}{serial:5d}{line[11:]}"


def write_topology_bonded_pdb(
    source: Path,
    dest: Path,
    prmtop: Path,
    *,
    include_resnames: set[str] | None = None,
) -> None:
    """Write a render-only PDB with CONECT records from the Amber topology."""
    other_lines, atoms = parse_pdb_atom_records(source)
    selected: list[tuple[int, int, str]] = []
    ordinal_to_serial: dict[int, int] = {}
    for ordinal, line in atoms:
        resn = line[17:21].strip()
        if include_resnames is not None and resn not in include_resnames:
            continue
        serial = len(selected) + 1
        if serial > 99999:
            raise ValueError(f"Too many selected atoms for PDB serials in {dest}")
        selected.append((ordinal, serial, renumber_pdb_atom_line(line, serial)))
        ordinal_to_serial[ordinal] = serial

    adjacency: dict[int, set[int]] = {serial: set() for _, serial, _ in selected}
    for ai, aj in read_prmtop_bonds(prmtop):
        if ai in ordinal_to_serial and aj in ordinal_to_serial:
            si = ordinal_to_serial[ai]
            sj = ordinal_to_serial[aj]
            adjacency[si].add(sj)
            adjacency[sj].add(si)

    lines = other_lines + [line for _, _, line in selected]
    lines.append("TER")
    for serial in sorted(adjacency):
        bonded = sorted(adjacency[serial])
        for start in range(0, len(bonded), 4):
            chunk = bonded[start : start + 4]
            if chunk:
                lines.append("CONECT" + f"{serial:5d}" + "".join(f"{j:5d}" for j in chunk))
    lines.append("END")
    dest.write_text("\n".join(lines) + "\n")


def pdb_coordinate_bounds(source: Path, *, pad: float = 1.5) -> tuple[float, float, float, float, float, float]:
    xs: list[float] = []
    ys: list[float] = []
    zs: list[float] = []
    for line in source.read_text().splitlines():
        if line.startswith(("ATOM", "HETATM")):
            xs.append(float(line[30:38]))
            ys.append(float(line[38:46]))
            zs.append(float(line[46:54]))
    return min(xs) - pad, max(xs) + pad, min(ys) - pad, max(ys) + pad, min(zs) - pad, max(zs) + pad


def render_snapshot_panels(paths: list[Path]) -> list[dict]:
    out_paths: list[dict] = []
    for panel_index, pdb in enumerate(paths):
        out_png = pdb.with_suffix(".snapshot.png")
        out_markers_json = pdb.with_suffix(".snapshot_markers.json")
        marker_dir = pdb.parent / f"_{pdb.stem}_snapshot_marker_layers"
        if marker_dir.exists():
            shutil.rmtree(marker_dir)
        marker_dir.mkdir()
        marker_offset = panel_index * 50
        with tempfile.TemporaryDirectory(prefix="md_bind_", dir="/tmp") as td:
            bonded_pdb = Path(td) / pdb.name
            write_topology_bonded_pdb(pdb, bonded_pdb, SNAPSHOT_PRMTOP)
            pml = Path(td) / "snap.pml"
            script = f"""
reinitialize
set suspend_updates, on
load {render.pymol_path(bonded_pdb)}, mol
hide everything, mol
bg_color white
set ray_opaque_background, off
set orthoscopic, off
set field_of_view, 24
set antialias, 2
set depth_cue, 0
set ray_shadow, on
set ray_trace_mode, 1
set ray_trace_color, black
set sphere_scale, 0.20
set stick_radius, 0.12
set spec_reflect, 0.25
set ambient, 0.23
set direct, 0.82

select resin, mol and resn R48 and not hydro
select pfas, mol and resn PFO and not hydro
select clions, mol and resn Cl- and not hydro
select clnear, (clions within 6.0 of pfas) and not hydro

set_color elem_h, [0.949, 0.949, 0.949]
set_color elem_c, [0.302, 0.302, 0.302]
set_color resin_c, [0.560, 0.560, 0.560]
set_color elem_n, [0.184, 0.396, 0.851]
set_color elem_o, [0.839, 0.153, 0.157]
set_color elem_f, [0.278, 0.722, 0.651]
set_color elem_cl, [0.173, 0.627, 0.173]

python
from pymol import cmd
import colorsys
import json
import math

pfas_xyz = [(a.coord[0], a.coord[1], a.coord[2]) for a in cmd.get_model("pfas").atom]
q_idx = []
for at in cmd.get_model("resin and elem N").atom:
    c_nbr = cmd.count_atoms(f"(neighbor index {{at.index}}) and elem C")
    if c_nbr == 4:
        nearest = min(
            math.dist((at.coord[0], at.coord[1], at.coord[2]), xyz)
            for xyz in pfas_xyz
        )
        q_idx.append((nearest, at.index))
q_sorted = sorted(q_idx)
q_all = [idx for _, idx in q_sorted]
q_near = [idx for _, idx in q_sorted[:3]]
if q_all:
    cmd.select("quatN_all", "index " + "+".join(str(i) for i in q_all))
else:
    cmd.select("quatN_all", "none")
if q_near:
    cmd.select("quatN_near", "index " + "+".join(str(i) for i in q_near))
else:
    cmd.select("quatN_near", "none")
python end

select shell_seed, ((resin within 5.5 of pfas) or quatN_near) and not hydro
select shell_net, shell_seed
select shell_net, shell_net or (neighbor shell_net)
select shell_net, shell_net or (neighbor shell_net)
select shell_net, shell_net or (neighbor shell_net)
select shell_net, shell_net or (neighbor shell_net)
select shell_net, shell_net or (neighbor shell_net)
select shell_net, shell_net or (neighbor shell_net)
select shell_net, shell_net or (neighbor shell_net)
select shell_net, shell_net or (neighbor shell_net)
select shell_net, shell_net and resin and not hydro
select quatN_in_net, quatN_all and shell_net
select quatN_substituents, (neighbor quatN_in_net) and resin and not hydro
select shell_net, shell_net or quatN_substituents
select bindview, (pfas or shell_net or clnear) and not hydro

show sticks, bindview
show spheres, bindview
set sphere_scale, 0.20, bindview
set stick_radius, 0.085, shell_net
set stick_radius, 0.125, pfas
set stick_transparency, 0.18, shell_net
set sphere_transparency, 0.18, shell_net
set sphere_scale, 0.17, shell_net
set sphere_scale, 0.22, pfas
set sphere_scale, 0.28, clnear

color resin_c, shell_net and elem C
color elem_c, pfas and elem C
color elem_n, bindview and elem N
color elem_o, bindview and elem O
color elem_f, bindview and elem F
color elem_cl, bindview and elem Cl

python
from pymol import cmd
cmd.select("quatN_view", "quatN_all and shell_net")
cmd.show("spheres", "quatN_view")
cmd.set("sphere_scale", 0.34, "quatN_view")
cmd.set("label_size", 11)
cmd.set("label_font_id", 7)
cmd.set("label_color", "black")
cmd.set("label_outline_color", "white")
cmd.set("dash_width", 1.3)
cmd.set("dash_gap", 0.30)
cmd.set("dash_radius", 0.018)
cmd.set("dash_color", "black")
bind_atoms = cmd.get_model("bindview").atom
cx = sum(a.coord[0] for a in bind_atoms) / len(bind_atoms)
cy = sum(a.coord[1] for a in bind_atoms) / len(bind_atoms)
cz = sum(a.coord[2] for a in bind_atoms) / len(bind_atoms)
bind_xy = [(a.coord[0], a.coord[1]) for a in bind_atoms]
placed_xy = []
label_objects = []
anchor_records = []

def add_anchor_marker(anchor_coord, label_text):
    marker_i = {marker_offset} + len(anchor_records)
    hue = (marker_i * 0.61803398875) % 1.0
    r, g, b = colorsys.hsv_to_rgb(hue, 0.82, 1.0)
    color_name = f"mkr_color_{{marker_i:03d}}"
    obj = f"mkr_anchor_{{marker_i:03d}}"
    cmd.set_color(color_name, [r, g, b])
    cmd.pseudoatom(obj, pos=anchor_coord)
    cmd.color(color_name, obj)
    cmd.hide("everything", obj)
    anchor_records.append({{
        "id": f"{{label_text}}_{{marker_i:03d}}",
        "text": label_text,
        "marker_index": marker_i,
        "color": [round(r * 255), round(g * 255), round(b * 255)],
    }})

def add_leader_label_at(anchor_coord, label_text, object_key, ordinal=0, source_selection=None):
    global placed_xy
    directions = [
        (1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0),
        (0.72, 0.72), (-0.72, 0.72), (0.72, -0.72), (-0.72, -0.72),
        (1.0, 0.45), (-1.0, 0.45), (1.0, -0.45), (-1.0, -0.45),
    ]
    radii = [6.0, 7.5, 9.0]
    candidates = []
    for radius in radii:
        for dx, dy in directions:
            x = anchor_coord[0] + radius * dx
            y = anchor_coord[1] + radius * dy
            z = anchor_coord[2] + 1.5 + 0.25 * (ordinal % 3)
            atom_clearance = min(((x - ax) ** 2 + (y - ay) ** 2) ** 0.5 for ax, ay in bind_xy)
            label_clearance = min(
                [((x - lx) ** 2 + (y - ly) ** 2) ** 0.5 for lx, ly in placed_xy] or [99.0]
            )
            outward = ((x - cx) ** 2 + (y - cy) ** 2) ** 0.5
            score = 3.0 * atom_clearance + 1.8 * label_clearance + 0.18 * outward - 0.20 * radius
            candidates.append((score, [x, y, z]))
    pos = max(candidates, key=lambda item: item[0])[1]
    placed_xy.append((pos[0], pos[1]))
    add_anchor_marker(anchor_coord, label_text)
    obj = f"lbl_{{object_key}}"
    label_objects.append(obj)
    cmd.pseudoatom(obj, pos=pos, label=label_text)
    cmd.hide("spheres", obj)
    if source_selection is None:
        anchor_obj = f"anchor_{{object_key}}"
        cmd.pseudoatom(anchor_obj, pos=anchor_coord)
        cmd.hide("spheres", anchor_obj)
        source_selection = anchor_obj
    cmd.distance(f"lead_{{object_key}}", source_selection, obj)
    cmd.hide("labels", f"lead_{{object_key}}")

def add_leader_label(at, prefix, ordinal=0):
    add_leader_label_at(at.coord, f"{{prefix}}{{at.id}}", f"{{prefix}}_{{at.id}}", ordinal, f"index {{at.index}}")

for j, at in enumerate(cmd.get_model("quatN_view").atom):
    add_leader_label(at, "N", j)
for j, at in enumerate(cmd.get_model("clnear").atom):
    add_leader_label(at, "Cl", j)
pfas_atoms = cmd.get_model("pfas").atom
resin_atoms = cmd.get_model("shell_net").atom
if pfas_atoms:
    px = sum(a.coord[0] for a in pfas_atoms) / len(pfas_atoms)
    py = sum(a.coord[1] for a in pfas_atoms) / len(pfas_atoms)
    pz = sum(a.coord[2] for a in pfas_atoms) / len(pfas_atoms)
    add_leader_label_at([px, py, pz], "PFOA", "PFOA", len(placed_xy))
if resin_atoms:
    rx = sum(a.coord[0] for a in resin_atoms) / len(resin_atoms)
    ry = sum(a.coord[1] for a in resin_atoms) / len(resin_atoms)
    rz = sum(a.coord[2] for a in resin_atoms) / len(resin_atoms)
    add_leader_label_at([rx, ry, rz], "R48", "R48", len(placed_xy))
if label_objects:
    cmd.select("annotation_view", "bindview or " + " or ".join(label_objects))
else:
    cmd.select("annotation_view", "bindview")
with open(r"{out_markers_json.as_posix()}", "w") as fh:
    json.dump(anchor_records, fh, indent=2)
python end

orient annotation_view
turn y, 20
turn x, -18
zoom annotation_view, 3.5
clip slab, 32
{"" if SHOW_STRUCTURE_LABELS else HIDE_LABELS_PML}
set suspend_updates, off
png {render.pymol_path(out_png)}, width=3200, height=2400, dpi=900, ray=1
python
from pymol import cmd
cmd.bg_color("black")
cmd.set("ray_opaque_background", 1)
cmd.set("ambient", 1.0)
cmd.set("direct", 0.0)
cmd.set("spec_reflect", 0.0)
for rec in anchor_records:
    name = "mkr_anchor_%03d" % rec["marker_index"]
    cmd.hide("everything", "all")
    cmd.hide("labels", "all")
    cmd.enable(name)
    cmd.show("spheres", name)
    cmd.set("sphere_scale", 0.70, name)
    cmd.png(r"{marker_dir.as_posix()}/marker_%03d.png" % rec["marker_index"], width=3200, height=2400, dpi=900, ray=1)
python end
quit
"""
            pml.write_text(script)
            cmd = [str(render.PYMOL_RUNNER), "run", "-n", render.PYMOL_ENV, "pymol", "-cq", str(pml)]
            env = os.environ.copy()
            env.setdefault("PYMOL_GIT_MOD", "0")
            res = subprocess.run(cmd, cwd=pdb.parent, text=True, capture_output=True, env=env, check=False)
            if res.returncode != 0:
                raise RuntimeError(f"PyMOL snapshot rendering failed for {pdb.name}.\nSTDERR:\n{res.stderr}")
        out_paths.append({"image": out_png, "marker_dir": marker_dir, "records": out_markers_json})
    return out_paths


def ammonium_site_labels(pdb: Path) -> list[str]:
    radii = {
        "C": 0.76,
        "N": 0.71,
        "O": 0.66,
        "F": 0.57,
        "S": 1.05,
        "P": 1.07,
        "CL": 1.02,
    }
    atoms: list[tuple[int, str, str, str, tuple[float, float, float]]] = []
    for line in pdb.read_text().splitlines():
        if line.startswith(("ATOM", "HETATM")):
            idx = int(line[6:11])
            name = line[12:16].strip()
            resn = line[17:21].strip()
            elem = line[76:78].strip().upper() or name[0].upper()
            xyz = (float(line[30:38]), float(line[38:46]), float(line[46:54]))
            atoms.append((idx, name, resn, elem, xyz))
    pfas = [(idx, xyz) for idx, _, resn, elem, xyz in atoms if resn == "PFO" and elem != "H"]
    resin = [(idx, elem, xyz) for idx, _, resn, elem, xyz in atoms if resn == "R48" and elem != "H"]
    if not pfas:
        return []

    atom_by_idx = {idx: (elem, xyz) for idx, _, _, elem, xyz in atoms}
    bonds: dict[int, set[int]] = {idx: set() for idx, _, _, _, _ in atoms}
    heavy = [(idx, resn, elem, xyz) for idx, _, resn, elem, xyz in atoms if elem != "H" and resn != "Cl-"]
    for i, (idx_i, _, elem_i, xyz_i) in enumerate(heavy):
        r_i = radii.get(elem_i, 0.77)
        for idx_j, _, elem_j, xyz_j in heavy[i + 1 :]:
            cutoff = r_i + radii.get(elem_j, 0.77) + 0.45
            if 0.4 < dist(xyz_i, xyz_j) <= cutoff:
                bonds[idx_i].add(idx_j)
                bonds[idx_j].add(idx_i)

    quat_n = []
    for idx, elem, xyz in resin:
        if elem != "N":
            continue
        carbon_neighbors = sum(1 for nbr in bonds[idx] if atom_by_idx[nbr][0] == "C")
        if carbon_neighbors == 4:
            nearest = min(dist(xyz, p_xyz) for _, p_xyz in pfas)
            quat_n.append((nearest, idx))
    q_near = {idx for _, idx in sorted(quat_n)[:3]}
    resin_ids = {idx for idx, _, _ in resin}
    shell = {
        idx
        for idx, elem, xyz in resin
        if idx in q_near or min(dist(xyz, p_xyz) for _, p_xyz in pfas) <= 5.5
    }
    for _ in range(8):
        shell |= {nbr for idx in list(shell) for nbr in bonds[idx]}
    shell &= resin_ids

    visible_q = [(nearest, idx) for nearest, idx in quat_n if idx in shell]
    return [f"N{idx}" for _, idx in sorted(visible_q)]


def white_border_crop_box(image: Image.Image, padding: int = 80) -> tuple[int, int, int, int]:
    bg = Image.new(image.mode, image.size, "white")
    diff = ImageChops.difference(image, bg).convert("L")
    bbox = diff.point(lambda px: 255 if px > 12 else 0).getbbox()
    if bbox is None:
        return (0, 0, image.width, image.height)
    left = max(bbox[0] - padding, 0)
    upper = max(bbox[1] - padding, 0)
    right = min(bbox[2] + padding, image.width)
    lower = min(bbox[3] + padding, image.height)
    return (left, upper, right, lower)


def crop_white_border(image: Image.Image, padding: int = 80) -> Image.Image:
    return image.crop(white_border_crop_box(image, padding))


def marker_centroid(image: Image.Image, threshold: int = 35) -> tuple[int, int] | None:
    try:
        import numpy as np

        arr = np.asarray(image.convert("RGB"), dtype=np.int16)
        chroma = arr.max(axis=2) - arr.min(axis=2)
        mask = (chroma > threshold) & (arr.max(axis=2) > 50)
        ys, xs = np.nonzero(mask)
        if len(xs) == 0:
            return None
        return round(float(xs.mean())), round(float(ys.mean()))
    except Exception:
        rgb = image.convert("RGB")
        px = rgb.load()
        xs: list[int] = []
        ys: list[int] = []
        for y in range(rgb.height):
            for x in range(rgb.width):
                r, g, b = px[x, y]
                if max(r, g, b) - min(r, g, b) > threshold and max(r, g, b) > 50:
                    xs.append(x)
                    ys.append(y)
        if not xs:
            return None
        return round(sum(xs) / len(xs)), round(sum(ys) / len(ys))


def save_titled_image(
    image: Image.Image,
    path: Path,
    *,
    title: str,
    figsize: tuple[float, float],
    dpi: int,
    pad: int,
) -> None:
    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_subplot(111)
    ax.imshow(image)
    ax.set_axis_off()
    ax.set_title(title, pad=pad, fontweight="bold")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def save_plain_image(
    image: Image.Image,
    path: Path,
    *,
    figsize: tuple[float, float],
    dpi: int,
) -> None:
    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_subplot(111)
    ax.imshow(image)
    ax.set_axis_off()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def load_font(size: int, bold: bool = True) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def make_snapshot_strip() -> None:
    panel_results = render_snapshot_panels(SNAPSHOT_PDBS)
    panel_w, panel_h = 3200, 2400
    margin_x, margin_y = 75, 90
    label_font = load_font(112)
    canvas = Image.new("RGB", (panel_w * 2, panel_h * 2), "white")
    marker_records: list[dict] = []
    for i, result in enumerate(panel_results):
        raw = Image.open(result["image"]).convert("RGB")
        crop_box = white_border_crop_box(raw)
        im = raw.crop(crop_box)
        base = ImageOps.contain(
            im,
            (panel_w - margin_x * 2, panel_h - margin_y * 2),
            Image.Resampling.LANCZOS,
        )
        scale = PANEL_SCALE_FACTORS[i]
        max_w = panel_w - margin_x * 2
        max_h = panel_h - margin_y * 2
        uniform_scale = min(scale, max_w / base.width, max_h / base.height)
        scaled_w = int(base.width * uniform_scale)
        scaled_h = int(base.height * uniform_scale)
        fitted = base.resize((scaled_w, scaled_h), Image.Resampling.LANCZOS)
        col = i % 2
        row = i // 2
        x0 = col * panel_w
        y0 = row * panel_h
        x = x0 + (panel_w - fitted.width) // 2
        y = y0 + (panel_h - fitted.height) // 2 + 45
        canvas.paste(fitted, (x, y))
        draw = ImageDraw.Draw(canvas)
        draw.text((x0 + 120, y0 + 95), SNAPSHOT_LABELS[i], fill="black", font=label_font)

        records = json.loads(Path(result["records"]).read_text())
        for record in records:
            marker_path = Path(result["marker_dir"]) / f"marker_{record['marker_index']:03d}.png"
            if not marker_path.exists():
                continue
            marker_im = Image.open(marker_path).convert("RGB").crop(crop_box)
            marker_base = ImageOps.contain(
                marker_im,
                (panel_w - margin_x * 2, panel_h - margin_y * 2),
                Image.Resampling.NEAREST,
            )
            marker_fitted = marker_base.resize((scaled_w, scaled_h), Image.Resampling.NEAREST)
            marker_canvas = Image.new("RGB", (panel_w * 2, panel_h * 2), "black")
            marker_canvas.paste(marker_fitted, (x, y))
            marker_canvas.paste("black", (panel_w - 5, 0, panel_w + 5, panel_h * 2))
            marker_canvas.paste("black", (0, panel_h - 5, panel_w * 2, panel_h + 5))
            marker_canvas = ImageOps.expand(marker_canvas, border=10, fill="black")
            marker_out = THIS_DIR / f"_tmp_figure14_snapshot_marker_{record['marker_index']:03d}.png"
            save_plain_image(
                marker_canvas,
                marker_out,
                figsize=(12.8, 9.8),
                dpi=700,
            )
            marker_records.append(
                {
                    "text": record["text"],
                    "marker_path": str(marker_out),
                }
            )

    pane_width = 10
    canvas.paste("black", (panel_w - pane_width // 2, 0, panel_w + pane_width // 2, panel_h * 2))
    canvas.paste("black", (0, panel_h - pane_width // 2, panel_w * 2, panel_h + pane_width // 2))
    canvas = ImageOps.expand(canvas, border=10, fill="black")

    save_plain_image(
        canvas,
        OUT_STRIP,
        figsize=(12.8, 9.8),
        dpi=700,
    )
    FINAL_DIR.mkdir(exist_ok=True)
    if OUT_STRIP.resolve() != FINAL_STRIP.resolve():
        shutil.copyfile(OUT_STRIP, FINAL_STRIP)

    for result in panel_results:
        Path(result["image"]).unlink(missing_ok=True)
        Path(result["records"]).unlink(missing_ok=True)
        shutil.rmtree(result["marker_dir"], ignore_errors=True)
    (THIS_DIR / "_tmp_figure14_snapshot_anchor_records.json").write_text(json.dumps(marker_records, indent=2))


def build_whole_system_pymol_script(
    input_pdb: Path,
    out_png: Path,
    box_bounds: tuple[float, float, float, float, float, float],
    marker_png: Path,
    marker_json: Path,
) -> str:
    x0, x1, y0, y1, z0, z1 = box_bounds
    return f"""
reinitialize
set suspend_updates, on
load {render.pymol_path(input_pdb)}, mol
hide everything, mol
bg_color white
set ray_opaque_background, off
set orthoscopic, off
set field_of_view, 14
set antialias, 2
set depth_cue, 0
set ray_shadow, on
set ray_trace_mode, 1
set ray_trace_color, black
set stick_radius, 0.12
set sphere_scale, 0.20
set spec_reflect, 0.25
set ambient, 0.23
set direct, 0.82

select water_sel, mol and resn WAT
select resin_sel, mol and resn R48
select pfas_sel, mol and resn PFO
select cl_sel, mol and resn Cl-

set_color elem_c, [0.302, 0.302, 0.302]
set_color resin_c, [0.560, 0.560, 0.560]
set_color elem_n, [0.184, 0.396, 0.851]
set_color elem_o, [0.839, 0.153, 0.157]
set_color elem_f, [0.278, 0.722, 0.651]
set_color elem_cl, [0.173, 0.627, 0.173]
set_color water_blue, [0.650, 0.820, 1.000]

show sticks, resin_sel and not hydro
show spheres, resin_sel and not hydro
set sphere_scale, 0.15, resin_sel
set stick_radius, 0.105, resin_sel
color resin_c, resin_sel and elem C
color elem_n, resin_sel and elem N
color elem_o, resin_sel and elem O

show sticks, pfas_sel and not hydro
show spheres, pfas_sel and not hydro
set sphere_scale, 0.31, pfas_sel
set stick_radius, 0.180, pfas_sel
color elem_c, pfas_sel and elem C
color elem_o, pfas_sel and elem O
color elem_f, pfas_sel and elem F

show spheres, cl_sel and not hydro
set sphere_scale, 0.44, cl_sel
color elem_cl, cl_sel

python
from pymol import cmd
from pymol.cgo import *
import colorsys
import json
import math

cmd.set("label_size", 8)
cmd.set("label_font_id", 7)
cmd.set("label_color", "black")
cmd.set("label_outline_color", "white")
cmd.set("dash_width", 1.7)
cmd.set("dash_gap", 0.28)
cmd.set("dash_radius", 0.025)
cmd.set("dash_color", "black")

x0, x1 = {x0:.6f}, {x1:.6f}
y0, y1 = {y0:.6f}, {y1:.6f}
z0, z1 = {z0:.6f}, {z1:.6f}
corners = [
    (x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0),
    (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1),
]
for corner in corners:
    cmd.pseudoatom("box_corner_pts", pos=corner)
cmd.hide("everything", "box_corner_pts")
hidden_edges = [(0,3),(2,3),(3,7)]
solid_edges = [
    (0,1),(1,2),(2,6),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5)
]
faces_idx = [
    (0,1,2), (0,2,3), (4,5,6), (4,6,7),
    (0,1,5), (0,5,4), (1,2,6), (1,6,5),
    (2,3,7), (2,7,6), (3,0,4), (3,4,7),
]
faces = [ALPHA, 0.18, BEGIN, TRIANGLES, COLOR, 0.68, 0.88, 0.90]
for tri in faces_idx:
    for idx in tri:
        faces.extend([VERTEX, *corners[idx]])
faces.append(END)
cmd.load_cgo(faces, "solvent_box_surface")

box = [LINEWIDTH, 0.65, BEGIN, LINES, COLOR, 0.35, 0.58, 0.62]
for i, j in solid_edges:
    box.extend([VERTEX, *corners[i], VERTEX, *corners[j]])
box.append(END)
cmd.load_cgo(box, "solvent_box")

def add_dashed_edge(cgo, start, end, segments=15):
    for seg in range(0, segments, 2):
        t0 = seg / segments
        t1 = min((seg + 1) / segments, 1.0)
        p0 = tuple(start[k] + (end[k] - start[k]) * t0 for k in range(3))
        p1 = tuple(start[k] + (end[k] - start[k]) * t1 for k in range(3))
        cgo.extend([VERTEX, *p0, VERTEX, *p1])

rear_box = [LINEWIDTH, 0.65, BEGIN, LINES, COLOR, 0.35, 0.58, 0.62]
for i, j in hidden_edges:
    add_dashed_edge(rear_box, corners[i], corners[j])
rear_box.append(END)
cmd.load_cgo(rear_box, "solvent_box_rear_edges")

def centroid(selection):
    atoms = cmd.get_model(selection).atom
    return [
        sum(a.coord[0] for a in atoms) / len(atoms),
        sum(a.coord[1] for a in atoms) / len(atoms),
        sum(a.coord[2] for a in atoms) / len(atoms),
    ]

pfas_c = centroid("pfas_sel and not hydro")
resin_c = centroid("resin_sel and not hydro")

pfas_atoms = cmd.get_model("pfas_sel and not hydro").atom
cl_atoms = cmd.get_model("cl_sel and not hydro").atom
n_atoms = cmd.get_model("resin_sel and elem N").atom
def nearest_to_pfas(atoms):
    ranked = []
    for at in atoms:
        nearest = min(math.dist(at.coord, p.coord) for p in pfas_atoms)
        ranked.append((nearest, at))
    return sorted(ranked, key=lambda item: item[0])[0][1] if ranked else None

near_cl = nearest_to_pfas(cl_atoms)
near_n = nearest_to_pfas(n_atoms)
anchor_records = []

def add_anchor_marker(anchor_coord, label_text):
    marker_i = 300 + len(anchor_records)
    hue = (marker_i * 0.61803398875) % 1.0
    r, g, b = colorsys.hsv_to_rgb(hue, 0.82, 1.0)
    color_name = f"mkr_color_{{marker_i:03d}}"
    obj = f"mkr_anchor_{{marker_i:03d}}"
    cmd.set_color(color_name, [r, g, b])
    cmd.pseudoatom(obj, pos=anchor_coord)
    cmd.color(color_name, obj)
    cmd.hide("everything", obj)
    anchor_records.append({{
        "id": f"{{label_text}}_{{marker_i:03d}}",
        "text": label_text,
        "marker_index": marker_i,
        "color": [round(r * 255), round(g * 255), round(b * 255)],
    }})

def add_whole_leader(at, prefix, offset):
    label_name = f"lbl_whole_{{prefix}}_{{at.id}}"
    pos = [at.coord[0] + offset[0], at.coord[1] + offset[1], at.coord[2] + offset[2]]
    add_anchor_marker(at.coord, f"{{prefix}}{{at.id}}")
    cmd.pseudoatom(label_name, pos=pos, label=f"{{prefix}}{{at.id}}")
    cmd.hide("spheres", label_name)
    cmd.distance(f"lead_whole_{{prefix}}_{{at.id}}", f"index {{at.index}}", label_name)
    cmd.hide("labels", f"lead_whole_{{prefix}}_{{at.id}}")

def add_named_leader(anchor_pos, name, offset):
    anchor_name = f"anchor_whole_{{name}}"
    label_name = f"lbl_whole_{{name}}"
    label_pos = [anchor_pos[0] + offset[0], anchor_pos[1] + offset[1], anchor_pos[2] + offset[2]]
    add_anchor_marker(anchor_pos, name)
    cmd.pseudoatom(anchor_name, pos=anchor_pos)
    cmd.hide("spheres", anchor_name)
    cmd.pseudoatom(label_name, pos=label_pos, label=name)
    cmd.hide("spheres", label_name)
    cmd.distance(f"lead_whole_{{name}}", anchor_name, label_name)
    cmd.hide("labels", f"lead_whole_{{name}}")

add_named_leader(pfas_c, "PFOA", [16.0, 10.0, 8.0])
add_named_leader(resin_c, "R48", [-18.0, -13.0, 8.0])
if near_cl is not None:
    add_whole_leader(near_cl, "Cl", [13.0, -9.0, 7.0])
with open(r"{marker_json.as_posix()}", "w") as fh:
    json.dump(anchor_records, fh, indent=2)
python end

orient (mol or box_corner_pts)
turn y, 35
turn x, -25
zoom (mol or box_corner_pts), 34
{"" if SHOW_STRUCTURE_LABELS else HIDE_LABELS_PML}
set suspend_updates, off
png {render.pymol_path(out_png)}, width=3400, height=2300, dpi=650, ray=1
python
from pymol import cmd
cmd.bg_color("black")
cmd.set("ray_opaque_background", 1)
cmd.set("ambient", 1.0)
cmd.set("direct", 0.0)
cmd.set("spec_reflect", 0.0)
for rec in anchor_records:
    name = "mkr_anchor_%03d" % rec["marker_index"]
    cmd.hide("everything", "all")
    cmd.hide("labels", "all")
    cmd.enable(name)
    cmd.show("spheres", name)
    cmd.set("sphere_scale", 1.45, name)
    cmd.png(r"{marker_png.parent.as_posix()}/whole_marker_%03d.png" % rec["marker_index"], width=3400, height=2300, dpi=650, ray=1)
python end
quit
"""


def make_whole_system_figure() -> None:
    out_png = THIS_DIR / "_tmp_whole.png"
    out_marker_png = THIS_DIR / "_tmp_whole_anchor_markers_raw.png"
    out_marker_json = THIS_DIR / "_tmp_whole_anchor_records.json"
    for old_marker in THIS_DIR.glob("whole_marker_*.png"):
        old_marker.unlink(missing_ok=True)
    with tempfile.TemporaryDirectory(prefix="md_whole_", dir="/tmp") as td:
        bonded_pdb = Path(td) / WHOLE_SYSTEM_PDB.name
        write_topology_bonded_pdb(
            WHOLE_SYSTEM_PDB,
            bonded_pdb,
            WHOLE_SYSTEM_PRMTOP,
            include_resnames={"R48", "PFO", "Cl-"},
        )
        script = build_whole_system_pymol_script(
            bonded_pdb,
            out_png,
            pdb_coordinate_bounds(WHOLE_SYSTEM_PDB),
            out_marker_png,
            out_marker_json,
        )
        pml = Path(td) / "whole.pml"
        pml.write_text(script)
        cmd = [str(render.PYMOL_RUNNER), "run", "-n", render.PYMOL_ENV, "pymol", "-cq", str(pml)]
        env = os.environ.copy()
        env.setdefault("PYMOL_GIT_MOD", "0")
        res = subprocess.run(cmd, cwd=WHOLE_SYSTEM_PDB.parent, text=True, capture_output=True, env=env, check=False)
        if res.returncode != 0:
            raise RuntimeError(f"PyMOL whole-system rendering failed.\nSTDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}")

    whole_img = Image.open(out_png).convert("RGB")
    whole_img = ImageOps.expand(whole_img, border=10, fill="black")

    save_titled_image(
        whole_img,
        OUT_WHOLE,
        title="Full System Representation",
        figsize=(12.5, 8.8),
        dpi=500,
        pad=10,
    )
    FINAL_DIR.mkdir(exist_ok=True)
    if OUT_WHOLE.resolve() != FINAL_WHOLE.resolve():
        shutil.copyfile(OUT_WHOLE, FINAL_WHOLE)
    if OUT_WHOLE.resolve() != FINAL_WHOLE_NUMBERED.resolve():
        shutil.copyfile(OUT_WHOLE, FINAL_WHOLE_NUMBERED)

    marker_records = []
    for record in json.loads(out_marker_json.read_text()):
        raw_marker = THIS_DIR / f"whole_marker_{record['marker_index']:03d}.png"
        if not raw_marker.exists():
            continue
        marker_img = Image.open(raw_marker).convert("RGB")
        marker_img = ImageOps.expand(marker_img, border=10, fill="black")
        marker_out = THIS_DIR / f"_tmp_figure14_whole_marker_{record['marker_index']:03d}.png"
        save_titled_image(
            marker_img,
            marker_out,
            title="Full System Representation",
            figsize=(12.5, 8.8),
            dpi=500,
            pad=10,
        )
        marker_records.append({"text": record["text"], "marker_path": str(marker_out)})
        raw_marker.unlink(missing_ok=True)
    out_png.unlink(missing_ok=True)
    out_marker_png.unlink(missing_ok=True)
    out_marker_json.write_text(json.dumps(marker_records, indent=2))


def add_panel_letter(image: Image.Image, letter: str, font_size: int = 150) -> Image.Image:
    panel = image.copy()
    draw = ImageDraw.Draw(panel)
    font = load_font(font_size)
    margin = max(55, font_size // 2)
    draw.text((margin, margin), letter, fill="black", font=font)
    return panel


def add_figure14_header(canvas: Image.Image, height: int = 245) -> Image.Image:
    out = Image.new("RGB", (canvas.width, canvas.height + height), "white")
    out.paste(canvas, (0, height))
    draw = ImageDraw.Draw(out)
    title_font = load_font(122, bold=True)
    font = load_font(92, bold=True)
    title = "Molecular Dynamics Trajectory of PFAS Interactions with Cholestyramine"
    title_box = draw.textbbox((0, 0), title, font=title_font)
    draw.text(
        ((canvas.width - (title_box[2] - title_box[0])) // 2, 42),
        title,
        fill="black",
        font=title_font,
    )
    items = [
        ("R48 resin C", "#8f8f8f"),
        ("PFOA C", "#4d4d4d"),
        ("N", "#2f65d9"),
        ("O", "#d62728"),
        ("F", "#47b8a6"),
        ("Cl", "#2ca02c"),
    ]
    widths = []
    for text, _ in items:
        box = draw.textbbox((0, 0), text, font=font)
        widths.append(96 + (box[2] - box[0]) + 95)
    total = sum(widths)
    x = max(80, (canvas.width - total) // 2)
    y = 248
    for (text, color), width in zip(items, widths):
        draw.ellipse((x, y, x + 56, y + 56), fill=color, outline="black", width=4)
        draw.text((x + 76, y - 22), text, fill="black", font=font)
        x += width
    return out


def write_label_anchor_files_from_markers(
    marker_records: list[dict],
    *,
    target_w: int,
    snap_h: int,
    gap: int,
    top_offset: int,
    canvas_size: tuple[int, int],
) -> list[dict]:
    """Write dragger anchors from isolated PyMOL marker renders."""
    labels: list[dict] = []
    seen: dict[str, int] = {}
    for record in marker_records:
        marker_path = Path(record["marker_path"])
        marker = Image.open(marker_path).convert("RGB")
        marker = ImageOps.contain(
            marker,
            (target_w, int(target_w * marker.height / marker.width)),
            Image.Resampling.NEAREST,
        )
        full = Image.new("RGB", canvas_size, "black")
        if "whole_marker" in marker_path.name:
            offset = ((target_w - marker.width) // 2, top_offset + snap_h + gap)
        else:
            offset = ((target_w - marker.width) // 2, top_offset)
        full.paste(marker, offset)
        anchor = marker_centroid(full)
        if anchor is None:
            continue
        text = record["text"]
        seen[text] = seen.get(text, 0) + 1
        labels.append(
            {
                "id": f"{text}_{seen[text]}",
                "text": text,
                "anchor": [anchor[0], anchor[1]],
                "position": [anchor[0], anchor[1]],
            }
        )

    payload = {
        "image": "Figure_14_MD_structural_snapshots_strip_no_structure_labels.png",
        "font_size": 60,
        "line_width": 2,
        "labels": labels,
    }
    FINAL_LABEL_ANCHORS_JSON.write_text(json.dumps(payload, indent=2))
    FINAL_LABEL_ANCHORS_JS.write_text(
        "window.FIGURE14_LABEL_DATA = "
        + json.dumps(payload, indent=2)
        + ";\n"
    )
    return labels


def structure_mask(image: Image.Image) -> Image.Image:
    """Mask non-background pixels so overlay labels can avoid rendered structure."""
    rgb = image.convert("RGB")
    px = rgb.load()
    mask = Image.new("L", rgb.size, 0)
    mp = mask.load()
    for y in range(rgb.height):
        for x in range(rgb.width):
            r, g, b = px[x, y]
            # Keep panel borders/title text out of the atom mask by requiring
            # either colored pixels or sufficiently dark gray molecular pixels.
            if (max(r, g, b) - min(r, g, b) > 18 and min(r, g, b) < 245) or max(r, g, b) < 110:
                mp[x, y] = 255
    return mask.filter(ImageFilter.MaxFilter(15))


def bbox_overlaps_mask(mask: Image.Image, box: tuple[int, int, int, int]) -> bool:
    left, top, right, bottom = box
    left = max(0, left)
    top = max(0, top)
    right = min(mask.width, right)
    bottom = min(mask.height, bottom)
    if left >= right or top >= bottom:
        return True
    return mask.crop((left, top, right, bottom)).getbbox() is not None


def rects_intersect(a: tuple[int, int, int, int], b: tuple[int, int, int, int], pad: int = 10) -> bool:
    return not (
        a[2] + pad < b[0]
        or b[2] + pad < a[0]
        or a[3] + pad < b[1]
        or b[3] + pad < a[1]
    )


def draw_dashed_line(
    draw: ImageDraw.ImageDraw,
    start: tuple[int, int],
    end: tuple[int, int],
    *,
    fill: str = "black",
    width: int = 2,
    dash: int = 14,
    gap: int = 9,
) -> None:
    sx, sy = start
    ex, ey = end
    dx = ex - sx
    dy = ey - sy
    length = (dx * dx + dy * dy) ** 0.5
    if length == 0:
        return
    ux = dx / length
    uy = dy / length
    pos = 0.0
    while pos < length:
        end_pos = min(pos + dash, length)
        x0 = int(round(sx + ux * pos))
        y0 = int(round(sy + uy * pos))
        x1 = int(round(sx + ux * end_pos))
        y1 = int(round(sy + uy * end_pos))
        draw.line((x0, y0, x1, y1), fill=fill, width=width)
        pos += dash + gap


def choose_label_position(
    label: str,
    anchor: tuple[int, int],
    panel_box: tuple[int, int, int, int],
    mask: Image.Image,
    font: ImageFont.ImageFont,
    occupied: list[tuple[int, int, int, int]],
) -> tuple[int, int, tuple[int, int, int, int]]:
    probe = Image.new("RGB", (1, 1))
    pd = ImageDraw.Draw(probe)
    text_box = pd.textbbox((0, 0), label, font=font)
    tw = text_box[2] - text_box[0]
    th = text_box[3] - text_box[1]
    ax, ay = anchor
    left, top, right, bottom = panel_box
    offsets = [
        (-260, -170), (210, -170), (-260, 160), (210, 160),
        (-330, 0), (260, 0), (0, -235), (0, 220),
        (-360, -250), (275, -250), (-360, 245), (275, 245),
        (-470, -90), (380, -90), (-470, 95), (380, 95),
    ]
    best = None
    best_score = -10**9
    for ox, oy in offsets:
        x = int(ax + ox)
        y = int(ay + oy)
        x = max(left + 34, min(x, right - tw - 34))
        y = max(top + 34, min(y, bottom - th - 34))
        box = (x - 14, y - 10, x + tw + 14, y + th + 12)
        overlap_structure = bbox_overlaps_mask(mask, box)
        overlap_labels = any(rects_intersect(box, used, pad=18) for used in occupied)
        dist2 = (x + tw / 2 - ax) ** 2 + (y + th / 2 - ay) ** 2
        edge_clearance = min(x - left, right - (x + tw), y - top, bottom - (y + th))
        score = edge_clearance * 1.3 - dist2 * 0.0008
        if overlap_structure:
            score -= 5000
        if overlap_labels:
            score -= 3500
        if score > best_score:
            best_score = score
            best = (x, y, box)
    assert best is not None
    return best


def add_overlay_labels(image: Image.Image, labels: list[dict], font_size: int = 60) -> Image.Image:
    out = image.copy().convert("RGB")
    draw = ImageDraw.Draw(out)
    font = load_font(font_size, bold=True)
    mask = structure_mask(out)
    occupied: list[tuple[int, int, int, int]] = []
    for item in labels:
        text = item["text"]
        anchor = item["anchor"]
        panel = item["panel"]
        if "pos" in item:
            x, y = item["pos"]
            tb_probe = ImageDraw.Draw(Image.new("RGB", (1, 1))).textbbox((x, y), text, font=font)
            box = (tb_probe[0] - 14, tb_probe[1] - 10, tb_probe[2] + 14, tb_probe[3] + 12)
        else:
            x, y, box = choose_label_position(text, anchor, panel, mask, font, occupied)
        occupied.append(box)
        probe = ImageDraw.Draw(Image.new("RGB", (1, 1)))
        tb = probe.textbbox((x, y), text, font=font)
        attach = (
            max(tb[0], min(anchor[0], tb[2])),
            max(tb[1], min(anchor[1], tb[3])),
        )
        draw_dashed_line(draw, anchor, attach, width=item.get("width", 2))
        draw.text((x, y), text, fill="black", font=font, stroke_width=3, stroke_fill="white")
    return out


def figure14_overlay_specs() -> tuple[list[dict], list[dict]]:
    """Approximate anchors in final composite pixel space for Figure 14 labels."""
    # Panel boxes are intentionally inset from the borders/titles so labels
    # stay in white space around the rendered structures.
    pane_a = [
        (75, 390, 3540, 2810),
        (3545, 390, 7005, 2810),
        (75, 2825, 3540, 5335),
        (3545, 2825, 7005, 5335),
    ]
    a_specs = [
        ("Cl1688", (1545, 1180), (1120, 650), pane_a[0]),
        ("N188", (1490, 1285), (610, 980), pane_a[0]),
        ("N177", (1545, 1360), (720, 1330), pane_a[0]),
        ("PFOA", (1940, 1520), (2380, 1060), pane_a[0]),
        ("N144", (2220, 1530), (2700, 1450), pane_a[0]),
        ("R48", (2230, 1850), (3210, 1510), pane_a[0]),
        ("N155", (2050, 1795), (2320, 2110), pane_a[0]),
        ("N166", (1995, 2310), (1920, 2500), pane_a[0]),
        ("N188", (5240, 1080), (5170, 670), pane_a[1]),
        ("N177", (5350, 1280), (5660, 1000), pane_a[1]),
        ("R48", (5090, 1450), (4560, 1120), pane_a[1]),
        ("PFOA", (4870, 1600), (4210, 1450), pane_a[1]),
        ("N199", (5030, 1955), (4450, 2360), pane_a[1]),
        ("N166", (5480, 2050), (5230, 2240), pane_a[1]),
        ("Cl1700", (5835, 2050), (6120, 2380), pane_a[1]),
        ("N155", (5715, 1810), (6320, 1590), pane_a[1]),
        ("N177", (1760, 3500), (1500, 3480), pane_a[2]),
        ("N188", (2070, 3550), (2040, 3370), pane_a[2]),
        ("N144", (2520, 3550), (2590, 3520), pane_a[2]),
        ("R48", (2420, 4240), (3040, 3910), pane_a[2]),
        ("N155", (2290, 4320), (2640, 4710), pane_a[2]),
        ("N166", (2060, 4630), (1440, 4890), pane_a[2]),
        ("N199", (1330, 4410), (920, 4410), pane_a[2]),
        ("PFOA", (1860, 4240), (1260, 3800), pane_a[2]),
        ("N177", (4930, 3430), (4460, 3240), pane_a[3]),
        ("N188", (5600, 3520), (5480, 3420), pane_a[3]),
        ("N144", (6260, 3520), (6150, 3410), pane_a[3]),
        ("R48", (4850, 3920), (4930, 3780), pane_a[3]),
        ("N199", (4850, 4460), (4230, 4490), pane_a[3]),
        ("N155", (5300, 4710), (4660, 5000), pane_a[3]),
        ("PFOA", (5580, 4600), (6060, 5000), pane_a[3]),
        ("N166", (5830, 4310), (6440, 3630), pane_a[3]),
    ]
    panel_b = (75, 6060, 7005, 10495)
    b_specs = [
        ("PFOA", (4070, 7970), (4510, 7800), panel_b),
        ("R48", (3140, 8200), (2740, 8280), panel_b),
        ("Cl1685", (4020, 8500), (4300, 8660), panel_b),
    ]
    return (
        [{"text": text, "anchor": anchor, "pos": pos, "panel": panel} for text, anchor, pos, panel in a_specs],
        [{"text": text, "anchor": anchor, "pos": pos, "panel": panel} for text, anchor, pos, panel in b_specs],
    )


def make_combined_md_figure() -> None:
    snap = Image.open(FINAL_STRIP).convert("RGB")
    whole = Image.open(FINAL_WHOLE).convert("RGB")
    snapshot_marker_path = THIS_DIR / "_tmp_figure14_snapshot_anchor_records.json"
    whole_marker_path = THIS_DIR / "_tmp_whole_anchor_records.json"
    snapshot_marker_records = json.loads(snapshot_marker_path.read_text()) if snapshot_marker_path.exists() else []
    whole_marker_records = json.loads(whole_marker_path.read_text()) if whole_marker_path.exists() else []

    target_w = max(snap.width, whole.width)
    snap = ImageOps.contain(snap, (target_w, int(target_w * snap.height / snap.width)), Image.Resampling.LANCZOS)
    whole = ImageOps.contain(whole, (target_w, int(target_w * whole.height / whole.width)), Image.Resampling.LANCZOS)

    whole = add_panel_letter(whole, "B")

    gap = 80
    header_h = 430
    panel_a_pad = 150
    content = Image.new("RGB", (target_w, panel_a_pad + snap.height + gap + whole.height), "white")
    content_draw = ImageDraw.Draw(content)
    content_draw.text((75, 18), "A", fill="black", font=load_font(150))
    content.paste(snap, ((target_w - snap.width) // 2, panel_a_pad))
    content.paste(whole, ((target_w - whole.width) // 2, panel_a_pad + snap.height + gap))
    canvas = add_figure14_header(content, height=header_h)
    no_label_path = FINAL_DIR / "Figure_14_MD_structural_snapshots_strip_no_structure_labels.png"
    labeled_reference_path = FINAL_DIR / "Figure_14_MD_structural_snapshots_strip_labeled_reference.png"
    canvas.save(no_label_path)
    old_header_h = 245
    manual_y_offset = header_h + panel_a_pad - old_header_h
    if FINAL_MANUAL_LABEL_LAYOUT.exists():
        manual = json.loads(FINAL_MANUAL_LABEL_LAYOUT.read_text())
        labels = []
        for item in manual.get("labels", []):
            labels.append(
                {
                    "text": item["text"],
                    "anchor": (item["anchor"][0], item["anchor"][1] + manual_y_offset),
                    "pos": (item["position"][0], item["position"][1] + manual_y_offset),
                    "panel": (0, 0, canvas.width, canvas.height),
                    "width": manual.get("line_width", 2),
                }
            )
        payload = {
            "image": "Figure_14_MD_structural_snapshots_strip_no_structure_labels.png",
            "font_size": manual.get("font_size", 60),
            "line_width": manual.get("line_width", 2),
            "labels": [
                {
                    "id": item["id"],
                    "text": item["text"],
                    "anchor": [item["anchor"][0], item["anchor"][1] + manual_y_offset],
                    "position": [item["position"][0], item["position"][1] + manual_y_offset],
                }
                for item in manual.get("labels", [])
            ],
        }
        FINAL_LABEL_ANCHORS_JSON.write_text(json.dumps(payload, indent=2))
        FINAL_LABEL_ANCHORS_JS.write_text("window.FIGURE14_LABEL_DATA = " + json.dumps(payload, indent=2) + ";\n")
        labeled = add_overlay_labels(canvas, labels, font_size=manual.get("font_size", 60))
    else:
        if snapshot_marker_records or whole_marker_records:
            write_label_anchor_files_from_markers(
                snapshot_marker_records + whole_marker_records,
                target_w=target_w,
                snap_h=snap.height,
                gap=gap,
                top_offset=header_h + panel_a_pad,
                canvas_size=canvas.size,
            )
        a_labels, b_labels = figure14_overlay_specs()
        for item in a_labels + b_labels:
            item["anchor"] = (item["anchor"][0], item["anchor"][1] + header_h + panel_a_pad)
            item["pos"] = (item["pos"][0], item["pos"][1] + header_h + panel_a_pad)
            item["panel"] = (
                item["panel"][0],
                item["panel"][1] + header_h + panel_a_pad,
                item["panel"][2],
                item["panel"][3] + header_h + panel_a_pad,
            )
        labeled = add_overlay_labels(canvas, a_labels + b_labels)
    labeled.save(labeled_reference_path)
    labeled.save(FINAL_STRIP_NUMBERED)
    for record in snapshot_marker_records + whole_marker_records:
        Path(record["marker_path"]).unlink(missing_ok=True)
    snapshot_marker_path.unlink(missing_ok=True)
    whole_marker_path.unlink(missing_ok=True)


if __name__ == "__main__":
    require_inputs([*SNAPSHOT_PDBS, WHOLE_SYSTEM_PDB])
    make_snapshot_strip()
    make_whole_system_figure()
    make_combined_md_figure()
    print("Structural MD figures generated.")
    print(f"Outputs:\n  - {OUT_STRIP}\n  - {OUT_WHOLE}\n  - {FINAL_STRIP}\n  - {FINAL_WHOLE}")
