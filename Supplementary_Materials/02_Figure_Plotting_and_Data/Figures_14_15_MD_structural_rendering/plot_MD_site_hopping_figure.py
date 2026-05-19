from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path
from math import dist

import matplotlib.pyplot as plt
from PIL import Image, ImageChops, ImageDraw, ImageFont, ImageOps

import sys

ROOT = Path(__file__).resolve().parents[2]
TOOLS_DIR = ROOT / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))
import render_structure_3d as render  # noqa: E402


THIS_DIR = Path(__file__).resolve().parent
FINAL_DIR = ROOT / "final_plots"
METRICS_DIR = ROOT / "work_amber" / "17_exchange_cycle_proxy" / "05_metrics"
VMD_DIR = ROOT / "work_amber" / "16_vmd_pfoa_exchange_inspection"

SNAPSHOT_PDBS = [
    METRICS_DIR / "pfoa_assoc_frame_001_resin_pfo_cl.pdb",
    METRICS_DIR / "pfoa_assoc_frame_025_resin_pfo_cl.pdb",
    METRICS_DIR / "pfoa_assoc_frame_050_resin_pfo_cl.pdb",
    METRICS_DIR / "pfoa_assoc_frame_100_resin_pfo_cl.pdb",
]
SNAPSHOT_LABELS = ["Frame 1", "Frame 25", "Frame 50", "Frame 100"]
PANEL_SCALE_FACTORS = [1.18, 1.16, 1.20, 1.00]
WHOLE_SYSTEM_PDB = VMD_DIR / "resin_pfoa_exchange_npt_1ns_final_imaged.pdb"

OUT_STRIP = THIS_DIR / "figure_MD_structural_snapshots_strip.png"
OUT_WHOLE = THIS_DIR / "figure_MD_whole_system_water_transparent.png"
FINAL_STRIP = FINAL_DIR / OUT_STRIP.name
FINAL_WHOLE = FINAL_DIR / OUT_WHOLE.name
FINAL_STRIP_NUMBERED = FINAL_DIR / "Figure_14_MD_structural_snapshots_strip.png"
FINAL_WHOLE_NUMBERED = FINAL_DIR / "Figure_15_MD_whole_system_water_transparent.png"


def require_inputs(paths: list[Path]) -> None:
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing required inputs:\n" + "\n".join(f"  - {m}" for m in missing))


def write_bonded_pdb(source: Path, dest: Path) -> None:
    """Write a render-only PDB with inferred covalent CONECT records."""
    radii = {
        "H": 0.31,
        "C": 0.76,
        "N": 0.71,
        "O": 0.66,
        "F": 0.57,
        "S": 1.05,
        "P": 1.07,
        "CL": 1.02,
    }
    atom_lines: list[str] = []
    atoms: list[tuple[int, str, str, tuple[float, float, float]]] = []
    other_lines: list[str] = []
    for line in source.read_text().splitlines():
        if line.startswith(("ATOM", "HETATM")):
            idx = int(line[6:11])
            resn = line[17:21].strip()
            elem = line[76:78].strip().upper() or line[12:16].strip()[0].upper()
            xyz = (float(line[30:38]), float(line[38:46]), float(line[46:54]))
            atom_lines.append(line)
            atoms.append((idx, resn, elem, xyz))
        elif not line.startswith(("CONECT", "END")):
            other_lines.append(line)

    bonds: dict[int, set[int]] = {idx: set() for idx, *_ in atoms}
    for i, (idx_i, resn_i, elem_i, xyz_i) in enumerate(atoms):
        if resn_i == "Cl-" or elem_i == "H":
            continue
        r_i = radii.get(elem_i, 0.77)
        for idx_j, resn_j, elem_j, xyz_j in atoms[i + 1 :]:
            if resn_j == "Cl-" or elem_j == "H":
                continue
            cutoff = r_i + radii.get(elem_j, 0.77) + 0.45
            if 0.4 < dist(xyz_i, xyz_j) <= cutoff:
                bonds[idx_i].add(idx_j)
                bonds[idx_j].add(idx_i)

    lines = other_lines + atom_lines
    for idx in sorted(bonds):
        bonded = sorted(bonds[idx])
        for start in range(0, len(bonded), 4):
            chunk = bonded[start : start + 4]
            if chunk:
                lines.append("CONECT" + f"{idx:5d}" + "".join(f"{j:5d}" for j in chunk))
    lines.append("END")
    dest.write_text("\n".join(lines) + "\n")


def render_snapshot_panels(paths: list[Path]) -> list[Path]:
    out_paths: list[Path] = []
    for pdb in paths:
        out_png = pdb.with_suffix(".snapshot.png")
        with tempfile.TemporaryDirectory(prefix="md_bind_", dir="/tmp") as td:
            bonded_pdb = Path(td) / pdb.name
            write_bonded_pdb(pdb, bonded_pdb)
            pml = Path(td) / "snap.pml"
            script = f"""
reinitialize
set suspend_updates, on
load {render.pymol_path(bonded_pdb)}, mol
hide everything, mol
bg_color white
set ray_opaque_background, off
set orthoscopic, on
set antialias, 2
set depth_cue, 0
set sphere_scale, 0.20
set stick_radius, 0.12
set spec_reflect, 0.25
set ambient, 0.35
set direct, 0.65

select resin, mol and resn R48 and not hydro
select pfas, mol and resn PFO and not hydro
select clions, mol and resn Cl- and not hydro
select clnear, (clions within 6.0 of pfas) and not hydro

set_color elem_h, [0.949, 0.949, 0.949]
set_color elem_c, [0.302, 0.302, 0.302]
set_color elem_n, [0.184, 0.396, 0.851]
set_color elem_o, [0.839, 0.153, 0.157]
set_color elem_f, [0.278, 0.722, 0.651]
set_color elem_cl, [0.173, 0.627, 0.173]

python
from pymol import cmd
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
set stick_transparency, 0.18, shell_net
set sphere_transparency, 0.18, shell_net
set sphere_scale, 0.28, clnear

color elem_c, bindview and elem C
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

def add_leader_label(at, prefix, ordinal=0):
    global placed_xy
    directions = [
        (1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0),
        (0.72, 0.72), (-0.72, 0.72), (0.72, -0.72), (-0.72, -0.72),
        (1.0, 0.45), (-1.0, 0.45), (1.0, -0.45), (-1.0, -0.45),
    ]
    radii = [5.5, 7.0, 8.5]
    candidates = []
    for radius in radii:
        for dx, dy in directions:
            x = at.coord[0] + radius * dx
            y = at.coord[1] + radius * dy
            z = at.coord[2] + 1.5 + 0.25 * (ordinal % 3)
            atom_clearance = min(((x - ax) ** 2 + (y - ay) ** 2) ** 0.5 for ax, ay in bind_xy)
            label_clearance = min(
                [((x - lx) ** 2 + (y - ly) ** 2) ** 0.5 for lx, ly in placed_xy] or [99.0]
            )
            outward = ((x - cx) ** 2 + (y - cy) ** 2) ** 0.5
            score = 2.2 * atom_clearance + 1.4 * label_clearance + 0.20 * outward
            candidates.append((score, [x, y, z]))
    pos = max(candidates, key=lambda item: item[0])[1]
    placed_xy.append((pos[0], pos[1]))
    obj = f"lbl_{{prefix}}_{{at.id}}"
    label_objects.append(obj)
    cmd.pseudoatom(obj, pos=pos, label=f"{{prefix}}{{at.id}}")
    cmd.hide("spheres", obj)
    cmd.distance(f"lead_{{prefix}}_{{at.id}}", f"index {{at.index}}", obj)
    cmd.hide("labels", f"lead_{{prefix}}_{{at.id}}")

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
    cmd.pseudoatom("lbl_pfoa", pos=[px + 5.0, py + 3.5, pz + 3.0], label="PFOA")
    cmd.hide("spheres", "lbl_pfoa")
    label_objects.append("lbl_pfoa")
if resin_atoms:
    rx = sum(a.coord[0] for a in resin_atoms) / len(resin_atoms)
    ry = sum(a.coord[1] for a in resin_atoms) / len(resin_atoms)
    rz = sum(a.coord[2] for a in resin_atoms) / len(resin_atoms)
    cmd.pseudoatom("lbl_r48", pos=[rx - 7.0, ry - 4.5, rz + 3.5], label="R48")
    cmd.hide("spheres", "lbl_r48")
    label_objects.append("lbl_r48")
if label_objects:
    cmd.select("annotation_view", "bindview or " + " or ".join(label_objects))
else:
    cmd.select("annotation_view", "bindview")
python end

orient annotation_view
turn y, 20
turn x, -18
zoom annotation_view, 3.5
clip slab, 32
set suspend_updates, off
png {render.pymol_path(out_png)}, width=3200, height=2400, dpi=900, ray=1
quit
"""
            pml.write_text(script)
            cmd = [str(render.PYMOL_RUNNER), "run", "-n", render.PYMOL_ENV, "pymol", "-cq", str(pml)]
            env = os.environ.copy()
            env.setdefault("PYMOL_GIT_MOD", "0")
            res = subprocess.run(cmd, cwd=pdb.parent, text=True, capture_output=True, env=env, check=False)
            if res.returncode != 0:
                raise RuntimeError(f"PyMOL snapshot rendering failed for {pdb.name}.\nSTDERR:\n{res.stderr}")
        out_paths.append(out_png)
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


def crop_white_border(image: Image.Image, padding: int = 80) -> Image.Image:
    bg = Image.new(image.mode, image.size, "white")
    diff = ImageChops.difference(image, bg).convert("L")
    bbox = diff.point(lambda px: 255 if px > 12 else 0).getbbox()
    if bbox is None:
        return image
    left = max(bbox[0] - padding, 0)
    upper = max(bbox[1] - padding, 0)
    right = min(bbox[2] + padding, image.width)
    lower = min(bbox[3] + padding, image.height)
    return image.crop((left, upper, right, lower))


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
    panel_paths = render_snapshot_panels(SNAPSHOT_PDBS)
    images = [crop_white_border(Image.open(p).convert("RGB")) for p in panel_paths]
    panel_w, panel_h = 3200, 2400
    margin_x, margin_y = 75, 90
    label_font = load_font(112)
    canvas = Image.new("RGB", (panel_w * 2, panel_h * 2), "white")
    for i, im in enumerate(images):
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

    pane_width = 10
    canvas.paste("black", (panel_w - pane_width // 2, 0, panel_w + pane_width // 2, panel_h * 2))
    canvas.paste("black", (0, panel_h - pane_width // 2, panel_w * 2, panel_h + pane_width // 2))
    canvas = ImageOps.expand(canvas, border=10, fill="black")

    fig = plt.figure(figsize=(12.8, 9.8), dpi=700)
    ax = fig.add_subplot(111)
    ax.imshow(canvas)
    ax.set_axis_off()
    ax.set_title("Molecular Dynamics Trajectory of PFAS Interactions with Cholestyramine", pad=12, fontweight="bold")
    fig.savefig(OUT_STRIP, bbox_inches="tight")
    FINAL_DIR.mkdir(exist_ok=True)
    fig.savefig(FINAL_STRIP, bbox_inches="tight")
    plt.close(fig)

    for p in panel_paths:
        p.unlink(missing_ok=True)


def build_whole_system_pymol_script(input_pdb: Path, out_png: Path) -> str:
    return f"""
reinitialize
set suspend_updates, on
load {render.pymol_path(input_pdb)}, mol
hide everything, mol
bg_color white
set ray_opaque_background, off
set orthoscopic, off
set field_of_view, 24
set antialias, 2
set depth_cue, 0
set stick_radius, 0.12
set sphere_scale, 0.20
set spec_reflect, 0.25
set ambient, 0.35
set direct, 0.65

select water_sel, mol and resn WAT
select resin_sel, mol and resn R48
select pfas_sel, mol and resn PFO
select cl_sel, mol and resn Cl-

set_color elem_c, [0.302, 0.302, 0.302]
set_color elem_n, [0.184, 0.396, 0.851]
set_color elem_o, [0.839, 0.153, 0.157]
set_color elem_f, [0.278, 0.722, 0.651]
set_color elem_cl, [0.173, 0.627, 0.173]
set_color water_blue, [0.650, 0.820, 1.000]

show spheres, water_sel and not hydro
set sphere_scale, 0.040, water_sel
color water_blue, water_sel
set sphere_transparency, 0.975, water_sel

show sticks, resin_sel and not hydro
show spheres, resin_sel and not hydro
set sphere_scale, 0.17, resin_sel
set stick_radius, 0.135, resin_sel
color elem_c, resin_sel and elem C
color elem_n, resin_sel and elem N
color elem_o, resin_sel and elem O

show sticks, pfas_sel and not hydro
show spheres, pfas_sel and not hydro
set sphere_scale, 0.29, pfas_sel
set stick_radius, 0.170, pfas_sel
color elem_c, pfas_sel and elem C
color elem_o, pfas_sel and elem O
color elem_f, pfas_sel and elem F

show spheres, cl_sel and not hydro
set sphere_scale, 0.44, cl_sel
color elem_cl, cl_sel

python
from pymol import cmd
from pymol.cgo import *
import math

cmd.set("label_size", 8)
cmd.set("label_color", "black")
cmd.set("label_outline_color", "white")
cmd.set("dash_width", 1.7)
cmd.set("dash_gap", 0.28)
cmd.set("dash_radius", 0.025)
cmd.set("dash_color", "black")

all_atoms = cmd.get_model("mol and not hydro").atom
xs = [a.coord[0] for a in all_atoms]
ys = [a.coord[1] for a in all_atoms]
zs = [a.coord[2] for a in all_atoms]
pad = 1.5
x0, x1 = min(xs) - pad, max(xs) + pad
y0, y1 = min(ys) - pad, max(ys) + pad
z0, z1 = min(zs) - pad, max(zs) + pad
corners = [
    (x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0),
    (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1),
]
for corner in corners:
    cmd.pseudoatom("box_corner_pts", pos=corner)
cmd.hide("everything", "box_corner_pts")
edges = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]
box = [LINEWIDTH, 0.8, BEGIN, LINES, COLOR, 0.0, 0.0, 0.0]
for i, j in edges:
    box.extend([VERTEX, *corners[i], VERTEX, *corners[j]])
box.append(END)
cmd.load_cgo(box, "solvent_box")

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
def add_whole_leader(at, prefix, offset):
    label_name = f"lbl_whole_{{prefix}}_{{at.id}}"
    pos = [at.coord[0] + offset[0], at.coord[1] + offset[1], at.coord[2] + offset[2]]
    cmd.pseudoatom(label_name, pos=pos, label=f"{{prefix}}{{at.id}}")
    cmd.hide("spheres", label_name)
    cmd.distance(f"lead_whole_{{prefix}}_{{at.id}}", f"index {{at.index}}", label_name)
    cmd.hide("labels", f"lead_whole_{{prefix}}_{{at.id}}")

def add_named_leader(anchor_pos, name, offset):
    anchor_name = f"anchor_whole_{{name}}"
    label_name = f"lbl_whole_{{name}}"
    label_pos = [anchor_pos[0] + offset[0], anchor_pos[1] + offset[1], anchor_pos[2] + offset[2]]
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
if near_n is not None:
    add_whole_leader(near_n, "N", [-13.0, 11.0, 7.0])
python end

orient (mol or box_corner_pts)
turn y, 18
turn x, -12
zoom (mol or box_corner_pts), 16
move y, -6
set suspend_updates, off
png {render.pymol_path(out_png)}, width=3400, height=2300, dpi=650, ray=1
quit
"""


def make_whole_system_figure() -> None:
    out_png = THIS_DIR / "_tmp_whole.png"
    with tempfile.TemporaryDirectory(prefix="md_whole_", dir="/tmp") as td:
        bonded_pdb = Path(td) / WHOLE_SYSTEM_PDB.name
        write_bonded_pdb(WHOLE_SYSTEM_PDB, bonded_pdb)
        script = build_whole_system_pymol_script(bonded_pdb, out_png)
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

    fig = plt.figure(figsize=(12.5, 8.8), dpi=500)
    ax = fig.add_subplot(111)
    ax.imshow(whole_img)
    ax.set_axis_off()
    ax.set_title("Full System Representation", pad=10, fontweight="bold")
    fig.savefig(OUT_WHOLE, bbox_inches="tight")
    FINAL_DIR.mkdir(exist_ok=True)
    fig.savefig(FINAL_WHOLE, bbox_inches="tight")
    fig.savefig(FINAL_WHOLE_NUMBERED, bbox_inches="tight")
    plt.close(fig)
    out_png.unlink(missing_ok=True)


def add_panel_letter(image: Image.Image, letter: str, font_size: int = 150) -> Image.Image:
    panel = image.copy()
    draw = ImageDraw.Draw(panel)
    font = load_font(font_size)
    margin = max(55, font_size // 2)
    draw.text((margin, margin), letter, fill="black", font=font)
    return panel


def make_combined_md_figure() -> None:
    snap = Image.open(FINAL_STRIP).convert("RGB")
    whole = Image.open(FINAL_WHOLE).convert("RGB")

    target_w = max(snap.width, whole.width)
    snap = ImageOps.contain(snap, (target_w, int(target_w * snap.height / snap.width)), Image.Resampling.LANCZOS)
    whole = ImageOps.contain(whole, (target_w, int(target_w * whole.height / whole.width)), Image.Resampling.LANCZOS)

    snap = add_panel_letter(snap, "A")
    whole = add_panel_letter(whole, "B")

    gap = 80
    canvas = Image.new("RGB", (target_w, snap.height + gap + whole.height), "white")
    canvas.paste(snap, ((target_w - snap.width) // 2, 0))
    canvas.paste(whole, ((target_w - whole.width) // 2, snap.height + gap))
    canvas.save(FINAL_STRIP_NUMBERED)


if __name__ == "__main__":
    require_inputs([*SNAPSHOT_PDBS, WHOLE_SYSTEM_PDB])
    make_snapshot_strip()
    make_whole_system_figure()
    make_combined_md_figure()
    print("Structural MD figures generated.")
    print(f"Outputs:\n  - {OUT_STRIP}\n  - {OUT_WHOLE}\n  - {FINAL_STRIP}\n  - {FINAL_WHOLE}")
