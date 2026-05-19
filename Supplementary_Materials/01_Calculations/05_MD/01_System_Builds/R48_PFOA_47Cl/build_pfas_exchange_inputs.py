from pathlib import Path
import math

import numpy as np


ROOT = Path("../..").resolve()
FINAL_PDB = ROOT / "work_amber/07_namd_resin_only/resin_only_npt_2ns_final_imaged.pdb"
PARAM = ROOT / "work_amber/03_param"
LOG = ROOT / "work_amber/logs"

PFAS = {
    "pfoa": {
        "resname": "PFO",
        "mol2": PARAM / "pfoa_gaff2.mol2",
        "frcmod": "../03_param/pfoa.frcmod",
        "leap_mol2": "../03_param/pfoa_gaff2.mol2",
        "prefix": "resin_pfoa_exchange",
    },
    "pfos": {
        "resname": "PFS",
        "mol2": PARAM / "pfos_gaff2.mol2",
        "frcmod": "../03_param/pfos.frcmod",
        "leap_mol2": "../03_param/pfos_gaff2.mol2",
        "prefix": "resin_pfos_exchange",
    },
}


def element_from_name_type(name, atom_type):
    letters = "".join(c for c in name if c.isalpha()).upper()
    at = atom_type.lower()
    if at.startswith("cl") or letters.startswith("CL"):
        return "CL"
    if at.startswith("br") or letters.startswith("BR"):
        return "BR"
    if at.startswith("s") or letters.startswith("S"):
        return "S"
    if at.startswith("o") or letters.startswith("O"):
        return "O"
    if at.startswith("n") or letters.startswith("N"):
        return "N"
    if at.startswith("c") or letters.startswith("C"):
        return "C"
    if at.startswith("h") or letters.startswith("H"):
        return "H"
    if at.startswith("f") or letters.startswith("F"):
        return "F"
    return letters[:2] if len(letters) > 1 else letters[:1]


def parse_pdb(path):
    atoms = []
    for line in path.read_text().splitlines():
        if not line.startswith(("ATOM", "HETATM")):
            continue
        name = line[12:16].strip()
        resname = line[17:20].strip()
        resid = int(line[22:26])
        elem = line[76:78].strip().upper()
        if not elem:
            elem = element_from_name_type(name, name)
        atoms.append(
            {
                "serial": int(line[6:11]),
                "name": name,
                "resname": resname,
                "resid": resid,
                "coord": np.array(
                    [float(line[30:38]), float(line[38:46]), float(line[46:54])],
                    dtype=float,
                ),
                "elem": elem,
            }
        )
    return atoms


def parse_mol2(path):
    lines = path.read_text().splitlines()
    atom_start = atom_end = bond_start = bond_end = None
    for i, line in enumerate(lines):
        if line.startswith("@<TRIPOS>ATOM"):
            atom_start = i + 1
        elif line.startswith("@<TRIPOS>BOND"):
            atom_end = i
            bond_start = i + 1
        elif bond_start is not None and line.startswith("@<TRIPOS>"):
            bond_end = i
            break
    if atom_start is None or atom_end is None or bond_start is None:
        raise ValueError(f"cannot parse mol2 sections in {path}")
    if bond_end is None:
        bond_end = len(lines)

    atoms = []
    for line in lines[atom_start:atom_end]:
        parts = line.split()
        atoms.append(
            {
                "id": int(parts[0]),
                "name": parts[1],
                "coord": np.array([float(parts[2]), float(parts[3]), float(parts[4])]),
                "type": parts[5],
                "charge": float(parts[8]),
                "elem": element_from_name_type(parts[1], parts[5]),
            }
        )
    bonds = []
    nbr = {a["id"] - 1: [] for a in atoms}
    for line in lines[bond_start:bond_end]:
        parts = line.split()
        if len(parts) < 4:
            continue
        i, j = int(parts[1]) - 1, int(parts[2]) - 1
        bonds.append((i, j, parts[3]))
        nbr[i].append(j)
        nbr[j].append(i)
    return atoms, bonds, nbr


def head_and_tail(mol_atoms, nbr):
    oxy = [i for i, a in enumerate(mol_atoms) if a["elem"] == "O"]
    sulfurs = [i for i, a in enumerate(mol_atoms) if a["elem"] == "S"]
    carbons = [i for i, a in enumerate(mol_atoms) if a["elem"] == "C"]
    if sulfurs:
        core = sulfurs[0]
        head_atoms = [i for i in nbr[core] if mol_atoms[i]["elem"] == "O"]
    else:
        core = None
        head_atoms = []
        for ci in carbons:
            o_nbrs = [j for j in nbr[ci] if mol_atoms[j]["elem"] == "O"]
            if len(o_nbrs) >= 2:
                core = ci
                head_atoms = o_nbrs
                break
    if core is None or len(head_atoms) < 2:
        raise ValueError("could not identify PFAS anionic head group")

    head_center = np.mean([mol_atoms[i]["coord"] for i in head_atoms], axis=0)
    tail = max(carbons, key=lambda i: np.linalg.norm(mol_atoms[i]["coord"] - head_center))
    tail_vec = mol_atoms[tail]["coord"] - head_center
    return core, head_atoms, head_center, tail, tail_vec


def unit(v):
    n = np.linalg.norm(v)
    if n < 1.0e-12:
        raise ValueError("zero-length vector")
    return v / n


def rotation_from_to(a, b):
    a = unit(a)
    b = unit(b)
    v = np.cross(a, b)
    c = float(np.dot(a, b))
    if c > 0.999999:
        return np.eye(3)
    if c < -0.999999:
        axis = np.cross(a, np.array([1.0, 0.0, 0.0]))
        if np.linalg.norm(axis) < 1.0e-8:
            axis = np.cross(a, np.array([0.0, 1.0, 0.0]))
        return rotation_about_axis(axis, math.pi)
    vx = np.array(
        [
            [0.0, -v[2], v[1]],
            [v[2], 0.0, -v[0]],
            [-v[1], v[0], 0.0],
        ]
    )
    return np.eye(3) + vx + vx @ vx * (1.0 / (1.0 + c))


def rotation_about_axis(axis, theta):
    axis = unit(axis)
    x, y, z = axis
    c = math.cos(theta)
    s = math.sin(theta)
    C = 1.0 - c
    return np.array(
        [
            [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
            [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
            [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
        ]
    )


def min_distance(a, b):
    diff = a[:, None, :] - b[None, :, :]
    d2 = np.sum(diff * diff, axis=2)
    return float(np.sqrt(np.min(d2)))


def format_pdb_atom(serial, name, resname, resid, coord, elem, record="ATOM"):
    return (
        f"{record:<6s}{serial:5d} {name:<4.4s} {resname:>3.3s} {resid:4d}"
        f"    {coord[0]:8.3f}{coord[1]:8.3f}{coord[2]:8.3f}"
        f"  1.00  0.00          {elem:>2.2s}"
    )


def write_residue_pdb(path, atoms):
    lines = []
    for serial, atom in enumerate(atoms, 1):
        lines.append(
            format_pdb_atom(
                serial,
                atom["name"],
                atom["resname"],
                atom["resid"],
                atom["coord"],
                atom["elem"],
            )
        )
    lines.append("END")
    path.write_text("\n".join(lines) + "\n")


all_atoms = parse_pdb(FINAL_PDB)
resin = [a for a in all_atoms if a["resname"] == "R48"]
chlorides = [a for a in all_atoms if a["resname"] == "Cl-"]
if len(resin) != 1644 or len(chlorides) != 48:
    raise SystemExit(f"unexpected final-frame composition: R48={len(resin)} Cl={len(chlorides)}")

n_atoms = [a for a in resin if a["elem"] == "N"]
resin_heavy = np.array([a["coord"] for a in resin if a["elem"] != "H"])
resin_and_cl_heavy_by_candidate = {}
resin_com = np.mean(resin_heavy, axis=0)

pfas_data = {}
for key, spec in PFAS.items():
    mol_atoms, bonds, nbr = parse_mol2(spec["mol2"])
    core, head_atoms, head_center, tail, tail_vec = head_and_tail(mol_atoms, nbr)
    pfas_data[key] = {
        "atoms": mol_atoms,
        "head_atoms": head_atoms,
        "head_center": head_center,
        "tail_vec": tail_vec,
        "coords": np.array([a["coord"] for a in mol_atoms]),
        "heavy_mask": np.array([a["elem"] != "H" for a in mol_atoms], dtype=bool),
    }

candidate_reports = []
placements = {}
for cl_idx, cl_atom in enumerate(chlorides):
    n_d = [(np.linalg.norm(cl_atom["coord"] - n["coord"]), n) for n in n_atoms]
    nearest_d, nearest_n = min(n_d, key=lambda x: x[0])
    if not 3.5 <= nearest_d <= 7.0:
        continue
    crowd = int(np.sum(np.linalg.norm(resin_heavy - cl_atom["coord"], axis=1) < 5.0))
    radial = float(np.linalg.norm(cl_atom["coord"] - resin_com))
    direction = unit(cl_atom["coord"] - nearest_n["coord"])
    head_target = nearest_n["coord"] + direction * min(nearest_d, 4.8)
    remaining_cl = np.array(
        [a["coord"] for i, a in enumerate(chlorides) if i != cl_idx],
        dtype=float,
    )
    environment = np.vstack([resin_heavy, remaining_cl])

    per_pfas = {}
    for key, data in pfas_data.items():
        centered = data["coords"] - data["head_center"]
        align = rotation_from_to(data["tail_vec"], direction)
        aligned = centered @ align.T
        best = None
        for deg in range(0, 360, 15):
            axial = rotation_about_axis(direction, math.radians(deg))
            placed = aligned @ axial.T + head_target
            heavy = placed[data["heavy_mask"]]
            closest = min_distance(heavy, environment)
            head_to_n = [
                float(np.linalg.norm(placed[i] - nearest_n["coord"]))
                for i in data["head_atoms"]
            ]
            score = closest - 0.03 * crowd + 0.01 * radial - 0.10 * abs(nearest_d - 5.0)
            trial = {
                "coords": placed,
                "angle": deg,
                "closest": closest,
                "head_to_n": head_to_n,
                "score": score,
            }
            if best is None or trial["score"] > best["score"]:
                best = trial
        per_pfas[key] = best

    combined_clash = min(v["closest"] for v in per_pfas.values())
    combined_score = combined_clash + 0.01 * radial - 0.03 * crowd - 0.10 * abs(nearest_d - 5.0)
    candidate_reports.append(
        {
            "cl_idx": cl_idx,
            "cl_atom": cl_atom,
            "nearest_n": nearest_n,
            "nearest_d": nearest_d,
            "head_target": head_target,
            "crowd": crowd,
            "radial": radial,
            "score": combined_score,
            "per_pfas": per_pfas,
        }
    )

if not candidate_reports:
    raise SystemExit("no chloride within 3.5-7.0 A of a quaternary N in the imaged final frame")

candidate_reports.sort(key=lambda x: x["score"], reverse=True)
selected = candidate_reports[0]
if min(v["closest"] for v in selected["per_pfas"].values()) < 1.6:
    raise SystemExit("best automated PFAS pose has a heavy-atom contact below 1.6 A; ask user before building")

report_lines = []
report_lines.append("PFAS exchange coordinate build")
report_lines.append(f"source final frame: {FINAL_PDB}")
report_lines.append(
    "removed chloride: "
    f"atom serial {selected['cl_atom']['serial']} residue {selected['cl_atom']['resid']} "
    f"coord {selected['cl_atom']['coord'][0]:.3f} {selected['cl_atom']['coord'][1]:.3f} {selected['cl_atom']['coord'][2]:.3f}"
)
report_lines.append(
    "nearest ammonium N: "
    f"{selected['nearest_n']['name']} atom serial {selected['nearest_n']['serial']} "
    f"distance {selected['nearest_d']:.3f} A"
)
report_lines.append(f"local heavy-atom crowding within 5 A: {selected['crowd']}")
report_lines.append(f"radial distance from resin heavy-atom COM: {selected['radial']:.3f} A")
report_lines.append(
    "PFAS headgroup target: "
    f"{selected['head_target'][0]:.3f} {selected['head_target'][1]:.3f} {selected['head_target'][2]:.3f} "
    f"({np.linalg.norm(selected['head_target'] - selected['cl_atom']['coord']):.3f} A from removed chloride)"
)
report_lines.append("top chloride candidates:")
for cand in candidate_reports[:10]:
    detail = ", ".join(
        f"{key} min_contact {val['closest']:.3f} A angle {val['angle']}"
        for key, val in cand["per_pfas"].items()
    )
    report_lines.append(
        f"  Cl serial {cand['cl_atom']['serial']:5d} near {cand['nearest_n']['name']:>4s} "
        f"{cand['nearest_d']:.3f} A crowd {cand['crowd']:2d} radial {cand['radial']:.2f}: {detail}"
    )

remaining_chlorides = [
    a for i, a in enumerate(chlorides) if i != selected["cl_idx"]
]

for key, spec in PFAS.items():
    prefix = spec["prefix"]
    placed = selected["per_pfas"][key]["coords"]
    mol_atoms = pfas_data[key]["atoms"]

    pfas_atoms = []
    for atom, coord in zip(mol_atoms, placed):
        pfas_atoms.append(
            {
                "name": atom["name"],
                "resname": spec["resname"],
                "resid": 2,
                "coord": coord,
                "elem": atom["elem"],
            }
        )

    cl_atoms = []
    for offset, atom in enumerate(remaining_chlorides, start=3):
        cl_atoms.append(
            {
                "name": "Cl-",
                "resname": "Cl-",
                "resid": offset,
                "coord": atom["coord"],
                "elem": "CL",
            }
        )

    combined = []
    for atom in resin:
        new_atom = dict(atom)
        new_atom["resid"] = 1
        combined.append(new_atom)
    combined.extend(pfas_atoms)
    combined.extend(cl_atoms)

    write_residue_pdb(Path(f"{prefix}_pfas_placed.pdb"), pfas_atoms)
    write_residue_pdb(Path(f"{prefix}_47cl.pdb"), cl_atoms)
    write_residue_pdb(Path(f"{prefix}_start.pdb"), combined)

    report_lines.append(
        f"{key.upper()} placement: best rotation {selected['per_pfas'][key]['angle']} deg, "
        f"min heavy contact {selected['per_pfas'][key]['closest']:.3f} A, "
        "head-O/SO distances to N "
        + ", ".join(f"{d:.3f}" for d in selected["per_pfas"][key]["head_to_n"])
        + " A"
    )

    leap = f"""source leaprc.gaff2
source leaprc.water.tip3p
loadamberparams frcmod.ionsjc_tip3p

loadamberparams ../03_param/chol48.frcmod
loadamberparams {spec['frcmod']}

R48 = loadmol2 ../03_param/chol48_gaff2.mol2
{spec['resname']} = loadmol2 {spec['leap_mol2']}

SYS = loadpdb {prefix}_start.pdb

check SYS
charge SYS

saveamberparm SYS dry_{prefix}.prmtop dry_{prefix}.inpcrd
savepdb SYS dry_{prefix}.pdb

solvatebox SYS TIP3PBOX 15.0

check SYS
charge SYS

saveamberparm SYS solvated_{prefix}.prmtop solvated_{prefix}.inpcrd
savepdb SYS solvated_{prefix}.pdb

quit
"""
    Path(f"build_{prefix}.leap").write_text(leap)

Path("pfas_exchange_build_report.txt").write_text("\n".join(report_lines) + "\n")
(LOG / "pfas_exchange_build_report.txt").write_text("\n".join(report_lines) + "\n")
print("\n".join(report_lines))
