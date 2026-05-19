from pathlib import Path
import math

import numpy as np
import parmed as pmd


TOP = Path("../05_pfas_exchange/solvated_resin_pfoa_exchange.prmtop")
PDB = Path("resin_pfoa_exchange_npt_short_final_imaged.pdb")
REPORT = Path("../logs/resin_pfoa_exchange_final_structure_check.txt")


def load_pdb(path):
    atoms = []
    box = None
    for line in path.read_text().splitlines():
        if line.startswith("CRYST1"):
            box = (
                float(line[6:15]),
                float(line[15:24]),
                float(line[24:33]),
            )
        elif line.startswith(("ATOM", "HETATM")):
            elem = line[76:78].strip().upper()
            atoms.append(
                {
                    "serial": int(line[6:11]),
                    "name": line[12:16].strip(),
                    "resname": line[17:20].strip(),
                    "resid": int(line[22:26]),
                    "coord": np.array(
                        [
                            float(line[30:38]),
                            float(line[38:46]),
                            float(line[46:54]),
                        ],
                        dtype=float,
                    ),
                    "elem": elem,
                }
            )
    if box is None:
        raise SystemExit("missing CRYST1 box")
    return atoms, box


def min_image_delta(a, b, box):
    dx = a - b
    for i, length in enumerate(box):
        dx[i] -= round(dx[i] / length) * length
    return dx


def dist(a, b, box):
    dx = min_image_delta(a, b, box)
    return float(math.sqrt(np.dot(dx, dx)))


parm = pmd.load_file(str(TOP))
atoms, box = load_pdb(PDB)
if len(atoms) != len(parm.atoms):
    raise SystemExit(f"atom count mismatch: PDB {len(atoms)} vs topology {len(parm.atoms)}")

coords = [a["coord"] for a in atoms]
res_counts = {}
for a in atoms:
    res_counts[a["resname"]] = res_counts.get(a["resname"], 0) + 1

max_bond = (0.0, None)
long_bonds = []
for bond in parm.bonds:
    i, j = bond.atom1.idx, bond.atom2.idx
    d = dist(coords[i], coords[j], box)
    if d > max_bond[0]:
        max_bond = (d, (i + 1, j + 1, bond.atom1.name, bond.atom2.name))
    if d > 2.2:
        long_bonds.append((d, i + 1, j + 1, bond.atom1.name, bond.atom2.name))

n4_idx = [a.idx for a in parm.atoms if a.residue.name == "R48" and a.type == "n4"]
pfo_idx = [i for i, a in enumerate(atoms) if a["resname"] == "PFO"]
pfo_o_idx = [i for i in pfo_idx if atoms[i]["elem"] == "O"]
pfo_f_idx = [i for i in pfo_idx if atoms[i]["elem"] == "F"]
cl_idx = [i for i, a in enumerate(atoms) if a["resname"] == "Cl-"]
water_o_idx = [i for i, a in enumerate(atoms) if a["resname"] == "WAT" and a["name"] == "O"]
resin_heavy_idx = [
    i for i, a in enumerate(atoms)
    if a["resname"] == "R48" and a["elem"] != "H"
]

head_n_contacts = []
for oi in pfo_o_idx:
    nearest = min((dist(coords[oi], coords[ni], box), oi, ni) for ni in n4_idx)
    head_n_contacts.append(nearest)
head_n_contacts.sort()

pfo_heavy = [i for i in pfo_idx if atoms[i]["elem"] != "H"]
resin_contacts = []
for pi in pfo_heavy:
    nearest = min((dist(coords[pi], coords[ri], box), pi, ri) for ri in resin_heavy_idx)
    resin_contacts.append(nearest)
resin_contacts.sort()

cl_n_dist = []
for ci in cl_idx:
    nearest = min((dist(coords[ci], coords[ni], box), ci, ni) for ni in n4_idx)
    cl_n_dist.append(nearest)
cl_n_dist.sort()

water_pfo_min = None
if water_o_idx and pfo_heavy:
    water_pfo_min = min(
        dist(coords[wi], coords[pi], box)
        for wi in water_o_idx
        for pi in pfo_heavy
    )

lines = []
lines.append("PFOA exchange final-frame structure check")
lines.append(f"atoms: {len(atoms)}")
lines.append(f"box: {box[0]:.3f} {box[1]:.3f} {box[2]:.3f} A")
lines.append(
    "residue atom counts: "
    + ", ".join(f"{k}: {v}" for k, v in sorted(res_counts.items()))
)
lines.append(f"R48 quaternary nitrogens: {len(n4_idx)}")
lines.append(f"PFO atoms: {len(pfo_idx)}; PFO oxygens: {len(pfo_o_idx)}; PFO fluorines: {len(pfo_f_idx)}")
lines.append(f"chlorides: {len(cl_idx)}")
lines.append(f"waters: {len(water_o_idx)}")
lines.append(f"max bonded distance: {max_bond[0]:.3f} A atoms {max_bond[1]}")
lines.append(f"bonds longer than 2.2 A using minimum image: {len(long_bonds)}")
if long_bonds:
    for d, i, j, ni, nj in long_bonds[:10]:
        lines.append(f"  long bond {i}-{j} {ni}-{nj}: {d:.3f} A")
lines.append("PFO oxygen nearest ammonium-N distances:")
for d, oi, ni in head_n_contacts:
    lines.append(f"  {atoms[oi]['name']:>4s} atom {oi + 1:5d} to {parm.atoms[ni].name:>4s} atom {ni + 1:5d}: {d:.3f} A")
lines.append("PFO heavy atom nearest resin-heavy contacts, closest 8:")
for d, pi, ri in resin_contacts[:8]:
    lines.append(
        f"  {atoms[pi]['name']:>4s} atom {pi + 1:5d} to "
        f"{atoms[ri]['name']:>4s} atom {ri + 1:5d}: {d:.3f} A"
    )
if cl_n_dist:
    vals = [x[0] for x in cl_n_dist]
    lines.append(
        "chloride nearest-N distance range/median: "
        f"{min(vals):.3f}/{vals[len(vals)//2]:.3f}/{max(vals):.3f} A"
    )
if water_pfo_min is not None:
    lines.append(f"nearest water-O to PFO heavy atom: {water_pfo_min:.3f} A")

REPORT.write_text("\n".join(lines) + "\n")
print("\n".join(lines))
