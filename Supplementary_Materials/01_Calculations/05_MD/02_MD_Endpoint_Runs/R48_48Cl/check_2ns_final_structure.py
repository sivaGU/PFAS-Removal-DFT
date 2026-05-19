from pathlib import Path
import math

import parmed as pmd


top_path = Path("../04_leap/solvated_resin_cl.prmtop")
pdb_path = Path("resin_only_npt_2ns_final.pdb")
report_path = Path("../logs/resin_only_2ns_structure_check.txt")

top = pmd.load_file(str(top_path))

coords = []
box = None
for line in pdb_path.read_text().splitlines():
    if line.startswith("CRYST1"):
        box = (
            float(line[6:15]),
            float(line[15:24]),
            float(line[24:33]),
        )
    elif line.startswith(("ATOM", "HETATM")):
        coords.append(
            (
                float(line[30:38]),
                float(line[38:46]),
                float(line[46:54]),
            )
        )

if len(coords) != len(top.atoms):
    raise SystemExit(f"atom count mismatch: PDB {len(coords)} vs topology {len(top.atoms)}")
if box is None:
    raise SystemExit("CRYST1 box missing from final PDB")


def min_image_delta(a, b):
    dx = [a[i] - b[i] for i in range(3)]
    for i, length in enumerate(box):
        if length > 0:
            dx[i] -= round(dx[i] / length) * length
    return dx


def dist(a, b):
    dx = min_image_delta(a, b)
    return math.sqrt(sum(v * v for v in dx))


max_bond = (0.0, None)
max_resin_bond = (0.0, None)
long_bonds = []
for bond in top.bonds:
    i = bond.atom1.idx
    j = bond.atom2.idx
    d = dist(coords[i], coords[j])
    if d > max_bond[0]:
        max_bond = (d, (i + 1, j + 1, bond.atom1.name, bond.atom2.name))
    if i < 1644 and j < 1644 and d > max_resin_bond[0]:
        max_resin_bond = (d, (i + 1, j + 1, bond.atom1.name, bond.atom2.name))
    if d > 2.2:
        long_bonds.append((d, i + 1, j + 1, bond.atom1.name, bond.atom2.name))

n4 = [atom.idx for atom in top.atoms if atom.type == "n4" and atom.residue.name == "R48"]
cl = [atom.idx for atom in top.atoms if atom.residue.name == "Cl-"]

n_c_dists = []
for ni in n4:
    atom = top.atoms[ni]
    for bonded in atom.bond_partners:
        if bonded.atomic_number == 6:
            n_c_dists.append(dist(coords[ni], coords[bonded.idx]))

nearest_cl_n = []
for ci in cl:
    nearest = min((dist(coords[ci], coords[ni]), ci + 1, ni + 1, top.atoms[ni].name) for ni in n4)
    nearest_cl_n.append(nearest)
nearest_cl_n.sort()

lines = []
lines.append("Resin-only 2 ns final-frame structure check")
lines.append(f"atoms: {len(coords)}")
lines.append(f"box: {box[0]:.3f} {box[1]:.3f} {box[2]:.3f} A")
lines.append(f"quaternary nitrogens: {len(n4)}")
lines.append(f"chlorides: {len(cl)}")
lines.append(f"max bonded distance: {max_bond[0]:.3f} A atoms {max_bond[1]}")
lines.append(f"max R48 bonded distance: {max_resin_bond[0]:.3f} A atoms {max_resin_bond[1]}")
lines.append(f"bonds longer than 2.2 A using minimum image: {len(long_bonds)}")
if n_c_dists:
    lines.append(
        "quaternary N-C bond range: "
        f"{min(n_c_dists):.3f}-{max(n_c_dists):.3f} A over {len(n_c_dists)} bonds"
    )
if nearest_cl_n:
    vals = [x[0] for x in nearest_cl_n]
    median = vals[len(vals) // 2]
    lines.append(
        "chloride nearest-N distance range/median: "
        f"{min(vals):.3f}/{median:.3f}/{max(vals):.3f} A"
    )
    lines.append("closest 10 chloride-to-N contacts:")
    for d, ci, ni, nname in nearest_cl_n[:10]:
        lines.append(f"  Cl atom {ci:5d} to {nname:>4s} atom {ni:5d}: {d:.3f} A")
if long_bonds[:10]:
    lines.append("first long bonded distances:")
    for d, i, j, ni, nj in long_bonds[:10]:
        lines.append(f"  {i:5d}-{j:5d} {ni}-{nj}: {d:.3f} A")

report_path.write_text("\n".join(lines) + "\n")
print("\n".join(lines))
