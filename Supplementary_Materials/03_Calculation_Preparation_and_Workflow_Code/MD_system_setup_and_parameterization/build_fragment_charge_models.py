import json
from collections import Counter, defaultdict
from pathlib import Path

from rdkit import Chem
from rdkit.Geometry import Point3D


SOURCE = Path("chol48_resin_only.sdf")
OUTDIR = Path("models")
REPORT = Path("../logs/fragment_model_build_report.txt")
MAP_JSON = Path("models/fragment_model_map.json")


def heavy_neighbors(mol, idx):
    return [n.GetIdx() for n in mol.GetAtomWithIdx(idx).GetNeighbors() if n.GetSymbol() != "H"]


def h_neighbors(mol, idx):
    return [n.GetIdx() for n in mol.GetAtomWithIdx(idx).GetNeighbors() if n.GetSymbol() == "H"]


def ring_path_roles(ring, ring_alpha, ring_benzyl, mol):
    graph = defaultdict(list)
    for atom in ring:
        for nbr in heavy_neighbors(mol, atom):
            if nbr in ring:
                graph[atom].append(nbr)

    paths = []

    def walk(path):
        cur = path[-1]
        if cur == ring_benzyl:
            paths.append(path[:])
            return
        for nxt in graph[cur]:
            if nxt in path:
                continue
            walk(path + [nxt])

    walk([ring_alpha])
    paths = [p for p in paths if len(p) == 4]
    if len(paths) != 2:
        raise RuntimeError(f"Expected two para ring paths, got {paths}")

    ortho = sorted([paths[0][1], paths[1][1]])
    meta = sorted([paths[0][2], paths[1][2]])
    return {
        "ar_alpha": [ring_alpha],
        "ar_ortho": ortho,
        "ar_meta": meta,
        "ar_benzyl": [ring_benzyl],
    }


def identify_units(mol):
    rings = [set(r) for r in mol.GetRingInfo().AtomRings()]
    units = []
    pendant_rings = []

    for n_atom in [a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() == "N"]:
        nbrs = heavy_neighbors(mol, n_atom)
        methyl = [
            idx
            for idx in nbrs
            if mol.GetAtomWithIdx(idx).GetSymbol() == "C" and len(heavy_neighbors(mol, idx)) == 1
        ]
        benzyl = [idx for idx in nbrs if idx not in methyl]
        if len(methyl) != 3 or len(benzyl) != 1:
            raise RuntimeError(f"Could not identify N substituents for atom {n_atom + 1}")
        benzyl = benzyl[0]
        ring_atom = [idx for idx in heavy_neighbors(mol, benzyl) if mol.GetAtomWithIdx(idx).GetIsAromatic()]
        if len(ring_atom) != 1:
            raise RuntimeError(f"Could not identify benzyl ring atom for atom {benzyl + 1}")
        ring_atom = ring_atom[0]
        ring = next(r for r in rings if ring_atom in r)

        exocyclic = []
        for atom in ring:
            for nbr in heavy_neighbors(mol, atom):
                if nbr not in ring:
                    exocyclic.append((atom, nbr))
        alpha_pairs = [
            (atom, nbr)
            for atom, nbr in exocyclic
            if nbr != benzyl and not mol.GetAtomWithIdx(nbr).GetIsAromatic()
        ]
        if len(alpha_pairs) != 1:
            raise RuntimeError(f"Could not identify backbone alpha atom for N {n_atom + 1}")
        ring_alpha, alpha = alpha_pairs[0]

        ring_benzyl = next(atom for atom, nbr in exocyclic if nbr == benzyl)
        ring_roles = ring_path_roles(ring, ring_alpha, ring_benzyl, mol)
        backbone = [
            idx
            for idx in heavy_neighbors(mol, alpha)
            if not mol.GetAtomWithIdx(idx).GetIsAromatic()
        ]
        if len(backbone) != 2:
            raise RuntimeError(f"Expected two backbone C neighbors for alpha {alpha + 1}")

        heavy_core = set([n_atom, benzyl, alpha, *methyl, *ring, *backbone])
        units.append(
            {
                "N": n_atom,
                "methyl": sorted(methyl),
                "benzyl": benzyl,
                "ring": sorted(ring),
                "ring_roles": ring_roles,
                "alpha": alpha,
                "backbone": sorted(backbone),
                "heavy_core": sorted(heavy_core),
            }
        )
        pendant_rings.append(ring)

    pendant_union = set().union(*pendant_rings)
    crosslink_rings = [sorted(r) for r in rings if r.isdisjoint(pendant_union)]
    if len(crosslink_rings) != 1:
        raise RuntimeError(f"Expected one crosslink ring, got {crosslink_rings}")
    crosslink_ring = crosslink_rings[0]

    owner = {}
    for unit_idx, unit in enumerate(units):
        for atom in unit["backbone"]:
            owner[atom] = unit_idx

    adjacency = defaultdict(set)
    for bond in mol.GetBonds():
        a = bond.GetBeginAtomIdx()
        b = bond.GetEndAtomIdx()
        if a in owner and b in owner and owner[a] != owner[b]:
            adjacency[owner[a]].add(owner[b])
            adjacency[owner[b]].add(owner[a])

    crosslink_set = set(crosslink_ring)
    cross_units = [
        idx
        for idx, unit in enumerate(units)
        if any(nbr in crosslink_set for nbr in heavy_neighbors(mol, unit["alpha"]))
    ]
    terminals = [
        idx
        for idx, unit in enumerate(units)
        if any(len(heavy_neighbors(mol, atom)) == 1 for atom in unit["backbone"])
    ]
    adjacent = sorted(set().union(*(adjacency[idx] for idx in cross_units)) - set(cross_units))
    categories = {}
    for idx in range(len(units)):
        if idx in cross_units:
            categories[idx] = "crosslink"
        elif idx in terminals:
            categories[idx] = "terminal"
        elif idx in adjacent:
            categories[idx] = "adjacent"
        else:
            categories[idx] = "internal"

    return units, crosslink_ring, adjacency, categories


def unit_heavy(units, ids):
    out = set()
    for idx in ids:
        out.update(units[idx]["heavy_core"])
    return out


def unit_core_with_hydrogens(mol, unit):
    atoms = set(unit["heavy_core"])
    for heavy in unit["heavy_core"]:
        atoms.update(h_neighbors(mol, heavy))
    return sorted(atoms)


def build_heavy_model(mol, selected_heavy, name):
    selected_heavy = sorted(selected_heavy)
    old_to_new = {old: new for new, old in enumerate(selected_heavy)}
    rw = Chem.RWMol()
    for old in selected_heavy:
        old_atom = mol.GetAtomWithIdx(old)
        atom = Chem.Atom(old_atom.GetSymbol())
        atom.SetFormalCharge(old_atom.GetFormalCharge())
        atom.SetIsAromatic(old_atom.GetIsAromatic())
        atom.SetNoImplicit(False)
        rw.AddAtom(atom)

    for bond in mol.GetBonds():
        a = bond.GetBeginAtomIdx()
        b = bond.GetEndAtomIdx()
        if a in old_to_new and b in old_to_new:
            rw.AddBond(old_to_new[a], old_to_new[b], bond.GetBondType())
            new_bond = rw.GetBondBetweenAtoms(old_to_new[a], old_to_new[b])
            new_bond.SetIsAromatic(bond.GetIsAromatic())

    heavy_mol = rw.GetMol()
    Chem.SanitizeMol(heavy_mol)

    conf = Chem.Conformer(len(selected_heavy))
    src_conf = mol.GetConformer()
    for new, old in enumerate(selected_heavy):
        pos = src_conf.GetAtomPosition(old)
        conf.SetAtomPosition(new, Point3D(pos.x, pos.y, pos.z))
    heavy_mol.AddConformer(conf, assignId=True)

    with_h = Chem.AddHs(heavy_mol, addCoords=True)
    with_h.SetProp("_Name", name)
    for atom in with_h.GetAtoms():
        if atom.GetIdx() < len(selected_heavy):
            atom.SetProp("full_atom_index", str(selected_heavy[atom.GetIdx()] + 1))
    return with_h, selected_heavy


def write_model(mol, selected_heavy, name, model_charge, core_groups):
    OUTDIR.mkdir(parents=True, exist_ok=True)
    sdf = OUTDIR / f"{name}.sdf"
    writer = Chem.SDWriter(str(sdf))
    writer.write(mol)
    writer.close()
    return {
        "sdf": str(sdf),
        "charge": model_charge,
        "heavy_full_indices_1based": [idx + 1 for idx in selected_heavy],
        "n_atoms": mol.GetNumAtoms(),
        "n_heavy": len(selected_heavy),
        "core_groups": core_groups,
    }


def main():
    mol = Chem.MolFromMolFile(str(SOURCE), removeHs=False, sanitize=True)
    if mol is None:
        raise SystemExit(f"Could not read {SOURCE}")

    units, crosslink_ring, adjacency, categories = identify_units(mol)
    category_members = defaultdict(list)
    for idx, category in categories.items():
        category_members[category].append(idx)

    internal_rep = next(
        idx
        for idx in category_members["internal"]
        if all(categories[nbr] == "internal" for nbr in adjacency[idx])
    )
    terminal_rep = category_members["terminal"][0]
    adjacent_rep = category_members["adjacent"][0]

    internal_units = sorted([internal_rep, *adjacency[internal_rep]])
    terminal_units = sorted([terminal_rep, *adjacency[terminal_rep]])
    adjacent_units = set([adjacent_rep, *adjacency[adjacent_rep]])
    adjacent_cross = next(idx for idx in adjacency[adjacent_rep] if categories[idx] == "crosslink")
    adjacent_units.update(category_members["crosslink"])
    for idx in list(adjacent_units):
        if categories[idx] == "crosslink":
            adjacent_units.update(adjacency[idx])
    adjacent_units = sorted(adjacent_units)

    cross_units = set(category_members["crosslink"])
    for idx in category_members["crosslink"]:
        cross_units.update(adjacency[idx])
    cross_units = sorted(cross_units)

    models = {}
    model_specs = [
        (
            "model_internal",
            unit_heavy(units, internal_units),
            len(internal_units),
            {"internal_template_unit": [internal_rep]},
        ),
        (
            "model_terminal",
            unit_heavy(units, terminal_units),
            len(terminal_units),
            {"terminal_template_unit": [terminal_rep]},
        ),
        (
            "model_adjacent",
            unit_heavy(units, adjacent_units).union(crosslink_ring),
            len(adjacent_units),
            {"adjacent_template_unit": [adjacent_rep]},
        ),
        (
            "model_crosslink",
            unit_heavy(units, cross_units).union(crosslink_ring),
            len(cross_units),
            {
                "crosslink_units": category_members["crosslink"],
                "crosslink_ring": [idx + 1 for idx in crosslink_ring],
            },
        ),
    ]

    for name, heavy, charge, core_groups in model_specs:
        model_mol, selected = build_heavy_model(mol, heavy, name)
        models[name] = write_model(model_mol, selected, name, charge, core_groups)

    metadata = {
        "source": str(SOURCE),
        "units": [
            {
                key: ([v + 1 for v in value] if isinstance(value, list) else value + 1)
                for key, value in unit.items()
                if key not in {"ring_roles"}
            }
            for unit in units
        ],
        "unit_ring_roles": [
            {role: [atom + 1 for atom in atoms] for role, atoms in unit["ring_roles"].items()}
            for unit in units
        ],
        "unit_adjacency": {str(idx): sorted(nbr + 1 for nbr in nbrs) for idx, nbrs in adjacency.items()},
        "categories": {str(idx + 1): categories[idx] for idx in range(len(units))},
        "category_counts": dict(Counter(categories.values())),
        "category_members": {key: [idx + 1 for idx in vals] for key, vals in category_members.items()},
        "crosslink_ring": [idx + 1 for idx in crosslink_ring],
        "representatives": {
            "internal": internal_rep + 1,
            "terminal": terminal_rep + 1,
            "adjacent": adjacent_rep + 1,
        },
        "models": models,
    }
    MAP_JSON.write_text(json.dumps(metadata, indent=2) + "\n")

    lines = [
        "Fragment charge model build report",
        f"source: {SOURCE}",
        f"units: {len(units)}",
        f"crosslink_ring_atoms: {[idx + 1 for idx in crosslink_ring]}",
        f"category_counts: {dict(Counter(categories.values()))}",
        f"representatives_1based: internal={internal_rep + 1}, terminal={terminal_rep + 1}, adjacent={adjacent_rep + 1}",
    ]
    for name, data in models.items():
        lines.append(
            f"{name}: sdf={data['sdf']} charge={data['charge']} atoms={data['n_atoms']} heavy={data['n_heavy']}"
        )
    REPORT.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
