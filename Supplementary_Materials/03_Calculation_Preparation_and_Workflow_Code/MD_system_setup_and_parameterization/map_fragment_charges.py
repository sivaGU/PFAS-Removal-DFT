import json
from collections import defaultdict
from pathlib import Path

from rdkit import Chem


FULL_SDF = Path("chol48_resin_only.sdf")
INPUT_MOL2 = Path("chol48_input.mol2")
MAP_JSON = Path("models/fragment_model_map.json")
OUT_MOL2 = Path("chol48_mapped_charges.mol2")
OUT_CHARGE = Path("chol48_mapped_charges.dat")
REPORT = Path("../logs/fragment_charge_mapping_report.txt")


def parse_mol2(path):
    atoms = []
    bonds = []
    section = None
    lines = path.read_text().splitlines()
    for line in lines:
        if line.startswith("@<TRIPOS>"):
            section = line.strip()
            continue
        if section == "@<TRIPOS>ATOM" and line.strip():
            parts = line.split()
            atoms.append(
                {
                    "id": int(parts[0]),
                    "name": parts[1],
                    "type": parts[5],
                    "charge": float(parts[-1]),
                    "line": line,
                }
            )
        elif section == "@<TRIPOS>BOND" and line.strip():
            parts = line.split()
            bonds.append((int(parts[1]), int(parts[2])))
    adjacency = defaultdict(list)
    for a, b in bonds:
        adjacency[a].append(b)
        adjacency[b].append(a)
    return atoms, adjacency, lines


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
    return {
        "ar_alpha": [ring_alpha],
        "ar_ortho": sorted([paths[0][1], paths[1][1]]),
        "ar_meta": sorted([paths[0][2], paths[1][2]]),
        "ar_benzyl": [ring_benzyl],
    }


def identify_units(mol):
    rings = [set(r) for r in mol.GetRingInfo().AtomRings()]
    units = []
    pendant = []
    for n_atom in [a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() == "N"]:
        nbrs = heavy_neighbors(mol, n_atom)
        methyl = [idx for idx in nbrs if mol.GetAtomWithIdx(idx).GetSymbol() == "C" and len(heavy_neighbors(mol, idx)) == 1]
        benzyl = [idx for idx in nbrs if idx not in methyl][0]
        ring_atom = [idx for idx in heavy_neighbors(mol, benzyl) if mol.GetAtomWithIdx(idx).GetIsAromatic()][0]
        ring = next(r for r in rings if ring_atom in r)
        exo = []
        for atom in ring:
            for nbr in heavy_neighbors(mol, atom):
                if nbr not in ring:
                    exo.append((atom, nbr))
        ring_alpha, alpha = [
            (atom, nbr)
            for atom, nbr in exo
            if nbr != benzyl and not mol.GetAtomWithIdx(nbr).GetIsAromatic()
        ][0]
        ring_benzyl = next(atom for atom, nbr in exo if nbr == benzyl)
        backbone = [idx for idx in heavy_neighbors(mol, alpha) if not mol.GetAtomWithIdx(idx).GetIsAromatic()]
        heavy_core = set([n_atom, benzyl, alpha, *methyl, *ring, *backbone])
        units.append(
            {
                "N": n_atom,
                "methyl": sorted(methyl),
                "benzyl": benzyl,
                "ring": sorted(ring),
                "ring_roles": ring_path_roles(ring, ring_alpha, ring_benzyl, mol),
                "alpha": alpha,
                "backbone": sorted(backbone),
                "heavy_core": sorted(heavy_core),
            }
        )
        pendant.append(ring)
    crosslink_ring = sorted([r for r in rings if r.isdisjoint(set().union(*pendant))][0])
    owner = {}
    for unit_idx, unit in enumerate(units):
        for atom in unit["backbone"]:
            owner[atom] = unit_idx
    adjacency = defaultdict(set)
    for bond in mol.GetBonds():
        a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if a in owner and b in owner and owner[a] != owner[b]:
            adjacency[owner[a]].add(owner[b])
            adjacency[owner[b]].add(owner[a])
    xset = set(crosslink_ring)
    cross_units = [idx for idx, unit in enumerate(units) if any(nbr in xset for nbr in heavy_neighbors(mol, unit["alpha"]))]
    terminals = [idx for idx, unit in enumerate(units) if any(len(heavy_neighbors(mol, atom)) == 1 for atom in unit["backbone"])]
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


def unit_heavy_roles(mol, unit, category, adjacency=None, categories=None):
    roles = {
        "alpha": [unit["alpha"]],
        "benzyl": [unit["benzyl"]],
        "N": [unit["N"]],
        "methyl_C": unit["methyl"],
    }
    roles.update(unit["ring_roles"])
    if category == "terminal":
        terminal = [atom for atom in unit["backbone"] if len(heavy_neighbors(mol, atom)) == 1]
        chain = [atom for atom in unit["backbone"] if atom not in terminal]
        roles["bb_terminal"] = terminal
        roles["bb_chain"] = chain
    elif category == "adjacent":
        unit_idx = None
        for idx, u in enumerate(UNITS_GLOBAL):
            if u is unit:
                unit_idx = idx
                break
        cross_neighbors = [nbr for nbr in adjacency[unit_idx] if categories[nbr] == "crosslink"]
        cross_unit = cross_neighbors[0]
        cross_heavy = set(UNITS_GLOBAL[cross_unit]["heavy_core"])
        bb_cross = [atom for atom in unit["backbone"] if any(nbr in cross_heavy for nbr in heavy_neighbors(mol, atom))]
        roles["bb_cross"] = bb_cross
        roles["bb_other"] = [atom for atom in unit["backbone"] if atom not in bb_cross]
    else:
        roles["bb"] = unit["backbone"]
    return roles


def model_charges(model_name, meta):
    mol2 = Path(meta["models"][model_name]["gaff2_mol2"])
    atoms, adjacency, _ = parse_mol2(mol2)
    heavy_full = [idx - 1 for idx in meta["models"][model_name]["heavy_full_indices_1based"]]
    heavy_count = len(heavy_full)
    full_to_model = {full: i + 1 for i, full in enumerate(heavy_full)}
    return atoms, adjacency, full_to_model, heavy_count


def averaged_template_for_unit(mol, unit, roles, atoms, adjacency, full_to_model):
    template = {}
    for role, full_heavy_atoms in roles.items():
        heavy_charges = []
        h_charges = []
        h_counts = []
        for full_idx in full_heavy_atoms:
            model_idx = full_to_model[full_idx]
            heavy_charges.append(atoms[model_idx - 1]["charge"])
            model_h = [
                nbr
                for nbr in adjacency[model_idx]
                if atoms[nbr - 1]["type"].lower().startswith("h") or atoms[nbr - 1]["name"].upper().startswith("H")
            ]
            h_full = h_neighbors(mol, full_idx)
            h_counts.append(len(h_full))
            if model_h:
                h_charges.extend(atoms[idx - 1]["charge"] for idx in model_h)
        template[role] = {
            "heavy_charge": sum(heavy_charges) / len(heavy_charges),
            "hydrogen_charge": (sum(h_charges) / len(h_charges)) if h_charges else None,
            "hydrogens_per_heavy": h_counts,
        }
    return template


def apply_template(charges, mol, unit, roles, template):
    assigned = set()
    for role, full_heavy_atoms in roles.items():
        heavy_q = template[role]["heavy_charge"]
        h_q = template[role]["hydrogen_charge"]
        for full_idx in full_heavy_atoms:
            charges[full_idx] = heavy_q
            assigned.add(full_idx)
            for h in h_neighbors(mol, full_idx):
                if h_q is None:
                    raise RuntimeError(f"No H charge for role {role}, full atom {full_idx + 1}")
                charges[h] = h_q
                assigned.add(h)
    return assigned


def direct_model_group_charges(mol, atoms, adjacency, full_to_model, group_atoms):
    mapped = {}
    for full_idx in sorted(group_atoms):
        model_idx = full_to_model[full_idx]
        mapped[full_idx] = atoms[model_idx - 1]["charge"]
        model_h = [
            nbr
            for nbr in adjacency[model_idx]
            if atoms[nbr - 1]["type"].lower().startswith("h") or atoms[nbr - 1]["name"].upper().startswith("H")
        ]
        full_h = h_neighbors(mol, full_idx)
        if full_h:
            if not model_h:
                raise RuntimeError(f"No model hydrogens for full atom {full_idx + 1}")
            h_q = sum(atoms[idx - 1]["charge"] for idx in model_h) / len(model_h)
            for h in full_h:
                mapped[h] = h_q
    return mapped


def normalize_group(charges, group_atoms, target, mol, label):
    current = sum(charges[idx] for idx in group_atoms)
    residual = target - current
    adjust_atoms = [
        idx
        for idx in group_atoms
        if mol.GetAtomWithIdx(idx).GetSymbol() == "H"
        and mol.GetAtomWithIdx(idx).GetNeighbors()[0].GetSymbol() == "C"
    ]
    if not adjust_atoms:
        adjust_atoms = [idx for idx in group_atoms if mol.GetAtomWithIdx(idx).GetSymbol() == "C"]
    delta = residual / len(adjust_atoms)
    for idx in adjust_atoms:
        charges[idx] += delta
    return current, residual, len(adjust_atoms), label


def write_mol2_with_charges(charges):
    lines = INPUT_MOL2.read_text().splitlines()
    out = []
    section = None
    atom_idx = 0
    charge_type_replaced = False
    for line in lines:
        if not charge_type_replaced and line.strip() == "GASTEIGER":
            out.append("USER_CHARGES")
            charge_type_replaced = True
            continue
        if line.startswith("@<TRIPOS>"):
            section = line.strip()
            out.append(line)
            continue
        if section == "@<TRIPOS>ATOM" and line.strip():
            atom_idx += 1
            parts = line.split()
            prefix = line[: line.rfind(parts[-1])]
            out.append(prefix + f"{charges[atom_idx - 1]:10.6f}")
        else:
            out.append(line)
    OUT_MOL2.write_text("\n".join(out) + "\n")
    OUT_CHARGE.write_text("\n".join(f"{q:.6f}" for q in charges) + "\n")


def main():
    global UNITS_GLOBAL
    mol = Chem.MolFromMolFile(str(FULL_SDF), removeHs=False, sanitize=True)
    units, crosslink_ring, adjacency, categories = identify_units(mol)
    UNITS_GLOBAL = units
    meta = json.loads(MAP_JSON.read_text())

    # Model outputs
    for name in ["model_internal", "model_terminal", "model_adjacent", "model_crosslink"]:
        gaff2 = Path("models") / f"{name}_gaff2.mol2"
        if not gaff2.exists():
            raise SystemExit(f"Missing model output {gaff2}")
        meta["models"][name]["gaff2_mol2"] = str(gaff2)

    charges = [None] * mol.GetNumAtoms()
    group_reports = []

    # Charge templates
    templates = {}
    for category, model_name, core_key in [
        ("internal", "model_internal", "internal_template_unit"),
        ("terminal", "model_terminal", "terminal_template_unit"),
        ("adjacent", "model_adjacent", "adjacent_template_unit"),
    ]:
        atoms, model_adj, full_to_model, _ = model_charges(model_name, meta)
        unit_idx = meta["models"][model_name]["core_groups"][core_key][0]
        roles = unit_heavy_roles(mol, units[unit_idx], category, adjacency, categories)
        templates[category] = averaged_template_for_unit(mol, units[unit_idx], roles, atoms, model_adj, full_to_model)

    for idx, unit in enumerate(units):
        if categories[idx] in {"internal", "terminal", "adjacent"}:
            roles = unit_heavy_roles(mol, unit, categories[idx], adjacency, categories)
            group_atoms = apply_template(charges, mol, unit, roles, templates[categories[idx]])
            group_reports.append(normalize_group(charges, group_atoms, 1.0, mol, f"unit_{idx + 1}_{categories[idx]}"))

    # Crosslink charges
    x_atoms, x_adj, x_full_to_model, _ = model_charges("model_crosslink", meta)
    cross_units = [idx for idx, cat in categories.items() if cat == "crosslink"]
    cross_heavy = set(crosslink_ring)
    for idx in cross_units:
        cross_heavy.update(units[idx]["heavy_core"])
    cross_group = set(cross_heavy)
    for heavy in list(cross_heavy):
        cross_group.update(h_neighbors(mol, heavy))
    direct = direct_model_group_charges(mol, x_atoms, x_adj, x_full_to_model, cross_heavy)
    for idx, charge in direct.items():
        charges[idx] = charge
    group_reports.append(normalize_group(charges, cross_group, 2.0, mol, "crosslink_units_plus_dvb_ring"))

    missing = [idx + 1 for idx, q in enumerate(charges) if q is None]
    if missing:
        raise SystemExit(f"Missing mapped charges for atoms: {missing[:50]} ...")

    total = sum(charges)
    residual = 48.0 - total
    # Charge correction
    adjust = [
        idx
        for idx, atom in enumerate(mol.GetAtoms())
        if atom.GetSymbol() == "H" and atom.GetNeighbors()[0].GetSymbol() == "C"
    ]
    delta = residual / len(adjust)
    for idx in adjust:
        charges[idx] += delta
    final_total = sum(charges)

    write_mol2_with_charges(charges)

    lines = [
        "Fragment charge mapping report",
        "charge model: AM1-BCC on capped model compounds; mapped by chemically equivalent repeat-unit roles",
        f"total_atoms: {mol.GetNumAtoms()}",
        f"initial_mapped_total_before_final_rounding: {total:.10f}",
        f"final_rounding_residual: {residual:.10f}",
        f"final_total_charge: {final_total:.10f}",
        "per_group_normalization_before_final_rounding:",
    ]
    for current, residual_group, n_adjust, label in group_reports:
        lines.append(f"  {label}: pre={current:.8f} target_residual={residual_group:.8f} adjusted_atoms={n_adjust}")
    lines.append("template_role_charges:")
    for category, template in templates.items():
        lines.append(f"  {category}:")
        for role, vals in sorted(template.items()):
            lines.append(
                f"    {role}: heavy={vals['heavy_charge']:.6f}"
                + (f" H={vals['hydrogen_charge']:.6f}" if vals["hydrogen_charge"] is not None else "")
            )
    REPORT.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
