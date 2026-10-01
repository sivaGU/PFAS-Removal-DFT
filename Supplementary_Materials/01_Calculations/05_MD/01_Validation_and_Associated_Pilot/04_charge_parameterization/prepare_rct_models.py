#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from statistics import mean

from rdkit import Chem, rdBase
from rdkit.Chem import AllChem, rdMolDescriptors

REPEAT_SMILES = "[*:1]CC(c1ccc(C[N+](C)(C)C)cc1)[*:2]"
RDKIT_SEED = 20260916
N_CONFORMERS = 12
UFF_MAX_ITERS = 1000


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dummy(mol: Chem.Mol, mapno: int):
    hits = [
        a.GetIdx()
        for a in mol.GetAtoms()
        if a.GetAtomicNum() == 0 and a.GetAtomMapNum() == mapno
    ]
    if len(hits) != 1:
        raise ValueError(f"Expected one dummy atom with map {mapno}; got {hits}")
    idx = hits[0]
    nbr = [n.GetIdx() for n in mol.GetAtomWithIdx(idx).GetNeighbors()]
    if len(nbr) != 1:
        raise ValueError(f"Dummy atom {mapno} must have exactly one neighbor")
    return idx, nbr[0]


def remove_atoms(rw: Chem.RWMol, indices):
    for idx in sorted(indices, reverse=True):
        rw.RemoveAtom(idx)


def labelled_unit(repeat_index: int) -> Chem.Mol:
    m = Chem.MolFromSmiles(REPEAT_SMILES)
    if m is None:
        raise RuntimeError("Could not parse repeat SMILES")
    m = Chem.Mol(m)
    for a in m.GetAtoms():
        a.SetIntProp("_repeat_index", repeat_index)
        a.SetIntProp("_unit_atom_index", a.GetIdx())
    return m


def connect(a: Chem.Mol, b: Chem.Mol) -> Chem.Mol:
    da, na = dummy(a, 2)
    db, nb = dummy(b, 1)
    off = a.GetNumAtoms()
    rw = Chem.RWMol(Chem.CombineMols(a, b))
    rw.AddBond(na, off + nb, Chem.BondType.SINGLE)
    remove_atoms(rw, [da, off + db])
    out = rw.GetMol()
    Chem.SanitizeMol(out)
    return out


def build_labelled_chain(n: int) -> Chem.Mol:
    if n < 1:
        raise ValueError("n must be >= 1")
    chain = labelled_unit(1)
    for r in range(2, n + 1):
        chain = connect(chain, labelled_unit(r))
    rw = Chem.RWMol(chain)
    remove_atoms(
        rw, [a.GetIdx() for a in chain.GetAtoms() if a.GetAtomicNum() == 0]
    )
    chain = rw.GetMol()
    Chem.SanitizeMol(chain)
    return chain


def aromatic_backbone_centers(m: Chem.Mol):
    centers = []
    for a in m.GetAtoms():
        if a.GetAtomicNum() != 6 or a.GetIsAromatic():
            continue
        if not any(n.GetIsAromatic() for n in a.GetNeighbors()):
            continue
        if any(n.GetAtomicNum() == 7 for n in a.GetNeighbors()):
            continue
        centers.append(a.GetIdx())
    return centers


def ordered_centers(m: Chem.Mol):
    centers = aromatic_backbone_centers(m)
    dmat = Chem.GetDistanceMatrix(m)
    adj = {c: [] for c in centers}
    for i, c in enumerate(centers):
        for x in centers[i + 1 :]:
            if int(dmat[c, x]) == 2:
                adj[c].append(x)
                adj[x].append(c)
    ends = [c for c in centers if len(adj[c]) == 1]
    starts = [c for c in ends if m.GetAtomWithIdx(c).GetTotalNumHs() == 1]
    if len(starts) != 1:
        raise ValueError(f"Could not uniquely identify CH3-end center: {starts}")
    order = [starts[0]]
    prev = None
    cur = starts[0]
    while True:
        nxt = [x for x in adj[cur] if x != prev]
        if not nxt:
            break
        if len(nxt) != 1:
            raise ValueError("Backbone center graph is not a simple path")
        prev, cur = cur, nxt[0]
        order.append(cur)
    if len(order) != len(centers):
        raise ValueError("Backbone center ordering failed")
    return order


def permutation_sign(actual, desired):

    if sorted(actual) != sorted(desired):
        raise ValueError(f"Neighbor sets differ: actual={actual}, desired={desired}")
    perm = [desired.index(x) for x in actual]
    inversions = sum(perm[i] > perm[j] for i in range(len(perm)) for j in range(i + 1, len(perm)))
    return -1 if inversions % 2 else 1


def local_neighbor_parity(m, c, prev_center, next_center, first=False):

    a = m.GetAtomWithIdx(c)
    nbrs = [n.GetIdx() for n in a.GetNeighbors()]
    aromatic = [n.GetIdx() for n in a.GetNeighbors() if n.GetIsAromatic()]
    aliph = [
        n.GetIdx()
        for n in a.GetNeighbors()
        if n.GetAtomicNum() == 6 and not n.GetIsAromatic()
    ]
    if len(aromatic) != 1 or len(aliph) != 2 or a.GetTotalNumHs() != 1:
        raise ValueError(f"Unexpected stereocenter environment at atom {c}")
    path_next = Chem.GetShortestPath(m, c, next_center)
    next_atom = path_next[1]
    if first:
        prev_atoms = [x for x in aliph if x != next_atom]
        if len(prev_atoms) != 1:
            raise ValueError("Could not identify repeat-1 terminal ligand")
        prev_atom = prev_atoms[0]
    else:
        path_prev = Chem.GetShortestPath(m, c, prev_center)
        prev_atom = path_prev[1]
    desired = [prev_atom, next_atom, aromatic[0]]
    return permutation_sign(nbrs, desired)


def apply_local_parity(m: Chem.Mol, signs):
    m = Chem.Mol(m)
    order = ordered_centers(m)
    if len(signs) != len(order) - 1:
        raise ValueError(
            f"Expected {len(order)-1} parity labels for {len(order)} repeats; got {len(signs)}"
        )
    for i, (c, sign) in enumerate(zip(order[:-1], signs)):
        p = local_neighbor_parity(
            m, c, order[i - 1] if i > 0 else None, order[i + 1], first=(i == 0)
        )
        effective_plus = (sign == "+") if p == 1 else (sign != "+")
        tag = (
            Chem.ChiralType.CHI_TETRAHEDRAL_CW
            if effective_plus
            else Chem.ChiralType.CHI_TETRAHEDRAL_CCW
        )
        m.GetAtomWithIdx(c).SetChiralTag(tag)
    Chem.AssignStereochemistry(m, cleanIt=True, force=True)
    return m


def atom_role_for_heavy(atom: Chem.Atom) -> str:
    if atom.GetAtomicNum() == 7 and atom.GetFormalCharge() == 1 and atom.GetDegree() == 4:
        return "Nq"
    if atom.GetAtomicNum() == 6:
        nplus = [
            n
            for n in atom.GetNeighbors()
            if n.GetAtomicNum() == 7 and n.GetFormalCharge() == 1
        ]
        if nplus:


            heavy_neighbors = [n for n in atom.GetNeighbors() if n.GetAtomicNum() > 1]
            if len(heavy_neighbors) == 1:
                return "NMe_C"
    return f"u{atom.GetIntProp('_unit_atom_index'):02d}_{atom.GetSymbol()}"


def add_hydrogens_and_roles(m: Chem.Mol) -> Chem.Mol:
    mh = Chem.AddHs(m)

    for atom in mh.GetAtoms():
        if atom.GetAtomicNum() == 1:
            continue
        atom.SetProp("_rct_role", atom_role_for_heavy(atom))


    for atom in mh.GetAtoms():
        if atom.GetAtomicNum() != 1:
            continue
        parent = atom.GetNeighbors()[0]
        atom.SetIntProp("_repeat_index", parent.GetIntProp("_repeat_index"))
        parent_role = parent.GetProp("_rct_role")
        role = "NMe_H" if parent_role == "NMe_C" else f"H@{parent_role}"
        atom.SetProp("_rct_role", role)
    return mh


def mapping_payload(m: Chem.Mol, n_repeats: int):
    atoms = []
    for a in m.GetAtoms():
        r = a.GetIntProp("_repeat_index")
        kind = "head" if r == 1 else ("tail" if r == n_repeats else "internal")
        atoms.append(
            {
                "atom_index_1based": a.GetIdx() + 1,
                "element": a.GetSymbol(),
                "repeat_index_1based": r,
                "repeat_kind": kind,
                "role": a.GetProp("_rct_role"),
            }
        )
    return atoms


def embed_reduced(m: Chem.Mol):
    params = AllChem.ETKDGv3()
    params.randomSeed = RDKIT_SEED
    params.useRandomCoords = True
    ids = list(AllChem.EmbedMultipleConfs(m, numConfs=N_CONFORMERS, params=params))
    if not ids:
        raise RuntimeError("RDKit failed to embed any pVBTMA5 conformers")
    results = AllChem.UFFOptimizeMoleculeConfs(
        m, numThreads=0, maxIters=UFF_MAX_ITERS
    )
    table = []
    for cid, (status, energy) in zip(ids, results):
        table.append(
            {
                "conformer_id": int(cid),
                "uff_status": int(status),
                "uff_energy": float(energy),
            }
        )

    candidates = [x for x in table if x["uff_status"] == 0] or table
    best = min(candidates, key=lambda x: x["uff_energy"])
    return int(best["conformer_id"]), table


def write_one_conformer_sdf(m: Chem.Mol, conf_id: int, path: Path):
    writer = Chem.SDWriter(str(path))
    writer.write(m, confId=conf_id)
    writer.close()


def reorder_accepted_coordinates(builder_explicit: Chem.Mol, accepted_path: Path):
    accepted_explicit = Chem.SDMolSupplier(str(accepted_path), removeHs=False)[0]
    if accepted_explicit is None:
        raise RuntimeError(f"Could not parse accepted SDF: {accepted_path}")
    builder_heavy = Chem.RemoveHs(Chem.Mol(builder_explicit))
    accepted_heavy = Chem.RemoveHs(Chem.Mol(accepted_explicit))
    match = accepted_heavy.GetSubstructMatch(builder_heavy, useChirality=False)
    if len(match) != builder_heavy.GetNumAtoms():
        raise RuntimeError("Could not map accepted pVBTMA12 heavy-atom graph to builder graph")

    builder_heavy_indices = [a.GetIdx() for a in builder_explicit.GetAtoms() if a.GetAtomicNum() != 1]
    accepted_heavy_indices = [a.GetIdx() for a in accepted_explicit.GetAtoms() if a.GetAtomicNum() != 1]
    b_to_a = {}
    for b_heavy_order, a_heavy_order in enumerate(match):
        b_idx = builder_heavy_indices[b_heavy_order]
        a_idx = accepted_heavy_indices[a_heavy_order]
        b_to_a[b_idx] = a_idx


    for b_idx, a_idx in list(b_to_a.items()):
        b_h = sorted(
            n.GetIdx()
            for n in builder_explicit.GetAtomWithIdx(b_idx).GetNeighbors()
            if n.GetAtomicNum() == 1
        )
        a_h = sorted(
            n.GetIdx()
            for n in accepted_explicit.GetAtomWithIdx(a_idx).GetNeighbors()
            if n.GetAtomicNum() == 1
        )
        if len(b_h) != len(a_h):
            raise RuntimeError(
                f"Hydrogen-count mismatch during coordinate mapping at builder atom {b_idx+1}"
            )
        b_to_a.update(dict(zip(b_h, a_h)))

    if len(b_to_a) != builder_explicit.GetNumAtoms():
        missing = sorted(set(range(builder_explicit.GetNumAtoms())) - set(b_to_a))
        raise RuntimeError(f"Incomplete full-atom coordinate mapping; missing {missing[:20]}")

    out = Chem.Mol(builder_explicit)
    conf = Chem.Conformer(out.GetNumAtoms())
    src_conf = accepted_explicit.GetConformer()
    for b_idx in range(out.GetNumAtoms()):
        conf.SetAtomPosition(b_idx, src_conf.GetAtomPosition(b_to_a[b_idx]))
    out.RemoveAllConformers()
    out.AddConformer(conf, assignId=True)
    return out, b_to_a


def validate_basic(m: Chem.Mol, n: int):
    formula = rdMolDescriptors.CalcMolFormula(m)
    q = sum(a.GetFormalCharge() for a in m.GetAtoms())
    qn = sum(
        a.GetAtomicNum() == 7 and a.GetFormalCharge() == 1 and a.GetDegree() == 4
        for a in m.GetAtoms()
    )
    expected_formula = {5: "C60H92N5+5", 12: "C144H218N12+12"}[n]
    return {
        "formula": formula,
        "expected_formula": expected_formula,
        "formula_pass": formula == expected_formula,
        "atoms_with_hydrogen": m.GetNumAtoms(),
        "heavy_atoms": sum(a.GetAtomicNum() != 1 for a in m.GetAtoms()),
        "formal_charge": q,
        "quaternary_ammonium_sites": qn,
        "connected_components": len(Chem.GetMolFrags(m)),
        "expected_checks_pass": formula == expected_formula
        and q == n
        and qn == n
        and len(Chem.GetMolFrags(m)) == 1,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--accepted-pvbtma12-sdf",
        type=Path,
        default=Path("../02_3d_structure_generation/pVBTMA12_gen3d_uffmin.sdf"),
    )
    ap.add_argument(
        "--accepted-stereo-spec",
        type=Path,
        default=Path("../01_model_definition/pVBTMA12_model_definition_stereo_spec.json"),
    )
    args = ap.parse_args()

    spec = json.loads(args.accepted_stereo_spec.read_text())
    seq12 = spec["local_parity_sequence_repeat_1_to_11"]
    if len(seq12) != 11:
        raise SystemExit("Accepted stereo specification does not contain 11 parity labels")
    seq5 = seq12[:4]


    m5 = apply_local_parity(build_labelled_chain(5), seq5)
    m5 = add_hydrogens_and_roles(m5)
    best_cid, conformers = embed_reduced(m5)
    p5_sdf = Path("pVBTMA5_rct_model_uffmin.sdf")
    write_one_conformer_sdf(m5, best_cid, p5_sdf)
    p5_smi = Path("pVBTMA5_rct_model.smi")
    p5_smi.write_text(Chem.MolToSmiles(Chem.RemoveHs(m5), canonical=True, isomericSmiles=True) + "\n")
    p5_map = Path("pVBTMA5_rct_atom_mapping.json")
    p5_map.write_text(json.dumps({"n_repeats": 5, "atoms": mapping_payload(m5, 5)}, indent=2) + "\n")


    m12 = apply_local_parity(build_labelled_chain(12), seq12)
    m12 = add_hydrogens_and_roles(m12)
    m12_reordered, b_to_a = reorder_accepted_coordinates(m12, args.accepted_pvbtma12_sdf)
    p12_sdf = Path("pVBTMA12_rct_reference.sdf")
    write_one_conformer_sdf(m12_reordered, 0, p12_sdf)
    p12_map = Path("pVBTMA12_rct_atom_mapping.json")
    p12_map.write_text(json.dumps({"n_repeats": 12, "atoms": mapping_payload(m12, 12)}, indent=2) + "\n")
    reorder_json = Path("pVBTMA12_accepted_to_rct_reorder.json")
    reorder_json.write_text(
        json.dumps(
            {
                "description": "builder atom index -> accepted Stage-02 SDF atom index; both 1-based",
                "builder_to_accepted_1based": {
                    str(k + 1): int(v + 1) for k, v in sorted(b_to_a.items())
                },
            },
            indent=2,
        )
        + "\n"
    )

    v5 = validate_basic(m5, 5)
    v12 = validate_basic(m12_reordered, 12)
    accepted_ref = Chem.MolFromSmiles(args.accepted_stereo_spec.with_name("pVBTMA12_model_definition.smi").read_text().strip())
    if accepted_ref is None:
        raise RuntimeError("Could not parse accepted pVBTMA12 model-definition SMILES")
    accepted_iso = Chem.MolToSmiles(accepted_ref, canonical=True, isomericSmiles=True)
    reordered_iso = Chem.MolToSmiles(Chem.RemoveHs(m12_reordered), canonical=True, isomericSmiles=True)
    v12["exact_stereoisomer_match_to_stage01"] = reordered_iso == accepted_iso
    v12["expected_checks_pass"] = v12["expected_checks_pass"] and v12["exact_stereoisomer_match_to_stage01"]
    v5_centers = Chem.FindMolChiralCenters(Chem.RemoveHs(m5), includeUnassigned=True, includeCIP=True)
    v5["assigned_stereocenters"] = len([x for x in v5_centers if x[1] != "?"])
    v5["expected_checks_pass"] = v5["expected_checks_pass"] and v5["assigned_stereocenters"] == 4
    report = {
        "rdkit_version": rdBase.rdkitVersion,
        "repeat_smiles": REPEAT_SMILES,
        "accepted_pvbtma12_local_parity": seq12,
        "pVBTMA5_local_parity": seq5,
        "pVBTMA5_conformer_generation": {
            "method": "RDKit ETKDGv3 followed by UFF minimization; starting geometry only for subsequent SQM",
            "random_seed": RDKIT_SEED,
            "n_conformers_requested": N_CONFORMERS,
            "uff_max_iterations": UFF_MAX_ITERS,
            "selected_conformer_id": best_cid,
            "selected_uff_energy": next(x["uff_energy"] for x in conformers if x["conformer_id"] == best_cid),
            "n_uff_converged": sum(x["uff_status"] == 0 for x in conformers),
            "conformers": conformers,
        },
        "pVBTMA5_validation": v5,
        "pVBTMA12_validation": v12,
        "pVBTMA12_coordinates": "accepted Stage-02 UFF-minimized coordinates reordered onto deterministic builder atom order",
        "outputs": [
            p5_sdf.name,
            p5_smi.name,
            p5_map.name,
            p12_sdf.name,
            p12_map.name,
            reorder_json.name,
        ],
    }
    report["all_checks_pass"] = v5["expected_checks_pass"] and v12["expected_checks_pass"]
    out = Path("rct_model_preparation.json")
    out.write_text(json.dumps(report, indent=2) + "\n")

    hashes = {p.name: sha256(p) for p in [p5_sdf, p5_smi, p5_map, p12_sdf, p12_map, reorder_json, out]}
    Path("rct_model_preparation_SHA256.json").write_text(json.dumps(hashes, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["all_checks_pass"] else 2)


if __name__ == "__main__":
    main()
