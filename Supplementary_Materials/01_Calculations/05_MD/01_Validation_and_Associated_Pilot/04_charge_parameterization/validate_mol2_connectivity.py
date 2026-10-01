#!/usr/bin/env python3

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

import parmed as pmd


def parse_sdf_v2000(path: Path):
    lines = path.read_text(errors="replace").splitlines()
    if len(lines) < 4:
        raise ValueError("SDF is too short")
    counts = lines[3]
    nat, nbd = int(counts[0:3]), int(counts[3:6])
    elements = []
    for line in lines[4 : 4 + nat]:
        element = line[31:34].strip() or line.split()[3]
        elements.append(element)
    bonds = set()
    for line in lines[4 + nat : 4 + nat + nbd]:
        try:
            a, b = int(line[0:3]) - 1, int(line[3:6]) - 1
        except Exception:
            f = line.split()
            a, b = int(f[0]) - 1, int(f[1]) - 1
        bonds.add(tuple(sorted((a, b))))
    return elements, bonds


def parse_mol2(path: Path):
    section = None
    atom_types, bonds = [], set()
    for line in path.read_text(errors="replace").splitlines():
        if line.startswith("@<TRIPOS>"):
            section = line.strip()
            continue
        if not line.strip():
            continue
        if section == "@<TRIPOS>ATOM":
            f = line.split()
            atom_types.append(f[5])
        elif section == "@<TRIPOS>BOND":
            f = line.split()
            bonds.add(tuple(sorted((int(f[1]) - 1, int(f[2]) - 1))))
    return atom_types, bonds


def components_and_adjacency(n, bonds):
    adj = [[] for _ in range(n)]
    for a, b in bonds:
        adj[a].append(b)
        adj[b].append(a)
    seen, ncomp = set(), 0
    for start in range(n):
        if start in seen:
            continue
        ncomp += 1
        stack = [start]
        seen.add(start)
        while stack:
            i = stack.pop()
            for j in adj[i]:
                if j not in seen:
                    seen.add(j)
                    stack.append(j)
    return ncomp, adj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mol2", type=Path)
    ap.add_argument("reference_sdf", type=Path)
    ap.add_argument("--expected-charge", type=float, default=None)
    ap.add_argument("--json-out", type=Path, default=None)
    args = ap.parse_args()

    ref_elements, ref_bonds = parse_sdf_v2000(args.reference_sdf)
    atom_types, mol2_bonds = parse_mol2(args.mol2)
    s = pmd.load_file(str(args.mol2))
    elements = []
    for atom in s.atoms:
        z = int(getattr(atom, "atomic_number", 0) or 0)
        symbol = {1: "H", 6: "C", 7: "N"}.get(z, f"Z{z}")
        elements.append(symbol)

    ncomp, adj = components_and_adjacency(len(elements), mol2_bonds)
    ref_ncomp, ref_adj = components_and_adjacency(len(ref_elements), ref_bonds)
    quat_n = sum(e == "N" and len(adj[i]) == 4 for i, e in enumerate(elements))
    ref_quat_n = sum(e == "N" and len(ref_adj[i]) == 4 for i, e in enumerate(ref_elements))
    heavy_bonds = sum(elements[a] != "H" and elements[b] != "H" for a, b in mol2_bonds)
    ref_heavy_bonds = sum(ref_elements[a] != "H" and ref_elements[b] != "H" for a, b in ref_bonds)
    charge = sum(float(a.charge) for a in s.atoms)

    connectivity_checks = {
        "atom_count_match": len(elements) == len(ref_elements),
        "indexwise_element_sequence_match": elements == ref_elements,
        "indexwise_bond_set_match": mol2_bonds == ref_bonds,
        "bond_count_match": len(mol2_bonds) == len(ref_bonds),
        "heavy_bond_count_match": heavy_bonds == ref_heavy_bonds,
        "connected_component_count_match": ncomp == ref_ncomp == 1,
        "degree4_nitrogen_count_match": quat_n == ref_quat_n,
    }
    charge_check = None
    if args.expected_charge is not None:
        charge_check = {
            "expected_charge": args.expected_charge,
            "observed_charge": charge,
            "delta": charge - args.expected_charge,
            "within_1e-6_diagnostic_only": abs(charge - args.expected_charge) <= 1e-6,
            "included_in_constitutional_pass": False,
            "interpretation": (
                "Charge is reported separately from constitutional connectivity and is "
                "not a pass/fail criterion in this validator. For the raw reduced "
                "AM1-BCC oligomer, small finite-precision residuals are handled during "
                "H/I/T library normalization. For the final transferred MOL2, assess "
                "the serialized total against the displayed-precision rounding bound "
                "reported by summarize_mol2_charges.py."
            ),
        }

    constitutional_pass = all(connectivity_checks.values())
    report = {
        "validation_scope": "Indexed element sequence and atom-pair constitutional connectivity only; bond orders and stereochemistry are not checked here.",
        "mol2": str(args.mol2),
        "reference_sdf": str(args.reference_sdf),
        "atoms": len(elements),
        "composition": dict(collections.Counter(elements)),
        "bonds": len(mol2_bonds),
        "heavy_atom_bonds": heavy_bonds,
        "connected_components": ncomp,
        "degree4_nitrogens": quat_n,
        "partial_charge_sum": charge,
        "gaff2_atom_type_counts": dict(collections.Counter(atom_types)),
        "reference": {
            "atoms": len(ref_elements),
            "composition": dict(collections.Counter(ref_elements)),
            "bonds": len(ref_bonds),
            "heavy_atom_bonds": ref_heavy_bonds,
            "connected_components": ref_ncomp,
            "degree4_nitrogens": ref_quat_n,
        },
        "connectivity_checks": connectivity_checks,
        "constitutional_connectivity_pass": constitutional_pass,
        "charge_check": charge_check,
    }
    text = json.dumps(report, indent=2, sort_keys=True)
    print(text)
    if args.json_out:
        args.json_out.write_text(text + "\n")
    raise SystemExit(0 if constitutional_pass else 2)


if __name__ == "__main__":
    main()
