#!/usr/bin/env python3

from __future__ import annotations

import argparse
import collections
import json
import math
import statistics
from pathlib import Path


def load_mapping(path: Path):
    obj = json.loads(path.read_text())
    atoms = obj["atoms"]
    idxs = [a["atom_index_1based"] for a in atoms]
    if idxs != list(range(1, len(atoms) + 1)):
        raise ValueError(f"Mapping {path} is not contiguous and 1-based")
    return obj


def mol2_data(path: Path):
    out = []
    section = None
    for line in path.read_text(errors="replace").splitlines():
        if line.startswith("@<TRIPOS>"):
            section = line.strip()
            continue
        if section != "@<TRIPOS>ATOM" or not line.strip():
            continue
        f = line.split()
        if len(f) < 9:
            raise ValueError(f"Malformed MOL2 atom line: {line!r}")
        out.append({"charge": float(f[-1]), "name": f[1], "type": f[5]})
    if not out:
        raise ValueError(f"No MOL2 atom records found in {path}")
    return out


def role_counter(atoms, repeat_index):
    return collections.Counter(
        a["role"] for a in atoms if a["repeat_index_1based"] == repeat_index
    )


def build_library(kind, source_repeats, mapping_atoms, source_charges, target_charge=1.0):
    representative = source_repeats[0]
    multiplicities = role_counter(mapping_atoms, representative)
    for r in source_repeats[1:]:
        if role_counter(mapping_atoms, r) != multiplicities:
            raise ValueError(f"Role multiplicity differs within {kind} source repeats")

    samples = collections.defaultdict(list)
    per_repeat_raw = {}
    for r in source_repeats:
        vals = []
        for m, q in zip(mapping_atoms, source_charges):
            if m["repeat_index_1based"] == r:
                samples[m["role"]].append(q)
                vals.append(q)
        per_repeat_raw[str(r)] = sum(vals)

    if set(samples) != set(multiplicities):
        raise ValueError(f"Role set mismatch while constructing {kind} library")

    role_stats = {}
    for role in sorted(samples):
        vals = samples[role]
        role_stats[role] = {
            "count_per_repeat": multiplicities[role],
            "n_source_atom_samples": len(vals),
            "raw_mean_charge": statistics.fmean(vals),
            "raw_population_sd": statistics.pstdev(vals) if len(vals) > 1 else 0.0,
            "raw_min": min(vals),
            "raw_max": max(vals),
        }

    n_atoms = sum(multiplicities.values())
    raw_sum = sum(
        role_stats[role]["raw_mean_charge"] * multiplicities[role]
        for role in multiplicities
    )
    correction = (target_charge - raw_sum) / n_atoms
    normalized_sum = 0.0
    for role in role_stats:
        q = role_stats[role]["raw_mean_charge"] + correction
        role_stats[role]["normalized_charge"] = q
        normalized_sum += q * multiplicities[role]

    return {
        "kind": kind,
        "source_repeats_1based": source_repeats,
        "target_formal_charge": target_charge,
        "atoms_per_repeat": n_atoms,
        "raw_source_repeat_charge_sums": per_repeat_raw,
        "raw_library_charge_sum": raw_sum,
        "uniform_per_atom_normalization": correction,
        "normalized_library_charge_sum": normalized_sum,
        "roles": role_stats,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reduced-mol2", type=Path, default=Path("rct5_antechamber_work/pVBTMA5_gaff2_am1bcc.mol2"))
    ap.add_argument("--reduced-mapping", type=Path, default=Path("pVBTMA5_rct_atom_mapping.json"))
    ap.add_argument("--target-mapping", type=Path, default=Path("pVBTMA12_rct_atom_mapping.json"))
    ap.add_argument("--library-out", type=Path, default=Path("rct_charge_library.json"))
    ap.add_argument("--charges-out", type=Path, default=Path("pVBTMA12_rct_charges.txt"))
    ap.add_argument("--validation-out", type=Path, default=Path("rct_charge_transfer_validation.json"))
    args = ap.parse_args()

    src_map = load_mapping(args.reduced_mapping)
    tgt_map = load_mapping(args.target_mapping)
    src = mol2_data(args.reduced_mol2)
    if len(src) != len(src_map["atoms"]):
        raise SystemExit("Reduced MOL2 atom count does not match mapping")
    source_charges = [x["charge"] for x in src]
    source_total = sum(source_charges)
    source_delta = source_total - 5.0


    if abs(source_delta) > 0.02:
        raise SystemExit(
            f"Reduced pVBTMA5 raw charge {source_total:.6f} differs from +5 by "
            f"{source_delta:+.6f} e, exceeding the 0.02 e safety limit"
        )

    libraries = {
        "head": build_library("head", [1], src_map["atoms"], source_charges),
        "internal": build_library("internal", [2, 3, 4], src_map["atoms"], source_charges),
        "tail": build_library("tail", [5], src_map["atoms"], source_charges),
    }


    target_repeat_sums = {}
    transferred = []
    missing = []
    for atom in tgt_map["atoms"]:
        kind = atom["repeat_kind"]
        role = atom["role"]
        if role not in libraries[kind]["roles"]:
            missing.append((atom["atom_index_1based"], kind, role))
            transferred.append(float("nan"))
        else:
            transferred.append(libraries[kind]["roles"][role]["normalized_charge"])
    if missing:
        raise SystemExit(f"Missing RCT library roles for target atoms: {missing[:20]}")


    multiplicity_checks = {}
    for r in range(1, tgt_map["n_repeats"] + 1):
        kind = "head" if r == 1 else ("tail" if r == tgt_map["n_repeats"] else "internal")
        observed = role_counter(tgt_map["atoms"], r)
        expected = collections.Counter(
            {role: data["count_per_repeat"] for role, data in libraries[kind]["roles"].items()}
        )
        multiplicity_checks[str(r)] = observed == expected
        if not multiplicity_checks[str(r)]:
            raise SystemExit(f"Target repeat {r} role multiplicity does not match {kind} library")

    for r in range(1, tgt_map["n_repeats"] + 1):
        q = sum(
            transferred[i]
            for i, a in enumerate(tgt_map["atoms"])
            if a["repeat_index_1based"] == r
        )
        target_repeat_sums[str(r)] = q

    target_total = sum(transferred)
    with args.charges_out.open("w") as fh:
        for q in transferred:
            fh.write(f"{q:.12f}\n")

    lib_report = {
        "method": "RCT-style pVBTMA5 H-I-I-I-T library derivation with residue-wise +1 normalization",
        "source_reduced_mol2": str(args.reduced_mol2),
        "source_reduced_serialized_charge_sum": source_total,
        "target_charge_per_repeat": 1.0,
        "libraries": libraries,
    }
    args.library_out.write_text(json.dumps(lib_report, indent=2, sort_keys=True) + "\n")

    internal_sds = [x["raw_population_sd"] for x in libraries["internal"]["roles"].values()]
    report = {
        "source_reduced_mol2": str(args.reduced_mol2),
        "source_atoms": len(src),
        "source_serialized_charge_sum": source_total,
        "source_delta_from_plus5": source_total - 5.0,
        "library_normalization": {
            kind: {
                "raw_library_charge_sum": lib["raw_library_charge_sum"],
                "uniform_per_atom_normalization": lib["uniform_per_atom_normalization"],
                "normalized_library_charge_sum": lib["normalized_library_charge_sum"],
                "atoms_per_repeat": lib["atoms_per_repeat"],
            }
            for kind, lib in libraries.items()
        },
        "internal_role_variability": {
            "maximum_population_sd_e": max(internal_sds),
            "mean_population_sd_e": statistics.fmean(internal_sds),
            "roles_with_sd_gt_0p01_e": sorted(
                role
                for role, data in libraries["internal"]["roles"].items()
                if data["raw_population_sd"] > 0.01
            ),
        },
        "target_atoms": len(transferred),
        "target_repeat_charge_sums": target_repeat_sums,
        "target_total_charge": target_total,
        "target_delta_from_plus12": target_total - 12.0,
        "target_role_multiplicity_checks": multiplicity_checks,
        "all_repeat_sums_within_1e-10_of_plus1": all(abs(q - 1.0) <= 1e-10 for q in target_repeat_sums.values()),
        "total_within_1e-10_of_plus12": abs(target_total - 12.0) <= 1e-10,
        "charges_file": str(args.charges_out),
    }
    report["transfer_validation_pass"] = (
        report["all_repeat_sums_within_1e-10_of_plus1"]
        and report["total_within_1e-10_of_plus12"]
        and all(multiplicity_checks.values())
        and all(math.isfinite(x) for x in transferred)
    )
    args.validation_out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    raise SystemExit(0 if report["transfer_validation_pass"] else 2)


if __name__ == "__main__":
    main()
