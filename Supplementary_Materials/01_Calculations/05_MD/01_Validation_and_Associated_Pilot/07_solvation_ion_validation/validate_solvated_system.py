#!/usr/bin/env python3
from __future__ import annotations

import collections
import json
import math
import re
from pathlib import Path

import parmed as pmd

AVOGADRO = 6.02214076e23
ANGSTROM3_TO_LITER = 1.0e-27

FINAL_TOP = "pVBTMA12_12Cl_015MNaCl_pilot.prmtop"
FINAL_CRD = "pVBTMA12_12Cl_015MNaCl_pilot.inpcrd"
SALT_JSON = "salt_setup.json"


def cell_volume_ang3(box):
    a, b, c, alpha, beta, gamma = [float(x) for x in box[:6]]
    ca = math.cos(math.radians(alpha)); cb = math.cos(math.radians(beta)); cg = math.cos(math.radians(gamma))
    factor = 1.0 + 2.0 * ca * cb * cg - ca * ca - cb * cb - cg * cg
    return a * b * c * math.sqrt(factor)


def element_counts(atoms):
    z_to_symbol = {1: "H", 6: "C", 7: "N", 8: "O", 9: "F", 11: "Na", 17: "Cl"}
    out = collections.Counter()
    for atom in atoms:
        z = int(getattr(atom, "atomic_number", 0) or 0)
        out[z_to_symbol.get(z, f"Z{z}")] += 1
    return out


def main():
    setup = json.loads(Path(SALT_JSON).read_text())
    p = pmd.load_file(FINAL_TOP, FINAL_CRD)
    q = sum(a.charge for a in p.atoms)
    counts = collections.Counter(r.name for r in p.residues)

    chloride_res = [r for r in p.residues if len(r.atoms) == 1 and int(getattr(r.atoms[0], "atomic_number", 0) or 0) == 17]
    sodium_res = [r for r in p.residues if len(r.atoms) == 1 and int(getattr(r.atoms[0], "atomic_number", 0) or 0) == 11]
    waters = [r for r in p.residues if r.name.upper() in {"WAT", "HOH", "TIP3", "TP3"}]
    polymer_res = [r for r in p.residues if r.name.upper() == "PVB"]
    polymer_atoms = [a for r in polymer_res for a in r.atoms]
    polymer_elements = element_counts(polymer_atoms)
    n4_sites = sum(1 for a in polymer_atoms if str(getattr(a, "type", "")).lower() == "n4")

    box_ok = p.box is not None and len(p.box) >= 6
    volume = cell_volume_ang3(p.box) if box_ok else float("nan")
    reference_volume = float(setup["box_volume_angstrom3"])
    box_volume_matches_reference = box_ok and abs(volume - reference_volume) < 1.0e-3
    added_pairs = int(setup["added_nacl_pairs"])
    achieved = added_pairs / (AVOGADRO * volume * ANGSTROM3_TO_LITER) if box_ok else float("nan")
    expected_cl = int(setup["expected_final_chloride_count"])
    expected_na = int(setup["expected_final_sodium_count"])


    charge_ok = abs(q) < 1.0e-4
    polymer_ok = (
        len(polymer_res) == 1
        and len(polymer_atoms) == 374
        and polymer_elements.get("C", 0) == 144
        and polymer_elements.get("H", 0) == 218
        and polymer_elements.get("N", 0) == 12
        and sum(polymer_elements.values()) == 374
        and n4_sites == 12
    )
    ions_ok = len(chloride_res) == expected_cl and len(sodium_res) == expected_na
    salt_ok = box_volume_matches_reference and abs(achieved - float(setup["achieved_nominal_added_nacl_molar"])) < 1.0e-8

    log_text = "\n".join(
        Path(name).read_text(errors="replace")
        for name in ("tleap_salt_stdout.log", "tleap_salt_stderr.log", "leap_salt.log")
        if Path(name).exists()
    )
    fatal_patterns = [
        r"\bFATAL\b",
        r"Could not find",
        r"does not have a type",
        r"No torsion terms",
        r"Unknown residue",
    ]
    fatal_hits = [pat for pat in fatal_patterns if re.search(pat, log_text, re.I)]
    leap_ok = not fatal_hits and ("Unit is OK" in log_text)

    print(f"atoms: {len(p.atoms)}")
    print(f"residues: {len(p.residues)}")
    print(f"partial_charge_sum: {q:.8f}")
    print(f"box: {p.box}")
    print(f"box_volume_angstrom3: {volume:.6f}")
    print(f"reference_box_volume_angstrom3: {reference_volume:.6f}")
    print(f"box_volume_matches_reference: {box_volume_matches_reference}")
    print(f"water_residues: {len(waters)}")
    print(f"polymer_residues_PVB: {len(polymer_res)}")
    print(f"polymer_atoms: {len(polymer_atoms)}")
    print(f"polymer_elements: {dict(sorted(polymer_elements.items()))}")
    print(f"polymer_n4_sites: {n4_sites}")
    print(f"neutralizing_chloride: {setup['neutralizing_chloride_count']}")
    print(f"added_nacl_pairs: {added_pairs}")
    print(f"sodium_residues_by_element: {len(sodium_res)}")
    print(f"chloride_residues_by_element: {len(chloride_res)}")
    print(f"expected_sodium: {expected_na}")
    print(f"expected_chloride: {expected_cl}")
    print(f"target_added_nacl_molar: {setup['target_added_nacl_molar']:.8f}")
    print(f"achieved_nominal_added_nacl_molar: {achieved:.8f}")
    print(f"charge_ok: {charge_ok}")
    print(f"polymer_ok: {polymer_ok}")
    print(f"ions_ok: {ions_ok}")
    print(f"salt_concentration_ok: {salt_ok}")
    print(f"periodic_box_ok: {box_ok}")
    print(f"water_present_ok: {len(waters) > 0}")
    print(f"leap_unit_ok: {leap_ok}")
    print(f"fatal_log_patterns: {fatal_hits}")
    print("residue_counts:")
    for k, v in sorted(counts.items()):
        print(f"  {k}: {v}")

    ok = charge_ok and polymer_ok and ions_ok and salt_ok and box_ok and len(waters) > 0 and leap_ok
    raise SystemExit(0 if ok else 2)


if __name__ == "__main__":
    main()
