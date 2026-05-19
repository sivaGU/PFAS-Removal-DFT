#!/usr/bin/env python3
from __future__ import annotations

import csv
import math
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FRAMES = ROOT.parent / "13_mmgba_pfoa_qualitative"
OUT = ROOT / "05_metrics"

PFOA_CARBOXYLATE_O = {"O", "O1"}
PFOA_TAIL = {
    "F", "F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8", "F9", "F10",
    "F11", "F12", "F13", "F14", "C", "C1", "C2", "C3", "C4", "C5", "C6",
}


def atom(line: str) -> dict:
    return {
        "serial": int(line[6:11]),
        "name": line[12:16].strip(),
        "resname": line[17:20].strip(),
        "xyz": np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])]),
        "element": line[76:78].strip(),
    }


def atoms_from_pdb(path: Path) -> list[dict]:
    return [atom(line) for line in path.open() if line.startswith(("ATOM", "HETATM"))]


def is_heavy(a: dict) -> bool:
    return a["element"] != "H" and not a["name"].startswith("H")


def min_pair(a_atoms: list[dict], b_atoms: list[dict]):
    best = None
    for a in a_atoms:
        for b in b_atoms:
            d = float(np.linalg.norm(a["xyz"] - b["xyz"]))
            if best is None or d < best[0]:
                best = (d, a, b)
    return best


def contacts(tail: list[dict], resin: list[dict], cutoff: float = 4.0) -> int:
    tail_xyz = np.array([a["xyz"] for a in tail])
    cutoff2 = cutoff * cutoff
    count = 0
    for r in resin:
        delta = tail_xyz - r["xyz"]
        if np.any(np.einsum("ij,ij->i", delta, delta) <= cutoff2):
            count += 1
    return count


def chloride_counts(center: np.ndarray, chlorides: list[dict]) -> tuple[int, int, int]:
    ds = [float(np.linalg.norm(cl["xyz"] - center)) for cl in chlorides]
    return tuple(sum(d <= cutoff for d in ds) for cutoff in (5.0, 7.0, 10.0))


rows = []
nearest_ns = Counter()
files = sorted(FRAMES.glob("bound_complex_frames.pdb.*"), key=lambda p: int(p.suffix[1:]))
for i, pdb in enumerate(files, start=1):
    atoms = atoms_from_pdb(pdb)
    carbox_o = [a for a in atoms if a["resname"] == "PFO" and a["name"] in PFOA_CARBOXYLATE_O]
    tail = [a for a in atoms if a["resname"] == "PFO" and a["name"] in PFOA_TAIL]
    ammonium_n = [a for a in atoms if a["resname"] == "R48" and a["name"].startswith("N")]
    chlorides = [a for a in atoms if a["resname"].startswith("Cl") or a["name"].upper().startswith("CL")]
    resin_heavy = [a for a in atoms if a["resname"] == "R48" and is_heavy(a)]

    d, o_atom, n_atom = min_pair(carbox_o, ammonium_n)
    nearest_ns[(n_atom["serial"], n_atom["name"])] += 1
    c5, c7, c10 = chloride_counts(n_atom["xyz"], chlorides)
    rows.append(
        {
            "frame": i,
            "nearest_carboxylate_O_to_any_resin_N_A": f"{d:.4f}",
            "nearest_carboxylate_O_name": o_atom["name"],
            "nearest_resin_N_serial": n_atom["serial"],
            "nearest_resin_N_name": n_atom["name"],
            "pfoa_tail_resin_heavy_contacts_4A": contacts(tail, resin_heavy),
            "chlorides_within_5A_of_nearest_N": c5,
            "chlorides_within_7A_of_nearest_N": c7,
            "chlorides_within_10A_of_nearest_N": c10,
        }
    )

csv_path = OUT / "pfoa_association_mechanism_metrics.csv"
with csv_path.open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)

summary_path = OUT / "pfoa_association_mechanism_metrics_summary.txt"
dist = [float(r["nearest_carboxylate_O_to_any_resin_N_A"]) for r in rows]
tail_contacts = [float(r["pfoa_tail_resin_heavy_contacts_4A"]) for r in rows]
with summary_path.open("w") as handle:
    handle.write(f"frames {len(rows)}\n")
    handle.write(f"nearest O-N mean {np.mean(dist):.4f} A\n")
    handle.write(f"nearest O-N median {np.median(dist):.4f} A\n")
    handle.write(f"nearest O-N min {np.min(dist):.4f} A\n")
    handle.write(f"nearest O-N max {np.max(dist):.4f} A\n")
    for cutoff in (3.5, 4.0, 5.0, 6.0):
        handle.write(f"frames nearest O-N <= {cutoff:.1f} A {sum(d <= cutoff for d in dist)}\n")
    handle.write(f"tail contacts mean {np.mean(tail_contacts):.4f}\n")
    handle.write(f"tail contacts min {np.min(tail_contacts):.0f}\n")
    handle.write(f"tail contacts max {np.max(tail_contacts):.0f}\n")
    handle.write("nearest resin N counts\n")
    for (serial, name), count in nearest_ns.most_common():
        handle.write(f"{serial} {name} {count}\n")

print(summary_path.read_text())
