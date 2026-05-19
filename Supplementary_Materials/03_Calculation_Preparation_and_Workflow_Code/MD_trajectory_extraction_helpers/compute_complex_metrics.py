#!/usr/bin/env python3
from __future__ import annotations

import csv
import math
import sys
from pathlib import Path

import numpy as np

SITE_ATOMS = {176, 177, 178, 179, 180}
PFO_CARBOXYLATE = {"O", "O1"}
PFO_TAIL = {
    "F", "F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8", "F9", "F10",
    "F11", "F12", "F13", "F14", "C", "C1", "C2", "C3", "C4", "C5", "C6",
}
CONTACT_CUTOFF = 4.0


def pdb_atom(line: str) -> dict:
    return {
        "serial": int(line[6:11]),
        "name": line[12:16].strip(),
        "resname": line[17:20].strip(),
        "x": float(line[30:38]),
        "y": float(line[38:46]),
        "z": float(line[46:54]),
        "element": line[76:78].strip(),
    }


def frames(path: Path):
    if not path.exists():
        parts = sorted(path.parent.glob(path.name + ".*"), key=lambda p: int(p.suffix[1:]))
        for part in parts:
            yield from frames(part)
        return
    frame = []
    saw_model = False
    with path.open() as handle:
        for line in handle:
            rec = line[:6].strip()
            if rec == "MODEL":
                saw_model = True
                frame = []
            elif rec in {"ATOM", "HETATM"}:
                frame.append(pdb_atom(line))
            elif rec == "ENDMDL":
                yield frame
                frame = []
    if frame and not saw_model:
        yield frame


def com(atoms: list[dict]) -> np.ndarray:
    xyz = np.array([[a["x"], a["y"], a["z"]] for a in atoms], dtype=float)
    return xyz.mean(axis=0)


def min_distance(point: np.ndarray, atoms: list[dict]) -> float:
    xyz = np.array([[a["x"], a["y"], a["z"]] for a in atoms], dtype=float)
    return float(np.linalg.norm(xyz - point, axis=1).min())


def contact_count(tail_atoms: list[dict], resin_atoms: list[dict]) -> int:
    tail = np.array([[a["x"], a["y"], a["z"]] for a in tail_atoms], dtype=float)
    resin = np.array([[a["x"], a["y"], a["z"]] for a in resin_atoms], dtype=float)
    count = 0
    cutoff2 = CONTACT_CUTOFF * CONTACT_CUTOFF
    for r in resin:
        delta = tail - r
        if np.any(np.einsum("ij,ij->i", delta, delta) <= cutoff2):
            count += 1
    return count


def is_heavy(atom: dict) -> bool:
    return atom["element"] != "H" and not atom["name"].startswith("H")


def main() -> int:
    if len(sys.argv) != 3:
        raise SystemExit("usage: compute_complex_metrics.py frames.pdb output.csv")
    infile = Path(sys.argv[1])
    outfile = Path(sys.argv[2])

    rows = []
    for idx, atoms in enumerate(frames(infile), start=1):
        site = [a for a in atoms if a["serial"] in SITE_ATOMS]
        carbox = [a for a in atoms if a["resname"] == "PFO" and a["name"] in PFO_CARBOXYLATE]
        tail = [a for a in atoms if a["resname"] == "PFO" and a["name"] in PFO_TAIL]
        chlorides = [a for a in atoms if a["resname"].startswith("Cl") or a["name"].upper().startswith("CL")]
        resin_heavy = [a for a in atoms if a["resname"] == "R48" and is_heavy(a)]
        if not site or len(carbox) != 2 or not tail or not chlorides:
            raise SystemExit(f"Missing expected atom selection in frame {idx}")

        site_com = com(site)
        carbox_com = com(carbox)
        pfoa_site = float(np.linalg.norm(carbox_com - site_com))
        chloride_site = min_distance(site_com, chlorides)
        contacts = contact_count(tail, resin_heavy)
        rows.append(
            {
                "frame": idx,
                "pfoa_carboxylate_site_distance_A": f"{pfoa_site:.4f}",
                "nearest_chloride_site_distance_A": f"{chloride_site:.4f}",
                "pfoa_tail_resin_heavy_contacts_4A": contacts,
            }
        )

    with outfile.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} frames to {outfile}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
