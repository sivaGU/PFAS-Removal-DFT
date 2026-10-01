#!/usr/bin/env python3
"""Figure 13 PyMOL structures"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent / "shared_helpers"))
import render_structure_3d as mol

SOURCE = HERE / "input_structures"
DEST = HERE / "structure_renders"
NAMES = ("lowest_sampled_grid", "configuration_step_01",
         "configuration_step_08", "configuration_step_12")
COLORS = ("#d62728", "#1f77b4", "#2ca02c")


def contacts_for(structure: mol.Structure) -> tuple[list[mol.Contact], tuple[float, float, float]]:
    elements, xyz = structure.elements, structure.coords
    if len(elements) != 53 or elements[25] != "N":
        raise ValueError("Unexpected 53-atom sampled structure or ammonium N (atom 26)")
    if [elements[i] for i in (17, 18, 52)] != ["O", "O", "Cl"]:
        raise ValueError("Unexpected O/O/Cl partner identities (atoms 18, 19, 53)")
    oxygens = sorted((float(np.linalg.norm(xyz[25] - xyz[i])), i)
                     for i in (17, 18))
    chloride = float(np.linalg.norm(xyz[25] - xyz[52]))
    values = (oxygens[0][0], oxygens[1][0], chloride)
    if not np.isfinite(values).all() or min(values) <= 0:
        raise ValueError(f"Invalid guide lengths: {values}")
    guides = [mol.Contact(i=25, j=oxygens[0][1], color=COLORS[0], label=""),
              mol.Contact(i=25, j=oxygens[1][1], color=COLORS[1], label=""),
              mol.Contact(i=25, j=52, color=COLORS[2], label="")]
    return guides, values


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mamba", type=Path, default=mol.PYMOL_RUNNER)
    parser.add_argument("--env", default=mol.PYMOL_ENV)
    parser.add_argument("--audit-only", action="store_true",
                        help="Validate and print distances without launching PyMOL")
    args = parser.parse_args()
    mol.PYMOL_RUNNER = args.mamba
    mol.PYMOL_ENV = args.env
    prepared = []
    for name in NAMES:
        path = SOURCE / f"{name}.xyz"
        structure = mol.read_xyz(path, 0)
        guides, values = contacts_for(structure)
        prepared.append((name, path, structure, guides))
        print(f"{name}: red O–N {values[0]:.3f} Å, blue O–N "
              f"{values[1]:.3f} Å, green N–Cl {values[2]:.3f} Å", flush=True)

    if args.audit_only:
        return

    DEST.mkdir(exist_ok=True)
    for name, path, structure, guides in prepared:
        is_selected = name != NAMES[0]
        output = DEST / f"{name}.png"
        mol.draw_structure_pymol(
            input_path=path, structure=structure, out_path=output,
            contacts=guides, hide_h=False, show_indices=False, title=None,
            frame=0, dpi=650, width=3200 if is_selected else 3600,
            height=2600 if is_selected else 2800, no_bonds=False,
            sphere_scale=.25, stick_radius=.13, ray=True,
            pymol_projection="orthoscopic", no_crop=is_selected, keep_script=None,
        )
        print(f"PyMOL rendered {output}", flush=True)


if __name__ == "__main__":
    main()
