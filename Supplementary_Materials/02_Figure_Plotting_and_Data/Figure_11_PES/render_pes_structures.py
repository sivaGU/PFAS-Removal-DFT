#!/usr/bin/env python3
"""Render the molecular structure panels used in Figure 11."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
HELPER = THIS_DIR.parent / "shared_helpers" / "render_structure_3d.py"
INPUT_DIR = THIS_DIR / "input_structures"
RENDER_DIR = THIS_DIR / "structure_renders"

RENDERS = [
    {
        "input": "panel_b_global_minimum.xyz",
        "output": "panel_b_global_minimum.png",
        "width": 2600,
        "height": 2050,
        "sphere": 0.25,
        "stick": 0.13,
        "contacts": [
            "26:18:#d62728:nolabel",
            "26:19:#1f77b4:nolabel",
            "26:53:#2ca02c:nolabel",
        ],
    },
    {
        "input": "panel_c_approach.xyz",
        "output": "panel_c_approach.png",
        "width": 1600,
        "height": 1150,
        "sphere": 0.22,
        "stick": 0.11,
        "contacts": [],
    },
    {
        "input": "panel_c_displacement.xyz",
        "output": "panel_c_displacement.png",
        "width": 1600,
        "height": 1150,
        "sphere": 0.22,
        "stick": 0.11,
        "contacts": [],
    },
    {
        "input": "panel_c_bound.xyz",
        "output": "panel_c_bound.png",
        "width": 1600,
        "height": 1150,
        "sphere": 0.22,
        "stick": 0.11,
        "contacts": [],
    },
]


def render_one(spec: dict[str, object]) -> Path:
    input_path = INPUT_DIR / str(spec["input"])
    output_path = RENDER_DIR / str(spec["output"])
    if not input_path.exists():
        raise FileNotFoundError(f"Missing source structure: {input_path}")
    if not HELPER.exists():
        raise FileNotFoundError(f"Missing shared render helper: {HELPER}")

    cmd = [
        sys.executable,
        str(HELPER),
        str(input_path),
        "--backend",
        "pymol",
        "--out",
        str(output_path),
        "--width",
        str(spec["width"]),
        "--height",
        str(spec["height"]),
        "--dpi",
        "650",
        "--pymol-projection",
        "orthoscopic",
        "--pymol-sphere-scale",
        str(spec["sphere"]),
        "--pymol-stick-radius",
        str(spec["stick"]),
    ]
    for contact in spec["contacts"]:
        cmd.extend(["--contact", str(contact)])

    RENDER_DIR.mkdir(exist_ok=True)
    subprocess.run(cmd, cwd=THIS_DIR, check=True)
    return output_path


def main() -> None:
    for spec in RENDERS:
        print(f"Rendering {spec['output']}...")
        render_one(spec)


if __name__ == "__main__":
    main()
