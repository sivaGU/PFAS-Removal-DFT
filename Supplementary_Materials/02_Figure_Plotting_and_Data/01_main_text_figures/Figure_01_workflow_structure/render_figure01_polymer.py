#!/usr/bin/env python3
"""Figure 1 polymer asset"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


HERE = Path(__file__).resolve().parent
SOURCE = HERE / "input_data/pVBTMA12_gaff2_rct.mol2"


def parse_mol2(path):
    sections = {}
    section = None
    for line in path.read_text().splitlines():
        if line.startswith("@<TRIPOS>"):
            section = line[9:]
            sections[section] = []
        elif section is not None and line.strip():
            sections[section].append(line.split())
    atoms = sections["ATOM"]
    bonds = sections["BOND"]
    if len(atoms) != 374 or len(bonds) != 385:
        raise ValueError("Expected the validated pVBTMA12 model (374 atoms, 385 bonds)")
    xyz = np.array([[float(x) for x in a[2:5]] for a in atoms])
    elements = [a[1][0].upper() for a in atoms]
    if (elements.count("N"), elements.count("C"), elements.count("H")) != (12, 144, 218):
        raise ValueError("Unexpected polymer elemental composition")
    edges = [(int(b[1])-1, int(b[2])-1) for b in bonds]
    return xyz, elements, edges


def render(output, dpi):
    xyz, elements, bonds = parse_mol2(SOURCE)
    xyz = xyz - xyz.mean(axis=0)
    carbon, nitrogen = "#4d4d4d", "#354f9c"
    colors = {"C": carbon, "N": nitrogen}
    fig = plt.figure(figsize=(4.6, 4.0), facecolor="none")
    ax = fig.add_axes((.015, .015, .97, .97), projection="3d", facecolor="none")
    for a, b in bonds:
        if "H" in (elements[a], elements[b]):
            continue
        midpoint = (xyz[a] + xyz[b])/2
        for start, stop, el in ((xyz[a], midpoint, elements[a]),
                                (midpoint, xyz[b], elements[b])):
            ax.plot(*np.array((start, stop)).T, color=colors[el], linewidth=2.35,
                    solid_capstyle="round", zorder=1)
    for el, size, edge in (("C", 23, "#333333"), ("N", 80, "#263875")):
        points = xyz[np.array(elements) == el]
        ax.scatter(*points.T, s=size, c=colors[el], edgecolor=edge,
                   linewidth=.35, depthshade=True, zorder=2)
    span = max(np.ptp(xyz, axis=0)) * .55
    center = (xyz.max(axis=0) + xyz.min(axis=0))/2
    ax.set(xlim=(center[0]-span,center[0]+span),
           ylim=(center[1]-span,center[1]+span),
           zlim=(center[2]-span,center[2]+span))
    ax.set_box_aspect((1, 1, 1), zoom=1.20)
    ax.view_init(elev=20, azim=-58)
    ax.set_axis_off()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi, transparent=True, bbox_inches="tight", pad_inches=.01)
    plt.close(fig)
    with Image.open(output) as raw:
        im = raw.convert("RGBA")
    mask = im.getchannel("A")
    bounds = mask.getbbox()
    if bounds is None:
        raise ValueError("Empty polymer render")
    padding = round(.035 * max(bounds[2]-bounds[0], bounds[3]-bounds[1]))
    bounds = (max(0, bounds[0]-padding), max(0, bounds[1]-padding),
              min(im.width, bounds[2]+padding), min(im.height, bounds[3]+padding))
    trimmed = output.with_name(output.stem + ".trimmed.png")
    im.crop(bounds).save(trimmed, dpi=(dpi, dpi))
    trimmed.replace(output)
    print(f"Rendered 12 N, 144 C and {sum('H' not in (elements[a],elements[b]) for a,b in bonds)} heavy-atom bonds: {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=HERE / "structure_renders/pVBTMA12_workflow_step06.png")
    parser.add_argument("--dpi", type=int, default=1200)
    args = parser.parse_args()
    render(args.output, args.dpi)
