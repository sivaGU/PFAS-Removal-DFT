"""Figure 4 molecular assets"""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "input_structures"
DEST = HERE / "structure_renders"
RADII = {"H": .31, "C": .76, "N": .71, "O": .66, "F": .57, "S": 1.05}
COLORS = {"H": "#e5e7e9", "C": "#3e454b", "N": "#2470b6",
          "O": "#d13536", "F": "#4db9ad", "S": "#d6a52c"}
PFAS = ("6-2_FTCA", "PFHxA", "PFOA", "PFOS")
METHODS = ("r2SCAN-3c", "wB97X-D3")


def read_xyz(path):
    rows = path.read_text().splitlines()
    count = int(rows[0]); atoms = [line.split() for line in rows[2:2+count]]
    assert len(atoms) == count
    elements = [row[0].title() for row in atoms]
    coords = np.array([[float(s) for s in row[1:4]] for row in atoms])
    return elements, coords


def project(elements, coords):
    nitrogen = [i for i,e in enumerate(elements) if e == "N"]
    oxygen = [i for i,e in enumerate(elements) if e == "O"]
    assert len(nitrogen) == 1 and len(oxygen) in (2, 3)
    ni = nitrogen[0]
    heavy = [i for i,e in enumerate(elements) if e != "H"]
    fluorine = [i for i,e in enumerate(elements) if e == "F"]
    origin = coords[ni]
    x = origin - coords[fluorine].mean(axis=0)
    x /= np.linalg.norm(x)
    center = coords[heavy] - coords[heavy].mean(axis=0)
    residual = center - np.outer(center @ x, x)
    _, _, vt = np.linalg.svd(residual, full_matrices=False)
    y = vt[0]
    y -= x * np.dot(y, x)
    y /= np.linalg.norm(y)
    z = np.cross(x, y)
    xy = (coords - coords[heavy].mean(axis=0)) @ np.array([x,y,z]).T
    assert xy[ni,0] > xy[fluorine,0].mean()
    return xy, ni, oxygen


def bonds(elements, xyz):
    for i in range(len(elements)):
        for j in range(i+1,len(elements)):
            if np.linalg.norm(xyz[i]-xyz[j]) <= 1.24*(RADII[elements[i]]+RADII[elements[j]]):
                if elements[i] == elements[j] == "H":
                    continue
                yield i,j


def render_one(method, pfas, output, dpi):
    src = SOURCE / f"{method}_{pfas}.xyz"
    elements, xyz = read_xyz(src)
    plane, ni, oxygen = project(elements,xyz)
    fig, ax = plt.subplots(figsize=(4.5, 2.7))
    fig.patch.set_alpha(0)
    ax.patch.set_alpha(0)
    depth = plane[:,2]
    for i,j in bonds(elements,xyz):
        width = 1.0 if "H" in (elements[i],elements[j]) else 2.5
        mid = (plane[i,:2]+plane[j,:2])/2
        for k, start, end in ((i,plane[i,:2],mid),(j,mid,plane[j,:2])):
            ax.plot([start[0],end[0]],[start[1],end[1]],
                    color=COLORS[elements[k]], lw=width, solid_capstyle="round", zorder=2)
    edges = list(bonds(elements, xyz))
    carbon_neighbors = {j if i == ni else i for i,j in edges
                        if ni in (i,j) and elements[j if i == ni else i] == "C"}
    resin_hydrogens = {j if i in carbon_neighbors else i for i,j in edges
                       if (i in carbon_neighbors or j in carbon_neighbors)
                       and elements[j if i in carbon_neighbors else i] == "H"}
    contacts = sorted((np.linalg.norm(xyz[o]-xyz[h]),o,h)
                      for o in oxygen for h in resin_hydrogens)
    for index in np.argsort(depth):
        el=elements[index]
        ax.scatter(*plane[index,:2],s=(11 if el=="H" else 43 if el=="C" else 49),
                   facecolor=COLORS[el], edgecolor="#666666" if el!="H" else "none",
                   lw=.3, zorder=3)
    mins,maxes=plane[:,:2].min(axis=0),plane[:,:2].max(axis=0)
    span=max(maxes-mins)
    center=(mins+maxes)/2
    ax.set(xlim=(center[0]-.59*span,center[0]+.59*span),
           ylim=(center[1]-.37*span,center[1]+.37*span))
    ax.set_aspect("equal")
    ax.axis("off")
    output.parent.mkdir(parents=True, exist_ok=True)
    raw=output.with_name(output.stem+".raw.png")
    fig.savefig(raw,dpi=dpi,transparent=True,bbox_inches="tight",pad_inches=.02)
    plt.close(fig)
    with Image.open(raw) as im:
        rgba=im.convert("RGBA")
        bbox=rgba.getchannel("A").getbbox()
        if not bbox: raise ValueError("Empty render")
        pad=int(.05*max(bbox[2]-bbox[0],bbox[3]-bbox[1]))
        bbox=(max(0,bbox[0]-pad),max(0,bbox[1]-pad),
              min(rgba.width,bbox[2]+pad),min(rgba.height,bbox[3]+pad))
        trimmed=output.with_name(output.stem+".trimmed.png")
        rgba.crop(bbox).save(trimmed,dpi=(dpi,dpi))
    trimmed.replace(output)
    raw.unlink()
    with Image.open(output) as check:
        check.verify()
    return [(round(float(d),3),o+1,h+1) for d,o,h in contacts[:4]]


def main(dpi=1200):
    records=[]
    for method in METHODS:
        for pfas in PFAS:
            out=DEST / f"{method}_{pfas}.png"
            contacts=render_one(method,pfas,out,dpi)
            records.append(dict(method=method,pfas="6:2 FTCA" if pfas=="6-2_FTCA" else pfas,
                geometry=SOURCE.joinpath(f"{method}_{pfas}.xyz").name,
                render=out.name,nearest_O_H_A=contacts[0][0],
                second_O_H_A=contacts[1][0],
                nearest_O_atom_index=contacts[0][1],nearest_H_atom_index=contacts[0][2],
                second_O_atom_index=contacts[1][1],second_H_atom_index=contacts[1][2]))
            print(method,pfas,contacts,out)
    with (HERE / "input_data/structure_contacts.csv").open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=records[0]);writer.writeheader();writer.writerows(records)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dpi",type=int,default=1200)
    args=parser.parse_args();main(args.dpi)
