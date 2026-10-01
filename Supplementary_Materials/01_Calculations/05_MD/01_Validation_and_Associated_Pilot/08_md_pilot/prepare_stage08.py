#!/usr/bin/env python3

from __future__ import annotations
from pathlib import Path
import json, math, re, sys
from collections import defaultdict, deque

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
STAGE07 = ROOT / "07_solvation_ion_validation"
STAGE04 = ROOT / "04_charge_parameterization"
PRMTOP = STAGE07 / "pVBTMA12_12Cl_015MNaCl_pilot.prmtop"
INPCRD = STAGE07 / "pVBTMA12_12Cl_015MNaCl_pilot.inpcrd"
MOL2 = STAGE04 / "pVBTMA12_gaff2_rct.mol2"
VAL07 = STAGE07 / "solvation_validation.txt"
COMP07 = STAGE07 / "stage07_completion.txt"
EXPECTED_ATOMS = 15973
EXPECTED_POLYMER_ATOMS = 374
EXPECTED_BOX = (55.366399, 60.206530, 61.625519)


def die(msg):
    raise SystemExit(f"ERROR: {msg}")


def parse_inpcrd(path: Path):
    lines = path.read_text(errors="replace").splitlines()
    if len(lines) < 3:
        die(f"Malformed INPCRD: {path}")
    m = re.match(r"\s*(\d+)", lines[1])
    if not m:
        die("Could not parse atom count from INPCRD line 2")
    natom = int(m.group(1))
    vals = []
    for line in lines[2:]:
        vals.extend(float(x) for x in line.split())
    need = 3 * natom
    if len(vals) < need + 3:
        die(f"INPCRD lacks periodic box values; parsed {len(vals)} numeric values after atom count")
    boxvals = vals[need:]
    if len(boxvals) < 3:
        die("Could not parse box lengths")
    xyz = tuple(boxvals[:3])
    ang = tuple(boxvals[3:6]) if len(boxvals) >= 6 else (90.0, 90.0, 90.0)
    return natom, xyz, ang


def parse_mol2(path: Path):
    lines = path.read_text(errors="replace").splitlines()
    try:
        ia = lines.index("@<TRIPOS>ATOM") + 1
        ib = lines.index("@<TRIPOS>BOND")
    except ValueError:
        die("MOL2 missing ATOM/BOND sections")
    atoms = {}
    for line in lines[ia:ib]:
        if not line.strip() or line.startswith("@<TRIPOS>"):
            continue
        p = line.split()
        idx = int(p[0]); name = p[1]; typ = p[5]
        element = re.match(r"[A-Za-z]+", name).group(0)[0].upper()
        atoms[idx] = {"name": name, "type": typ, "element": element}
    bonds = []
    for line in lines[ib+1:]:
        if line.startswith("@<TRIPOS>"):
            break
        if not line.strip():
            continue
        p = line.split()
        bonds.append((int(p[1]), int(p[2]), p[3]))
    return atoms, bonds


def derive_repeat_groups(atoms, bonds):


    adj = defaultdict(set)
    for a,b,_ in bonds:
        adj[a].add(b); adj[b].add(a)
    aliph = {i for i,a in atoms.items() if a["type"] == "c3" and a["element"] == "C"}

    cand = {i for i in aliph if any(j in aliph for j in adj[i])}
    cadj = {i:{j for j in adj[i] if j in cand} for i in cand}


    endpoints = [i for i in cand if len(cadj[i]) == 1]
    best = []
    for s in endpoints:
        parent = {s:None}; q = deque([s])
        while q:
            u=q.popleft()
            for v in cadj[u]:
                if v not in parent:
                    parent[v]=u; q.append(v)
        for t in endpoints:
            if t not in parent: continue
            path=[]; x=t
            while x is not None:
                path.append(x); x=parent[x]
            path.reverse()
            if len(path)>len(best): best=path
    if len(best) != 24:
        die(f"Expected 24-carbon backbone path, derived {len(best)} atoms: {best}")


    def aromatic_neighbor(i):
        return any(atoms[j]["type"] == "ca" for j in adj[i])
    phases=[]
    for phase in (0,1):
        anchors=best[phase::2]
        phases.append((sum(aromatic_neighbor(i) for i in anchors), anchors))
    score, anchors=max(phases, key=lambda x:x[0])
    if len(anchors)!=12 or score!=12:
        die(f"Could not derive 12 aryl-bearing repeat anchors (score={score}, anchors={anchors})")


    if best.index(anchors[0]) > best.index(anchors[-1]):
        anchors=list(reversed(anchors))


    if anchors[0] > anchors[-1]:
        anchors=list(reversed(anchors))


    dist_by_anchor=[]
    for anchor in anchors:
        d={anchor:0}; q=deque([anchor])
        while q:
            u=q.popleft()
            for v in adj[u]:
                if v not in d:
                    d[v]=d[u]+1; q.append(v)
        dist_by_anchor.append(d)
    groups={k+1:[] for k in range(12)}
    for atom in sorted(atoms):
        ds=[d.get(atom,10**9) for d in dist_by_anchor]
        k=min(range(12), key=lambda x:(ds[x],x))
        groups[k+1].append(atom)

    flat=sorted(i for g in groups.values() for i in g)
    if flat != list(range(1, EXPECTED_POLYMER_ATOMS+1)):
        die("Repeat partition does not cover polymer atoms 1-374 exactly")
    return best, anchors, groups


def main():
    for p in (PRMTOP, INPCRD, MOL2, VAL07, COMP07):
        if not p.is_file() or p.stat().st_size == 0:
            die(f"Required upstream file missing or empty: {p}")
    valtext=(VAL07.read_text(errors="replace")+"\n"+COMP07.read_text(errors="replace")).lower()
    if "fail" in valtext and "pass" not in valtext:
        die("Stage 07 reports contain a failure marker without a pass marker; inspect before Stage 08")

    natom, box, ang = parse_inpcrd(INPCRD)
    if natom != EXPECTED_ATOMS:
        die(f"Stage 07 atom count is {natom}; expected {EXPECTED_ATOMS}")
    if any(abs(a-b)>0.02 for a,b in zip(box, EXPECTED_BOX)):
        die(f"Stage 07 box differs materially from accepted box: {box}")
    if any(abs(a-90.0)>1e-3 for a in ang):
        die(f"Expected orthorhombic box, got angles {ang}")

    atoms,bonds=parse_mol2(MOL2)
    if len(atoms)!=EXPECTED_POLYMER_ATOMS:
        die(f"Production MOL2 contains {len(atoms)} atoms; expected 374")
    backbone,anchors,groups=derive_repeat_groups(atoms,bonds)
    heavy={r:[i for i in ids if atoms[i]["element"]!="H"] for r,ids in groups.items()}

    system_conf = HERE / "stage08_system.generated.conf"
    system_conf.write_text(
        f'set stage07_prmtop "../07_solvation_ion_validation/{PRMTOP.name}"\n'
        f'set stage07_inpcrd "../07_solvation_ion_validation/{INPCRD.name}"\n'
        f'set boxX {box[0]:.6f}\nset boxY {box[1]:.6f}\nset boxZ {box[2]:.6f}\n'
    )
    record={
        "stage07_prmtop":str(PRMTOP.relative_to(ROOT)),
        "stage07_inpcrd":str(INPCRD.relative_to(ROOT)),
        "production_mol2":str(MOL2.relative_to(ROOT)),
        "natom":natom,
        "box_A":box,"box_angles_deg":ang,
        "backbone_atom_indices_1based":backbone,
        "repeat_anchor_atom_indices_1based":anchors,
        "repeat_groups_1based":{str(k):v for k,v in groups.items()},
        "repeat_heavy_groups_1based":{str(k):v for k,v in heavy.items()},
        "terminal_repeats":[1,12],"central_repeats":[5,6,7,8],
        "contact_cutoff_A":4.5,
        "predeclared_review_flags":{"terminal_center_occupancy_gt":0.25,"terminal_center_contiguous_ps_gt":100.0},
        "periodic_image":{"hard_fail_below_A":9.0,"review_below_A":11.0},
    }
    (HERE/"repeat_groups.json").write_text(json.dumps(record,indent=2)+"\n")
    print(json.dumps({"status":"PASS","natom":natom,"box_A":box,"repeat_group_sizes":{k:len(v) for k,v in groups.items()},"anchors":anchors},indent=2))

if __name__ == "__main__": main()
