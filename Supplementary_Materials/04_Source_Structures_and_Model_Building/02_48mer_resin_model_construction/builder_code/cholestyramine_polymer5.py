"""Cholestyramine-like resin builder with DVB crosslinks."""

# =========================
# SETTINGS (EDIT THIS ONLY)
# =========================
SETTINGS = {
    "TOTAL_MONOMERS": 48,

    "CROSSLINK_FRACTION": 0.02,
    "ENSURE_CONNECTED_NETWORK": True,
    "INTERCHAIN_ONLY": True,
    "AVOID_END_UNITS": 2,
    "FAIL_ON_SKIPPED_CROSSLINKS": False,

    "SEED": 1,
    "TOPOLOGY_ONLY": True,

    "SMALL_EMBED_MAX_ITERS": 600,
    "SMALL_EMBED_ATTEMPTS": 30,

    "SKIP_GEOMETRY_RELAXATION": True,
    "STRAIGHT_BACKBONE_ALTERNATE_180": True,
    "DYNAMIC_BRANCH_DIRECTIONS": True,
    "DYNAMIC_DIRECTION_POOL_SIZE": 48,
    "DYNAMIC_DIRECTION_TRIES_PER_CHAIN": 18,
    "MIN_INTERCHAIN_ATOM_DIST_A": 1.25,

    "LOCAL_RELAX": True,
    "LOCAL_ITERS": 25,
    "FREEZE_RADIUS_BONDS": 4,

    "RELAX_RETRIES": 3,
    "RELAX_JITTER_A": 0.05,
    "RELAX_BACKOFF": 0.5,

    "CONTINUOUS_EMBED": True,
    "CONTINUOUS_EMBED_MAX_ITERS": 400,
    "CONTINUOUS_EMBED_ATTEMPTS": 5,
    "CONTINUOUS_EMBED_RETRIES": 2,
    "CONTINUOUS_EMBED_AFTER_IONS": True,

    "ION_RELAX_AFTER_ADD": True,
    "ION_RELAX_ITERS": 40,
    "ION_RELAX_RETRIES": 1,
    "ION_RELAX_JITTER_A": 0.05,

    "BOND_LENGTH_A": 1.54,
    "INITIAL_CHAIN_SPACING": 14.0,
    "AUTO_PARALLEL_SPACING": True,
    "INTERCHAIN_CLEARANCE_A": 3.0,
    "MAX_XLINK_DIST_A": 11.0,
    "AUTO_ADAPT_XLINK_DIST": True,
    "AUTO_ADAPT_XLINK_STEP_A": 2.0,
    "AUTO_ADAPT_XLINK_MAX_A": 24.0,
    "XLINK_HARD_MAX_DIST_A": 7.5,
    "XLINK_RANK_PENALTY_A": 1.5,
    "XLINK_MIN_UNIT_SEPARATION": 4,
    "PLACEMENT_MIN_DIST_A": 1.2,
    "PLACEMENT_TRIES": 30,

    "ADD_COUNTERIONS": False,
    "COUNTERION_SMILES": "[Cl-]",
    "ION_DISTANCE_A": 4.0,
    "ION_MIN_DIST_A": 2.2,
    "ION_MIN_ANCHOR_DIST_A": 3.2,
    "ION_TRIES": 60,

    "ADD_HS_AT_END": True,
    "FAIL_ON_CLOSE_CONTACTS": True,
    "CLOSE_CONTACT_MIN_DIST_A": 0.9,
    "CLOSE_CONTACT_HEAVY_ONLY": True,

    "FINAL_GLOBAL_RELAX": True,
    "FINAL_GLOBAL_ITERS": 200,
    "FINAL_GLOBAL_RETRIES": 2,
    "FINAL_GLOBAL_JITTER_A": 0.08,
    "POST_ION_GLOBAL_RELAX": True,
    "POST_ION_GLOBAL_ITERS": 120,
    "POST_ION_GLOBAL_RETRIES": 1,
    "POST_ION_GLOBAL_JITTER_A": 0.06,
    "INTERMEDIATE_GLOBAL_RELAX": True,
    "INTERMEDIATE_GLOBAL_ITERS": 120,
    "INTERMEDIATE_GLOBAL_RETRIES": 2,
    "INTERMEDIATE_GLOBAL_JITTER_A": 0.06,

    "OUT_SDF": "cholestyramine_multichain.sdf",
    "OUT_PDB": "cholestyramine_multichain.pdb",
    "OUT_SMILES": "cholestyramine_multichain.smi",
    "OUT_TOPO_SMILES": "cholestyramine_multichain_topology.smi",
    "OUT_TOPO_MOL": "cholestyramine_multichain_topology.mol",

    "SILENCE_RDKIT_WARNINGS": True,
}

# =========================
# CODE (DON'T EDIT BELOW)
# =========================
import math
import random
import sys
from typing import List, Optional, Set, Tuple

from rdkit import Chem, rdBase
from rdkit.Chem import AllChem, Descriptors
from rdkit.Chem.rdchem import AtomValenceException
from rdkit.Geometry import Point3D


MONOMER_SMILES = "CC(c1ccc(C[N+](C)(C)C)cc1)C"

DVB_RING_SMILES = "c1ccccc1"

# ---------------- Progress bar ----------------
class ProgressBar:
    def __init__(self, total: int, width: int = 44, label: str = "Assembling"):
        self.total = max(1, int(total))
        self.width = int(width)
        self.label = label
        self.current = 0
        self._last_pct = -1

    def advance(self, delta: int):
        self.current = min(self.total, self.current + int(delta))
        self.draw()

    def draw(self, force: bool = False):
        frac = self.current / self.total
        pct = int(round(frac * 100))
        if not force and pct == self._last_pct:
            return
        self._last_pct = pct
        filled = int(round(frac * self.width))
        bar = "â–ˆ" * filled + " " * (self.width - filled)
        sys.stdout.write(f"\r{self.label}: |{bar}| {pct:3d}% ({self.current}/{self.total} heavy atoms)")
        sys.stdout.flush()

    def finish(self, msg: str = "Done"):
        self.current = self.total
        self.draw(force=True)
        sys.stdout.write(f"\n{msg}\n")
        sys.stdout.flush()


# ---------------- Utilities ----------------
def partition_counts(total: int, parts: int) -> List[int]:
    """Split total into parts."""
    if parts <= 0:
        return []
    base = total // parts
    rem = total % parts
    return [base + (1 if i < rem else 0) for i in range(parts)]


def evenly_spaced_indices(n: int, k: int, avoid_ends: int) -> List[int]:
    if k <= 0:
        return []
    start = max(0, avoid_ends)
    end = max(start + 1, n - avoid_ends)
    if end - start <= 1:
        return [start] * k
    span = (end - start - 1)
    return [start + round((i + 1) * span / (k + 1)) for i in range(k)]


def ensure_conformer(mol: Chem.Mol) -> Chem.Mol:
    if mol.GetNumConformers() == 0:
        conf = Chem.Conformer(mol.GetNumAtoms())
        mol.AddConformer(conf, assignId=True)
    return mol


def copy_coords(dst: Chem.Mol, src: Chem.Mol, dst_offset: int):
    dst = ensure_conformer(dst)
    src = ensure_conformer(src)
    dconf = dst.GetConformer()
    sconf = src.GetConformer()
    for i in range(src.GetNumAtoms()):
        p = sconf.GetAtomPosition(i)
        dconf.SetAtomPosition(dst_offset + i, Point3D(p.x, p.y, p.z))


def translate_atoms(mol: Chem.Mol, atom_indices: List[int], delta: Point3D):
    mol = ensure_conformer(mol)
    conf = mol.GetConformer()
    for i in atom_indices:
        p = conf.GetAtomPosition(i)
        conf.SetAtomPosition(i, Point3D(p.x + delta.x, p.y + delta.y, p.z + delta.z))


def centroid(conf: Chem.Conformer, atom_indices: List[int]) -> Point3D:
    sx = sy = sz = 0.0
    n = max(1, len(atom_indices))
    for i in atom_indices:
        p = conf.GetAtomPosition(i)
        sx += p.x; sy += p.y; sz += p.z
    return Point3D(sx / n, sy / n, sz / n)


def random_rotation_matrix(rng: random.Random):
    u1 = rng.random()
    u2 = rng.random()
    u3 = rng.random()
    q1 = math.sqrt(1 - u1) * math.sin(2 * math.pi * u2)
    q2 = math.sqrt(1 - u1) * math.cos(2 * math.pi * u2)
    q3 = math.sqrt(u1) * math.sin(2 * math.pi * u3)
    q4 = math.sqrt(u1) * math.cos(2 * math.pi * u3)

    r11 = 1 - 2*(q3*q3 + q4*q4)
    r12 = 2*(q2*q3 - q1*q4)
    r13 = 2*(q2*q4 + q1*q3)

    r21 = 2*(q2*q3 + q1*q4)
    r22 = 1 - 2*(q2*q2 + q4*q4)
    r23 = 2*(q3*q4 - q1*q2)

    r31 = 2*(q2*q4 - q1*q3)
    r32 = 2*(q3*q4 + q1*q2)
    r33 = 1 - 2*(q2*q2 + q3*q3)

    return ((r11, r12, r13),
            (r21, r22, r23),
            (r31, r32, r33))


def rotate_atoms_about_point(mol: Chem.Mol, atom_indices: List[int], R, origin: Point3D):
    mol = ensure_conformer(mol)
    conf = mol.GetConformer()
    for i in atom_indices:
        p = conf.GetAtomPosition(i)
        x = p.x - origin.x
        y = p.y - origin.y
        z = p.z - origin.z
        rx = R[0][0]*x + R[0][1]*y + R[0][2]*z
        ry = R[1][0]*x + R[1][1]*y + R[1][2]*z
        rz = R[2][0]*x + R[2][1]*y + R[2][2]*z
        conf.SetAtomPosition(i, Point3D(rx + origin.x, ry + origin.y, rz + origin.z))


def get_atoms_within_bonds(mol: Chem.Mol, seeds: List[int], radius: int) -> Set[int]:
    visited: Set[int] = set(seeds)
    frontier: Set[int] = set(seeds)
    for _ in range(max(0, radius)):
        nxt: Set[int] = set()
        for i in frontier:
            ai = mol.GetAtomWithIdx(i)
            for nb in ai.GetNeighbors():
                j = nb.GetIdx()
                if j not in visited:
                    visited.add(j)
                    nxt.add(j)
        frontier = nxt
        if not frontier:
            break
    return visited


def jitter_atoms(mol: Chem.Mol, atom_indices: Set[int], rng: random.Random, amp: float):
    mol = ensure_conformer(mol)
    conf = mol.GetConformer()
    for i in atom_indices:
        p = conf.GetAtomPosition(i)
        dx = rng.uniform(-amp, amp)
        dy = rng.uniform(-amp, amp)
        dz = rng.uniform(-amp, amp)
        conf.SetAtomPosition(i, Point3D(p.x + dx, p.y + dy, p.z + dz))


def robust_local_minimize(mol: Chem.Mol,
                          free_atoms: Set[int],
                          rng: random.Random,
                          iters: int,
                          freeze_radius: int,
                          retries: int,
                          jitter_amp: float,
                          backoff: float) -> Chem.Mol:
    """
    Robust local minimizer wrapper:
    - Uses UFF with fixed points.
    - If optimizer fails (including BFGS 'bad direction'), it retries:
        * jittering the free atoms slightly
        * reducing iterations
    - If still failing, returns mol unchanged (do NOT crash build).
    """
    mol = ensure_conformer(mol)
    if iters <= 0:
        return mol
    if not free_atoms:
        return mol

    attempt_iters = int(iters)

    for attempt in range(max(1, retries)):
        try:
            ff = AllChem.UFFGetMoleculeForceField(mol)
            all_atoms = set(range(mol.GetNumAtoms()))
            fixed = all_atoms - free_atoms
            for i in fixed:
                ff.AddFixedPoint(int(i))
            ff.Initialize()
            ff.Minimize(int(max(1, attempt_iters)))
            return mol
        except Exception:
            jitter_atoms(mol, free_atoms, rng=rng, amp=jitter_amp)
            attempt_iters = max(1, int(attempt_iters * backoff))

    return mol


def global_minimize(mol: Chem.Mol,
                    rng: random.Random,
                    iters: int,
                    retries: int,
                    jitter_amp: float) -> Chem.Mol:
    """Whole-molecule UFF relaxation with jitter fallback; no fixed atoms."""
    mol = ensure_conformer(mol)
    if iters <= 0:
        return mol

    attempt_iters = int(iters)
    for _ in range(max(1, retries)):
        try:
            ff = AllChem.UFFGetMoleculeForceField(mol)
            ff.Initialize()
            ff.Minimize(int(max(1, attempt_iters)))
            return mol
        except Exception:
            jitter_atoms(mol, set(range(mol.GetNumAtoms())), rng=rng, amp=jitter_amp)
            attempt_iters = max(1, int(attempt_iters * 0.5))
    return mol


def embed_small(smiles: str, seed: int, max_iters: int, attempts: int) -> Chem.Mol:
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        raise RuntimeError(f"Failed to parse SMILES: {smiles}")
    m = Chem.AddHs(m)
    params = AllChem.ETKDGv3()
    params.useRandomCoords = True
    params.randomSeed = int(seed)
    params.maxIterations = int(max_iters)
    if hasattr(params, "maxAttempts"):
        params.maxAttempts = int(attempts)
    code = AllChem.EmbedMolecule(m, params)
    if code != 0:
        AllChem.Compute2DCoords(m)
    m = Chem.RemoveHs(m)
    return ensure_conformer(m)


def reembed_full(mol: Chem.Mol,
                 rng: random.Random,
                 max_iters: int,
                 attempts: int,
                 retries: int) -> Chem.Mol:
    """
    Re-embed the full molecule using distance geometry.
    Returns the new molecule if embedding succeeds, otherwise returns the input unchanged.
    """
    max_iters = int(max(1, max_iters))
    attempts = int(max(1, attempts))
    retries = int(max(0, retries))

    params = AllChem.ETKDGv3()
    params.useRandomCoords = True
    params.maxIterations = max_iters
    if hasattr(params, "maxAttempts"):
        params.maxAttempts = attempts

    for _ in range(max(1, retries + 1)):
        m = Chem.Mol(mol)
        m.RemoveAllConformers()
        params.randomSeed = int(rng.randint(1, 2**31 - 1))
        code = AllChem.EmbedMolecule(m, params)
        if code == 0:
            return ensure_conformer(m)

    return mol


def terminal_methyl_carbons(mol: Chem.Mol) -> List[int]:
    out = []
    for a in mol.GetAtoms():
        if a.GetSymbol() != "C":
            continue
        if a.GetIsAromatic():
            continue
        if a.GetDegree() != 1:
            continue
        out.append(a.GetIdx())
    out.sort()
    return out


def unit_vector_random(rng: random.Random) -> Point3D:
    while True:
        x1 = rng.uniform(-1.0, 1.0)
        x2 = rng.uniform(-1.0, 1.0)
        s = x1*x1 + x2*x2
        if 0 < s < 1:
            z = 1 - 2*s
            f = 2 * math.sqrt(1 - s)
            x = x1 * f
            y = x2 * f
            norm = math.sqrt(x*x + y*y + z*z)
            return Point3D(x / norm, y / norm, z / norm)


def normalize_vec(v: Point3D) -> Point3D:
    n = math.sqrt(v.x * v.x + v.y * v.y + v.z * v.z)
    if n < 1e-12:
        return Point3D(1.0, 0.0, 0.0)
    return Point3D(v.x / n, v.y / n, v.z / n)


def rotation_matrix_axis_angle(axis: Point3D, angle_rad: float):
    ux, uy, uz = axis.x, axis.y, axis.z
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    t = 1.0 - c
    return (
        (t * ux * ux + c, t * ux * uy - s * uz, t * ux * uz + s * uy),
        (t * ux * uy + s * uz, t * uy * uy + c, t * uy * uz - s * ux),
        (t * ux * uz - s * uy, t * uy * uz + s * ux, t * uz * uz + c),
    )


def align_fragment_axis(mol: Chem.Mol,
                        atom_indices: List[int],
                        start_idx: int,
                        end_idx: int,
                        target_dir: Point3D):
    mol = ensure_conformer(mol)
    conf = mol.GetConformer()
    p0 = conf.GetAtomPosition(start_idx)
    p1 = conf.GetAtomPosition(end_idx)

    src = normalize_vec(Point3D(p1.x - p0.x, p1.y - p0.y, p1.z - p0.z))
    dst = normalize_vec(target_dir)

    dot = max(-1.0, min(1.0, src.x * dst.x + src.y * dst.y + src.z * dst.z))
    if abs(dot - 1.0) < 1e-8:
        return

    if abs(dot + 1.0) < 1e-8:
        ref = Point3D(0.0, 0.0, 1.0) if abs(src.z) < 0.9 else Point3D(0.0, 1.0, 0.0)
        axis = normalize_vec(
            Point3D(
                src.y * ref.z - src.z * ref.y,
                src.z * ref.x - src.x * ref.z,
                src.x * ref.y - src.y * ref.x,
            )
        )
        R = rotation_matrix_axis_angle(axis, math.pi)
    else:
        axis = normalize_vec(
            Point3D(
                src.y * dst.z - src.z * dst.y,
                src.z * dst.x - src.x * dst.z,
                src.x * dst.y - src.y * dst.x,
            )
        )
        angle = math.acos(dot)
        R = rotation_matrix_axis_angle(axis, angle)

    rotate_atoms_about_point(mol, atom_indices, R, origin=p0)


def chain_direction_for_index(i: int) -> Point3D:
    dirs = [
        Point3D(1.0, 0.0, 0.0),
        Point3D(0.0, 1.0, 0.0),
        Point3D(0.0, 0.0, 1.0),
        Point3D(-1.0, 0.0, 0.0),
        Point3D(0.0, -1.0, 0.0),
        Point3D(0.0, 0.0, -1.0),
        Point3D(1.0, 1.0, 0.0),
        Point3D(-1.0, 1.0, 0.0),
        Point3D(1.0, 0.0, 1.0),
        Point3D(0.0, 1.0, 1.0),
        Point3D(1.0, -1.0, 0.0),
        Point3D(0.0, 1.0, -1.0),
    ]
    return normalize_vec(dirs[i % len(dirs)])


def direction_pool(num_dirs: int) -> List[Point3D]:
    """Generate candidate branch directions on a sphere."""
    num_dirs = max(12, int(num_dirs))
    pool: List[Point3D] = [chain_direction_for_index(i) for i in range(12)]
    if num_dirs <= len(pool):
        return pool[:num_dirs]

    golden = math.pi * (3.0 - math.sqrt(5.0))
    extra = num_dirs - len(pool)
    for k in range(extra):
        z = 1.0 - 2.0 * ((k + 0.5) / extra)
        r = math.sqrt(max(0.0, 1.0 - z * z))
        theta = golden * k
        pool.append(normalize_vec(Point3D(r * math.cos(theta), r * math.sin(theta), z)))
    return pool


def min_distance_translated_chain_to_mol(chain: Chem.Mol,
                                         mol: Chem.Mol,
                                         delta: Point3D,
                                         early_stop_below: Optional[float] = None) -> float:
    """Minimum atom-atom distance between translated chain and existing mol."""
    chain = ensure_conformer(chain)
    mol = ensure_conformer(mol)
    cconf = chain.GetConformer()
    mconf = mol.GetConformer()
    md2 = float("inf")
    stop2 = None if early_stop_below is None else float(early_stop_below) ** 2

    for i in range(chain.GetNumAtoms()):
        pc = cconf.GetAtomPosition(i)
        x = pc.x + delta.x
        y = pc.y + delta.y
        z = pc.z + delta.z
        for j in range(mol.GetNumAtoms()):
            pm = mconf.GetAtomPosition(j)
            dx = x - pm.x
            dy = y - pm.y
            dz = z - pm.z
            d2 = dx * dx + dy * dy + dz * dz
            if d2 < md2:
                md2 = d2
                if (stop2 is not None) and (md2 < stop2):
                    return math.sqrt(md2)
    return math.sqrt(md2) if md2 < float("inf") else float("inf")


def first_close_contact(mol: Chem.Mol,
                        min_dist_a: float,
                        heavy_only: bool = True) -> Optional[Tuple[int, int, float]]:
    """
    Return first close-contact pair (i, j, distance) if any pair is below min_dist_a.
    Uses a simple 3D grid for near-linear scaling.
    """
    mol = ensure_conformer(mol)
    conf = mol.GetConformer()
    cell = max(0.1, float(min_dist_a))
    thresh2 = float(min_dist_a) ** 2
    grid = {}

    for i in range(mol.GetNumAtoms()):
        ai = mol.GetAtomWithIdx(i)
        if heavy_only and ai.GetAtomicNum() <= 1:
            continue
        p = conf.GetAtomPosition(i)
        cx = int(math.floor(p.x / cell))
        cy = int(math.floor(p.y / cell))
        cz = int(math.floor(p.z / cell))

        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    key = (cx + dx, cy + dy, cz + dz)
                    for j in grid.get(key, []):
                        pj = conf.GetAtomPosition(j)
                        ddx = p.x - pj.x
                        ddy = p.y - pj.y
                        ddz = p.z - pj.z
                        d2 = ddx * ddx + ddy * ddy + ddz * ddz
                        if d2 < thresh2:
                            return (j, i, math.sqrt(d2))

        grid.setdefault((cx, cy, cz), []).append(i)
    return None


def orthonormal_plane_basis(direction: Point3D) -> Tuple[Point3D, Point3D]:
    u = normalize_vec(direction)
    ref = Point3D(0.0, 0.0, 1.0) if abs(u.z) < 0.9 else Point3D(0.0, 1.0, 0.0)
    e1 = normalize_vec(
        Point3D(
            u.y * ref.z - u.z * ref.y,
            u.z * ref.x - u.x * ref.z,
            u.x * ref.y - u.y * ref.x,
        )
    )
    e2 = normalize_vec(
        Point3D(
            u.y * e1.z - u.z * e1.y,
            u.z * e1.x - u.x * e1.z,
            u.x * e1.y - u.y * e1.x,
        )
    )
    return e1, e2


def estimate_chain_radius_perp(chain: Chem.Mol, direction: Point3D) -> float:
    chain = ensure_conformer(chain)
    conf = chain.GetConformer()
    u = normalize_vec(direction)
    max_r2 = 0.0
    for i in range(chain.GetNumAtoms()):
        p = conf.GetAtomPosition(i)
        t = p.x * u.x + p.y * u.y + p.z * u.z
        rx = p.x - t * u.x
        ry = p.y - t * u.y
        rz = p.z - t * u.z
        r2 = rx * rx + ry * ry + rz * rz
        if r2 > max_r2:
            max_r2 = r2
    return math.sqrt(max_r2)


def grid_offsets_parallel(n: int, spacing: float, direction: Point3D) -> List[Point3D]:
    side = int(math.ceil(math.sqrt(n)))
    e1, e2 = orthonormal_plane_basis(direction)
    out: List[Point3D] = []
    for i in range(n):
        a = (i % side) * spacing
        b = (i // side) * spacing
        out.append(
            Point3D(
                a * e1.x + b * e2.x,
                a * e1.y + b * e2.y,
                a * e1.z + b * e2.z,
            )
        )
    return out


def build_chain_incremental(length: int,
                            monomer_template: Chem.Mol,
                            bond_length: float,
                            chain_direction: Point3D,
                            alternate_180: bool) -> Chem.Mol:
    """Straight-backbone builder: deterministic monomer placement along chain_direction."""
    if length < 1:
        raise ValueError("CHAIN_LENGTH must be >= 1")

    chain = Chem.Mol(monomer_template)
    chain = ensure_conformer(chain)

    ends = terminal_methyl_carbons(chain)
    if len(ends) < 2:
        raise RuntimeError("Monomer template must have two terminal methyl carbons.")
    chain_atoms = list(range(chain.GetNumAtoms()))
    align_fragment_axis(chain, chain_atoms, ends[0], ends[-1], chain_direction)
    conf0 = chain.GetConformer()
    p_start = conf0.GetAtomPosition(ends[0])
    translate_atoms(chain, chain_atoms, Point3D(-p_start.x, -p_start.y, -p_start.z))

    tail = ends[-1]

    for _ in range(1, length):
        mon = Chem.Mol(monomer_template)
        mon = ensure_conformer(mon)

        mon_ends = terminal_methyl_carbons(mon)
        head = mon_ends[0]
        far = mon_ends[-1]

        mon_atoms_local = list(range(mon.GetNumAtoms()))
        align_fragment_axis(mon, mon_atoms_local, head, far, chain_direction)
        conf_m = mon.GetConformer()
        p_head_local = conf_m.GetAtomPosition(head)
        translate_atoms(mon, mon_atoms_local, Point3D(-p_head_local.x, -p_head_local.y, -p_head_local.z))
        if alternate_180 and (_ % 2 == 1):
            axis = normalize_vec(chain_direction)
            Rflip = rotation_matrix_axis_angle(axis, math.pi)
            rotate_atoms_about_point(mon, mon_atoms_local, Rflip, origin=Point3D(0.0, 0.0, 0.0))

        combo = Chem.CombineMols(chain, mon)
        combo = ensure_conformer(combo)
        copy_coords(combo, chain, 0)
        off = chain.GetNumAtoms()
        copy_coords(combo, mon, off)

        mon_atoms = list(range(off, off + mon.GetNumAtoms()))
        conf = combo.GetConformer()
        p_tail = conf.GetAtomPosition(tail)
        p_head = conf.GetAtomPosition(off + head)

        u = normalize_vec(chain_direction)
        target = Point3D(p_tail.x + u.x * bond_length,
                         p_tail.y + u.y * bond_length,
                         p_tail.z + u.z * bond_length)
        delta = Point3D(target.x - p_head.x, target.y - p_head.y, target.z - p_head.z)
        translate_atoms(combo, mon_atoms, delta)

        rw = Chem.RWMol(combo)
        if rw.GetBondBetweenAtoms(tail, off + head) is None:
            rw.AddBond(tail, off + head, Chem.BondType.SINGLE)
        out = rw.GetMol()
        Chem.SanitizeMol(out)
        out = ensure_conformer(out)

        chain = out
        tail = off + far

    return chain


def grid_offsets(n: int, spacing: float) -> List[Point3D]:
    side = int(math.ceil(math.sqrt(n)))
    return [Point3D((i % side) * spacing, (i // side) * spacing, 0.0) for i in range(n)]


def backbone_methine_candidates(chain: Chem.Mol) -> List[int]:
    """Find methine anchors."""
    out = []
    for a in chain.GetAtoms():
        if a.GetSymbol() != "C":
            continue
        if a.GetIsAromatic():
            continue
        if a.GetDegree() != 3:
            continue
        if a.GetTotalNumHs() < 1:
            continue
        if any(n.GetIsAromatic() for n in a.GetNeighbors()):
            out.append(a.GetIdx())
    out.sort()
    return out


def available_methines_global(mol: Chem.Mol,
                              candidates_local: List[int],
                              chain_offset: int,
                              exclude: Set[int]) -> List[int]:
    """Return global indices of methine candidates that still have a free H and correct environment."""
    avail: List[int] = []
    for local_idx in candidates_local:
        g = chain_offset + local_idx
        if g in exclude:
            continue
        a = mol.GetAtomWithIdx(g)
        if a.GetSymbol() != "C":
            continue
        if a.GetIsAromatic():
            continue
        if a.GetDegree() > 3:
            continue
        if a.GetTotalNumHs() < 1:
            continue
        if not any(n.GetIsAromatic() for n in a.GetNeighbors()):
            continue
        avail.append(g)
    return avail


def pick_nearest_methine_pair(mol: Chem.Mol,
                              candidates_a: List[int],
                              candidates_b: List[int],
                              preferred_rank_a: int,
                              preferred_rank_b: int,
                              max_dist: float,
                              rank_map_a: Optional[dict] = None,
                              rank_map_b: Optional[dict] = None,
                              rank_penalty_a: float = 0.15) -> Tuple[Optional[int], Optional[int], float]:
    """
    Choose the closest anchor pair (Euclidean) with a mild penalty for deviating from preferred ranks.
    Returns (a_idx, b_idx, distance) or (None, None, inf) if none found.
    """
    conf = mol.GetConformer()
    best = (None, None, float("inf"))
    for ia_idx, ga in enumerate(candidates_a):
        pa = conf.GetAtomPosition(ga)
        ra = rank_map_a.get(ga, ia_idx) if rank_map_a is not None else ia_idx
        rank_pen_a = abs(ra - preferred_rank_a) * rank_penalty_a
        for ib_idx, gb in enumerate(candidates_b):
            if ga == gb:
                continue
            rb = rank_map_b.get(gb, ib_idx) if rank_map_b is not None else ib_idx
            pb = conf.GetAtomPosition(gb)
            dx = pa.x - pb.x
            dy = pa.y - pb.y
            dz = pa.z - pb.z
            d = math.sqrt(dx*dx + dy*dy + dz*dz)
            score = d + rank_pen_a + abs(rb - preferred_rank_b) * rank_penalty_a
            if d <= max_dist and score < best[2]:
                best = (ga, gb, d)
    return best


def pick_available_methine_anchor(mol: Chem.Mol,
                                  candidates_local: List[int],
                                  chain_offset: int,
                                  preferred_rank: int,
                                  exclude: Set[int]) -> Optional[int]:
    """
    Choose a methine that still has a free H (available valence) in the combined molecule.
    - candidates_local: indices within the chain (pre-offset).
    - chain_offset: offset to map local -> global atom indices.
    - preferred_rank: preferred position within the candidate list (nearest to target unit).
    - exclude: global atom indices that cannot be used (e.g., already chosen for the other anchor).
    Returns a global atom index or None if none available.
    """
    available: List[Tuple[int, int]] = []
    for rank, local_idx in enumerate(candidates_local):
        g = chain_offset + local_idx
        if g in exclude:
            continue
        a = mol.GetAtomWithIdx(g)
        if a.GetSymbol() != "C":
            continue
        if a.GetIsAromatic():
            continue
        if a.GetDegree() > 3:
            continue
        if a.GetTotalNumHs() < 1:
            continue
        if not any(n.GetIsAromatic() for n in a.GetNeighbors()):
            continue
        dist = abs(rank - preferred_rank)
        available.append((dist, g))

    if not available:
        return None

    available.sort(key=lambda t: (t[0], t[1]))
    return available[0][1]


def rank_map_for_chain(candidates_local: List[int], chain_offset: int) -> dict:
    out = {}
    for rank, local_idx in enumerate(candidates_local):
        out[chain_offset + local_idx] = rank
    return out


def filter_by_rank_separation(candidate_globals: List[int],
                              rank_map: dict,
                              used_ranks: List[int],
                              min_sep: int) -> List[int]:
    if min_sep <= 0 or not used_ranks:
        return list(candidate_globals)
    out: List[int] = []
    for g in candidate_globals:
        r = rank_map.get(g, -10**9)
        if all(abs(r - ur) >= min_sep for ur in used_ranks):
            out.append(g)
    return out


def add_dvb_ring_crosslink(mol: Chem.Mol,
                           ring_template: Chem.Mol,
                           a_idx: int,
                           b_idx: int,
                           rng: random.Random,
                           local_relax: bool,
                           local_iters: int,
                           freeze_radius: int,
                           relax_retries: int,
                           relax_jitter: float,
                           relax_backoff: float,
                           continuous_embed: bool,
                           embed_max_iters: int,
                           embed_attempts: int,
                           embed_retries: int) -> Chem.Mol:
    """
    Add a DVB-like crosslink by inserting a benzene ring and bonding para-like positions to two backbone methines.

    Implementation:
    - Combine mol + ring.
    - Rotate ring randomly, translate its midpoint to midpoint of anchors.
    - Bond ring atom 0 to a_idx and ring atom 3 to b_idx (para positions in a 6-member ring).
    - Local relax around new bonds with robust minimizer.
    """
    mol = ensure_conformer(mol)
    ring = Chem.Mol(ring_template)
    ring = ensure_conformer(ring)

    combo = Chem.CombineMols(mol, ring)
    combo = ensure_conformer(combo)
    copy_coords(combo, mol, 0)
    off = mol.GetNumAtoms()
    copy_coords(combo, ring, off)

    ring_atoms = list(range(off, off + ring.GetNumAtoms()))
    conf = combo.GetConformer()
    c0 = centroid(conf, ring_atoms)

    R = random_rotation_matrix(rng)
    rotate_atoms_about_point(combo, ring_atoms, R, origin=c0)

    conf = combo.GetConformer()
    pA = conf.GetAtomPosition(a_idx)
    pB = conf.GetAtomPosition(b_idx)
    mid = Point3D((pA.x + pB.x) / 2.0, (pA.y + pB.y) / 2.0, (pA.z + pB.z) / 2.0)

    conf = combo.GetConformer()
    c1 = centroid(conf, ring_atoms)
    delta = Point3D(mid.x - c1.x, mid.y - c1.y, mid.z - c1.z)
    translate_atoms(combo, ring_atoms, delta)

    conf = combo.GetConformer()
    existing = [i for i in range(mol.GetNumAtoms()) if i not in (a_idx, b_idx)]
    if existing:
        dmin = min_distance_between_sets(conf, ring_atoms, existing)
        if dmin < 1.1:
            shift = Point3D(mid.x - c1.x, mid.y - c1.y, mid.z - c1.z)
            scale = 0.3
            translate_atoms(combo, ring_atoms, Point3D(shift.x * scale, shift.y * scale, shift.z * scale))

    ring_a = off + 0
    ring_b = off + 3

    if a_idx == b_idx:
        raise ValueError("Crosslink anchors must be distinct.")

    for idx in (a_idx, b_idx):
        a = mol.GetAtomWithIdx(idx)
        if a.GetTotalNumHs() < 1:
            raise ValueError(f"Anchor atom {idx} has no available hydrogen for new bond.")
        if a.GetDegree() > 3:
            raise ValueError(f"Anchor atom {idx} already has degree {a.GetDegree()}, cannot add bond.")

    rw = Chem.RWMol(combo)
    if rw.GetBondBetweenAtoms(a_idx, ring_a) is None:
        rw.AddBond(a_idx, ring_a, Chem.BondType.SINGLE)
    if rw.GetBondBetweenAtoms(b_idx, ring_b) is None:
        rw.AddBond(b_idx, ring_b, Chem.BondType.SINGLE)

    out = rw.GetMol()
    Chem.SanitizeMol(out)
    out = ensure_conformer(out)

    if continuous_embed:
        out = reembed_full(
            out,
            rng=rng,
            max_iters=embed_max_iters,
            attempts=embed_attempts,
            retries=embed_retries,
        )

    if local_relax:
        seeds = [a_idx, b_idx, ring_a, ring_b]
        free = get_atoms_within_bonds(out, seeds=seeds, radius=freeze_radius)
        out = robust_local_minimize(
            out, free_atoms=free, rng=rng,
            iters=local_iters, freeze_radius=freeze_radius,
            retries=relax_retries, jitter_amp=relax_jitter, backoff=relax_backoff
        )

    return out


def min_distance_to_atoms(conf: Chem.Conformer, point: Point3D, atom_indices: List[int]) -> float:
    md = float("inf")
    for i in atom_indices:
        p = conf.GetAtomPosition(i)
        dx = p.x - point.x
        dy = p.y - point.y
        dz = p.z - point.z
        d = math.sqrt(dx*dx + dy*dy + dz*dz)
        if d < md:
            md = d
    return md


def min_distance_between_sets(conf: Chem.Conformer, setA: List[int], setB: List[int]) -> float:
    md = float("inf")
    for i in setA:
        p = conf.GetAtomPosition(i)
        for j in setB:
            q = conf.GetAtomPosition(j)
            dx = p.x - q.x
            dy = p.y - q.y
            dz = p.z - q.z
            d = math.sqrt(dx*dx + dy*dy + dz*dz)
            if d < md:
                md = d
    return md


def add_counterions(mol: Chem.Mol,
                    rng: random.Random,
                    ion_smiles: str,
                    distance_a: float,
                    min_dist_a: float,
                    min_anchor_dist_a: float,
                    tries: int) -> Chem.Mol:
    """
    Fast straight-mode counterion placement:
    - Computes all ion positions first (no repeated CombineMols in loop).
    - Uses heavy-atom coordinates as steric references.
    - Adds all ions in one molecule-combine step.
    """
    mol = ensure_conformer(mol)
    q = Chem.GetFormalCharge(mol)

    ion = Chem.MolFromSmiles(ion_smiles)
    if ion is None or ion.GetNumAtoms() != 1:
        raise ValueError("COUNTERION_SMILES must parse to a single-atom ion, e.g. '[Cl-]'.")

    anchors = [a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() == "N" and a.GetFormalCharge() > 0]
    if not anchors:
        anchors = [a.GetIdx() for a in mol.GetAtoms()
                   if a.GetSymbol() == "N" and (not a.GetIsAromatic()) and a.GetDegree() == 4]
    if not anchors:
        if q > 0:
            sys.stdout.write("[warn] No ammonium anchors found; skipping counterion placement.\n")
            sys.stdout.flush()
        return mol

    if q != 0 and q != len(anchors):
        sys.stdout.write(
            f"[warn] Formal charge ({q}) != ammonium anchors ({len(anchors)}); placing one counterion per anchor.\n"
        )
        sys.stdout.flush()

    out = mol
    conf = out.GetConformer()
    heavy_existing_points: List[Point3D] = []
    for a in out.GetAtoms():
        if a.GetAtomicNum() > 1:
            heavy_existing_points.append(conf.GetAtomPosition(a.GetIdx()))
    placed_ion_points: List[Point3D] = []

    def pick_backbone_carbon(n_atom: Chem.Atom) -> Optional[int]:
        candidates: List[Tuple[int, int]] = []
        for nb in n_atom.GetNeighbors():
            if nb.GetSymbol() != "C":
                continue
            score = 0
            if any(n.GetIsAromatic() for n in nb.GetNeighbors()):
                score += 2
            if nb.GetDegree() > 1:
                score += 1
            candidates.append((score, nb.GetIdx()))
        if not candidates:
            return None
        candidates.sort(key=lambda t: (-t[0], t[1]))
        return candidates[0][1]

    def point_ok(pt: Point3D) -> bool:
        md2 = min_dist_a * min_dist_a
        for p in heavy_existing_points:
            dx = p.x - pt.x
            dy = p.y - pt.y
            dz = p.z - pt.z
            if dx*dx + dy*dy + dz*dz < md2:
                return False
        for p in placed_ion_points:
            dx = p.x - pt.x
            dy = p.y - pt.y
            dz = p.z - pt.z
            if dx*dx + dy*dy + dz*dz < md2:
                return False
        return True

    def orthonormal_basis(dir_u: Point3D) -> Tuple[Point3D, Point3D]:
        ref = Point3D(0.0, 0.0, 1.0) if abs(dir_u.z) < 0.9 else Point3D(0.0, 1.0, 0.0)
        v1 = normalize_vec(
            Point3D(
                dir_u.y * ref.z - dir_u.z * ref.y,
                dir_u.z * ref.x - dir_u.x * ref.z,
                dir_u.x * ref.y - dir_u.y * ref.x,
            )
        )
        v2 = normalize_vec(
            Point3D(
                dir_u.y * v1.z - dir_u.z * v1.y,
                dir_u.z * v1.x - dir_u.x * v1.z,
                dir_u.x * v1.y - dir_u.y * v1.x,
            )
        )
        return v1, v2

    for a_idx in anchors:
        pN = conf.GetAtomPosition(a_idx)

        n_atom = out.GetAtomWithIdx(a_idx)
        c_idx = pick_backbone_carbon(n_atom)
        if c_idx is not None:
            pC = conf.GetAtomPosition(c_idx)
            vx = pN.x - pC.x
            vy = pN.y - pC.y
            vz = pN.z - pC.z
            norm = math.sqrt(vx*vx + vy*vy + vz*vz)
            if norm > 1e-6:
                ux, uy, uz = vx / norm, vy / norm, vz / norm
            else:
                u = unit_vector_random(rng)
                ux, uy, uz = u.x, u.y, u.z
        else:
            u = unit_vector_random(rng)
            ux, uy, uz = u.x, u.y, u.z

        u_dir = normalize_vec(Point3D(ux, uy, uz))
        v1, v2 = orthonormal_basis(u_dir)
        base_d = max(min_anchor_dist_a, distance_a)

        placed = False
        max_trials = max(1, min(int(tries), 12))
        for t in range(max_trials):
            ang = (2.0 * math.pi * t) / max_trials
            lateral = 0.35 + 0.1 * (t % 3)
            dir_try = normalize_vec(
                Point3D(
                    u_dir.x + lateral * (math.cos(ang) * v1.x + math.sin(ang) * v2.x),
                    u_dir.y + lateral * (math.cos(ang) * v1.y + math.sin(ang) * v2.y),
                    u_dir.z + lateral * (math.cos(ang) * v1.z + math.sin(ang) * v2.z),
                )
            )
            pt = Point3D(pN.x + dir_try.x * base_d, pN.y + dir_try.y * base_d, pN.z + dir_try.z * base_d)
            if point_ok(pt):
                placed_ion_points.append(pt)
                placed = True
                break

        if not placed:
            pt = Point3D(pN.x + u_dir.x * base_d, pN.y + u_dir.y * base_d, pN.z + u_dir.z * base_d)
            placed_ion_points.append(pt)

    if not placed_ion_points:
        return out

    ion_atom_template = ion.GetAtomWithIdx(0)
    ion_rw = Chem.RWMol()
    for _ in placed_ion_points:
        ion_rw.AddAtom(Chem.Atom(ion_atom_template))
    ion_cluster = ion_rw.GetMol()
    ion_conf = Chem.Conformer(len(placed_ion_points))
    for i, pt in enumerate(placed_ion_points):
        ion_conf.SetAtomPosition(i, pt)
    ion_cluster.AddConformer(ion_conf, assignId=True)

    combo = Chem.CombineMols(out, ion_cluster)
    combo = ensure_conformer(combo)
    copy_coords(combo, out, 0)
    copy_coords(combo, ion_cluster, out.GetNumAtoms())

    Chem.SanitizeMol(combo)
    return combo


def build_chain_topology(length: int, monomer_template: Chem.Mol, progress: ProgressBar, atoms_per_monomer: int) -> Chem.Mol:
    """Build a linear chain using connectivity only (no coordinates)."""
    if length < 1:
        raise ValueError("CHAIN_LENGTH must be >= 1")

    chain = Chem.Mol(monomer_template)
    ends = terminal_methyl_carbons(chain)
    if len(ends) < 2:
        raise RuntimeError("Monomer template must have two terminal methyl carbons.")
    tail = ends[-1]
    progress.advance(atoms_per_monomer)

    for _ in range(1, length):
        mon = Chem.Mol(monomer_template)
        mon_ends = terminal_methyl_carbons(mon)
        head = mon_ends[0]
        far = mon_ends[-1]

        combo = Chem.CombineMols(chain, mon)
        off = chain.GetNumAtoms()
        rw = Chem.RWMol(combo)
        if rw.GetBondBetweenAtoms(tail, off + head) is None:
            rw.AddBond(tail, off + head, Chem.BondType.SINGLE)
        chain = rw.GetMol()
        Chem.SanitizeMol(chain)
        tail = off + far
        progress.advance(atoms_per_monomer)

    return chain


def add_dvb_ring_crosslink_topology(mol: Chem.Mol, ring_template: Chem.Mol, a_idx: int, b_idx: int) -> Chem.Mol:
    """Insert a DVB ring by topology only (no geometry constraints)."""
    if a_idx == b_idx:
        raise ValueError("Crosslink anchors must be distinct.")
    for idx in (a_idx, b_idx):
        a = mol.GetAtomWithIdx(idx)
        if a.GetTotalNumHs() < 1:
            raise ValueError(f"Anchor atom {idx} has no available hydrogen for new bond.")
        if a.GetDegree() > 3:
            raise ValueError(f"Anchor atom {idx} already has degree {a.GetDegree()}, cannot add bond.")

    combo = Chem.CombineMols(mol, ring_template)
    off = mol.GetNumAtoms()
    ring_a = off + 0
    ring_b = off + 3

    rw = Chem.RWMol(combo)
    if rw.GetBondBetweenAtoms(a_idx, ring_a) is None:
        rw.AddBond(a_idx, ring_a, Chem.BondType.SINGLE)
    if rw.GetBondBetweenAtoms(b_idx, ring_b) is None:
        rw.AddBond(b_idx, ring_b, Chem.BondType.SINGLE)
    out = rw.GetMol()
    Chem.SanitizeMol(out)
    return out


def add_counterions_topology(mol: Chem.Mol, ion_smiles: str) -> Chem.Mol:
    """Add one disconnected counterion per cationic ammonium anchor (topology only)."""
    ion = Chem.MolFromSmiles(ion_smiles)
    if ion is None or ion.GetNumAtoms() != 1:
        raise ValueError("COUNTERION_SMILES must parse to a single-atom ion, e.g. '[Cl-]'.")

    anchors = [a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() == "N" and a.GetFormalCharge() > 0]
    if not anchors:
        anchors = [a.GetIdx() for a in mol.GetAtoms()
                   if a.GetSymbol() == "N" and (not a.GetIsAromatic()) and a.GetDegree() == 4]
    if not anchors:
        return mol

    out = mol
    for _ in anchors:
        out = Chem.CombineMols(out, ion)
    Chem.SanitizeMol(out)
    return out


def main_topology():
    if SETTINGS.get("SILENCE_RDKIT_WARNINGS", True):
        rdBase.DisableLog("rdApp.warning")

    rng = random.Random(int(SETTINGS["SEED"]))

    total_units = int(SETTINGS["TOTAL_MONOMERS"])
    if total_units < 1:
        raise ValueError("TOTAL_MONOMERS must be >= 1")

    xlink_target = float(SETTINGS["CROSSLINK_FRACTION"])
    nlinks = int(round(total_units * xlink_target))
    if nlinks < 1:
        nlinks = 0

    C = 1 if nlinks == 0 else (1 + nlinks)
    chain_lengths = partition_counts(total_units, C)
    if not chain_lengths or sum(chain_lengths) != total_units:
        raise RuntimeError("Failed to partition TOTAL_MONOMERS into chain lengths.")

    mon_t = Chem.MolFromSmiles(MONOMER_SMILES)
    ring_t = Chem.MolFromSmiles(DVB_RING_SMILES)
    if mon_t is None or ring_t is None:
        raise RuntimeError("Failed to parse monomer or DVB ring SMILES.")

    atoms_per_monomer = mon_t.GetNumAtoms()
    atoms_per_crosslink = ring_t.GetNumAtoms()
    total_heavy_est = sum(chain_lengths) * atoms_per_monomer + nlinks * atoms_per_crosslink
    progress = ProgressBar(total_heavy_est, label="Assembling Topology")

    chains: List[Chem.Mol] = []
    for L in chain_lengths:
        chains.append(build_chain_topology(L, mon_t, progress=progress, atoms_per_monomer=atoms_per_monomer))

    combined = Chem.Mol(chains[0])
    chain_offsets: List[int] = [0]
    for i in range(1, C):
        prev_atoms = combined.GetNumAtoms()
        combined = Chem.CombineMols(combined, chains[i])
        chain_offsets.append(prev_atoms)
    Chem.SanitizeMol(combined)

    methines_by_chain: List[List[int]] = [backbone_methine_candidates(ch) for ch in chains]
    planned: List[Tuple[int, int, int, int]] = []
    avoid = int(SETTINGS["AVOID_END_UNITS"])

    if SETTINGS["ENSURE_CONNECTED_NETWORK"] and C >= 2:
        for ci in range(C - 1):
            Li = chain_lengths[ci]
            Lj = chain_lengths[ci + 1]
            ui = evenly_spaced_indices(Li, 1, avoid)[0] if Li > 0 else 0
            uj = evenly_spaced_indices(Lj, 1, avoid)[0] if Lj > 0 else 0
            planned.append((ci, ci + 1, ui, uj))

    remaining = nlinks - len(planned)
    for _ in range(max(0, remaining)):
        if SETTINGS["INTERCHAIN_ONLY"]:
            if C < 2:
                continue
            i, j = rng.sample(range(C), 2)
        else:
            i = rng.randrange(C)
            j = rng.randrange(C)
            if C > 1 and i == j:
                j = (j + 1) % C
        Li = chain_lengths[i]
        Lj = chain_lengths[j]
        ui = rng.randrange(avoid, max(avoid + 1, Li - avoid))
        uj = rng.randrange(avoid, max(avoid + 1, Lj - avoid))
        planned.append((i, j, ui, uj))

    mol = combined
    skipped_crosslinks = 0
    for (ci, cj, ui, uj) in planned:
        mi = methines_by_chain[ci]
        mj = methines_by_chain[cj]
        if not mi or not mj:
            skipped_crosslinks += 1
            continue

        ia = min(ui, len(mi) - 1)
        ib = min(uj, len(mj) - 1)
        a_idx = pick_available_methine_anchor(mol, mi, chain_offsets[ci], ia, exclude=set())
        if a_idx is None:
            skipped_crosslinks += 1
            continue
        b_idx = pick_available_methine_anchor(mol, mj, chain_offsets[cj], ib, exclude={a_idx})
        if b_idx is None:
            skipped_crosslinks += 1
            continue

        try:
            mol = add_dvb_ring_crosslink_topology(mol, ring_t, a_idx, b_idx)
            progress.advance(atoms_per_crosslink)
        except (AtomValenceException, ValueError):
            skipped_crosslinks += 1

    if skipped_crosslinks:
        msg = f"Skipped {skipped_crosslinks} crosslinks due to unavailable anchors or valence issues."
        if SETTINGS.get("FAIL_ON_SKIPPED_CROSSLINKS", False):
            raise RuntimeError(msg)
        sys.stdout.write(f"[warn] {msg}\n")
        sys.stdout.flush()

    progress.finish("Topology build complete (heavy atoms).")

    if SETTINGS["ADD_COUNTERIONS"]:
        mol = add_counterions_topology(mol, ion_smiles=str(SETTINGS["COUNTERION_SMILES"]))

    if SETTINGS["ADD_HS_AT_END"]:
        mol = Chem.AddHs(mol)

    mw = Descriptors.MolWt(Chem.RemoveHs(mol))
    sys.stdout.write(
        f"Final topology: MW≈{mw:.1f} g/mol  formal_charge={Chem.GetFormalCharge(mol)}  atoms={mol.GetNumAtoms()}\n"
    )
    sys.stdout.flush()

    topo_smi = SETTINGS.get("OUT_TOPO_SMILES", SETTINGS["OUT_SMILES"])
    smiles = Chem.MolToSmiles(Chem.RemoveHs(mol), isomericSmiles=True)
    with open(topo_smi, "w", encoding="utf-8") as f:
        f.write(smiles + "\n")
    sys.stdout.write(f"Wrote: {topo_smi}\n")

    topo_mol = SETTINGS.get("OUT_TOPO_MOL", "")
    if topo_mol:
        try:
            Chem.MolToMolFile(Chem.RemoveHs(mol), topo_mol, kekulize=False)
            sys.stdout.write(f"Wrote: {topo_mol}\n")
        except Exception as e:
            sys.stdout.write(f"[warn] MOL write failed: {e}\n")


def main():
    if SETTINGS.get("TOPOLOGY_ONLY", True):
        main_topology()
        return

    if SETTINGS.get("SILENCE_RDKIT_WARNINGS", True):
        rdBase.DisableLog("rdApp.warning")

    rng = random.Random(int(SETTINGS["SEED"]))

    total_units = int(SETTINGS["TOTAL_MONOMERS"])
    if total_units < 1:
        raise ValueError("TOTAL_MONOMERS must be >= 1")

    xlink_target = float(SETTINGS["CROSSLINK_FRACTION"])
    nlinks = int(round(total_units * xlink_target))
    if nlinks < 1:
        nlinks = 0

    if nlinks == 0:
        C = 1
    else:
        C = 1 + nlinks

    chain_lengths = partition_counts(total_units, C)
    if not chain_lengths or sum(chain_lengths) != total_units:
        raise RuntimeError("Failed to partition TOTAL_MONOMERS into chain lengths.")

    skip_geometry_relax = True

    mon_t = embed_small(MONOMER_SMILES,
                        seed=SETTINGS["SEED"] + 10,
                        max_iters=SETTINGS["SMALL_EMBED_MAX_ITERS"],
                        attempts=SETTINGS["SMALL_EMBED_ATTEMPTS"])
    ring_t = embed_small(DVB_RING_SMILES,
                         seed=SETTINGS["SEED"] + 20,
                         max_iters=SETTINGS["SMALL_EMBED_MAX_ITERS"],
                         attempts=SETTINGS["SMALL_EMBED_ATTEMPTS"])

    atoms_per_monomer = mon_t.GetNumAtoms()
    atoms_per_crosslink = ring_t.GetNumAtoms()

    total_heavy_est = sum(chain_lengths) * atoms_per_monomer + nlinks * atoms_per_crosslink
    progress = ProgressBar(total_heavy_est, label="Assembling")

    chains: List[Chem.Mol] = []
    chosen_dirs: List[Point3D] = []
    dynamic_dirs = bool(SETTINGS.get("DYNAMIC_BRANCH_DIRECTIONS", True))
    pool = direction_pool(int(SETTINGS.get("DYNAMIC_DIRECTION_POOL_SIZE", 48)))
    tries_per_chain = int(max(1, SETTINGS.get("DYNAMIC_DIRECTION_TRIES_PER_CHAIN", len(pool))))
    min_interchain_dist = float(SETTINGS.get("MIN_INTERCHAIN_ATOM_DIST_A", 1.25))

    base_spacing = float(SETTINGS["INITIAL_CHAIN_SPACING"])
    offsets = grid_offsets(C, base_spacing)

    combined: Optional[Chem.Mol] = None
    chain_offsets: List[int] = []

    for i, L in enumerate(chain_lengths):
        candidate_indices = list(range(len(pool)))
        if dynamic_dirs and i > 0:
            rng.shuffle(candidate_indices)
            candidate_indices = candidate_indices[:min(len(candidate_indices), tries_per_chain)]
        else:
            candidate_indices = [i % len(pool)]

        best_chain: Optional[Chem.Mol] = None
        best_dir: Optional[Point3D] = None
        best_md = -1.0
        offset = offsets[i]

        for di in candidate_indices:
            cand_dir = pool[di]
            cand_chain = build_chain_incremental(
                length=L,
                monomer_template=mon_t,
                bond_length=float(SETTINGS["BOND_LENGTH_A"]),
                chain_direction=cand_dir,
                alternate_180=bool(SETTINGS.get("STRAIGHT_BACKBONE_ALTERNATE_180", True)),
            )

            if combined is None:
                md = float("inf")
            else:
                md = min_distance_translated_chain_to_mol(
                    cand_chain,
                    combined,
                    delta=offset,
                    early_stop_below=min_interchain_dist,
                )

            if md > best_md:
                best_md = md
                best_chain = cand_chain
                best_dir = cand_dir
            if md >= min_interchain_dist:
                break

        if best_chain is None or best_dir is None:
            raise RuntimeError("Failed to place a chain with a valid direction.")

        if (combined is not None) and (best_md < min_interchain_dist):
            sys.stdout.write(
                f"[warn] Chain {i}: best inter-chain min distance {best_md:.2f} A "
                f"< target {min_interchain_dist:.2f} A.\n"
            )
            sys.stdout.flush()

        chains.append(best_chain)
        chosen_dirs.append(best_dir)
        progress.advance(atoms_per_monomer * L)

        if combined is None:
            combined = Chem.Mol(best_chain)
            combined = ensure_conformer(combined)
            copy_coords(combined, best_chain, 0)
            translate_atoms(combined, list(range(combined.GetNumAtoms())), offset)
            chain_offsets.append(0)
        else:
            prev_atoms = combined.GetNumAtoms()
            tmp = Chem.CombineMols(combined, best_chain)
            tmp = ensure_conformer(tmp)
            copy_coords(tmp, combined, 0)
            copy_coords(tmp, best_chain, prev_atoms)
            translate_atoms(tmp, list(range(prev_atoms, prev_atoms + best_chain.GetNumAtoms())), offset)
            combined = tmp
            chain_offsets.append(prev_atoms)

    if combined is None:
        raise RuntimeError("No chains were constructed.")

    combined = ensure_conformer(combined)
    Chem.SanitizeMol(combined)

    methines_by_chain: List[List[int]] = [backbone_methine_candidates(ch) for ch in chains]

    planned: List[Tuple[int, int, int, int]] = []
    avoid = int(SETTINGS["AVOID_END_UNITS"])

    if SETTINGS["ENSURE_CONNECTED_NETWORK"] and C >= 2:
        for ci in range(C - 1):
            Li = chain_lengths[ci]
            Lj = chain_lengths[ci + 1]
            ui = evenly_spaced_indices(Li, 1, avoid)[0] if Li > 0 else 0
            uj = evenly_spaced_indices(Lj, 1, avoid)[0] if Lj > 0 else 0
            planned.append((ci, ci + 1, ui, uj))

    remaining = nlinks - len(planned)
    for _ in range(max(0, remaining)):
        if SETTINGS["INTERCHAIN_ONLY"]:
            if C < 2:
                continue
            i, j = rng.sample(range(C), 2)
        else:
            i = rng.randrange(C)
            j = rng.randrange(C)
            if C > 1 and i == j:
                j = (j + 1) % C
        Li = chain_lengths[i]
        Lj = chain_lengths[j]
        ui = rng.randrange(avoid, max(avoid + 1, Li - avoid))
        uj = rng.randrange(avoid, max(avoid + 1, Lj - avoid))
        planned.append((i, j, ui, uj))

    mol = combined
    base_xlink_dist = float(SETTINGS["MAX_XLINK_DIST_A"])
    auto_adapt_xlink_dist = bool(SETTINGS.get("AUTO_ADAPT_XLINK_DIST", False)) and skip_geometry_relax
    auto_adapt_step = float(max(0.1, SETTINGS.get("AUTO_ADAPT_XLINK_STEP_A", 2.0)))
    auto_adapt_max = float(max(base_xlink_dist, SETTINGS.get("AUTO_ADAPT_XLINK_MAX_A", base_xlink_dist)))
    xlink_hard_max = float(max(0.1, SETTINGS.get("XLINK_HARD_MAX_DIST_A", auto_adapt_max)))
    auto_adapt_max = min(auto_adapt_max, xlink_hard_max)
    rank_penalty_a = float(max(0.0, SETTINGS.get("XLINK_RANK_PENALTY_A", 1.5)))
    min_unit_sep = int(max(0, SETTINGS.get("XLINK_MIN_UNIT_SEPARATION", 4)))
    used_ranks_by_chain: List[List[int]] = [[] for _ in range(C)]
    skipped_crosslinks = 0
    for (ci, cj, ui, uj) in planned:
        mi = methines_by_chain[ci]
        mj = methines_by_chain[cj]
        if not mi or not mj:
            skipped_crosslinks += 1
            sys.stdout.write(f"[warn] No methine candidates on chain {ci if not mi else cj}; crosslink skipped.\n")
            sys.stdout.flush()
            continue

        ia = min(ui, len(mi) - 1)
        ib = min(uj, len(mj) - 1)

        exclude: Set[int] = set()
        avail_i = available_methines_global(mol, mi, chain_offsets[ci], exclude)
        avail_j = available_methines_global(mol, mj, chain_offsets[cj], exclude)
        if not avail_i or not avail_j:
            skipped_crosslinks += 1
            sys.stdout.write(f"[warn] No available methines with free H on chain {ci if not avail_i else cj}; crosslink skipped.\n")
            sys.stdout.flush()
            continue

        rank_map_i = rank_map_for_chain(mi, chain_offsets[ci])
        rank_map_j = rank_map_for_chain(mj, chain_offsets[cj])
        sep_i = filter_by_rank_separation(
            avail_i,
            rank_map_i,
            used_ranks=used_ranks_by_chain[ci],
            min_sep=min_unit_sep,
        )
        sep_j = filter_by_rank_separation(
            avail_j,
            rank_map_j,
            used_ranks=used_ranks_by_chain[cj],
            min_sep=min_unit_sep,
        )
        if sep_i:
            avail_i = sep_i
        if sep_j:
            avail_j = sep_j

        attempt_max_dist = base_xlink_dist
        a_idx = None
        b_idx = None
        dist = float("inf")
        while True:
            a_idx, b_idx, dist = pick_nearest_methine_pair(
                mol,
                avail_i,
                avail_j,
                ia,
                ib,
                max_dist=attempt_max_dist,
                rank_map_a=rank_map_i,
                rank_map_b=rank_map_j,
                rank_penalty_a=rank_penalty_a,
            )
            if (a_idx is not None) and (b_idx is not None) and (not math.isinf(dist)):
                break
            if (not auto_adapt_xlink_dist) or (attempt_max_dist >= auto_adapt_max):
                break
            attempt_max_dist = min(auto_adapt_max, attempt_max_dist + auto_adapt_step)

        if a_idx is None or b_idx is None or math.isinf(dist):
            skipped_crosslinks += 1
            sys.stdout.write(f"[warn] Crosslink skipped: no anchor pair within {attempt_max_dist:.1f} A.\n")
            sys.stdout.flush()
            continue
        if dist > xlink_hard_max:
            skipped_crosslinks += 1
            sys.stdout.write(
                f"[warn] Crosslink skipped: anchor distance {dist:.2f} A exceeds hard max {xlink_hard_max:.2f} A.\n"
            )
            sys.stdout.flush()
            continue
        if attempt_max_dist > base_xlink_dist:
            sys.stdout.write(
                f"[info] Auto-adapted MAX_XLINK_DIST_A for this crosslink: {base_xlink_dist:.1f} -> {attempt_max_dist:.1f} A.\n"
            )
            sys.stdout.flush()

        used_ranks_by_chain[ci].append(rank_map_i.get(a_idx, ia))
        used_ranks_by_chain[cj].append(rank_map_j.get(b_idx, ib))

        try:
            mol = add_dvb_ring_crosslink(
                mol=mol,
                ring_template=ring_t,
                a_idx=a_idx,
                b_idx=b_idx,
                rng=rng,
                local_relax=False,
                local_iters=int(SETTINGS["LOCAL_ITERS"]),
                freeze_radius=int(SETTINGS["FREEZE_RADIUS_BONDS"]),
                relax_retries=int(SETTINGS["RELAX_RETRIES"]),
                relax_jitter=float(SETTINGS["RELAX_JITTER_A"]),
                relax_backoff=float(SETTINGS["RELAX_BACKOFF"]),
                continuous_embed=False,
                embed_max_iters=int(SETTINGS["CONTINUOUS_EMBED_MAX_ITERS"]),
                embed_attempts=int(SETTINGS["CONTINUOUS_EMBED_ATTEMPTS"]),
                embed_retries=int(SETTINGS["CONTINUOUS_EMBED_RETRIES"]),
            )
            progress.advance(atoms_per_crosslink)
        except AtomValenceException as e:
            skipped_crosslinks += 1
            sys.stdout.write(f"[warn] Crosslink skipped due to valence error: {e}\n")
            sys.stdout.flush()
        except ValueError as e:
            skipped_crosslinks += 1
            sys.stdout.write(f"[warn] Crosslink skipped: {e}\n")
            sys.stdout.flush()

    if skipped_crosslinks:
        msg = f"Skipped {skipped_crosslinks} crosslinks due to unavailable anchors or valence issues."
        if SETTINGS.get("FAIL_ON_SKIPPED_CROSSLINKS", False):
            raise RuntimeError(msg)
        sys.stdout.write(f"[warn] {msg}\n")
        sys.stdout.flush()

    progress.finish("Connectivity build complete (heavy atoms).")

    if SETTINGS.get("INTERMEDIATE_GLOBAL_RELAX", False) and (not skip_geometry_relax):
        sys.stdout.write("Intermediate global relaxation (heavy atoms)...\n")
        sys.stdout.flush()
        mol = global_minimize(
            mol=mol,
            rng=rng,
            iters=int(SETTINGS["INTERMEDIATE_GLOBAL_ITERS"]),
            retries=int(SETTINGS["INTERMEDIATE_GLOBAL_RETRIES"]),
            jitter_amp=float(SETTINGS["INTERMEDIATE_GLOBAL_JITTER_A"]),
        )

    if SETTINGS["ADD_COUNTERIONS"]:
        sys.stdout.write("Adding counterions...\n")
        sys.stdout.flush()
        mol = add_counterions(
            mol=mol,
            rng=rng,
            ion_smiles=str(SETTINGS["COUNTERION_SMILES"]),
            distance_a=float(SETTINGS["ION_DISTANCE_A"]),
            min_dist_a=float(SETTINGS["ION_MIN_DIST_A"]),
            min_anchor_dist_a=float(SETTINGS["ION_MIN_ANCHOR_DIST_A"]),
            tries=int(SETTINGS["ION_TRIES"]),
        )

    if SETTINGS["ADD_HS_AT_END"]:
        sys.stdout.write("Adding hydrogens at end...\n")
        sys.stdout.flush()
        mol = Chem.AddHs(mol, addCoords=True)

    if SETTINGS.get("FAIL_ON_CLOSE_CONTACTS", False):
        min_close = float(SETTINGS.get("CLOSE_CONTACT_MIN_DIST_A", 0.9))
        heavy_only = bool(SETTINGS.get("CLOSE_CONTACT_HEAVY_ONLY", True))
        cc = first_close_contact(mol, min_dist_a=min_close, heavy_only=heavy_only)
        if cc is not None:
            i, j, d = cc
            mode = "heavy atoms" if heavy_only else "all atoms"
            raise RuntimeError(
                f"Close-contact check failed ({mode}): atoms {i}-{j} distance {d:.3f} A < {min_close:.3f} A."
            )

    mw = Descriptors.MolWt(Chem.RemoveHs(mol))
    sys.stdout.write(
        f"Final: MWâ‰ˆ{mw:.1f} g/mol  formal_charge={Chem.GetFormalCharge(mol)}  atoms={mol.GetNumAtoms()}\n"
    )
    sys.stdout.flush()

    sdf = SETTINGS["OUT_SDF"]
    w = Chem.SDWriter(sdf)
    w.write(mol)
    w.close()
    sys.stdout.write(f"Wrote: {sdf}\n")

    pdb = SETTINGS["OUT_PDB"]
    try:
        Chem.MolToPDBFile(mol, pdb)
        sys.stdout.write(f"Wrote: {pdb}\n")
    except Exception as e:
        sys.stdout.write(f"[warn] PDB write failed: {e} (SDF written successfully.)\n")

    smi = SETTINGS["OUT_SMILES"]
    try:
        smi_mol = Chem.RemoveHs(mol)
        smiles = Chem.MolToSmiles(smi_mol, isomericSmiles=True)
        with open(smi, "w", encoding="utf-8") as f:
            f.write(smiles + "\n")
        sys.stdout.write(f"Wrote: {smi}\n")
    except Exception as e:
        sys.stdout.write(f"[warn] SMILES write failed: {e}\n")


if __name__ == "__main__":
    main()

