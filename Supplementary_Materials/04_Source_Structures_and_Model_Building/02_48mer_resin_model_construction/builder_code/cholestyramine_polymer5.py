"""Build the topology SMILES for the 48-unit cholestyramine resin model.

This is the topology-only version of the builder used for the manuscript model.
It generates the 48-unit cationic resin connectivity as a canonical SMILES
string. The 3D coordinates used later in the workflow were generated from this
SMILES in a separate OpenBabel/Avogadro-based structure preparation step.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path
from typing import List, Optional, Set, Tuple

from rdkit import Chem, rdBase
from rdkit.Chem import Descriptors
from rdkit.Chem.rdchem import AtomValenceException


# Settings
SETTINGS = {
    "TOTAL_MONOMERS": 48,
    "CROSSLINK_FRACTION": 0.02,
    "ENSURE_CONNECTED_NETWORK": True,
    "INTERCHAIN_ONLY": True,
    "AVOID_END_UNITS": 2,
    "FAIL_ON_SKIPPED_CROSSLINKS": False,
    "SEED": 1,
    "ADD_COUNTERIONS": False,
    "COUNTERION_SMILES": "[Cl-]",
    "ADD_HS_AT_END": True,
    "OUT_TOPO_SMILES": "cholestyramine_multichain_topology.smi",
    "OUT_TOPO_MOL": "cholestyramine_multichain_topology.mol",
    "SILENCE_RDKIT_WARNINGS": True,
}


MONOMER_SMILES = "CC(c1ccc(C[N+](C)(C)C)cc1)C"
DVB_RING_SMILES = "c1ccccc1"
SCRIPT_DIR = Path(__file__).resolve().parent
REFERENCE_SMILES = SCRIPT_DIR.parent / "structures" / "cholestyramine_multichain_topology.smi"


class ProgressBar:
    def __init__(self, total: int, width: int = 44, label: str = "Assembling Topology"):
        self.total = max(1, int(total))
        self.width = int(width)
        self.label = label
        self.current = 0
        self._last_pct = -1

    def advance(self, delta: int) -> None:
        self.current = min(self.total, self.current + int(delta))
        self.draw()

    def draw(self, force: bool = False) -> None:
        frac = self.current / self.total
        pct = int(round(frac * 100))
        if not force and pct == self._last_pct:
            return
        self._last_pct = pct
        filled = int(round(frac * self.width))
        bar = "#" * filled + " " * (self.width - filled)
        sys.stdout.write(
            f"\r{self.label}: |{bar}| {pct:3d}% ({self.current}/{self.total} heavy atoms)"
        )
        sys.stdout.flush()

    def finish(self, msg: str = "Done") -> None:
        self.current = self.total
        self.draw(force=True)
        sys.stdout.write(f"\n{msg}\n")
        sys.stdout.flush()


def partition_counts(total: int, parts: int) -> List[int]:
    """Split a total count as evenly as possible across parts."""
    if parts <= 0:
        return []
    base = total // parts
    rem = total % parts
    return [base + (1 if i < rem else 0) for i in range(parts)]


def evenly_spaced_indices(n: int, k: int, avoid_ends: int) -> List[int]:
    """Return k approximately evenly spaced unit indices while avoiding chain ends."""
    if k <= 0:
        return []
    start = max(0, avoid_ends)
    end = max(start + 1, n - avoid_ends)
    if end - start <= 1:
        return [start] * k
    span = end - start - 1
    return [start + round((i + 1) * span / (k + 1)) for i in range(k)]


def terminal_methyl_carbons(mol: Chem.Mol) -> List[int]:
    """Find nonaromatic terminal methyl carbons used to connect monomer units."""
    out: List[int] = []
    for atom in mol.GetAtoms():
        if atom.GetSymbol() != "C":
            continue
        if atom.GetIsAromatic():
            continue
        if atom.GetDegree() != 1:
            continue
        out.append(atom.GetIdx())
    out.sort()
    return out


def backbone_methine_candidates(chain: Chem.Mol) -> List[int]:
    """Find backbone methine atoms suitable for DVB-like crosslink attachment."""
    out: List[int] = []
    for atom in chain.GetAtoms():
        if atom.GetSymbol() != "C":
            continue
        if atom.GetIsAromatic():
            continue
        if atom.GetDegree() != 3:
            continue
        if atom.GetTotalNumHs() < 1:
            continue
        if any(neighbor.GetIsAromatic() for neighbor in atom.GetNeighbors()):
            out.append(atom.GetIdx())
    out.sort()
    return out


def pick_available_methine_anchor(
    mol: Chem.Mol,
    candidates_local: List[int],
    chain_offset: int,
    preferred_rank: int,
    exclude: Set[int],
) -> Optional[int]:
    """Pick the available methine closest to the requested rank."""
    available: List[Tuple[int, int]] = []
    for rank, local_idx in enumerate(candidates_local):
        global_idx = chain_offset + local_idx
        if global_idx in exclude:
            continue
        atom = mol.GetAtomWithIdx(global_idx)
        if atom.GetSymbol() != "C":
            continue
        if atom.GetIsAromatic():
            continue
        if atom.GetDegree() > 3:
            continue
        if atom.GetTotalNumHs() < 1:
            continue
        if not any(neighbor.GetIsAromatic() for neighbor in atom.GetNeighbors()):
            continue
        available.append((abs(rank - preferred_rank), global_idx))

    if not available:
        return None

    available.sort(key=lambda item: (item[0], item[1]))
    return available[0][1]


def build_chain_topology(
    length: int,
    monomer_template: Chem.Mol,
    progress: ProgressBar,
    atoms_per_monomer: int,
) -> Chem.Mol:
    """Build one linear resin chain using connectivity only."""
    if length < 1:
        raise ValueError("Chain length must be at least 1.")

    chain = Chem.Mol(monomer_template)
    ends = terminal_methyl_carbons(chain)
    if len(ends) < 2:
        raise RuntimeError("Monomer template must have two terminal methyl carbons.")
    tail = ends[-1]
    progress.advance(atoms_per_monomer)

    for _ in range(1, length):
        monomer = Chem.Mol(monomer_template)
        monomer_ends = terminal_methyl_carbons(monomer)
        head = monomer_ends[0]
        far = monomer_ends[-1]

        combined = Chem.CombineMols(chain, monomer)
        offset = chain.GetNumAtoms()
        rw = Chem.RWMol(combined)
        if rw.GetBondBetweenAtoms(tail, offset + head) is None:
            rw.AddBond(tail, offset + head, Chem.BondType.SINGLE)
        chain = rw.GetMol()
        Chem.SanitizeMol(chain)
        tail = offset + far
        progress.advance(atoms_per_monomer)

    return chain


def add_dvb_ring_crosslink_topology(
    mol: Chem.Mol,
    ring_template: Chem.Mol,
    a_idx: int,
    b_idx: int,
) -> Chem.Mol:
    """Insert a DVB-like benzene crosslink between two backbone methines."""
    if a_idx == b_idx:
        raise ValueError("Crosslink anchors must be distinct.")

    for idx in (a_idx, b_idx):
        atom = mol.GetAtomWithIdx(idx)
        if atom.GetTotalNumHs() < 1:
            raise ValueError(f"Anchor atom {idx} has no available hydrogen.")
        if atom.GetDegree() > 3:
            raise ValueError(f"Anchor atom {idx} already has degree {atom.GetDegree()}.")

    combined = Chem.CombineMols(mol, ring_template)
    offset = mol.GetNumAtoms()
    ring_a = offset
    ring_b = offset + 3

    rw = Chem.RWMol(combined)
    if rw.GetBondBetweenAtoms(a_idx, ring_a) is None:
        rw.AddBond(a_idx, ring_a, Chem.BondType.SINGLE)
    if rw.GetBondBetweenAtoms(b_idx, ring_b) is None:
        rw.AddBond(b_idx, ring_b, Chem.BondType.SINGLE)
    out = rw.GetMol()
    Chem.SanitizeMol(out)
    return out


def add_counterions_topology(mol: Chem.Mol, ion_smiles: str) -> Chem.Mol:
    """Add one disconnected counterion per cationic ammonium group."""
    ion = Chem.MolFromSmiles(ion_smiles)
    if ion is None or ion.GetNumAtoms() != 1:
        raise ValueError("COUNTERION_SMILES must parse to a single atom ion, e.g. '[Cl-]'.")

    ammoniums = [
        atom.GetIdx()
        for atom in mol.GetAtoms()
        if atom.GetSymbol() == "N" and atom.GetFormalCharge() > 0
    ]
    if not ammoniums:
        ammoniums = [
            atom.GetIdx()
            for atom in mol.GetAtoms()
            if atom.GetSymbol() == "N" and not atom.GetIsAromatic() and atom.GetDegree() == 4
        ]

    out = mol
    for _ in ammoniums:
        out = Chem.CombineMols(out, ion)
    Chem.SanitizeMol(out)
    return out


def build_resin_topology() -> Chem.Mol:
    """Build and return the final resin topology molecule."""
    rng = random.Random(int(SETTINGS["SEED"]))

    total_units = int(SETTINGS["TOTAL_MONOMERS"])
    if total_units < 1:
        raise ValueError("TOTAL_MONOMERS must be at least 1.")

    nlinks = int(round(total_units * float(SETTINGS["CROSSLINK_FRACTION"])))
    if nlinks < 1:
        nlinks = 0

    chain_count = 1 if nlinks == 0 else 1 + nlinks
    chain_lengths = partition_counts(total_units, chain_count)
    if not chain_lengths or sum(chain_lengths) != total_units:
        raise RuntimeError("Failed to partition TOTAL_MONOMERS into chain lengths.")

    monomer_template = Chem.MolFromSmiles(MONOMER_SMILES)
    ring_template = Chem.MolFromSmiles(DVB_RING_SMILES)
    if monomer_template is None or ring_template is None:
        raise RuntimeError("Failed to parse monomer or DVB ring SMILES.")

    atoms_per_monomer = monomer_template.GetNumAtoms()
    atoms_per_crosslink = ring_template.GetNumAtoms()
    total_heavy_est = total_units * atoms_per_monomer + nlinks * atoms_per_crosslink
    progress = ProgressBar(total_heavy_est)

    chains = [
        build_chain_topology(length, monomer_template, progress, atoms_per_monomer)
        for length in chain_lengths
    ]

    combined = Chem.Mol(chains[0])
    chain_offsets: List[int] = [0]
    for idx in range(1, chain_count):
        previous_atoms = combined.GetNumAtoms()
        combined = Chem.CombineMols(combined, chains[idx])
        chain_offsets.append(previous_atoms)
    Chem.SanitizeMol(combined)

    methines_by_chain = [backbone_methine_candidates(chain) for chain in chains]
    avoid = int(SETTINGS["AVOID_END_UNITS"])
    planned: List[Tuple[int, int, int, int]] = []

    if SETTINGS["ENSURE_CONNECTED_NETWORK"] and chain_count >= 2:
        for chain_idx in range(chain_count - 1):
            left_len = chain_lengths[chain_idx]
            right_len = chain_lengths[chain_idx + 1]
            left_unit = evenly_spaced_indices(left_len, 1, avoid)[0] if left_len > 0 else 0
            right_unit = evenly_spaced_indices(right_len, 1, avoid)[0] if right_len > 0 else 0
            planned.append((chain_idx, chain_idx + 1, left_unit, right_unit))

    remaining = nlinks - len(planned)
    for _ in range(max(0, remaining)):
        if SETTINGS["INTERCHAIN_ONLY"]:
            if chain_count < 2:
                continue
            left_chain, right_chain = rng.sample(range(chain_count), 2)
        else:
            left_chain = rng.randrange(chain_count)
            right_chain = rng.randrange(chain_count)
            if chain_count > 1 and left_chain == right_chain:
                right_chain = (right_chain + 1) % chain_count

        left_len = chain_lengths[left_chain]
        right_len = chain_lengths[right_chain]
        left_unit = rng.randrange(avoid, max(avoid + 1, left_len - avoid))
        right_unit = rng.randrange(avoid, max(avoid + 1, right_len - avoid))
        planned.append((left_chain, right_chain, left_unit, right_unit))

    mol = combined
    skipped_crosslinks = 0
    for left_chain, right_chain, left_unit, right_unit in planned:
        left_methines = methines_by_chain[left_chain]
        right_methines = methines_by_chain[right_chain]
        if not left_methines or not right_methines:
            skipped_crosslinks += 1
            continue

        left_rank = min(left_unit, len(left_methines) - 1)
        right_rank = min(right_unit, len(right_methines) - 1)
        left_anchor = pick_available_methine_anchor(
            mol,
            left_methines,
            chain_offsets[left_chain],
            left_rank,
            exclude=set(),
        )
        if left_anchor is None:
            skipped_crosslinks += 1
            continue

        right_anchor = pick_available_methine_anchor(
            mol,
            right_methines,
            chain_offsets[right_chain],
            right_rank,
            exclude={left_anchor},
        )
        if right_anchor is None:
            skipped_crosslinks += 1
            continue

        try:
            mol = add_dvb_ring_crosslink_topology(
                mol,
                ring_template,
                left_anchor,
                right_anchor,
            )
            progress.advance(atoms_per_crosslink)
        except (AtomValenceException, ValueError):
            skipped_crosslinks += 1

    if skipped_crosslinks:
        msg = f"Skipped {skipped_crosslinks} crosslinks due to unavailable anchors or valence issues."
        if SETTINGS["FAIL_ON_SKIPPED_CROSSLINKS"]:
            raise RuntimeError(msg)
        sys.stdout.write(f"[warn] {msg}\n")

    progress.finish("Topology build complete (heavy atoms).")

    if SETTINGS["ADD_COUNTERIONS"]:
        mol = add_counterions_topology(mol, str(SETTINGS["COUNTERION_SMILES"]))

    if SETTINGS["ADD_HS_AT_END"]:
        mol = Chem.AddHs(mol)

    return mol


def main() -> None:
    if SETTINGS["SILENCE_RDKIT_WARNINGS"]:
        rdBase.DisableLog("rdApp.warning")

    mol = build_resin_topology()
    heavy_mol = Chem.RemoveHs(mol)
    smiles = Chem.MolToSmiles(heavy_mol, isomericSmiles=True)

    # SMILES validation
    if REFERENCE_SMILES.exists():
        reference_smiles = REFERENCE_SMILES.read_text(encoding="utf-8").strip()
        reference_mol = Chem.MolFromSmiles(reference_smiles)
        if reference_mol is None:
            raise RuntimeError(f"Could not parse reference SMILES: {REFERENCE_SMILES}")
        reference_canonical = Chem.MolToSmiles(reference_mol, isomericSmiles=True)
        generated_canonical = Chem.MolToSmiles(Chem.MolFromSmiles(smiles), isomericSmiles=True)
        if reference_canonical != generated_canonical:
            raise RuntimeError("Generated topology does not match the reference SMILES graph.")

    molecular_weight = Descriptors.MolWt(heavy_mol)
    sys.stdout.write(
        f"Final topology: MW={molecular_weight:.1f} g/mol  "
        f"formal_charge={Chem.GetFormalCharge(mol)}  atoms={mol.GetNumAtoms()}\n"
    )

    smiles_path = str(SETTINGS["OUT_TOPO_SMILES"])
    with open(smiles_path, "w", encoding="utf-8") as handle:
        handle.write(smiles + "\n")
    sys.stdout.write(f"Wrote: {smiles_path}\n")

    mol_path = str(SETTINGS.get("OUT_TOPO_MOL", ""))
    if mol_path:
        try:
            Chem.MolToMolFile(heavy_mol, mol_path, kekulize=False)
            sys.stdout.write(f"Wrote: {mol_path}\n")
        except Exception as exc:
            sys.stdout.write(f"[warn] MOL write failed: {exc}\n")


if __name__ == "__main__":
    main()
