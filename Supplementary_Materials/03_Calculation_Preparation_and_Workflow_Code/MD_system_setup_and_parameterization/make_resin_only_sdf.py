from pathlib import Path

from rdkit import Chem
from rdkit.Chem import rdmolops

source = Path("../../07-48mer/01_InitialStruc/48mer_init.sdf")
out = Path("chol48_resin_only.sdf")
report = Path("../logs/resin_only_sdf_report.txt")

mol = Chem.MolFromMolFile(str(source), removeHs=False, sanitize=False)
if mol is None:
    raise SystemExit(f"ERROR: RDKit could not read {source}")

editable = Chem.RWMol(mol)
remove = [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetSymbol().upper() == "CL"]
for idx in sorted(remove, reverse=True):
    editable.RemoveAtom(idx)

resin = editable.GetMol()
try:
    Chem.SanitizeMol(resin)
    sanitize = "OK"
except Exception as exc:
    sanitize = f"WARNING: {exc!r}"

block = Chem.MolToMolBlock(resin, forceV3000=True)
out.write_text(block + "\n$$$$\n")

frags = Chem.GetMolFrags(resin)
formal_charge = rdmolops.GetFormalCharge(resin)
n_plus = sum(1 for atom in resin.GetAtoms() if atom.GetAtomicNum() == 7 and atom.GetFormalCharge() == 1)

lines = [
    f"source: {source}",
    f"output: {out}",
    f"removed_chloride_atoms: {len(remove)}",
    f"num_atoms: {resin.GetNumAtoms()}",
    f"num_bonds: {resin.GetNumBonds()}",
    f"num_fragments: {len(frags)}",
    f"formal_charge: {formal_charge}",
    f"positively_charged_nitrogens: {n_plus}",
    f"sanitize: {sanitize}",
]
report.write_text("\n".join(lines) + "\n")
print("\n".join(lines))
