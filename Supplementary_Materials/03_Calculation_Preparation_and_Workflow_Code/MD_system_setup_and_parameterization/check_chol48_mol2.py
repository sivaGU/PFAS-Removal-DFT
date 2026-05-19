from pathlib import Path

from rdkit import Chem
from rdkit.Chem import rdmolops

path = Path("chol48_input.mol2")
mol = Chem.MolFromMol2File(str(path), removeHs=False, sanitize=False)
if mol is None:
    raise SystemExit("ERROR: RDKit could not read chol48_input.mol2")

try:
    Chem.SanitizeMol(mol)
    print("sanitize: OK")
except Exception as e:
    print("sanitize: WARNING", repr(e))

frags = Chem.GetMolFrags(mol)
charge = rdmolops.GetFormalCharge(mol)
n_plus = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 7 and a.GetFormalCharge() == 1)
n_atoms = mol.GetNumAtoms()
n_bonds = mol.GetNumBonds()

print("num_atoms", n_atoms)
print("num_bonds", n_bonds)
print("num_fragments", len(frags))
print("formal_charge", charge)
print("positively_charged_nitrogens", n_plus)

if len(frags) > 1:
    print("WARNING: multiple molecular fragments detected")
if n_plus == 0:
    print("WARNING: no positively charged nitrogens detected")
