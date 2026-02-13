"""
Structure preparation utilities for PFAS-cholestyramine calculations.
Converts SMILES to XYZ, prepares structures for ORCA calculations.
"""
from pathlib import Path
from typing import Optional, List
import tempfile

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from rdkit.Chem import rdMolDescriptors
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False


def smiles_to_xyz(smiles: str, output_file: Path, optimize: bool = True) -> bool:
    """
    Convert SMILES string to XYZ file format.
    
    Args:
        smiles: SMILES string
        output_file: Path for output XYZ file
        optimize: Whether to optimize geometry with MMFF94
    
    Returns:
        True if successful, False otherwise
    """
    if not RDKIT_AVAILABLE:
        return False
    
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False
    
    # Add hydrogens
    mol = Chem.AddHs(mol)
    
    # Generate 3D coordinates
    if optimize:
        try:
            AllChem.EmbedMolecule(mol, AllChem.ETKDG())
            AllChem.MMFFOptimizeMolecule(mol)
        except Exception:
            try:
                AllChem.EmbedMolecule(mol, randomSeed=42)
            except Exception:
                return False
    else:
        try:
            AllChem.EmbedMolecule(mol, randomSeed=42)
        except Exception:
            return False
    
    # Get conformer
    conf = mol.GetConformer()
    if conf is None:
        return False
    
    # Write XYZ file
    lines = [str(mol.GetNumAtoms())]
    lines.append(f"Generated from SMILES: {smiles}")
    
    for atom in mol.GetAtoms():
        symbol = atom.GetSymbol()
        pos = conf.GetAtomPosition(atom.GetIdx())
        lines.append(f"{symbol:3s} {pos.x:15.10f} {pos.y:15.10f} {pos.z:15.10f}")
    
    output_file.write_text("\n".join(lines))
    return True


def prepare_pfas_structure(
    smiles: str,
    pfas_name: str,
    output_dir: Path,
    deprotonate: bool = True,
) -> Optional[Path]:
    """
    Prepare PFAS structure for DFT calculation.
    
    Args:
        smiles: SMILES string of PFAS
        pfas_name: Name of PFAS molecule
        output_dir: Output directory
        deprotonate: Whether to deprotonate (PFAS are anions at pH 7.4)
    
    Returns:
        Path to XYZ file or None if failed
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # For now, use SMILES as-is (deprotonation would require pKa knowledge)
    # In production, would use OpenBabel or similar to handle protonation states
    xyz_file = output_dir / f"{pfas_name}.xyz"
    
    if smiles_to_xyz(smiles, xyz_file, optimize=True):
        return xyz_file
    return None


def prepare_cholestyramine_structure(
    model_type: str = "BTMA",
    output_dir: Path = Path("."),
) -> Optional[Path]:
    """
    Prepare cholestyramine structure (BTMA or Extended monomer).
    
    Args:
        model_type: "BTMA" or "Extended"
        output_dir: Output directory
    
    Returns:
        Path to XYZ file or None if failed
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Simplified structures - in production would load from PubChem or structure database
    if model_type == "BTMA":
        # Benzyltrimethylammonium cation (simplified)
        btma_smiles = "[N+](C)(C)(C)Cc1ccccc1"
        xyz_file = output_dir / "BTMA.xyz"
    elif model_type == "Extended":
        # Extended monomer - would need actual structure
        # For now, use BTMA as placeholder
        btma_smiles = "[N+](C)(C)(C)Cc1ccccc1"
        xyz_file = output_dir / "Cholestyramine_monomer.xyz"
    else:
        return None
    
    if smiles_to_xyz(btma_smiles, xyz_file, optimize=True):
        return xyz_file
    return None


def prepare_complex_structure(
    pfas_xyz: Path,
    cholestyramine_xyz: Path,
    output_file: Path,
    distance: float = 3.0,
) -> bool:
    """
    Prepare PFAS-cholestyramine complex structure.
    
    Args:
        pfas_xyz: Path to PFAS XYZ file
        cholestyramine_xyz: Path to cholestyramine XYZ file
        output_file: Path for output complex XYZ file
        distance: Initial distance between headgroups (Angstrom)
    
    Returns:
        True if successful, False otherwise
    """
    # Read structures
    pfas_lines = pfas_xyz.read_text().strip().split("\n")
    chol_lines = cholestyramine_xyz.read_text().strip().split("\n")
    
    pfas_natoms = int(pfas_lines[0])
    chol_natoms = int(chol_lines[0])
    
    # Parse coordinates
    pfas_coords = []
    for line in pfas_lines[2:2+pfas_natoms]:
        parts = line.split()
        pfas_coords.append((parts[0], float(parts[1]), float(parts[2]), float(parts[3])))
    
    chol_coords = []
    for line in chol_lines[2:2+chol_natoms]:
        parts = line.split()
        chol_coords.append((parts[0], float(parts[1]), float(parts[2]), float(parts[3])))
    
    # Simple positioning: place PFAS headgroup near ammonium nitrogen
    # Find ammonium N in cholestyramine (first N+)
    n_idx = None
    for i, (symbol, x, y, z) in enumerate(chol_coords):
        if symbol == "N":
            n_idx = i
            break
    
    if n_idx is None:
        return False
    
    # Find PFAS headgroup (O or S in headgroup)
    head_idx = None
    for i, (symbol, x, y, z) in enumerate(pfas_coords):
        if symbol in ["O", "S"]:
            head_idx = i
            break
    
    if head_idx is None:
        return False
    
    # Translate PFAS so headgroup is at distance from N
    n_pos = chol_coords[n_idx][1:4]
    head_pos = pfas_coords[head_idx][1:4]
    
    # Vector from head to N
    try:
        import numpy as np
    except ImportError:
        # Fallback if numpy not available
        return False
    
    vec = np.array(n_pos) - np.array(head_pos)
    vec_norm = np.linalg.norm(vec)
    if vec_norm > 0:
        vec = vec / vec_norm
    
    # New head position
    new_head_pos = np.array(n_pos) - vec * distance
    
    # Translation vector
    trans = new_head_pos - np.array(head_pos)
    
    # Translate all PFAS atoms
    translated_pfas = []
    for symbol, x, y, z in pfas_coords:
        new_pos = np.array([x, y, z]) + trans
        translated_pfas.append((symbol, new_pos[0], new_pos[1], new_pos[2]))
    
    # Write complex XYZ
    total_atoms = pfas_natoms + chol_natoms
    lines = [str(total_atoms)]
    lines.append("PFAS-Cholestyramine complex")
    
    # Add PFAS atoms
    for symbol, x, y, z in translated_pfas:
        lines.append(f"{symbol:3s} {x:15.10f} {y:15.10f} {z:15.10f}")
    
    # Add cholestyramine atoms
    for symbol, x, y, z in chol_coords:
        lines.append(f"{symbol:3s} {x:15.10f} {y:15.10f} {z:15.10f}")
    
    output_file.write_text("\n".join(lines))
    return True
