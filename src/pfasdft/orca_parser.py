"""
ORCA output file parser for extracting DFT calculation results.
Parses energies, EDA components, NOCV pairs, NBO results, etc.
"""
import re
from pathlib import Path
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass


@dataclass
class ORCAResults:
    """Parsed ORCA calculation results."""
    # Energies (Hartree)
    scf_energy: Optional[float] = None
    final_energy: Optional[float] = None
    gibbs_free_energy: Optional[float] = None
    enthalpy: Optional[float] = None
    
    # EDA components (kcal/mol)
    e_electrostatic: Optional[float] = None
    e_pauli: Optional[float] = None
    e_orbital: Optional[float] = None
    e_dispersion: Optional[float] = None
    e_solvation: Optional[float] = None
    e_preparation: Optional[float] = None
    e_binding_total: Optional[float] = None
    
    # NOCV pairs (kcal/mol)
    nocv_pairs: Optional[List[float]] = None
    
    # NBO results
    nbo_donor_acceptor: Optional[str] = None
    nbo_stabilization_energy: Optional[float] = None
    
    # Geometry optimization
    optimization_converged: bool = False
    n_imaginary_frequencies: int = 0
    
    # Errors
    error: str = ""


def parse_orca_output(output_file: Path) -> ORCAResults:
    """
    Parse ORCA output file and extract calculation results.
    
    Args:
        output_file: Path to ORCA .out file
    
    Returns:
        ORCAResults object with parsed data
    """
    results = ORCAResults()
    
    if not output_file.exists():
        results.error = f"Output file not found: {output_file}"
        return results
    
    content = output_file.read_text(encoding='utf-8', errors='ignore')
    
    # Parse final SCF energy
    scf_pattern = r'FINAL SINGLE POINT ENERGY\s+(-?\d+\.\d+)'
    match = re.search(scf_pattern, content)
    if match:
        results.scf_energy = float(match.group(1))
        results.final_energy = results.scf_energy
    
    # Parse Gibbs free energy (from frequency calculation)
    gibbs_pattern = r'Final Gibbs free energy\s+(-?\d+\.\d+)'
    match = re.search(gibbs_pattern, content)
    if match:
        results.gibbs_free_energy = float(match.group(1))
    
    # Parse enthalpy
    enthalpy_pattern = r'Total Enthalpy\s+(-?\d+\.\d+)'
    match = re.search(enthalpy_pattern, content)
    if match:
        results.enthalpy = float(match.group(1))
    
    # Parse optimization convergence
    if "ORCA TERMINATED NORMALLY" in content:
        results.optimization_converged = True
    
    # Parse imaginary frequencies
    imag_freq_pattern = r'(\d+)\s+imaginary frequencies'
    match = re.search(imag_freq_pattern, content)
    if match:
        results.n_imaginary_frequencies = int(match.group(1))
    
    # Parse EDA components
    eda_section = extract_eda_section(content)
    if eda_section:
        results.e_electrostatic = eda_section.get("electrostatic")
        results.e_pauli = eda_section.get("pauli")
        results.e_orbital = eda_section.get("orbital")
        results.e_dispersion = eda_section.get("dispersion")
        results.e_solvation = eda_section.get("solvation")
        results.e_preparation = eda_section.get("preparation")
        results.e_binding_total = eda_section.get("total")
        results.nocv_pairs = eda_section.get("nocv_pairs")
    
    # Parse NBO results
    nbo_section = extract_nbo_section(content)
    if nbo_section:
        results.nbo_donor_acceptor = nbo_section.get("donor_acceptor")
        results.nbo_stabilization_energy = nbo_section.get("stabilization_energy")
    
    return results


def extract_eda_section(content: str) -> Optional[Dict]:
    """Extract EDA-NOCV analysis section from ORCA output."""
    # Look for EDA section
    eda_start = content.find("ENERGY DECOMPOSITION ANALYSIS")
    if eda_start == -1:
        return None
    
    eda_section = content[eda_start:eda_start+5000]  # Get ~5000 chars after EDA
    
    results = {}
    
    # Parse EDA components (in kcal/mol)
    patterns = {
        "electrostatic": r'Electrostatic\s+(-?\d+\.\d+)',
        "pauli": r'Pauli\s+(-?\d+\.\d+)',
        "orbital": r'Orbital\s+(-?\d+\.\d+)',
        "dispersion": r'Dispersion\s+(-?\d+\.\d+)',
        "solvation": r'Solvation\s+(-?\d+\.\d+)',
        "preparation": r'Preparation\s+(-?\d+\.\d+)',
        "total": r'Total\s+(-?\d+\.\d+)',
    }
    
    for key, pattern in patterns.items():
        match = re.search(pattern, eda_section, re.IGNORECASE)
        if match:
            results[key] = float(match.group(1))
    
    # Parse NOCV pairs
    nocv_pairs = []
    nocv_pattern = r'NOCV Pair\s+\d+.*?(-?\d+\.\d+)\s+kcal/mol'
    matches = re.finditer(nocv_pattern, eda_section, re.IGNORECASE)
    for match in matches:
        nocv_pairs.append(float(match.group(1)))
    
    if nocv_pairs:
        results["nocv_pairs"] = nocv_pairs[:5]  # Top 5 pairs
    
    return results if results else None


def extract_nbo_section(content: str) -> Optional[Dict]:
    """Extract NBO analysis section from ORCA output."""
    # Look for NBO section
    nbo_start = content.find("NATURAL BOND ORBITAL")
    if nbo_start == -1:
        return None
    
    nbo_section = content[nbo_start:nbo_start+3000]  # Get ~3000 chars after NBO
    
    results = {}
    
    # Parse donor-acceptor interaction
    donor_acceptor_pattern = r'(\w+)\s+LP\s+.*?->\s+(\w+)\s+BD\*'
    match = re.search(donor_acceptor_pattern, nbo_section, re.IGNORECASE)
    if match:
        results["donor_acceptor"] = f"{match.group(1)} lone pair -> {match.group(2)} antibond"
    
    # Parse stabilization energy
    stab_pattern = r'Stabilization energy\s+(-?\d+\.\d+)\s+kcal/mol'
    match = re.search(stab_pattern, nbo_section, re.IGNORECASE)
    if match:
        results["stabilization_energy"] = float(match.group(1))
    
    return results if results else None


def calculate_exchange_energy(
    complex_results: ORCAResults,
    btma_cl_results: ORCAResults,
    anion_results: ORCAResults,
    chloride_results: ORCAResults,
    use_gibbs: bool = True,
) -> Tuple[Optional[float], Optional[float]]:
    """
    Calculate anion exchange energies.
    
    ΔEexchange = E(R4N+X-) - E(R4N+Cl-) - E(X-) + E(Cl-)
    ΔGexchange = G(R4N+X-) - G(R4N+Cl-) - G(X-) + G(Cl-)
    
    Returns:
        (delta_e_exchange, delta_g_exchange) in kcal/mol
    """
    # Convert Hartree to kcal/mol (1 Hartree = 627.509 kcal/mol)
    hartree_to_kcal = 627.509
    
    # Electronic energy
    if all([r.final_energy is not None for r in [complex_results, btma_cl_results, anion_results, chloride_results]]):
        delta_e = (
            complex_results.final_energy -
            btma_cl_results.final_energy -
            anion_results.final_energy +
            chloride_results.final_energy
        )
        delta_e_kcal = delta_e * hartree_to_kcal
    else:
        delta_e_kcal = None
    
    # Gibbs free energy
    if use_gibbs:
        if all([r.gibbs_free_energy is not None for r in [complex_results, btma_cl_results, anion_results, chloride_results]]):
            delta_g = (
                complex_results.gibbs_free_energy -
                btma_cl_results.gibbs_free_energy -
                anion_results.gibbs_free_energy +
                chloride_results.gibbs_free_energy
            )
            delta_g_kcal = delta_g * hartree_to_kcal
        else:
            delta_g_kcal = None
    else:
        delta_g_kcal = None
    
    return delta_e_kcal, delta_g_kcal


def parse_multiple_outputs(output_dir: Path, pattern: str = "*.out") -> Dict[str, ORCAResults]:
    """
    Parse multiple ORCA output files in a directory.
    
    Args:
        output_dir: Directory containing ORCA output files
        pattern: File pattern to match (default: "*.out")
    
    Returns:
        Dictionary mapping filename to ORCAResults
    """
    results = {}
    output_dir = Path(output_dir)
    
    for out_file in output_dir.glob(pattern):
        name = out_file.stem
        results[name] = parse_orca_output(out_file)
    
    return results
