"""
ORCA input file generator for PFAS-cholestyramine DFT calculations.
Generates input files following the methodology from the PFAS Removal DFT paper.
"""
from pathlib import Path
from typing import Optional, Dict, List
from dataclasses import dataclass


@dataclass
class ORCAConfig:
    """Configuration for ORCA calculations."""
    functional: str = "r2SCAN-3c"  # or "wB97X-D3"
    basis_set: str = "def2-TZVPD"  # for wB97X-D3, auto for r2SCAN-3c
    solvent: str = "Water"
    dielectric: float = 72.5
    temperature: float = 310.15
    optimization: bool = True
    frequency: bool = False
    eda_nocv: bool = False
    nbo: bool = False
    tightopt: bool = True
    defgrid: int = 3
    rijcosx: bool = True
    aux_basis: str = "def2/J"
    charge: int = 0
    multiplicity: int = 1
    nprocs: int = 4
    mem: int = 4000  # MB


def generate_orca_input(
    xyz_file: Path,
    output_file: Path,
    config: ORCAConfig,
    calculation_type: str = "opt",
    title: Optional[str] = None,
) -> str:
    """
    Generate ORCA input file for PFAS-cholestyramine calculations.
    
    Args:
        xyz_file: Path to XYZ structure file
        output_file: Path for output .inp file
        config: ORCA configuration
        calculation_type: Type of calculation (opt, freq, sp, eda, nbo, goat)
        title: Optional title for calculation
    
    Returns:
        ORCA input file content as string
    """
    lines = []
    
    # Title
    if title:
        lines.append(f"! {title}")
    else:
        lines.append(f"! {calculation_type.upper()} calculation")
    
    # Functional and basis set
    if config.functional == "r2SCAN-3c":
        lines.append(f"! {config.functional}")
        # r2SCAN-3c has built-in basis set
    elif config.functional == "wB97X-D3":
        lines.append(f"! {config.functional} {config.basis_set}")
        if config.rijcosx:
            lines.append(f"! RIJCOSX {config.aux_basis}")
    else:
        lines.append(f"! {config.functional} {config.basis_set}")
    
    # Calculation type
    if calculation_type == "opt":
        if config.tightopt:
            lines.append("! TightOpt")
        else:
            lines.append("! Opt")
    elif calculation_type == "freq":
        lines.append("! Opt Freq")
    elif calculation_type == "sp":
        lines.append("! SP")
    elif calculation_type == "eda":
        lines.append("! SP")
        lines.append("! EDA")
    elif calculation_type == "nbo":
        lines.append("! SP")
        lines.append("! NBO")
    elif calculation_type == "goat":
        lines.append("! Opt")
        lines.append("! GFN2-xTB")
    
    # Solvation
    if config.solvent:
        lines.append(f"%cpcm")
        lines.append(f"  smd true")
        lines.append(f"  smdsolvent \"{config.solvent}\"")
        lines.append(f"  epsilon {config.dielectric}")
        lines.append(f"end")
    
    # Numerical settings
    if config.defgrid > 0:
        lines.append(f"%method")
        lines.append(f"  DefGrid{config.defgrid}")
        lines.append(f"end")
    
    # Parallelization
    lines.append(f"%pal")
    lines.append(f"  nprocs {config.nprocs}")
    lines.append(f"end")
    
    # Memory
    lines.append(f"%maxcore {config.mem}")
    
    # EDA-NOCV block
    if config.eda_nocv and calculation_type == "eda":
        lines.append(f"%eda")
        lines.append(f"  NOCV true")
        lines.append(f"end")
    
    # NBO block
    if config.nbo and calculation_type == "nbo":
        lines.append(f"%nbo")
        lines.append(f"  file {output_file.stem}")
        lines.append(f"end")
    
    # GOAT block
    if calculation_type == "goat":
        lines.append(f"%goat")
        lines.append(f"  method \"GFN2-xTB\"")
        lines.append(f"  nconformers 50")
        lines.append(f"end")
    
    # Charge and multiplicity
    lines.append(f"* xyzfile {config.charge} {config.multiplicity} {xyz_file.name}")
    
    return "\n".join(lines)


def generate_workflow_inputs(
    base_name: str,
    pfas_xyz: Path,
    cholestyramine_xyz: Path,
    complex_xyz: Optional[Path],
    output_dir: Path,
    config: ORCAConfig,
) -> Dict[str, Path]:
    """
    Generate input files for complete PFAS binding workflow.
    
    Workflow:
    1. GOAT global optimization (GFN2-xTB)
    2. r2SCAN-3c optimization
    3. wB97X-D3 optimization
    4. Frequency calculation
    5. EDA-NOCV analysis
    6. NBO analysis
    
    Returns:
        Dictionary mapping calculation type to input file path
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    inputs = {}
    
    # 1. GOAT global optimization
    if complex_xyz:
        goat_config = ORCAConfig(
            functional="GFN2-xTB",
            solvent="Water",
            optimization=True,
            tightopt=False,
        )
        goat_input = generate_orca_input(
            complex_xyz,
            output_dir / f"{base_name}_GOAT.inp",
            goat_config,
            calculation_type="goat",
            title=f"{base_name} - GOAT Global Optimization",
        )
        (output_dir / f"{base_name}_GOAT.inp").write_text(goat_input)
        inputs["goat"] = output_dir / f"{base_name}_GOAT.inp"
    
    # 2. r2SCAN-3c optimization
    r2scan_config = ORCAConfig(
        functional="r2SCAN-3c",
        solvent="Water",
        dielectric=72.5,
        optimization=True,
        tightopt=True,
    )
    opt_xyz = complex_xyz or pfas_xyz
    r2scan_input = generate_orca_input(
        opt_xyz,
        output_dir / f"{base_name}_r2SCAN3c_opt.inp",
        r2scan_config,
        calculation_type="opt",
        title=f"{base_name} - r2SCAN-3c Optimization",
    )
    (output_dir / f"{base_name}_r2SCAN3c_opt.inp").write_text(r2scan_input)
    inputs["r2scan_opt"] = output_dir / f"{base_name}_r2SCAN3c_opt.inp"
    
    # 3. wB97X-D3 optimization
    wb97_config = ORCAConfig(
        functional="wB97X-D3",
        basis_set="def2-TZVPD",
        solvent="Water",
        dielectric=72.5,
        optimization=True,
        tightopt=True,
        rijcosx=True,
    )
    wb97_input = generate_orca_input(
        opt_xyz,
        output_dir / f"{base_name}_wB97X-D3_opt.inp",
        wb97_config,
        calculation_type="opt",
        title=f"{base_name} - wB97X-D3 Optimization",
    )
    (output_dir / f"{base_name}_wB97X-D3_opt.inp").write_text(wb97_input)
    inputs["wb97_opt"] = output_dir / f"{base_name}_wB97X-D3_opt.inp"
    
    # 4. Frequency calculation (on optimized structure)
    freq_config = ORCAConfig(
        functional="wB97X-D3",
        basis_set="def2-TZVPD",
        solvent="Water",
        dielectric=72.5,
        optimization=False,
        frequency=True,
        tightopt=True,
        rijcosx=True,
    )
    freq_input = generate_orca_input(
        opt_xyz,  # Should use optimized structure
        output_dir / f"{base_name}_wB97X-D3_freq.inp",
        freq_config,
        calculation_type="freq",
        title=f"{base_name} - wB97X-D3 Frequency",
    )
    (output_dir / f"{base_name}_wB97X-D3_freq.inp").write_text(freq_input)
    inputs["freq"] = output_dir / f"{base_name}_wB97X-D3_freq.inp"
    
    # 5. EDA-NOCV analysis
    eda_config = ORCAConfig(
        functional="wB97X-D3",
        basis_set="def2-TZVPD",
        solvent="Water",
        dielectric=72.5,
        optimization=False,
        eda_nocv=True,
        tightopt=False,
        rijcosx=True,
    )
    eda_input = generate_orca_input(
        opt_xyz,  # Should use optimized structure
        output_dir / f"{base_name}_EDA-NOCV.inp",
        eda_config,
        calculation_type="eda",
        title=f"{base_name} - EDA-NOCV Analysis",
    )
    (output_dir / f"{base_name}_EDA-NOCV.inp").write_text(eda_input)
    inputs["eda"] = output_dir / f"{base_name}_EDA-NOCV.inp"
    
    # 6. NBO analysis
    nbo_config = ORCAConfig(
        functional="wB97X-D3",
        basis_set="def2-TZVPD",
        solvent="Water",
        dielectric=72.5,
        optimization=False,
        nbo=True,
        tightopt=False,
        rijcosx=True,
    )
    nbo_input = generate_orca_input(
        opt_xyz,  # Should use optimized structure
        output_dir / f"{base_name}_NBO.inp",
        nbo_config,
        calculation_type="nbo",
        title=f"{base_name} - NBO Analysis",
    )
    (output_dir / f"{base_name}_NBO.inp").write_text(nbo_input)
    inputs["nbo"] = output_dir / f"{base_name}_NBO.inp"
    
    return inputs


def generate_exchange_calculation_inputs(
    base_name: str,
    complex_xyz: Path,
    anion_xyz: Path,
    chloride_xyz: Path,
    btma_cl_xyz: Path,
    output_dir: Path,
    config: ORCAConfig,
) -> Dict[str, Path]:
    """
    Generate input files for anion exchange energy calculation.
    
    Calculates: ΔGexchange = G(R4N+X-) - G(R4N+Cl-) - G(X-) + G(Cl-)
    
    Returns:
        Dictionary mapping component to input file path
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    inputs = {}
    
    # Components needed:
    # 1. R4N+X- (complex)
    # 2. R4N+Cl- (BTMA-Cl)
    # 3. X- (anion)
    # 4. Cl- (chloride)
    
    components = {
        "complex": (complex_xyz, f"{base_name}_complex"),
        "btma_cl": (btma_cl_xyz, f"{base_name}_BTMA-Cl"),
        "anion": (anion_xyz, f"{base_name}_anion"),
        "chloride": (chloride_xyz, f"{base_name}_Cl"),
    }
    
    for key, (xyz_path, name) in components.items():
        freq_config = ORCAConfig(
            functional=config.functional,
            basis_set=config.basis_set,
            solvent=config.solvent,
            dielectric=config.dielectric,
            optimization=True,
            frequency=True,
            tightopt=config.tightopt,
            rijcosx=config.rijcosx,
        )
        inp_content = generate_orca_input(
            xyz_path,
            output_dir / f"{name}_freq.inp",
            freq_config,
            calculation_type="freq",
            title=f"{name} - Frequency Calculation",
        )
        (output_dir / f"{name}_freq.inp").write_text(inp_content)
        inputs[key] = output_dir / f"{name}_freq.inp"
    
    return inputs
