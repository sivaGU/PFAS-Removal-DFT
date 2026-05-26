from __future__ import annotations

import io
import re
import zipfile
from dataclasses import dataclass

from .orca_templates import CALC_TYPES, OrcaSettings, build_orca_input
from .xyz_utils import XyzStructure


def safe_name(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9+_.-]+", "_", name.strip())
    return cleaned.strip("_") or "structure"


@dataclass(frozen=True)
class FileBundle:
    files: dict[str, str]

    def as_zip_bytes(self) -> bytes:
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for filename, content in self.files.items():
                zf.writestr(filename, content)
        return buffer.getvalue()


def single_calculation_bundle(
    *,
    structure: XyzStructure,
    calc_type: str,
    charge: int,
    multiplicity: int,
    settings: OrcaSettings,
    pfas_atom_count: int | None = None,
) -> FileBundle:
    stem = safe_name(structure.name)
    xyz_name = f"{stem}.xyz"
    inp_name = f"{stem}_{safe_name(calc_type)}.inp"
    inp = build_orca_input(
        title=f"{structure.name} - {calc_type}",
        calc_type=calc_type,
        xyz_filename=xyz_name,
        charge=charge,
        multiplicity=multiplicity,
        settings=settings,
        pfas_atom_count=pfas_atom_count,
        total_atom_count=structure.atom_count,
    )
    return FileBundle({xyz_name: structure.text, inp_name: inp})


def full_workflow_bundle(
    *,
    pfas_name: str,
    model_name: str,
    pfas: XyzStructure,
    complex_structure: XyzStructure,
    settings: OrcaSettings,
    include_goat: bool,
    include_r2scan_opt: bool,
    include_wb97xd3_opt: bool,
    include_frequency: bool,
    include_eda: bool,
    include_nbo: bool,
) -> FileBundle:
    root = f"{safe_name(model_name)}_{safe_name(pfas_name)}"
    complex_xyz = f"{root}_R4N+X-.xyz"
    pfas_xyz = f"{safe_name(pfas_name)}_X-.xyz"
    files: dict[str, str] = {
        f"structures/{complex_xyz}": complex_structure.text,
        f"structures/{pfas_xyz}": pfas.text,
    }
    tasks = [
        (include_goat, "01_GOAT_GFN2-xTB", "GOAT global optimization (GFN2-xTB)", "complex", 0, 1),
        (include_r2scan_opt, "02_r2SCAN-3c_opt", "r2SCAN-3c geometry optimization", "complex", 0, 1),
        (include_wb97xd3_opt, "03_wB97X-D3_opt", "wB97X-D3 geometry optimization", "complex", 0, 1),
        (include_frequency, "04_frequency", "Frequency calculation", "complex", 0, 1),
        (include_eda, "05_EDA-NOCV", "EDA-NOCV analysis", "complex", 0, 1),
        (include_nbo, "06_NBO", "NBO analysis", "complex", 0, 1),
    ]
    for enabled, folder, calc_type, _, charge, mult in tasks:
        if not enabled:
            continue
        inp = build_orca_input(
            title=f"{model_name} {pfas_name} complex - {calc_type}",
            calc_type=calc_type,
            xyz_filename=f"../structures/{complex_xyz}",
            charge=charge,
            multiplicity=mult,
            settings=settings,
            pfas_atom_count=pfas.atom_count,
            total_atom_count=complex_structure.atom_count,
        )
        files[f"{folder}/{root}_{safe_name(calc_type)}.inp"] = inp
    return FileBundle(files)


def exchange_bundle(
    *,
    pfas_name: str,
    model_name: str,
    pfas: XyzStructure,
    complex_structure: XyzStructure,
    chloride_complex: XyzStructure,
    chloride: XyzStructure,
    settings: OrcaSettings,
) -> FileBundle:
    root = f"{safe_name(model_name)}_{safe_name(pfas_name)}_exchange"
    components = [
        ("R4N+X-", complex_structure, 0, 1),
        ("R4N+Cl-", chloride_complex, 0, 1),
        ("X-", pfas, -1, 1),
        ("Cl-", chloride, -1, 1),
    ]
    files: dict[str, str] = {}
    for label, structure, charge, mult in components:
        stem = safe_name(label.replace("X", pfas_name))
        xyz_name = f"structures/{stem}.xyz"
        inp_name = f"frequency_inputs/{stem}_freq.inp"
        files[xyz_name] = structure.text
        files[inp_name] = build_orca_input(
            title=f"{root} component {label}",
            calc_type="Frequency calculation",
            xyz_filename=f"../{xyz_name}",
            charge=charge,
            multiplicity=mult,
            settings=settings,
        )
    return FileBundle(files)


def default_workflow_choices() -> dict[str, bool]:
    return {
        "GOAT global optimization (GFN2-xTB)": True,
        "r2SCAN-3c geometry optimization": True,
        "wB97X-D3 geometry optimization": True,
        "Frequency calculation": True,
        "EDA-NOCV analysis": True,
        "NBO analysis": True,
    }
