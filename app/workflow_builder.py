from __future__ import annotations

import io
import re
import zipfile
from dataclasses import dataclass

from .orca_templates import OrcaSettings, build_orca_input, fragment_method_file
from .xyz_utils import XyzStructure


def safe_name(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9+_.-]+", "_", name.strip())
    return cleaned.strip("_") or "structure"


def component_stem(label: str, pfas_name: str) -> str:
    return safe_name(label.replace("X", pfas_name))


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
    files = {xyz_name: structure.text, inp_name: inp}
    if calc_type == "EDA-NOCV analysis":
        files["frag1method.txt"] = fragment_method_file(settings)
        files["frag2method.txt"] = fragment_method_file(settings)
    return FileBundle(files)


def interaction_bundle(
    *,
    pfas_name: str,
    model_name: str,
    pfas: XyzStructure,
    complex_structure: XyzStructure,
    settings: OrcaSettings,
    include_eda: bool,
    include_nbo: bool,
) -> FileBundle:
    root = f"{safe_name(model_name)}_{safe_name(pfas_name)}"
    complex_xyz = f"{root}_R4N+X-.xyz"
    files: dict[str, str] = {}
    tasks = [
        (include_eda, "01_EDA-NOCV", "EDA-NOCV analysis", 0, 1),
        (include_nbo, "02_NBO", "NBO analysis", 0, 1),
    ]
    for enabled, folder, calc_type, charge, mult in tasks:
        if not enabled:
            continue
        files[f"{folder}/{complex_xyz}"] = complex_structure.text
        inp = build_orca_input(
            title=f"{model_name} {pfas_name} complex - {calc_type}",
            calc_type=calc_type,
            xyz_filename=complex_xyz,
            charge=charge,
            multiplicity=mult,
            settings=settings,
            pfas_atom_count=pfas.atom_count,
            total_atom_count=complex_structure.atom_count,
        )
        files[f"{folder}/{root}_{safe_name(calc_type)}.inp"] = inp
        if calc_type == "EDA-NOCV analysis":
            files[f"{folder}/frag1method.txt"] = fragment_method_file(settings)
            files[f"{folder}/frag2method.txt"] = fragment_method_file(settings)
    return FileBundle(files)


def exchange_bundle(
    *,
    pfas_name: str,
    model_name: str,
    pfas: XyzStructure,
    complex_structure: XyzStructure,
    chloride_complex: XyzStructure,
    chloride: XyzStructure,
    geometry_calc_type: str,
    geometry_settings: OrcaSettings,
    frequency_settings: OrcaSettings,
    include_goat: bool = False,
    goat_settings: OrcaSettings | None = None,
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
        stem = component_stem(label, pfas_name)
        xyz_name = f"{stem}.xyz"

        opt_folder = f"01_geometry_optimization/{stem}"
        files[f"{opt_folder}/{xyz_name}"] = structure.text
        files[f"{opt_folder}/{stem}_opt.inp"] = build_orca_input(
            title=f"{root} component {label} - geometry optimization",
            calc_type=geometry_calc_type,
            xyz_filename=xyz_name,
            charge=charge,
            multiplicity=mult,
            settings=geometry_settings,
        )

        freq_folder = f"02_frequency/{stem}"
        files[f"{freq_folder}/{xyz_name}"] = structure.text
        files[f"{freq_folder}/{stem}_freq.inp"] = build_orca_input(
            title=f"{root} component {label}",
            calc_type="Frequency calculation",
            xyz_filename=xyz_name,
            charge=charge,
            multiplicity=mult,
            settings=frequency_settings,
        )

        if include_goat and label != "Cl-":
            goat_folder = f"00_GOAT_GFN2-xTB/{stem}"
            files[f"{goat_folder}/{xyz_name}"] = structure.text
            files[f"{goat_folder}/{stem}_GOAT.inp"] = build_orca_input(
                title=f"{root} component {label} - GOAT global optimization",
                calc_type="GOAT global optimization (GFN2-xTB)",
                xyz_filename=xyz_name,
                charge=charge,
                multiplicity=mult,
                settings=goat_settings or geometry_settings,
            )
    return FileBundle(files)
