from __future__ import annotations

from pathlib import Path

from .xyz_utils import XyzStructure, load_xyz


ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"

PFAS_NAMES = ["PFOA", "PFOS", "PFHxA", "FHEA"]
MODEL_NAMES = ["BTMA", "Extended Monomer"]

MODEL_DIR = {
    "BTMA": "BTMA",
    "Extended Monomer": "Extended_Monomer",
}


def pfas_xyz_path(pfas: str) -> Path:
    return EXAMPLES / "pfas_xyz" / f"{pfas}.xyz"


def complex_xyz_path(model: str, pfas: str) -> Path:
    return EXAMPLES / "complexes" / MODEL_DIR[model] / f"R4N+{pfas}-.xyz"


def chloride_complex_path(model: str) -> Path:
    return EXAMPLES / "chloride_complexes" / f"{MODEL_DIR[model]}_R4N+Cl-.xyz"


def chloride_path() -> Path:
    return EXAMPLES / "chloride" / "Cl-.xyz"


def load_pfas(pfas: str) -> XyzStructure:
    return load_xyz(pfas_xyz_path(pfas), pfas)


def load_complex(model: str, pfas: str) -> XyzStructure:
    return load_xyz(complex_xyz_path(model, pfas), f"R4N+{pfas}-")


def load_chloride_complex(model: str) -> XyzStructure:
    return load_xyz(chloride_complex_path(model), "R4N+Cl-")


def load_chloride() -> XyzStructure:
    return load_xyz(chloride_path(), "Cl-")


def validate_examples() -> list[str]:
    missing: list[str] = []
    for pfas in PFAS_NAMES:
        if not pfas_xyz_path(pfas).exists():
            missing.append(str(pfas_xyz_path(pfas)))
        for model in MODEL_NAMES:
            if not complex_xyz_path(model, pfas).exists():
                missing.append(str(complex_xyz_path(model, pfas)))
    for model in MODEL_NAMES:
        if not chloride_complex_path(model).exists():
            missing.append(str(chloride_complex_path(model)))
    if not chloride_path().exists():
        missing.append(str(chloride_path()))
    return missing
