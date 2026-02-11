"""
PFAS Removal DFT prediction module.
Predicts binding affinities and energy decomposition for PFAS-cholestyramine complexes.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Union, Dict
import json

import pandas as pd
import numpy as np
from rdkit import Chem


@dataclass
class PredictResult:
    """Result of a single PFAS binding prediction."""
    is_valid: bool
    pfas_name: str
    smiles: str
    canonical_smiles: str
    
    # Binding energies (kcal/mol)
    delta_e_exchange: Optional[float] = None
    delta_g_exchange: Optional[float] = None
    
    # EDA components (kcal/mol)
    e_electrostatic: Optional[float] = None
    e_pauli: Optional[float] = None
    e_orbital: Optional[float] = None
    e_dispersion: Optional[float] = None
    e_solvation: Optional[float] = None
    e_preparation: Optional[float] = None
    e_binding_total: Optional[float] = None
    
    # NOCV pairs (top 5 stabilization energies in kcal/mol)
    nocv_pairs: Optional[List[float]] = None
    
    # NBO analysis
    nbo_donor_acceptor: Optional[str] = None
    nbo_stabilization_energy: Optional[float] = None
    
    # Model info
    functional: Optional[str] = None
    model_type: Optional[str] = None  # "BTMA" or "Extended"
    
    error: str = ""


def _canonicalize_smiles(smiles: str) -> Optional[str]:
    """Canonicalize SMILES."""
    if not smiles or not isinstance(smiles, str):
        return None
    mol = Chem.MolFromSmiles(smiles.strip())
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)


def _identify_pfas(smiles: str) -> str:
    """Identify PFAS molecule from SMILES."""
    canon = _canonicalize_smiles(smiles)
    if canon is None:
        return "Unknown"
    
    # Known PFAS SMILES patterns (simplified - would need actual canonical SMILES)
    pfas_patterns = {
        "PFOS": ["C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)S(=O)(=O)[O-]"],
        "PFOA": ["C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(=O)[O-]"],
        "PFHxA": ["C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(=O)[O-]"],
        "FHEA": ["C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)CC(=O)[O-]"],
    }
    
    # For now, return based on length or other heuristics
    # In production, would use exact SMILES matching or substructure search
    return "PFAS"


class PFASDFTPredictor:
    """Loaded predictor state (models, reference data)."""
    
    def __init__(
        self,
        reference_data: Dict,
        model_config: Dict,
    ):
        self.reference_data = reference_data
        self.model_config = model_config
    
    def predict(self, smiles: str, pfas_name: Optional[str] = None, model_type: str = "Extended") -> PredictResult:
        """Run prediction for one PFAS SMILES."""
        canon = _canonicalize_smiles(smiles)
        if canon is None:
            return PredictResult(
                is_valid=False,
                pfas_name=pfas_name or "Unknown",
                smiles=smiles,
                canonical_smiles="",
                error="Invalid SMILES",
            )
        
        # Identify PFAS if not provided
        if not pfas_name:
            pfas_name = _identify_pfas(canon)
        
        # Look up reference data if available
        result = PredictResult(
            is_valid=True,
            pfas_name=pfas_name,
            smiles=smiles,
            canonical_smiles=canon,
            model_type=model_type,
            functional=self.model_config.get("functional", "ωB97X-D3"),
        )
        
        # Try to find reference data for this PFAS
        ref_key = pfas_name.upper()
        if ref_key in self.reference_data:
            ref = self.reference_data[ref_key]
            result.delta_e_exchange = ref.get("delta_e_exchange")
            result.delta_g_exchange = ref.get("delta_g_exchange")
            result.e_electrostatic = ref.get("e_electrostatic")
            result.e_pauli = ref.get("e_pauli")
            result.e_orbital = ref.get("e_orbital")
            result.e_dispersion = ref.get("e_dispersion")
            result.e_solvation = ref.get("e_solvation")
            result.e_preparation = ref.get("e_preparation")
            result.e_binding_total = ref.get("e_binding_total")
            result.nocv_pairs = ref.get("nocv_pairs", [])
            result.nbo_donor_acceptor = ref.get("nbo_donor_acceptor")
            result.nbo_stabilization_energy = ref.get("nbo_stabilization_energy")
        else:
            result.error = f"No reference data available for {pfas_name}. Please upload calculation results."
        
        return result


def load_predictor(artifact_dir: Union[str, Path]) -> PFASDFTPredictor:
    """Load PFAS DFT predictor from artifact directory."""
    base = Path(artifact_dir)
    art = base / "artifacts"
    if not art.exists():
        art = base
    
    # Load reference data
    ref_path = art / "reference_data.json"
    reference_data = {}
    if ref_path.exists():
        with open(ref_path, "r", encoding="utf-8") as f:
            reference_data = json.load(f)
    else:
        # Default reference data from paper (placeholder values)
        reference_data = {
            "PFOS": {
                "delta_e_exchange": -5.62,
                "delta_g_exchange": None,  # Would be calculated
                "e_electrostatic": -67.2,
                "e_pauli": 7.9,
                "e_orbital": -6.2,
                "e_dispersion": -2.7,
                "e_solvation": -45.8,
                "e_preparation": 15.5,
                "e_binding_total": -101.2,
                "nocv_pairs": [-0.76, -0.69, -0.48, -0.28, -0.23],
                "nbo_donor_acceptor": "O lone pair → σ* C-H",
                "nbo_stabilization_energy": None,
            },
            "PFOA": {
                "delta_e_exchange": -5.65,
                "delta_g_exchange": None,
                "e_electrostatic": -68.0,
                "e_pauli": 9.1,
                "e_orbital": -7.0,
                "e_dispersion": -3.0,
                "e_solvation": -50.4,
                "e_preparation": 16.0,
                "e_binding_total": -106.2,
                "nocv_pairs": [-0.78, -0.74, -0.48, -0.46, -0.38],
                "nbo_donor_acceptor": "O lone pair → σ* C-H",
                "nbo_stabilization_energy": None,
            },
            "PFHXA": {
                "delta_e_exchange": -4.13,
                "delta_g_exchange": None,
                "e_electrostatic": -72.0,
                "e_pauli": 12.2,
                "e_orbital": -7.0,
                "e_dispersion": -3.5,
                "e_solvation": -48.8,
                "e_preparation": 13.7,
                "e_binding_total": -109.2,
                "nocv_pairs": [-0.78, -0.82, -0.48, -0.46, -0.37],
                "nbo_donor_acceptor": "O lone pair → σ* C-H",
                "nbo_stabilization_energy": None,
            },
            "FHEA": {
                "delta_e_exchange": -6.99,
                "delta_g_exchange": None,
                "e_electrostatic": -75.5,
                "e_pauli": 9.8,
                "e_orbital": -9.1,
                "e_dispersion": -1.8,
                "e_solvation": -50.1,
                "e_preparation": 16.2,
                "e_binding_total": -113.7,
                "nocv_pairs": [-1.79, -1.14, -0.78, -0.67, -0.59],
                "nbo_donor_acceptor": "O lone pair → σ* C-H",
                "nbo_stabilization_energy": None,
            },
        }
    
    # Load model config
    config_path = art / "model_config.json"
    model_config = {}
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            model_config = json.load(f)
    else:
        model_config = {
            "functional": "ωB97X-D3",
            "basis_set": "def2-TZVPD",
            "solvent": "Water (SMD/C-PCM)",
            "dielectric": 72.5,
            "temperature": 310.15,
        }
    
    return PFASDFTPredictor(
        reference_data=reference_data,
        model_config=model_config,
    )


def predict_single(
    smiles: str,
    pfas_name: Optional[str] = None,
    model_type: str = "Extended",
    artifact_dir: Union[str, Path] = ".",
    predictor: Optional[PFASDFTPredictor] = None,
) -> PredictResult:
    """Predict for a single PFAS SMILES."""
    if predictor is None:
        predictor = load_predictor(artifact_dir)
    return predictor.predict(smiles, pfas_name=pfas_name, model_type=model_type)


def predict_batch(
    smiles_list: List[str],
    pfas_names: Optional[List[str]] = None,
    model_type: str = "Extended",
    artifact_dir: Union[str, Path] = ".",
    predictor: Optional[PFASDFTPredictor] = None,
) -> List[PredictResult]:
    """Predict for a list of PFAS SMILES."""
    if predictor is None:
        predictor = load_predictor(artifact_dir)
    if pfas_names is None:
        pfas_names = [None] * len(smiles_list)
    return [
        predictor.predict(smiles, pfas_name=name, model_type=model_type)
        for smiles, name in zip(smiles_list, pfas_names)
    ]
