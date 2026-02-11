"""
PFAS Removal DFT CLI. Predict from command line.
Usage:
  python -m src.pfasdft.cli --smiles "C(=O)([O-])C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)" --pfas-name PFOA --output out.csv
  python -m src.pfasdft.cli --input example_inputs.csv --output out.csv
"""
import argparse
import csv
import sys
from pathlib import Path

from .predict import load_predictor, predict_batch, predict_single


def main():
    parser = argparse.ArgumentParser(description="PFAS Removal DFT binding affinity predictor")
    parser.add_argument("--smiles", nargs="+", help="SMILES strings to predict")
    parser.add_argument("--pfas-name", nargs="+", help="PFAS names (PFOS, PFOA, PFHxA, FHEA) corresponding to SMILES")
    parser.add_argument("--input", help="Input CSV file (must have SMILES column)")
    parser.add_argument("--output", help="Output CSV file")
    parser.add_argument("--artifact-dir", default=".", help="Path to artifact directory")
    parser.add_argument("--model-type", default="Extended", choices=["BTMA", "Extended"], help="Model type")
    args = parser.parse_args()

    artifact_dir = Path(args.artifact_dir).resolve()
    if not (artifact_dir / "artifacts").exists() and not (artifact_dir / "reference_data.json").exists():
        art_check = artifact_dir / "artifacts"
        if not art_check.exists():
            sys.stderr.write(f"Warning: artifacts not found in {artifact_dir}\n")
            sys.stderr.write("Using default reference data from paper.\n")

    predictor = load_predictor(artifact_dir)

    if args.smiles:
        pfas_names = args.pfas_name if args.pfas_name else [None] * len(args.smiles)
        if len(pfas_names) != len(args.smiles):
            sys.stderr.write("Error: number of PFAS names must match number of SMILES\n")
            sys.exit(1)
        
        results = []
        for smiles, name in zip(args.smiles, pfas_names):
            results.append(predict_single(smiles, pfas_name=name, model_type=args.model_type, predictor=predictor))
        
        rows = []
        for r in results:
            rows.append({
                "pfas_name": r.pfas_name,
                "smiles": r.smiles,
                "canonical_smiles": r.canonical_smiles,
                "delta_e_exchange": f"{r.delta_e_exchange:.4f}" if r.delta_e_exchange is not None else "",
                "delta_g_exchange": f"{r.delta_g_exchange:.4f}" if r.delta_g_exchange is not None else "",
                "e_electrostatic": f"{r.e_electrostatic:.4f}" if r.e_electrostatic is not None else "",
                "e_pauli": f"{r.e_pauli:.4f}" if r.e_pauli is not None else "",
                "e_orbital": f"{r.e_orbital:.4f}" if r.e_orbital is not None else "",
                "e_dispersion": f"{r.e_dispersion:.4f}" if r.e_dispersion is not None else "",
                "e_solvation": f"{r.e_solvation:.4f}" if r.e_solvation is not None else "",
                "e_preparation": f"{r.e_preparation:.4f}" if r.e_preparation is not None else "",
                "e_binding_total": f"{r.e_binding_total:.4f}" if r.e_binding_total is not None else "",
                "model_type": r.model_type or "",
                "functional": r.functional or "",
                "error": r.error,
            })
    elif args.input:
        inp = Path(args.input)
        if not inp.exists():
            sys.stderr.write(f"Error: input file not found: {inp}\n")
            sys.exit(1)
        import pandas as pd
        df = pd.read_csv(inp)
        col = next((c for c in df.columns if c.lower() in ("smiles", "canonical_smiles", "smi") or c == "SMILES"), None)
        if col is None:
            sys.stderr.write(f"Error: CSV must have SMILES column. Found: {list(df.columns)}\n")
            sys.exit(1)
        
        pfas_col = next((c for c in df.columns if c.lower() in ("pfas_name", "pfas", "name")), None)
        pfas_names = df[pfas_col].tolist() if pfas_col else [None] * len(df)
        
        smiles_list = df[col].astype(str).tolist()
        results = predict_batch(smiles_list, pfas_names=pfas_names, model_type=args.model_type, predictor=predictor)
        
        rows = []
        for r in results:
            rows.append({
                "pfas_name": r.pfas_name,
                "smiles": r.smiles,
                "canonical_smiles": r.canonical_smiles,
                "delta_e_exchange": f"{r.delta_e_exchange:.4f}" if r.delta_e_exchange is not None else "",
                "delta_g_exchange": f"{r.delta_g_exchange:.4f}" if r.delta_g_exchange is not None else "",
                "e_electrostatic": f"{r.e_electrostatic:.4f}" if r.e_electrostatic is not None else "",
                "e_pauli": f"{r.e_pauli:.4f}" if r.e_pauli is not None else "",
                "e_orbital": f"{r.e_orbital:.4f}" if r.e_orbital is not None else "",
                "e_dispersion": f"{r.e_dispersion:.4f}" if r.e_dispersion is not None else "",
                "e_solvation": f"{r.e_solvation:.4f}" if r.e_solvation is not None else "",
                "e_preparation": f"{r.e_preparation:.4f}" if r.e_preparation is not None else "",
                "e_binding_total": f"{r.e_binding_total:.4f}" if r.e_binding_total is not None else "",
                "model_type": r.model_type or "",
                "functional": r.functional or "",
                "error": r.error,
            })
    else:
        parser.print_help()
        sys.exit(0)

    out_path = args.output
    if out_path:
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"Wrote {len(rows)} rows to {out_path}")
    else:
        import csv as csvmod
        writer = csvmod.DictWriter(sys.stdout, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
