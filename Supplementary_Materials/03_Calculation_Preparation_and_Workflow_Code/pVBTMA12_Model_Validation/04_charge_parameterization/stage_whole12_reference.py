#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

DEST_NAME = "whole12_direct_am1bcc_reference"
FINAL_MOL2 = "pVBTMA12_gaff2_am1bcc.mol2"
REQUIRED = (FINAL_MOL2, "sqm.in", "sqm.out")
OPTIONAL_EXACT = (
    "antechamber_stdout.log",
    "antechamber_stderr.log",
    "ATOMTYPE.INF",
    "run_antechamber.sh",
)
OPTIONAL_GLOBS = (
    "ANTECHAMBER*",
    "*charge_summary*.json",
    "*connectivity_validation*.json",
)
SKIP_PARTS = {"_safety_backups", "backups", "backup", "__pycache__"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def candidate_previous_attempts(stage04: Path) -> list[Path]:
    roots: list[Path] = []
    validation_root = stage04.parent
    probes = [
        validation_root / "previous_attempts",
        validation_root.parent / "previous_attempts",
        validation_root.parent.parent / "previous_attempts",
    ]
    for p in probes:
        if p.is_dir() and p.resolve() not in [x.resolve() for x in roots]:
            roots.append(p)
    return roots


def filtered_matches(root: Path, pattern: str) -> list[Path]:
    found = []
    for p in root.rglob(pattern):
        if not p.is_file():
            continue
        if any(part in SKIP_PARTS for part in p.parts):
            continue
        found.append(p)
    return sorted(found)


def find_stage04_root(mol2: Path) -> Path:
    for parent in [mol2.parent, *mol2.parents]:
        if parent.name == "04_charge_parameterization":
            return parent
    return mol2.parent


def choose_named(stage_root: Path, filename: str, mol2_dir: Path) -> Path | None:
    matches = filtered_matches(stage_root, filename)
    if not matches:
        return None
    direct = [p for p in matches if p.parent == mol2_dir]
    if len(direct) == 1:
        return direct[0]
    if len(direct) > 1:
        raise SystemExit(f"Ambiguous {filename}: multiple files beside final MOL2: {direct}")

    ranked = sorted((len(p.relative_to(stage_root).parts), str(p), p) for p in matches)
    best_depth = ranked[0][0]
    best = [p for depth, _, p in ranked if depth == best_depth]
    if len(best) != 1:
        lines = "\n".join(f"  {p}" for p in best)
        raise SystemExit(f"Ambiguous {filename}; supply --source-mol2 for the intended attempt.\n{lines}")
    return best[0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--previous-attempts",
        type=Path,
        action="append",
        default=[],
        help="Directory to search. May be supplied more than once.",
    )
    ap.add_argument(
        "--source-mol2",
        type=Path,
        default=None,
        help="Explicit archived pVBTMA12_gaff2_am1bcc.mol2 if auto-discovery is ambiguous.",
    )
    ap.add_argument(
        "--destination",
        type=Path,
        default=Path(DEST_NAME),
        help=f"Destination reference directory (default: {DEST_NAME}).",
    )
    args = ap.parse_args()

    stage04 = Path.cwd().resolve()
    dest = args.destination.resolve()
    if dest.exists():
        raise SystemExit(
            f"Destination already exists: {dest}\n"
            "Preserve/back it up first and remove it only if a deliberate restage is required."
        )

    if args.source_mol2:
        mol2 = args.source_mol2.expanduser().resolve()
        if not mol2.is_file():
            raise SystemExit(f"Explicit source MOL2 not found: {mol2}")
    else:
        roots = [p.expanduser().resolve() for p in args.previous_attempts]
        if not roots:
            roots = candidate_previous_attempts(stage04)
        if not roots:
            raise SystemExit(
                "No previous_attempts directory was auto-discovered. Rerun with "
                "--previous-attempts /path/to/previous_attempts or --source-mol2 /path/to/"
                + FINAL_MOL2
            )
        mol2_matches: list[Path] = []
        for root in roots:
            if not root.is_dir():
                continue
            mol2_matches.extend(filtered_matches(root, FINAL_MOL2))

        uniq = []
        seen = set()
        for p in mol2_matches:
            r = p.resolve()
            if r not in seen:
                seen.add(r)
                uniq.append(r)
        if len(uniq) != 1:
            detail = "\n".join(f"  {p}" for p in uniq) if uniq else "  (none)"
            raise SystemExit(
                "Expected exactly one archived direct whole-12-mer MOL2. Found "
                f"{len(uniq)}:\n{detail}\n"
                "Rerun with --source-mol2 pointing to the intended file."
            )
        mol2 = uniq[0]

    source_stage = find_stage04_root(mol2)
    mol2_dir = mol2.parent
    selected: dict[str, Path] = {FINAL_MOL2: mol2}

    for name in ("sqm.in", "sqm.out"):
        p = choose_named(source_stage, name, mol2_dir)
        if p is None:
            raise SystemExit(f"Required archived reference file not found under {source_stage}: {name}")
        selected[name] = p

    for name in OPTIONAL_EXACT:
        p = choose_named(source_stage, name, mol2_dir)
        if p is not None:
            outname = "source_run_antechamber.sh" if name == "run_antechamber.sh" else name
            selected[outname] = p

    optional_extra: dict[str, Path] = {}
    for pattern in OPTIONAL_GLOBS:
        for p in filtered_matches(source_stage, pattern):
            if p in selected.values():
                continue

            key = p.name
            if key in optional_extra or key in selected:
                rel = "__".join(p.relative_to(source_stage).parts)
                key = rel
            optional_extra[key] = p

    dest.mkdir(parents=True)
    (dest / "antechamber_intermediates").mkdir()
    manifest_files = []

    for outname, src in selected.items():
        dst = dest / outname
        shutil.copy2(src, dst)
        manifest_files.append(
            {
                "destination": str(dst.relative_to(dest)),
                "source": str(src),
                "bytes": dst.stat().st_size,
                "sha256": sha256(dst),
            }
        )

    for outname, src in optional_extra.items():
        dst = dest / "antechamber_intermediates" / outname
        shutil.copy2(src, dst)
        manifest_files.append(
            {
                "destination": str(dst.relative_to(dest)),
                "source": str(src),
                "bytes": dst.stat().st_size,
                "sha256": sha256(dst),
            }
        )

    sqm_text = (dest / "sqm.out").read_text(errors="replace")
    sqm_status = {
        "geometry_converged_string_present": "geometry converged" in sqm_text.lower(),
        "calculation_completed_string_present": "calculation completed" in sqm_text.lower(),
    }
    manifest = {
        "purpose": (
            "Self-contained diagnostic reference for validating the RCT charge-transfer "
            "approximation against the previously completed direct whole-pVBTMA12 AM1-BCC calculation. "
            "These files are not inputs to RCT charge derivation."
        ),
        "source_stage04_directory": str(source_stage),
        "files": sorted(manifest_files, key=lambda x: x["destination"]),
        "sqm_status": sqm_status,
    }
    (dest / "reference_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    (dest / "README.md").write_text(
        "# Direct whole-pVBTMA12 AM1-BCC reference\n\n"
        "This directory is a copied, read-only working reference from the archived pre-RCT Stage 04. "
        "It makes the current `02_pVBTMA12_validation` branch self-contained for a diagnostic comparison "
        "between the RCT-transferred charges and a direct whole-12-mer AM1-BCC calculation.\n\n"
        "The reference is **not** used to derive the RCT charge libraries and is not an independent gold "
        "standard for AM1-BCC accuracy. It validates the reduced-oligomer charge-transfer approximation.\n\n"
        "Required reference artifacts are the direct whole-chain `pVBTMA12_gaff2_am1bcc.mol2`, `sqm.in`, "
        "and `sqm.out`. Additional Antechamber intermediates/logs are retained when available. "
        "`reference_manifest.json` records source paths, sizes, and SHA256 hashes.\n"
    )

    print(f"Staged direct whole-12-mer reference in: {dest}")
    print(f"Source Stage 04: {source_stage}")
    print(f"Copied {len(manifest_files)} reference artifacts.")
    print(f"SQM geometry-converged marker: {sqm_status['geometry_converged_string_present']}")
    print(f"SQM calculation-completed marker: {sqm_status['calculation_completed_string_present']}")


if __name__ == "__main__":
    main()
