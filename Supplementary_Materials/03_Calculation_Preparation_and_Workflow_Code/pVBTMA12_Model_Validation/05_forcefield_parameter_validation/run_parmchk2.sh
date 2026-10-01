#!/usr/bin/env bash
set -euo pipefail
if [[ -f "${HOME}/miniforge3/etc/profile.d/conda.sh" ]]; then
  source "${HOME}/miniforge3/etc/profile.d/conda.sh"
elif command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
else
  echo "ERROR: conda initialization not found" >&2
  exit 2
fi
conda activate AmberTools25

INPUT="../04_charge_parameterization/pVBTMA12_gaff2_rct.mol2"
[[ -s "$INPUT" ]] || { echo "ERROR: Stage-04 RCT MOL2 not found: $INPUT" >&2; exit 2; }

STAMP="$(date +%Y%m%d_%H%M%S)"
BACKUP="../_safety_backups/${STAMP}_stage05_forcefield_parameter_validation"
mkdir -p "$BACKUP"
for f in pVBTMA12.frcmod parmchk2_stdout.log parmchk2_stderr.log frcmod_validation.txt; do
  [[ -e "$f" ]] && cp -a "$f" "$BACKUP/"
done

parmchk2 \
  -i "$INPUT" \
  -f mol2 -o pVBTMA12.frcmod -s gaff2 \
  > parmchk2_stdout.log 2> parmchk2_stderr.log
python validate_frcmod.py pVBTMA12.frcmod > frcmod_validation.txt
