#!/usr/bin/env bash
set -euo pipefail

RESUME_AFTER_SQM=0
if [[ "${1:-}" == "--resume-after-sqm" ]]; then
  RESUME_AFTER_SQM=1
  shift
fi
if [[ $# -ne 0 ]]; then
  echo "Usage: bash run_stage04_rct.sh [--resume-after-sqm]" >&2
  exit 2
fi

if [[ -f "${HOME}/miniforge3/etc/profile.d/conda.sh" ]]; then
  source "${HOME}/miniforge3/etc/profile.d/conda.sh"
elif command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
else
  echo "ERROR: conda initialization not found" >&2
  exit 2
fi
conda activate AmberTools25

RDKIT_PY="${RDKIT_PY:-/usr/bin/python3}"
AMBER_PY="$(command -v python)"
if [[ ! -x "$RDKIT_PY" ]]; then
  echo "ERROR: RDKit-capable Python not found at $RDKIT_PY" >&2
  exit 2
fi

STAMP="$(date +%Y%m%d_%H%M%S)"
if [[ "$RESUME_AFTER_SQM" -eq 1 ]]; then
  BACKUP="../_safety_backups/${STAMP}_stage04_rct_resume"
else
  BACKUP="../_safety_backups/${STAMP}_stage04_rct"
fi
mkdir -p "$BACKUP"



COMMON_OUTPUTS=(
  pVBTMA5_charge_summary.json pVBTMA5_charge_summary.txt
  pVBTMA5_connectivity_validation.json pVBTMA5_connectivity_validation.txt
  rct_charge_library.json pVBTMA12_rct_charges.txt
  rct_charge_transfer_validation.json
  derive_rct_charges_stdout.log derive_rct_charges_stderr.log
  pVBTMA12_gaff2_rct.mol2
  pVBTMA12_charge_summary.json pVBTMA12_charge_summary.txt
  pVBTMA12_connectivity_validation.json pVBTMA12_connectivity_validation.txt
  rct_vs_archived_whole12.json rct_vs_archived_whole12.txt
  rct12_transfer_work
)
for f in "${COMMON_OUTPUTS[@]}"; do
  if [[ -e "$f" ]]; then
    cp -a "$f" "$BACKUP/"
  fi
done

if [[ "$RESUME_AFTER_SQM" -eq 0 ]]; then

  FULL_RUN_OUTPUTS=(
    pVBTMA5_rct_model.smi pVBTMA5_rct_model_uffmin.sdf
    pVBTMA5_rct_atom_mapping.json pVBTMA12_rct_reference.sdf
    pVBTMA12_rct_atom_mapping.json pVBTMA12_accepted_to_rct_reorder.json
    rct_model_preparation.json rct_model_preparation_SHA256.json
    prepare_rct_models_stdout.log prepare_rct_models_stderr.log
    rct5_antechamber_work
  )
  for f in "${FULL_RUN_OUTPUTS[@]}"; do
    if [[ -e "$f" ]]; then
      cp -a "$f" "$BACKUP/"
    fi
  done

  "$RDKIT_PY" prepare_rct_models.py \
    > prepare_rct_models_stdout.log \
    2> prepare_rct_models_stderr.log

  rm -rf rct5_antechamber_work
  mkdir -p rct5_antechamber_work
  cp pVBTMA5_rct_model_uffmin.sdf rct5_antechamber_work/
  pushd rct5_antechamber_work >/dev/null
  antechamber \
    -i pVBTMA5_rct_model_uffmin.sdf -fi sdf \
    -o pVBTMA5_gaff2_am1bcc.mol2 -fo mol2 \
    -c bcc -nc 5 -at gaff2 -rn PVR -s 2 -pf n \
    > antechamber_stdout.log 2> antechamber_stderr.log
  popd >/dev/null
else


  REQUIRED=(
    pVBTMA5_rct_model_uffmin.sdf
    pVBTMA5_rct_atom_mapping.json
    pVBTMA12_rct_reference.sdf
    pVBTMA12_rct_atom_mapping.json
    rct5_antechamber_work/pVBTMA5_gaff2_am1bcc.mol2
  )
  for f in "${REQUIRED[@]}"; do
    if [[ ! -s "$f" ]]; then
      echo "ERROR: --resume-after-sqm requires existing nonempty file: $f" >&2
      exit 2
    fi
  done
  echo "Resume mode: preserving existing pVBTMA5 SQM/Antechamber outputs; no SQM run will be launched."
fi



"$AMBER_PY" summarize_mol2_charges.py \
  rct5_antechamber_work/pVBTMA5_gaff2_am1bcc.mol2 \
  --target-charge 5 \
  --json-out pVBTMA5_charge_summary.json \
  > pVBTMA5_charge_summary.txt

"$AMBER_PY" validate_mol2_connectivity.py \
  rct5_antechamber_work/pVBTMA5_gaff2_am1bcc.mol2 \
  pVBTMA5_rct_model_uffmin.sdf \
  --expected-charge 5 \
  --json-out pVBTMA5_connectivity_validation.json \
  > pVBTMA5_connectivity_validation.txt


"$AMBER_PY" derive_rct_charges.py \
  > derive_rct_charges_stdout.log \
  2> derive_rct_charges_stderr.log



rm -rf rct12_transfer_work
mkdir -p rct12_transfer_work
cp pVBTMA12_rct_reference.sdf pVBTMA12_rct_charges.txt rct12_transfer_work/
pushd rct12_transfer_work >/dev/null
antechamber \
  -i pVBTMA12_rct_reference.sdf -fi sdf \
  -o pVBTMA12_gaff2_rct.mol2 -fo mol2 \
  -c rc -cf pVBTMA12_rct_charges.txt \
  -nc 12 -at gaff2 -rn PVB -s 2 -pf n \
  > antechamber_stdout.log 2> antechamber_stderr.log
popd >/dev/null
cp rct12_transfer_work/pVBTMA12_gaff2_rct.mol2 ./pVBTMA12_gaff2_rct.mol2

"$AMBER_PY" summarize_mol2_charges.py \
  pVBTMA12_gaff2_rct.mol2 \
  --target-charge 12 \
  --json-out pVBTMA12_charge_summary.json \
  > pVBTMA12_charge_summary.txt

"$AMBER_PY" validate_mol2_connectivity.py \
  pVBTMA12_gaff2_rct.mol2 \
  pVBTMA12_rct_reference.sdf \
  --expected-charge 12 \
  --json-out pVBTMA12_connectivity_validation.json \
  > pVBTMA12_connectivity_validation.txt




DIRECT_WHOLE12="whole12_direct_am1bcc_reference/pVBTMA12_gaff2_am1bcc.mol2"
if [[ -f "$DIRECT_WHOLE12" ]]; then
  "$AMBER_PY" compare_rct_to_archived_whole12.py \
    --archived-mol2 "$DIRECT_WHOLE12" \
    --json-out rct_vs_direct_whole12.json \
    > rct_vs_direct_whole12.txt
else
  echo "Self-contained direct whole-12-mer reference not staged; diagnostic comparison skipped. Run stage_whole12_reference.py, then run compare_rct_to_archived_whole12.py directly." \
    > rct_vs_direct_whole12.txt
fi

echo "Stage 04 RCT workflow completed. Review all Stage-04 validation outputs before Stage 05."
