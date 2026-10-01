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

STAMP="$(date +%Y%m%d_%H%M%S)"
BACKUP="../_safety_backups/${STAMP}_stage07_solvation_ion_validation/07_solvation_ion_validation"
mkdir -p "$BACKUP"
for f in \
  pVBTMA12_solvated_noions.prmtop pVBTMA12_solvated_noions.inpcrd pVBTMA12_solvated_noions.pdb \
  pVBTMA12_12Cl_015MNaCl_pilot.prmtop pVBTMA12_12Cl_015MNaCl_pilot.inpcrd pVBTMA12_12Cl_015MNaCl_pilot.pdb \
  salt_setup.json build_chloride_pilot_salted.generated.leap salt_setup_stdout.log \
  leap.log \
  tleap_box_stdout.log tleap_box_stderr.log leap_box.log \
  tleap_salt_stdout.log tleap_salt_stderr.log leap_salt.log \
  solvation_validation.txt stage07_completion.txt; do
  [[ -e "$f" ]] && cp -a "$f" "$BACKUP/$f"
done

rm -f leap.log

tleap -f build_chloride_pilot.leap > tleap_box_stdout.log 2> tleap_box_stderr.log
[[ -f leap.log ]] && cp -a leap.log leap_box.log

python prepare_salt_setup.py --target-molar 0.150 > salt_setup_stdout.log

rm -f leap.log
tleap -f build_chloride_pilot_salted.generated.leap > tleap_salt_stdout.log 2> tleap_salt_stderr.log
[[ -f leap.log ]] && cp -a leap.log leap_salt.log

python validate_solvated_system.py > solvation_validation.txt

{
  echo "Stage 07 completed successfully."
  echo "Target external NaCl: 0.150 M"
  python - <<'PY'
import json
j=json.load(open('salt_setup.json'))
print(f"Added NaCl pairs: {j['added_nacl_pairs']}")
print(f"Neutralizing Cl-: {j['neutralizing_chloride_count']}")
print(f"Nominal achieved external NaCl: {j['achieved_nominal_added_nacl_molar']:.8f} M")
print(f"Initial box volume: {j['box_volume_angstrom3']:.6f} A^3")
PY
} > stage07_completion.txt
cat stage07_completion.txt
