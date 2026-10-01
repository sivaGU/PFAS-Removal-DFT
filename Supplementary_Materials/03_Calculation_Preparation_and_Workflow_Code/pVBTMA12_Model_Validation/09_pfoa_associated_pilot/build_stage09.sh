#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p build _safety_backups
STAGE08="../08_md_pilot"
STAGE07="../07_solvation_ion_validation"


VMD_BIN="${VMD_BIN:-vmd}"
if [[ ! -x "$VMD_BIN" ]]; then VMD_BIN="$(command -v "$VMD_BIN" 2>/dev/null || true)"; fi
[[ -n "$VMD_BIN" && -x "$VMD_BIN" ]] || { echo 'ERROR: VMD executable not found; set VMD_BIN=/path/to/vmd or a command on PATH' >&2; exit 2; }


need_amber=0
command -v "${TLEAP_BIN:-tleap}" >/dev/null 2>&1 || need_amber=1
python3 -c 'import parmed' >/dev/null 2>&1 || need_amber=1
if [[ "$need_amber" -eq 1 ]]; then
  if [[ -f "${HOME}/miniforge3/etc/profile.d/conda.sh" ]]; then
    source "${HOME}/miniforge3/etc/profile.d/conda.sh"
  elif command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
  else
    echo 'ERROR: AmberTools dependencies missing and conda initialization unavailable' >&2; exit 2
  fi
  conda activate AmberTools25
fi
TLEAP_BIN="${TLEAP_BIN:-tleap}"
if [[ ! -x "$TLEAP_BIN" ]]; then TLEAP_BIN="$(command -v "$TLEAP_BIN" 2>/dev/null || true)"; fi
[[ -n "$TLEAP_BIN" && -x "$TLEAP_BIN" ]] || { echo 'ERROR: tleap not found; activate AmberTools25 or set TLEAP_BIN' >&2; exit 2; }
python3 -c 'import parmed, numpy' >/dev/null 2>&1 || { echo 'ERROR: Python ParmEd/numpy unavailable after AmberTools setup' >&2; exit 2; }

for f in "$STAGE08/stage08_completion.txt" "$STAGE08/05_acceptance_npt_1ns/acceptance1ns.restart.coor" "$STAGE08/05_acceptance_npt_1ns/acceptance1ns.restart.xsc" "$STAGE07/pVBTMA12_12Cl_015MNaCl_pilot.prmtop"; do
  [[ -s "$f" ]] || { echo "ERROR: missing required file: $f" >&2; exit 2; }
done
grep -q 'FINAL STATUS: ACCEPTED' "$STAGE08/stage08_completion.txt" || { echo 'ERROR: Stage 08 is not explicitly ACCEPTED' >&2; exit 2; }
[[ -s INPUT_SHA256SUMS.txt ]] || { echo 'ERROR: INPUT_SHA256SUMS.txt missing' >&2; exit 2; }
[[ -s DEPENDENCY_SHA256SUMS.txt ]] || { echo 'ERROR: DEPENDENCY_SHA256SUMS.txt missing' >&2; exit 2; }
sha256sum -c INPUT_SHA256SUMS.txt
sha256sum -c DEPENDENCY_SHA256SUMS.txt
python3 validate_stage09_configs.py

if compgen -G 'build/*' >/dev/null; then
  ts=$(date +%Y%m%d_%H%M%S); backup="_safety_backups/${ts}_stage09_build"; mkdir -p "$backup"; cp -a build "$backup/"
fi
rm -rf build && mkdir build
"$VMD_BIN" -dispdev text -e export_stage08_endpoint.tcl -args "$STAGE07/pVBTMA12_12Cl_015MNaCl_pilot.prmtop" "$STAGE08/05_acceptance_npt_1ns/acceptance1ns.restart.coor" build/stage08_endpoint.pdb > build/vmd_export.log 2>&1
[[ -s build/stage08_endpoint.pdb ]] || { echo 'ERROR: VMD endpoint export did not create a PDB' >&2; tail -100 build/vmd_export.log >&2; exit 3; }
python3 prepare_stage09.py | tee build/prepare_stage09.stdout
"$TLEAP_BIN" -f build/build_stage09.generated.leap > build/tleap.log 2>&1
if grep -Eiq 'Errors[[:space:]]*=[[:space:]]*[1-9][0-9]*|FATAL|Could not find|does not have a type' build/tleap.log; then
  echo 'ERROR: LEaP reported an error; inspect build/tleap.log' >&2; tail -100 build/tleap.log >&2; exit 3
fi
[[ -s build/pVBTMA12_PFOA_assoc.prmtop && -s build/pVBTMA12_PFOA_assoc.inpcrd ]] || { echo 'ERROR: LEaP did not produce topology/coordinates' >&2; tail -100 build/tleap.log >&2; exit 3; }
python3 validate_stage09_build.py | tee build/build_validation.stdout
sha256sum inputs/pfoa/* inputs/polymer/* build/pVBTMA12_PFOA_assoc.prmtop build/pVBTMA12_PFOA_assoc.inpcrd > build/SHA256SUMS.txt
cat > build/stage09_build_completion.txt <<EOF2
STATUS: PASS
Prepared PFOA-associated pVBTMA12 endpoint from the ACCEPTED Stage 08 final coordinates/cell.
PFOA was deliberately preassociated; no spontaneous binding or dynamic chloride displacement is claimed.
One remote chloride was removed only for stoichiometric charge neutrality.
See construction_report.json, build_validation.json, tleap.log, and SHA256SUMS.txt.
EOF2
