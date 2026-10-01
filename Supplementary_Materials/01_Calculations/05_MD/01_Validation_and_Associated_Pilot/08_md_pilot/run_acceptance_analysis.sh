#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"


VMD_BIN="${VMD_BIN:-}"
if [[ -z "$VMD_BIN" ]]; then
  if command -v vmd >/dev/null 2>&1; then
    VMD_BIN="$(command -v vmd)"
  fi
fi
[[ -n "$VMD_BIN" && -x "$VMD_BIN" ]] || { echo "ERROR: VMD executable not found. Set VMD_BIN=/path/to/vmd" >&2; exit 2; }

if [[ -f "${HOME}/miniforge3/etc/profile.d/conda.sh" ]]; then
  source "${HOME}/miniforge3/etc/profile.d/conda.sh"
elif command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
else
  echo "ERROR: conda initialization not found" >&2
  exit 2
fi
conda activate AmberTools25
command -v cpptraj >/dev/null 2>&1 || { echo "ERROR: cpptraj not found in AmberTools25" >&2; exit 2; }


python validate_namd_log.py 00_minimize/minimize.log --minimize
python validate_namd_log.py 01_heat_100K/heat100.log --expected-last-step 12500
python validate_namd_log.py 02_heat_200K/heat200.log --expected-last-step 12500
python validate_namd_log.py 03_heat_310K/heat310.log --expected-last-step 25000
python validate_namd_log.py 04_equil_npt_500ps/equil500ps.log --expected-last-step 250000
python validate_namd_log.py 05_acceptance_npt_1ns/acceptance1ns.log --expected-last-step 500000

[[ -s repeat_groups.json ]] || python prepare_stage08.py > prepare_stage08_report.txt
[[ -s 05_acceptance_npt_1ns/acceptance1ns.dcd ]] || { echo "ERROR: acceptance DCD missing" >&2; exit 2; }
[[ -s 05_acceptance_npt_1ns/acceptance1ns.xst ]] || { echo "ERROR: acceptance XST missing" >&2; exit 2; }

need_backup=0
for x in analysis acceptance_analysis.generated.cpptraj stage08_completion.txt; do
  [[ -e "$x" ]] && need_backup=1
done
if [[ "$need_backup" -eq 1 ]]; then
  stamp="$(date +%Y%m%d_%H%M%S)"
  backup="../_safety_backups/${stamp}_stage08_analysis_rerun/08_md_pilot"
  mkdir -p "$backup"
  for x in analysis acceptance_analysis.generated.cpptraj stage08_completion.txt; do
    if [[ -e "$x" ]]; then
      mkdir -p "$backup/$(dirname "$x")"
      cp -a "$x" "$backup/$x"
    fi
  done
  echo "Existing Stage 08 analysis artifacts backed up under $backup"
fi

rm -rf analysis
mkdir -p analysis
rm -f acceptance_analysis.generated.cpptraj stage08_completion.txt

python prepare_acceptance_analysis.py
cpptraj -i acceptance_analysis.generated.cpptraj > analysis/cpptraj_acceptance.log 2>&1
python validate_acceptance_analysis.py | tee analysis/box_validation_stdout.log

"$VMD_BIN" -dispdev text -e validate_vmd_pbc.tcl > analysis/vmd_pbc.log 2>&1
python validate_vmd_pbc.py | tee analysis/vmd_pbc_validation_stdout.log

python summarize_acceptance.py | tee analysis/acceptance_summary_stdout.log

cat > stage08_completion.txt <<EOT
Stage 08 dynamics completed through the 1.0 ns chloride-form acceptance trajectory and analysis-only validation completed successfully.
CPPTRAJ PBC-sensitive actions were evaluated before RMS fitting, preventing fitted coordinate/cell rotation from contaminating imaging or box analysis.
VMD/PBCTools independently read the NAMD DCD periodic cell and passed the XST-matched cell cross-check and polymer reconstruction smoke test.
This trajectory is not a production replica.
Review analysis/acceptance_summary.txt, analysis/box_validation.txt, analysis/vmd_pbc_validation.txt, and the trajectory before final pVBTMA12 model freeze.
Completed: $(date -Is)
EOT
cat stage08_completion.txt
