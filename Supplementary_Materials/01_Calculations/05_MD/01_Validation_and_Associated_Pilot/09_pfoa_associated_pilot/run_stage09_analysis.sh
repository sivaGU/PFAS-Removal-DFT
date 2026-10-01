#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

VMD_BIN="${VMD_BIN:-vmd}"
if [[ ! -x "$VMD_BIN" ]]; then VMD_BIN="$(command -v "$VMD_BIN" 2>/dev/null || true)"; fi
[[ -n "$VMD_BIN" && -x "$VMD_BIN" ]] || { echo 'ERROR: VMD executable not found; set VMD_BIN=/path/to/vmd or a command on PATH' >&2; exit 2; }
CPPTRAJ_BIN="${CPPTRAJ_BIN:-cpptraj}"
if [[ ! -x "$CPPTRAJ_BIN" ]]; then CPPTRAJ_BIN="$(command -v "$CPPTRAJ_BIN" 2>/dev/null || true)"; fi
if [[ -z "$CPPTRAJ_BIN" ]]; then
  if command -v cpptraj >/dev/null 2>&1; then CPPTRAJ_BIN="$(command -v cpptraj)"
  else
    if [[ -f "${HOME}/miniforge3/etc/profile.d/conda.sh" ]]; then source "${HOME}/miniforge3/etc/profile.d/conda.sh"
    elif command -v conda >/dev/null 2>&1; then eval "$(conda shell.bash hook)"
    else echo 'ERROR: cpptraj not found and conda initialization unavailable' >&2; exit 2; fi
    conda activate AmberTools25
    CPPTRAJ_BIN="$(command -v cpptraj || true)"
  fi
fi
[[ -n "$CPPTRAJ_BIN" && -x "$CPPTRAJ_BIN" ]] || { echo 'ERROR: cpptraj unavailable' >&2; exit 2; }

python3 validate_namd_log.py 00_minimize/minimize.log --minimize
python3 validate_namd_log.py 01_relax_nvt_100ps/relax100ps.log --expected-last-step 50000
python3 validate_namd_log.py 02_equil_npt_500ps/equil500ps.log --expected-last-step 250000
python3 validate_namd_log.py 03_acceptance_npt_1ns/acceptance1ns.log --expected-last-step 500000
[[ -s 03_acceptance_npt_1ns/acceptance1ns.dcd && -s 03_acceptance_npt_1ns/acceptance1ns.xst ]] || { echo 'ERROR: acceptance DCD/XST missing' >&2; exit 2; }

if [[ -e analysis || -e stage09_completion.txt || -e stage09_analysis.generated.cpptraj ]]; then
  stamp="$(date +%Y%m%d_%H%M%S)"; backup="_safety_backups/${stamp}_stage09_analysis_rerun"; mkdir -p "$backup"
  for x in analysis stage09_completion.txt stage09_analysis.generated.cpptraj; do [[ -e "$x" ]] && cp -a "$x" "$backup/"; done
fi
rm -rf analysis; mkdir -p analysis; rm -f stage09_completion.txt stage09_analysis.generated.cpptraj
python3 prepare_stage09_analysis.py
"$CPPTRAJ_BIN" -i stage09_analysis.generated.cpptraj > analysis/cpptraj.log 2>&1
if grep -Eiq 'Box is too skewed|Imaging disabled|Error:' analysis/cpptraj.log; then echo 'ERROR: prohibited CPPTRAJ/PBC warning' >&2; exit 4; fi
"$VMD_BIN" -dispdev text -e validate_vmd_pbc.tcl > analysis/vmd_pbc.log 2>&1
python3 validate_vmd_pbc.py
python3 summarize_stage09.py
cat > stage09_completion.txt <<EOF2
TECHNICAL STATUS: PASS
Stage 09 prepared-PFOA 1 ns acceptance trajectory and PBC validation completed.
Association persistence is reported as an observation and is not a technical acceptance gate.
The trajectory begins from a deliberately preassociated PFOA state and does not establish spontaneous binding or chloride displacement.
Stage 10 must not be started automatically; review analysis/stage09_summary.txt first.
EOF2
