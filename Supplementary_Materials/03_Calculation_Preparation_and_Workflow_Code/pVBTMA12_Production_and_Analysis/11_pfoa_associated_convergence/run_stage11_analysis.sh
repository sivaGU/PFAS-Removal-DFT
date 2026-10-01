
#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"; S10="../10_pfoa_associated_production"
[[ -f "$S10/stage10_completion.txt" ]] || { echo 'ERROR: Stage 10 completion marker missing' >&2; exit 2; }
CPPTRAJ_BIN="${CPPTRAJ_BIN:-cpptraj}"; [[ -x "$CPPTRAJ_BIN" ]] || CPPTRAJ_BIN="$(command -v "$CPPTRAJ_BIN" 2>/dev/null || true)"
if [[ -z "$CPPTRAJ_BIN" ]]; then
  if [[ -f "${HOME}/miniforge3/etc/profile.d/conda.sh" ]]; then source "${HOME}/miniforge3/etc/profile.d/conda.sh"; conda activate AmberTools25; CPPTRAJ_BIN="$(command -v cpptraj || true)"; fi
fi
[[ -n "$CPPTRAJ_BIN" && -x "$CPPTRAJ_BIN" ]] || { echo 'ERROR: cpptraj unavailable; set CPPTRAJ_BIN' >&2; exit 2; }
VMD_BIN="${VMD_BIN:-vmd}"; [[ -x "$VMD_BIN" ]] || VMD_BIN="$(command -v "$VMD_BIN" 2>/dev/null || true)"; [[ -n "$VMD_BIN" && -x "$VMD_BIN" ]] || { echo 'ERROR: VMD unavailable; set VMD_BIN' >&2; exit 2; }
python3 - <<'PY'
import numpy, matplotlib
print('Python analysis dependencies: PASS')
PY
if [[ -e analysis || -e stage11_completion.txt ]]; then stamp="$(date +%Y%m%d_%H%M%S)"; backup="_safety_backups/${stamp}_stage11_analysis"; mkdir -p "$backup"; [[ -e analysis ]] && cp -a analysis "$backup/"; [[ -e stage11_completion.txt ]] && cp -a stage11_completion.txt "$backup/"; fi
rm -rf analysis; rm -f stage11_completion.txt; mkdir -p analysis
python3 prepare_stage11_analysis.py
for r in 01 02 03; do "$CPPTRAJ_BIN" -i "analysis/replica_${r}.cpptraj" > "analysis/replica_${r}/cpptraj.log" 2>&1; if grep -Eiq 'Box is too skewed|Imaging disabled|Error:' "analysis/replica_${r}/cpptraj.log"; then echo "ERROR: prohibited CPPTRAJ/PBC warning in replica $r" >&2; exit 4; fi; done
"$VMD_BIN" -dispdev text -e validate_vmd_pbc.tcl > analysis/vmd_pbc.log 2>&1
grep -q 'VMD_PBC_STATUS PASS' analysis/vmd_pbc.log || { echo 'ERROR: VMD/PBCTools validation did not pass' >&2; exit 4; }
python3 validate_stage11_pbc.py
python3 summarize_convergence.py
decision="$(python3 -c 'import json; print(json.load(open("analysis/stage11_summary.json"))["convergence_decision"])')"
ns="$(python3 -c 'import json; print(json.load(open("analysis/stage11_summary.json"))["production_ns_per_replica"])')"
cat > stage11_completion.txt <<EOF
FINAL STATUS: ANALYSIS COMPLETE
PRODUCTION ANALYZED: 3 x ${ns} ns
CONVERGENCE DECISION: ${decision}
See analysis/stage11_summary.txt, analysis/convergence_diagnostics.csv,
analysis/tail_contact_hydration_by_frame.csv, and
analysis/tail_contact_hydration_blocks.png.
Scientific interpretation remains prepared associated-state persistence/rearrangement only.
EOF
cat stage11_completion.txt
