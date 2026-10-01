#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
[[ -s build/stage09_build_completion.txt ]] || { echo 'ERROR: run build_stage09.sh successfully first.' >&2; exit 2; }
grep -q '^STATUS: PASS' build/stage09_build_completion.txt || { echo 'ERROR: Stage 09 build is not PASS' >&2; exit 2; }
python3 validate_stage09_configs.py
NAMD_BIN="${NAMD_BIN:-namd3}"
if [[ ! -x "$NAMD_BIN" ]]; then NAMD_BIN="$(command -v "$NAMD_BIN" 2>/dev/null || true)"; fi
if [[ -z "$NAMD_BIN" ]]; then
  for c in namd3 namd3_cuda; do
    if command -v "$c" >/dev/null 2>&1; then NAMD_BIN="$(command -v "$c")"; break; fi
  done
fi
[[ -n "$NAMD_BIN" && -x "$NAMD_BIN" ]] || { echo 'ERROR: NAMD3 executable not found; set NAMD_BIN=/path/to/namd3 or a command on PATH' >&2; exit 2; }
NAMD_ARGS_STRING="${NAMD_ARGS:-}"
read -r -a NAMD_EXTRA <<< "$NAMD_ARGS_STRING"
echo "NAMD_BIN=$NAMD_BIN"
echo "NAMD_ARGS=$NAMD_ARGS_STRING"
outputs=(00_minimize 01_relax_nvt_100ps 02_equil_npt_500ps 03_acceptance_npt_1ns analysis stage09_completion.txt stage09_analysis.generated.cpptraj)
need_backup=0
for x in "${outputs[@]}"; do [[ -e "$x" ]] && need_backup=1; done
if [[ "$need_backup" -eq 1 ]]; then
  stamp="$(date +%Y%m%d_%H%M%S)"; backup="_safety_backups/${stamp}_stage09_acceptance_rerun"; mkdir -p "$backup"
  for x in "${outputs[@]}"; do [[ -e "$x" ]] && cp -a "$x" "$backup/"; done
  echo "Existing Stage 09 dynamic artifacts backed up under $backup"
fi
rm -rf 00_minimize 01_relax_nvt_100ps 02_equil_npt_500ps 03_acceptance_npt_1ns analysis
rm -f stage09_completion.txt stage09_analysis.generated.cpptraj
mkdir -p 00_minimize 01_relax_nvt_100ps 02_equil_npt_500ps 03_acceptance_npt_1ns analysis
run_stage(){
  local conf="$1" log="$2" steps="$3" mode="${4:-run}"
  echo "===== START $conf $(date -Is) ====="
  set +e
  "$NAMD_BIN" "${NAMD_EXTRA[@]}" "$conf" > "$log" 2>&1
  local rc=$?
  set -e
  if [[ $rc -ne 0 ]]; then tail -100 "$log" >&2; exit "$rc"; fi
  if [[ "$mode" == min ]]; then python3 validate_namd_log.py "$log" --minimize
  else python3 validate_namd_log.py "$log" --expected-last-step "$steps"; fi
  echo "===== END $conf $(date -Is) ====="
}
run_stage 00_minimize.conf 00_minimize/minimize.log 10000 min
run_stage 01_relax_nvt_100ps.conf 01_relax_nvt_100ps/relax100ps.log 50000
run_stage 02_equil_npt_500ps.conf 02_equil_npt_500ps/equil500ps.log 250000
run_stage 03_acceptance_npt_1ns.conf 03_acceptance_npt_1ns/acceptance1ns.log 500000
bash run_stage09_analysis.sh
