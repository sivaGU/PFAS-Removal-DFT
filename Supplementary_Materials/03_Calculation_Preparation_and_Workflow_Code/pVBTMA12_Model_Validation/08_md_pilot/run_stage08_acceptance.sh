#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"


NAMD_BIN="${NAMD_BIN:-}"
if [[ -z "$NAMD_BIN" ]]; then
  for c in namd3 namd3_cuda; do
    if command -v "$c" >/dev/null 2>&1; then NAMD_BIN="$(command -v "$c")"; break; fi
  done
fi
[[ -n "$NAMD_BIN" && -x "$NAMD_BIN" ]] || { echo "ERROR: NAMD3 executable not found. Set NAMD_BIN=/path/to/namd3" >&2; exit 2; }
NAMD_ARGS_STRING="${NAMD_ARGS:-}"
read -r -a NAMD_EXTRA <<< "$NAMD_ARGS_STRING"

echo "NAMD_BIN=$NAMD_BIN"
echo "NAMD_ARGS=$NAMD_ARGS_STRING"
echo "NAMD executable metadata:"
ls -l "$NAMD_BIN"
if command -v sha256sum >/dev/null 2>&1; then
  sha256sum "$NAMD_BIN" || true
fi





python validate_stage08_configs.py


outputs=(00_minimize 01_heat_100K 02_heat_200K 03_heat_310K 04_equil_npt_500ps 05_acceptance_npt_1ns analysis stage08_system.generated.conf repeat_groups.json prepare_stage08_report.txt acceptance_analysis.generated.cpptraj stage08_completion.txt)
need_backup=0
for x in "${outputs[@]}"; do [[ -e "$x" ]] && need_backup=1; done
if [[ "$need_backup" -eq 1 ]]; then
  stamp="$(date +%Y%m%d_%H%M%S)"
  backup="../_safety_backups/${stamp}_stage08_acceptance_rerun/08_md_pilot"
  mkdir -p "$backup"
  for x in "${outputs[@]}"; do
    if [[ -e "$x" ]]; then
      mkdir -p "$backup/$(dirname "$x")"
      cp -a "$x" "$backup/$x"
    fi
  done
  echo "Existing Stage 08 dynamic outputs backed up under $backup"
fi

rm -rf 00_minimize 01_heat_100K 02_heat_200K 03_heat_310K 04_equil_npt_500ps 05_acceptance_npt_1ns analysis
mkdir -p 00_minimize 01_heat_100K 02_heat_200K 03_heat_310K 04_equil_npt_500ps 05_acceptance_npt_1ns analysis

python prepare_stage08.py | tee prepare_stage08_report.txt

run_stage () {
  local conf="$1" log="$2" steps="$3" mode="${4:-run}"
  echo "===== START $conf $(date -Is) ====="
  "$NAMD_BIN" "${NAMD_EXTRA[@]}" "$conf" > "$log" 2>&1
  local rc=$?
  echo "NAMD exit status: $rc"
  [[ $rc -eq 0 ]] || { tail -100 "$log" >&2; exit "$rc"; }
  if [[ "$mode" == "min" ]]; then
    python validate_namd_log.py "$log" --minimize
  else
    python validate_namd_log.py "$log" --expected-last-step "$steps"
  fi
  echo "===== END $conf $(date -Is) ====="
}

run_stage 00_minimize.conf              00_minimize/minimize.log                    10000 min
run_stage 01_heat_100K.conf             01_heat_100K/heat100.log                    12500
run_stage 02_heat_200K.conf             02_heat_200K/heat200.log                    12500
run_stage 03_heat_310K.conf             03_heat_310K/heat310.log                    25000
run_stage 04_equil_npt_500ps.conf       04_equil_npt_500ps/equil500ps.log          250000
run_stage 05_acceptance_npt_1ns.conf    05_acceptance_npt_1ns/acceptance1ns.log    500000



bash run_acceptance_analysis.sh
