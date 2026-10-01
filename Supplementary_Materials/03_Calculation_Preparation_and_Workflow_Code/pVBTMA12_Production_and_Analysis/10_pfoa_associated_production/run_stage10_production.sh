
#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
python3 validate_stage10_package.py
NAMD_BIN="${NAMD_BIN:-namd3}"
if [[ ! -x "$NAMD_BIN" ]]; then NAMD_BIN="$(command -v "$NAMD_BIN" 2>/dev/null || true)"; fi
[[ -n "$NAMD_BIN" && -x "$NAMD_BIN" ]] || { echo 'ERROR: NAMD3 not found; set NAMD_BIN' >&2; exit 2; }
read -r -a namd_args <<< "${NAMD_ARGS:-}"
run_segment() {
  local conf="$1" prefix="$2" expected="$3"
  local log="${prefix}.log"
  mkdir -p "$(dirname "$prefix")"
  if [[ -s "$log" ]] && python3 validate_namd_log.py "$log" --expected-last-step "$expected" >/dev/null 2>&1; then
    for ext in coor vel xsc dcd xst; do [[ -s "${prefix}.${ext}" ]] || { echo "ERROR: validated log but missing ${prefix}.${ext}" >&2; exit 3; }; done
    echo "SKIP complete: $prefix"; return
  fi
  if find "$(dirname "$prefix")" -mindepth 1 -maxdepth 1 -type f -print -quit | grep -q .; then
    echo "ERROR: partial output exists for $prefix; preserve or move it before rerun" >&2; exit 3
  fi
  echo "RUN: $conf"
  "$NAMD_BIN" "${namd_args[@]}" "$conf" > "$log" 2>&1
  python3 validate_namd_log.py "$log" --expected-last-step "$expected"
  for ext in coor vel xsc dcd xst; do [[ -s "${prefix}.${ext}" ]] || { echo "ERROR: missing ${prefix}.${ext}" >&2; exit 3; }; done
}
for r in 01 02 03; do
  run_segment "replica_${r}/00_equil_npt_500ps.conf" "replica_${r}/00_equil_npt_500ps/equil" 250000
  for c in 01 02 03 04 05; do
    run_segment "replica_${r}/${c}_prod_1ns.conf" "replica_${r}/${c}_prod_1ns/prod_${c}" "$((250000 + 10#$c * 500000))"
  done
done
bash validate_stage10_outputs.sh --target-ns 5
