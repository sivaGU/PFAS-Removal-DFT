
#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
[[ -f replica_01/06_prod_1ns.conf ]] || { echo 'ERROR: run enable_extension_to_10ns.py --confirm-extend first' >&2; exit 2; }
NAMD_BIN="${NAMD_BIN:-namd3}"; [[ -x "$NAMD_BIN" ]] || NAMD_BIN="$(command -v "$NAMD_BIN" 2>/dev/null || true)"
[[ -n "$NAMD_BIN" && -x "$NAMD_BIN" ]] || { echo 'ERROR: NAMD3 not found; set NAMD_BIN' >&2; exit 2; }
read -r -a namd_args <<< "${NAMD_ARGS:-}"
for r in 01 02 03; do for c in 06 07 08 09 10; do
  prefix="replica_${r}/${c}_prod_1ns/prod_${c}"; conf="replica_${r}/${c}_prod_1ns.conf"; expected=$((250000 + 10#$c * 500000)); mkdir -p "$(dirname "$prefix")"
  if [[ -s "${prefix}.log" ]] && python3 validate_namd_log.py "${prefix}.log" --expected-last-step "$expected" >/dev/null 2>&1; then echo "SKIP complete: $prefix"; continue; fi
  if find "$(dirname "$prefix")" -mindepth 1 -maxdepth 1 -type f -print -quit | grep -q .; then echo "ERROR: partial output exists for $prefix" >&2; exit 3; fi
  "$NAMD_BIN" "${namd_args[@]}" "$conf" > "${prefix}.log" 2>&1
  python3 validate_namd_log.py "${prefix}.log" --expected-last-step "$expected"
done; done
bash validate_stage10_outputs.sh --target-ns 10
