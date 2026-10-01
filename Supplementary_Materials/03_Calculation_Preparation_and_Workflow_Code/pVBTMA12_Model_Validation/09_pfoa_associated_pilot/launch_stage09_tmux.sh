#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
SESSION="${STAGE09_TMUX_SESSION:-pVBTMA12_stage09_pfoa_acceptance}"
command -v tmux >/dev/null || { echo 'tmux not found' >&2; exit 2; }
if tmux has-session -t "$SESSION" 2>/dev/null; then echo "session already exists: $SESSION" >&2; exit 2; fi
resolve_cmd(){
  local requested="$1" fallback="$2" out=""
  if [[ -n "$requested" ]]; then
    if [[ -x "$requested" ]]; then out="$requested"; else out="$(command -v "$requested" 2>/dev/null || true)"; fi
  fi
  if [[ -z "$out" ]]; then out="$(command -v "$fallback" 2>/dev/null || true)"; fi
  printf '%s' "$out"
}
NAMD_BIN="$(resolve_cmd "${NAMD_BIN:-}" namd3)"
VMD_BIN="$(resolve_cmd "${VMD_BIN:-}" vmd)"
CPPTRAJ_BIN="$(resolve_cmd "${CPPTRAJ_BIN:-}" cpptraj)"
[[ -n "$NAMD_BIN" ]] || { echo 'NAMD3 not found; set NAMD_BIN' >&2; exit 2; }
[[ -n "$VMD_BIN" ]] || { echo 'VMD not found; set VMD_BIN' >&2; exit 2; }

{
  printf 'export NAMD_BIN=%q\n' "$NAMD_BIN"
  printf 'export NAMD_ARGS=%q\n' "${NAMD_ARGS:-}"
  printf 'export VMD_BIN=%q\n' "$VMD_BIN"
  printf 'export CPPTRAJ_BIN=%q\n' "$CPPTRAJ_BIN"
  printf 'export PATH=%q\n' "$PATH"
} > stage09_launch.env
export NAMD_BIN VMD_BIN
[[ -n "$CPPTRAJ_BIN" ]] && export CPPTRAJ_BIN
export NAMD_ARGS="${NAMD_ARGS:-}"
tmux new-session -d -s "$SESSION" "cd '$PWD' && bash stage09_tmux_worker.sh"
printf '%s\n' "$SESSION" > stage09_tmux_session.txt
echo "launched $SESSION"
