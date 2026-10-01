
#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"; SESSION="${STAGE10_TMUX_SESSION:-pVBTMA12_stage10_3x5ns}"
command -v tmux >/dev/null || { echo 'tmux not found' >&2; exit 2; }
tmux has-session -t "$SESSION" 2>/dev/null && { echo "session already exists: $SESSION" >&2; exit 2; }
NAMD_BIN="${NAMD_BIN:-namd3}"; [[ -x "$NAMD_BIN" ]] || NAMD_BIN="$(command -v "$NAMD_BIN" 2>/dev/null || true)"
[[ -n "$NAMD_BIN" && -x "$NAMD_BIN" ]] || { echo 'NAMD3 not found; set NAMD_BIN' >&2; exit 2; }
{ printf 'export NAMD_BIN=%q\n' "$NAMD_BIN"; printf 'export NAMD_ARGS=%q\n' "${NAMD_ARGS:-}"; printf 'export PATH=%q\n' "$PATH"; } > stage10_launch.env
tmux new-session -d -s "$SESSION" "cd '$PWD' && bash stage10_tmux_worker.sh"
printf '%s\n' "$SESSION" > stage10_tmux_session.txt; echo "launched $SESSION"
