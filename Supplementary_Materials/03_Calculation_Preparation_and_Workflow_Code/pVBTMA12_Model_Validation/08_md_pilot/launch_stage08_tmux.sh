#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

command -v tmux >/dev/null 2>&1 || { echo "ERROR: tmux is not installed or not on PATH" >&2; exit 2; }

SESSION="${STAGE08_TMUX_SESSION:-pVBTMA12_stage08_acceptance}"
if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "ERROR: tmux session '$SESSION' already exists. Inspect it before launching another Stage 08 job." >&2
  exit 2
fi


if [[ -e stage08_launcher.log || -e stage08_launcher.exit || -e stage08_launch.env ]]; then
  stamp="$(date +%Y%m%d_%H%M%S)"
  backup="../_safety_backups/${stamp}_stage08_launcher_rerun/08_md_pilot"
  mkdir -p "$backup"
  for f in stage08_launcher.log stage08_launcher.exit stage08_launch.env; do
    [[ -e "$f" ]] && cp -a "$f" "$backup/$f"
  done
  echo "Previous launcher artifacts backed up under $backup"
fi

: > stage08_launcher.log
rm -f stage08_launcher.exit



{
  printf 'export PATH=%q\n' "$PATH"
  printf 'export NAMD_BIN=%q\n' "${NAMD_BIN:-}"
  printf 'export NAMD_ARGS=%q\n' "${NAMD_ARGS:-}"
  if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    printf 'export CUDA_VISIBLE_DEVICES=%q\n' "$CUDA_VISIBLE_DEVICES"
  fi
} > stage08_launch.env

chmod +x stage08_tmux_worker.sh
WORKER="$PWD/stage08_tmux_worker.sh"
tmux new-session -d -s "$SESSION" "bash '$WORKER'"

printf '%s\n' "$SESSION" > stage08_tmux_session.txt

echo "Started Stage 08 in detached tmux session: $SESSION"
echo "Use: bash stage08_status.sh"
echo "Attach interactively if needed: tmux attach -t $SESSION"
