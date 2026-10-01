#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
SESSION="${STAGE08_TMUX_SESSION:-$(cat stage08_tmux_session.txt 2>/dev/null || echo pVBTMA12_stage08_acceptance)}"

if [[ -f stage08_launcher.exit ]]; then
  rc="$(tr -d '[:space:]' < stage08_launcher.exit)"
  echo "Stage 08 worker has exited. Recorded exit status: $rc"
elif tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "Stage 08 tmux session is running: $SESSION"
else
  echo "WARNING: no exit marker and tmux session '$SESSION' is absent. Inspect stage08_launcher.log." >&2
fi

echo
if [[ -s stage08_launcher.log ]]; then
  echo "--- stage08_launcher.log (last 30 lines) ---"
  tail -30 stage08_launcher.log
else
  echo "stage08_launcher.log is empty or absent."
fi

echo
for log in \
  05_acceptance_npt_1ns/acceptance1ns.log \
  04_equil_npt_500ps/equil500ps.log \
  03_heat_310K/heat310.log \
  02_heat_200K/heat200.log \
  01_heat_100K/heat100.log \
  00_minimize/minimize.log; do
  if [[ -s "$log" ]]; then
    echo "--- active/latest NAMD log: $log (last 20 lines) ---"
    tail -20 "$log"
    break
  fi
done

if [[ -s stage08_completion.txt ]]; then
  echo
  echo "--- stage08_completion.txt ---"
  cat stage08_completion.txt
fi
