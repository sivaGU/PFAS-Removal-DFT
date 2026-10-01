#!/usr/bin/env bash
set -u
cd "$(dirname "$0")"
SESSION="${STAGE09_TMUX_SESSION:-$(cat stage09_tmux_session.txt 2>/dev/null || echo pVBTMA12_stage09_pfoa_acceptance)}"
if tmux has-session -t "$SESSION" 2>/dev/null; then echo "RUNNING: $SESSION"; else echo "NOT RUNNING: $SESSION"; fi
[[ -f stage09_launcher.exit ]] && echo "exit: $(cat stage09_launcher.exit)" || echo 'exit: not yet recorded'
for f in stage09_launcher.log 03_acceptance_npt_1ns/acceptance1ns.log analysis/stage09_summary.txt; do [[ -f "$f" ]] && { echo "--- $f ---"; tail -n 20 "$f"; }; done
