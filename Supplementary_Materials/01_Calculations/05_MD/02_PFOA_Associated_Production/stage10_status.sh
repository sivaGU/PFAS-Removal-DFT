
#!/usr/bin/env bash
set -u
cd "$(dirname "$0")"; SESSION="${STAGE10_TMUX_SESSION:-$(cat stage10_tmux_session.txt 2>/dev/null || echo pVBTMA12_stage10_3x5ns)}"
tmux has-session -t "$SESSION" 2>/dev/null && echo "RUNNING: $SESSION" || echo "NOT RUNNING: $SESSION"
[[ -f stage10_launcher.exit ]] && echo "exit: $(cat stage10_launcher.exit)" || echo 'exit: not yet recorded'
[[ -f stage10_completion.txt ]] && cat stage10_completion.txt
[[ -f stage10_launcher.log ]] && { echo '--- launcher tail ---'; tail -n 25 stage10_launcher.log; }
