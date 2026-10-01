
#!/usr/bin/env bash
set -u
cd "$(dirname "$0")"; [[ -f stage10_launch.env ]] && source stage10_launch.env
rm -f stage10_launcher.exit
{ echo "===== Stage 10 started: $(date -Is) ====="; bash run_stage10_production.sh; rc=$?; echo "===== Stage 10 finished: $(date -Is); exit=$rc ====="; printf '%s\n' "$rc" > stage10_launcher.exit; exit "$rc"; } >> stage10_launcher.log 2>&1
