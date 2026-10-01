
#!/usr/bin/env bash
set -u
cd "$(dirname "$0")"; [[ -f stage10_extension_launch.env ]] && source stage10_extension_launch.env
rm -f stage10_extension.exit
{ echo "===== Stage 10 extension started: $(date -Is) ====="; bash run_stage10_extension.sh; rc=$?; echo "===== Stage 10 extension finished: $(date -Is); exit=$rc ====="; printf '%s\n' "$rc" > stage10_extension.exit; exit "$rc"; } >> stage10_extension.log 2>&1
