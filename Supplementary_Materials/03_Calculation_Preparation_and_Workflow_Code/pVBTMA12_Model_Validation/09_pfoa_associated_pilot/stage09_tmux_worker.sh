#!/usr/bin/env bash
set -u
cd "$(dirname "$0")"
[[ -f stage09_launch.env ]] && source stage09_launch.env
rm -f stage09_launcher.exit
{
  echo "===== Stage 09 tmux worker started: $(date -Is) ====="
  echo "host=$(hostname)"; echo "pid=$$"; echo "pwd=$PWD"
  echo "NAMD_BIN=${NAMD_BIN:-<auto>}"; echo "NAMD_ARGS=${NAMD_ARGS:-}"
  bash run_stage09_acceptance.sh
  rc=$?
  echo "===== Stage 09 tmux worker finished: $(date -Is); exit=$rc ====="
  printf '%s\n' "$rc" > stage09_launcher.exit
  exit "$rc"
} >> stage09_launcher.log 2>&1
