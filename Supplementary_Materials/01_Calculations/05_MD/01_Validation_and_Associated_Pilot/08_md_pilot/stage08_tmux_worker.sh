#!/usr/bin/env bash


set -u
cd "$(dirname "$0")"

if [[ -f stage08_launch.env ]]; then
  # shellcheck disable=SC1091
  source stage08_launch.env
fi

rm -f stage08_launcher.exit
{
  echo "===== Stage 08 tmux worker started: $(date -Is) ====="
  echo "host=$(hostname)"
  echo "pid=$$"
  echo "pwd=$PWD"
  echo "NAMD_BIN=${NAMD_BIN:-<auto>}"
  echo "NAMD_ARGS=${NAMD_ARGS:-}"
  bash run_stage08_acceptance.sh
  rc=$?
  echo "===== Stage 08 tmux worker finished: $(date -Is); exit=$rc ====="
  printf '%s\n' "$rc" > stage08_launcher.exit
  exit "$rc"
} >> stage08_launcher.log 2>&1
