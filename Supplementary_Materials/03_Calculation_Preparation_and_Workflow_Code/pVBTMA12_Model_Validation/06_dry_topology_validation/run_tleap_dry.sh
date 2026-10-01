#!/usr/bin/env bash
set -euo pipefail
if [[ -f "${HOME}/miniforge3/etc/profile.d/conda.sh" ]]; then
  source "${HOME}/miniforge3/etc/profile.d/conda.sh"
elif command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
else
  echo "ERROR: conda initialization not found" >&2
  exit 2
fi
conda activate AmberTools25

STAMP="$(date +%Y%m%d_%H%M%S)"
BACKUP="../_safety_backups/${STAMP}_stage06_dry_topology_validation"
STAGE_BACKUP="$BACKUP/06_dry_topology_validation"
mkdir -p "$STAGE_BACKUP"
for f in \
  pVBTMA12_dry.prmtop pVBTMA12_dry.inpcrd pVBTMA12_dry.pdb \
  tleap_stdout.log tleap_stderr.log leap.log dry_topology_validation.txt; do
  [[ -e "$f" ]] && cp -a "$f" "$STAGE_BACKUP/"
done

tleap -f build_dry.leap > tleap_stdout.log 2> tleap_stderr.log
python validate_dry_topology.py > dry_topology_validation.txt
