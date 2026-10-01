
#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
python3 validate_stage10_outputs.py "$@"
