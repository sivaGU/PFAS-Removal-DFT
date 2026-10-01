#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64000
#SBATCH --time=12:00:00
#SBATCH --partition=batch


shopt -s extglob

# --- 1. Environment Setup ---
module purge
module load openbabel || true

export OB_BIN="/oscar/rt/9.6/25/spack/x86_64_v3/openbabel-3.1.1-agyrcjjydkhgct2f4huneznokq6ayxry/bin"
export PATH="$OB_BIN:$PATH"


export TOPO_SMI="${TOPO_SMI:-}"


export rundir=${SLURM_SUBMIT_DIR:-$(pwd)}
export job="openbabel_geom"
logfile="$rundir/${job}.out"


symlink_path="$HOME/jobtmp"
if [[ -L "$symlink_path" ]]; then
    base_root=$(readlink -f "$symlink_path")
else
    base_root="$symlink_path"
fi

if [[ ! -d "$base_root" ]]; then
    mkdir -p "$base_root"
fi

cd "$base_root" || exit 1

timestamp=$(date +%Y%m%d_%H%M%S)
dirname_base="${job}_${timestamp}"
dirname="$dirname_base"
counter=1
while [[ -e "$dirname" ]]; do
    dirname="${dirname_base}_${counter}"
    ((counter++))
done

mkdir -p "$dirname"
if [[ $? -ne 0 ]]; then
    echo "Error: Failed to create scratch directory inside $base_root" >> "$logfile"
    exit 1
fi

tmpdir="$(pwd)/$dirname"

# --- 3. Staging Files ---
echo "Staging files from $rundir to $tmpdir..."
cp -r "$rundir"/* "$tmpdir" 2>/dev/null || true

# --- 4. Execution ---
cd "$tmpdir" || exit 1

{
    echo "start: $(date)"
    echo "job: $job"
    echo "host: $(hostname)"
    echo "scratch: $tmpdir"
    echo "TOPO_SMI override: ${TOPO_SMI:-<auto-detect>}"
    echo "OpenBabel bin: $OB_BIN"
} > "$logfile"

if ! command -v obabel >/dev/null 2>&1; then
    echo "Error: obabel not found in PATH" | tee -a "$logfile"
    exit 1
fi


if [[ -n "$TOPO_SMI" ]]; then
    if [[ -f "$TOPO_SMI" ]]; then
        topo_input="$TOPO_SMI"
    else
        echo "Error: TOPO_SMI was set but not found: $TOPO_SMI" | tee -a "$logfile"
        exit 1
    fi
else
    shopt -s nullglob
    smi_files=( *.smi )
    shopt -u nullglob
    if [[ ${#smi_files[@]} -eq 0 ]]; then
        echo "Error: no .smi file found in submit directory." | tee -a "$logfile"
        exit 1
    fi
    if [[ ${#smi_files[@]} -gt 1 ]]; then
        echo "Error: multiple .smi files found. Set TOPO_SMI to choose one." | tee -a "$logfile"
        printf '%s\n' "${smi_files[@]}" | tee -a "$logfile"
        exit 1
    fi
    topo_input="${smi_files[0]}"
fi

echo "Topology input: $topo_input" | tee -a "$logfile"
base="${topo_input%.*}"

# 4b) Generate 3D geometry
obabel "$topo_input" -O "${base}_gen3d.sdf" --gen3d 2>&1 | tee -a "$logfile"

# 4c) Quick force-field minimization (UFF)
obabel "${base}_gen3d.sdf" -O "${base}_gen3d_uffmin.sdf" --minimize --ff UFF --steps 1000 2>&1 | tee -a "$logfile"


obabel "${base}_gen3d_uffmin.sdf" -O "${base}_gen3d_mmffmin.sdf" --minimize --ff MMFF94 --steps 500 2>&1 | tee -a "$logfile" || true

# 4e) Convenient xyz export
obabel "${base}_gen3d_uffmin.sdf" -O "${base}_gen3d_uffmin.xyz" 2>&1 | tee -a "$logfile"

if [[ -f "${base}_gen3d_mmffmin.sdf" ]]; then
    obabel "${base}_gen3d_mmffmin.sdf" -O "${base}_gen3d_mmffmin.xyz" 2>&1 | tee -a "$logfile" || true
fi

echo "end: $(date)" >> "$logfile"

# --- 5. Retrieval ---
echo "Retrieving files..." >> "$logfile"
if command -v rsync >/dev/null 2>&1; then
    rsync -a --exclude='*.tmp' . "$rundir/"
else
    echo "Warning: rsync not found. Falling back to cp..." >> "$logfile"
    cp -r . "$rundir/"
fi

# --- 6. Cleanup ---
cd "$rundir" || exit 1
if [[ "$tmpdir" == "$base_root"* ]] && [[ -d "$tmpdir" ]]; then
    echo "Cleaning up scratch directory: $tmpdir" >> "$logfile"
    rm -rf "$tmpdir"
fi
