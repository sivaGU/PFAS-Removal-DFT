#!/usr/bin/env bash
set -euo pipefail

source /home/zwa/miniforge3/etc/profile.d/conda.sh
conda activate AmberTools25

ante-MMPBSA.py \
  -p ../05_pfas_exchange/solvated_resin_pfoa_exchange.prmtop \
  -c bound_complex.prmtop \
  -r bound_receptor.prmtop \
  -l bound_ligand.prmtop \
  -s ':WAT' \
  -n ':PFO' \
  --radii=mbondi2 \
  > ../logs/ante_mmpbsa_bound.log 2>&1

ante-MMPBSA.py \
  -p ../11_pfoa_unbound_reference/solvated_resin_pfoa_unbound_reference.prmtop \
  -c unbound_complex.prmtop \
  -r unbound_receptor.prmtop \
  -l unbound_ligand.prmtop \
  -s ':WAT' \
  -n ':PFO' \
  --radii=mbondi2 \
  > ../logs/ante_mmpbsa_unbound.log 2>&1

MMPBSA.py -O \
  -i mmpbsa_qualitative.in \
  -cp bound_complex.prmtop \
  -rp bound_receptor.prmtop \
  -lp bound_ligand.prmtop \
  -y bound_complex_100.nc \
  -o FINAL_RESULTS_BOUND_MMPBSA.dat \
  -eo bound_mmpbsa_per_frame.csv \
  > ../logs/mmpbsa_bound.log 2>&1

MMPBSA.py -O \
  -i mmpbsa_qualitative.in \
  -cp unbound_complex.prmtop \
  -rp unbound_receptor.prmtop \
  -lp unbound_ligand.prmtop \
  -y unbound_complex_50.nc \
  -o FINAL_RESULTS_UNBOUND_MMPBSA.dat \
  -eo unbound_mmpbsa_per_frame.csv \
  > ../logs/mmpbsa_unbound.log 2>&1

grep -A20 'DELTA TOTAL' FINAL_RESULTS_BOUND_MMPBSA.dat > bound_delta_total_summary.txt || true
grep -A20 'DELTA TOTAL' FINAL_RESULTS_UNBOUND_MMPBSA.dat > unbound_delta_total_summary.txt || true
