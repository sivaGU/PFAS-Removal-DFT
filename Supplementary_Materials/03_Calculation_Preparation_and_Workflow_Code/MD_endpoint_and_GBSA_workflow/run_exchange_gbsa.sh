#!/usr/bin/env bash
set -euo pipefail

source /home/zwa/miniforge3/etc/profile.d/conda.sh
conda activate AmberTools25

ante-MMPBSA.py \
  -p ../../05_pfas_exchange/solvated_resin_pfoa_exchange.prmtop \
  -c r48_pfoa_47cl.prmtop \
  -s ':WAT' \
  --radii=mbondi2 \
  > ../logs/ante_r48_pfoa_47cl.log 2>&1

ante-MMPBSA.py \
  -p ../../04_leap/solvated_resin_cl.prmtop \
  -c r48_48cl.prmtop \
  -s ':WAT' \
  --radii=mbondi2 \
  > ../logs/ante_r48_48cl.log 2>&1

ante-MMPBSA.py \
  -p ../01_build_bulk/pfoa_aq.prmtop \
  -c pfoa_aq.prmtop \
  -s ':WAT' \
  --radii=mbondi2 \
  > ../logs/ante_pfoa_aq.log 2>&1

ante-MMPBSA.py \
  -p ../01_build_bulk/cl_aq.prmtop \
  -c cl_aq.prmtop \
  -s ':WAT' \
  --radii=mbondi2 \
  > ../logs/ante_cl_aq.log 2>&1

cpptraj -i extract_large_terms.cpptraj > ../logs/extract_large_terms.log 2>&1
cpptraj -i extract_bulk_terms.cpptraj > ../logs/extract_bulk_terms.log 2>&1

MMPBSA.py -O \
  -i mmpbsa_stability.in \
  -cp r48_pfoa_47cl.prmtop \
  -y r48_pfoa_47cl_100.nc \
  -o FINAL_R48_PFOA_47CL_GBSA.dat \
  -eo r48_pfoa_47cl_gbsa_per_frame.csv \
  > ../logs/gbsa_r48_pfoa_47cl.log 2>&1

MMPBSA.py -O \
  -i mmpbsa_stability.in \
  -cp r48_48cl.prmtop \
  -y r48_48cl_100.nc \
  -o FINAL_R48_48CL_GBSA.dat \
  -eo r48_48cl_gbsa_per_frame.csv \
  > ../logs/gbsa_r48_48cl.log 2>&1

MMPBSA.py -O \
  -i mmpbsa_stability.in \
  -cp pfoa_aq.prmtop \
  -y pfoa_aq_50.nc \
  -o FINAL_PFOA_AQ_GBSA.dat \
  -eo pfoa_aq_gbsa_per_frame.csv \
  > ../logs/gbsa_pfoa_aq.log 2>&1

MMPBSA.py -O \
  -i mmpbsa_stability.in \
  -cp cl_aq.prmtop \
  -y cl_aq_50.nc \
  -o FINAL_CL_AQ_GBSA.dat \
  -eo cl_aq_gbsa_per_frame.csv \
  > ../logs/gbsa_cl_aq.log 2>&1

python ../05_metrics/combine_exchange_gbsa.py
