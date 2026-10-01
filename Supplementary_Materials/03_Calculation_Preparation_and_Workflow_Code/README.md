# Calculation and Analysis Code

`pVBTMA12_Model_Validation` contains model-definition, charge-transfer, Amber
parameterization/LEaP, pilot validation, and PFOA-associated construction
scripts from MD Stages 01-09.
`pVBTMA12_Production_and_Analysis` contains the Stage 10 launcher/validation
and Stage 11 CPPTRAJ/VMD/PBC and summary scripts. Matching input/output files
are organized under `../01_Calculations/05_MD` using the same stage numbers.

These are methods/provenance copies. Some launcher scripts expect their
original stage-relative directory layout and installed AmberTools/NAMD/VMD
environment; copying a script out of that context does not itself rerun the
workflow. Figure plotters and curated adjacent CSVs are in
`../02_Figure_Plotting_and_Data`.
