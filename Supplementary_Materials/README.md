# Supplementary Materials

This directory contains copied supplementary files for the PFAS removal DFT/MD
study. Files here are organized for manuscript support and repository upload;
the original working directories outside `Supplementary_Materials` are not
modified by this archive.

## Directory overview

- `00_README_and_File_Maps/`: navigation files describing the contents of this
  supplement and how calculation folders, plotting/data files, and source
  structures relate to manuscript outputs.
- `01_Calculations/`: pruned calculation folders containing main inputs,
  outputs, and input/output geometries where available.
- `02_Figure_Plotting_and_Data/`: plotting scripts paired with the extracted
  data/transcripts used to create the manuscript figures. Main figure image
  files are intentionally not included here.
- `03_Calculation_Preparation_and_Workflow_Code/`: scripts used to prepare
  calculation inputs, build/parameterize MD systems, or generate intermediate
  analysis files.
- `04_Source_Structures_and_Model_Building/`: initial structures, source
  structures, and the latest 48mer resin builder code.
- `05_MD_Animations/`: rendered MD trajectory animations and the scripts used
  to generate them.
- `06_Table_Workbooks/`: Excel workbook versions of the supplementary CSV
  tables, with one worksheet per table and source-path cross-references.

## Packaging note

Most calculation outputs are left as readable plain text in this directory.
Large text-like files that would otherwise exceed single-file upload limits are
stored as gzip-compressed files (`.gz`) and can be decompressed with
`gzip -d <filename>.gz` if needed. Additional intermediate files and large
outputs can be provided upon request.
