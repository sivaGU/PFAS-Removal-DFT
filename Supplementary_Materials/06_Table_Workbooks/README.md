# Table workbooks

This folder contains Excel workbooks generated from the CSV tables in the supplementary archive. Original CSV files were left in place.

Each workbook groups related CSV tables. Each data table is on its own worksheet. A `source_paths` worksheet records source CSV paths and duplicate CSV paths that were not repeated. Data worksheets are labeled sequentially as `Table S1`, `Table S2`, etc.; see `table_index.md` for the full cross-reference.

The workbooks use plain worksheets with frozen header rows, autofilters, and formatted headers. They intentionally do not use formal Excel Table objects, which keeps the files more compatible with Excel's repair checker.

The visual formatting is designed to import cleanly into Google Sheets. Google Sheets' native "table" object may still need to be applied inside Google Sheets after import because that feature is not reliably preserved as a portable `.xlsx` object.

## Workbooks

- `01_BTMA_exchange_energetics.xlsx`: 2 data sheet(s).
- `02_EDA_component_summaries.xlsx`: 1 data sheet(s).
- `03_Extended_monomer_exchange.xlsx`: 2 data sheet(s).
- `04_PES_grid_data.xlsx`: 1 data sheet(s).
- `05_MD_GBSA_exchange_cycle.xlsx`: 6 data sheet(s).
- `06_MD_structural_metrics.xlsx`: 1 data sheet(s).
