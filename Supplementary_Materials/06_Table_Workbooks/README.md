# Table workbooks

This folder contains the consolidated Excel workbook version of the labeled
supplementary tables. Source data files remain in their corresponding
calculation or figure-data folders and are cross-referenced in
`table_index.md`.

The consolidated workbook `Supplementary_Tables_S1-S11.xlsx` contains one
worksheet for each labeled supplementary table. Each worksheet begins with the
visible table title (`Table S1`, `Table S2`, etc.) followed immediately by the
column headers and data. Worksheet tab names are shortened for Excel
compatibility; see `table_index.md` for the full table-to-worksheet and
source-data cross-reference.

The workbooks use plain worksheets with frozen header rows, autofilters, and formatted headers. They intentionally do not use formal Excel Table objects, which keeps the files more compatible with Excel's repair checker.

The visual formatting is designed to import cleanly into Google Sheets. Google Sheets' native "table" object may still need to be applied inside Google Sheets after import because that feature is not reliably preserved as a portable `.xlsx` object.

## Workbook

- `Supplementary_Tables_S1-S11.xlsx`: 11 data sheet(s).
