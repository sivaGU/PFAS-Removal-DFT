# Supplementary Table Workbook

Updated 30 September 2026 to match Supplementary_Materials_toxics-4550545_reconciled.docx and the common-reference DFT data used in manuscript 06-3.

Supplementary_Tables_S1-S11.xlsx retains the eleven numbered worksheets. Table S5 includes the system definition (S5a) and structural/parameterization validation (S5b). Table S11 includes energy sensitivity (S11a) and conformer provenance (S11b). See table_index.md for definitions and sources.

## Numerical conventions

Tables S1 and S2 use one common aqueous isolated-PFAS reference per species and functional across the resin models. The existing extended-model records supply the shared hybrid references and the r²SCAN-3c PFOA reference. Both bound-state solvent cycles retain aqueous isolated ions. Electronic and Gibbs source energies remain paired from coherent calculation records. Exchange energies are calculated by editable Excel formulas from the source energies, using 627.509474 kcal/mol per Eh.

This reference reconciliation is separate from rounding corrections. Hybrid BTMA Gibbs exchanges are 2.0711, 4.5325, 8.0739, and 5.7285 kcal/mol for 6:2 FTCA, PFHxA, PFOA, and PFOS. The corresponding model-expansion changes are -0.1679, -2.8548, -8.5277, and -4.6835 kcal/mol.

Table S3 contains full-precision EDA source values, formula-derived component sums, and Bond Energy minus sum. A gCP term not separately reported is marked as unavailable rather than zero. The residual has no assigned physical interpretation.

Table S5b distinguishes the strict serialized-charge check from its documented acceptance tolerance. Solvation checks cover the chloride-form precursor. MD sampling diagnostics remain separate and do not establish global convergence.

Existing PES and MD numerical records are preserved. Figure S1 belongs to the companion supplementary Word document and plotting package, not this table workbook.

## Sources

- 02_Figure_Plotting_and_Data/03_reporting_tables/DFT_reference_reconciliation/common_reference_cycles.csv
- 02_Figure_Plotting_and_Data/03_reporting_tables/EDA/current_eda_full_precision.csv
- Supplementary_Materials_toxics-4550545_reconciled.docx, Tables S5b and S11b
- Original workbook source paths and full-precision MD/PES records are preserved.

No additional quantum calculation or MD run was performed for this update.
