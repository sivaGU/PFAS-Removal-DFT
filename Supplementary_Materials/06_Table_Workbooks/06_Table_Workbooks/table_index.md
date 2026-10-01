# Supplementary Table Index

The workbook is [Supplementary_Tables_S1-S11.xlsx](Supplementary_Tables_S1-S11.xlsx). It retains eleven numbered worksheets and matches the reconciled supplementary Word tables.

| Worksheet | Content |
|---|---|
| S1_BTMA_exchange | S1a-c. BTMA electronic/Gibbs source energies and formula-derived exchange quantities with common aqueous references |
| S2_DVB_BTMA_exchange | S2a-c. Hybrid BTMA and DVB-BTMA exchange in water and 1-octanol, using common aqueous isolated ions |
| S3_EDA_components | S3a-b. Full-precision ORCA EDA terms, component sum, Bond Energy, unassigned residual, and source paths |
| S4_PFOA_PES | S4. Constrained two-distance PFOA/BTMA/chloride PES and absolute xTB energies |
| S5_pVBTMA12_model_system | S5a at rows 1-33. System definition. S5b at rows 36-49. Structural and parameterization validation |
| S6_MD_protocol | S6. Molecular dynamics protocol and three 10 ns replicas |
| S7_replica_summary | S7. Full-precision per-replica summaries, correlations, and effective sample sizes |
| S8_block_summary | S8. Ten 1 ns block summaries per replica |
| S9_convergence_diagnostics | S9. Early/late shifts, replica spread, and effective sample size diagnostics |
| S10_contact_occupancy | S10. Fraction of sampled frames with a headgroup oxygen strictly below 5 Å from any oligomer ammonium nitrogen |
| S11_GOAT_sensitivity | S11a at rows 1-11. Source energies and one-species substitutions. S11b at rows 14-24. GOAT indices, search energies, and starting/refined RMSDs |

## Definitions

Energy units are Eh or kcal/mol as labeled. Exchange = PFAS complex + free chloride - chloride complex - free PFAS. The conversion is 627.509474 kcal/mol per Eh.

S3 component sums exclude terms not separately printed. Their difference from Bond Energy is an arithmetic residual, not preparation energy.

S5b reports the formal +12 polymer charge separately from the serialized sum +11.999994 e. The strict 1e-6 test was not met, but the residual is within the documented serialization bound. Chloride-form precursor checks are distinct from the PFOA-associated production composition in S5a.

For S7, distances are in Å, tail contacts/waters are counts, and sampling is every 2 ps. IAT includes half the zero-lag contribution and the initial positive normalized autocorrelation sequence. ESS = n / (2 × IAT in frames). S8 uses these same metric definitions. S9 diagnostics do not establish global convergence.

S11 GOAT indices are zero-based. Search energies are relative to the original GOAT minimum for each species. RMSDs are symmetry-aware heavy-atom values relative to the original candidate. Quasi-RRHO uses a 100 cm⁻¹ reference frequency and a separate 1 cm⁻¹ cutoff. The sensitivity values are not an ensemble free energy or a hybrid-level uncertainty interval.
