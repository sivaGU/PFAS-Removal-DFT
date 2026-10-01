# Source Structures and Model-Building Map

Paths are relative to `Supplementary_Materials/04_Source_Structures_and_Model_Building`.

| Folder | Purpose and origin |
|---|---|
| `01_initial_small_molecule_structures/pubchem`, `processed`, `complexes` | Source and prepared small-molecule and DFT structures. Matching geometries for individual calculations also accompany their inputs and outputs under `01_Calculations`. |
| `02_pVBTMA12_current_model/01_model_definition` | Constitutional and stereo-explicit model definitions, graph builders, and stereochemical specification. `pVBTMA12_model_definition.mol` is the model graph. |
| `02_pVBTMA12_current_model/02_accepted_UFF_structure` | UFF-minimized SDF and XYZ structure used for parameterization. |
| `02_pVBTMA12_current_model/03_structure_validation` | Indexed structure/connectivity validation records and validator. |
| `02_pVBTMA12_current_model/04_RCT_charge_models` | RCT 5-mer charge-source SDF/MOL2, atom maps, charge-transfer records, and full 12-mer `pVBTMA12_gaff2_rct.mol2`. |
| `02_pVBTMA12_current_model/05_dry_and_solvated` | Stage 06 dry polymer PDB and Stage 07 chloride/NaCl solvated PDB. Matching LEaP inputs, logs and Amber topology/coordinates are under `01_Calculations/05_MD/01_Validation_and_Associated_Pilot`. |
| `02_pVBTMA12_current_model/06_PFOA_associated` | Stage 09 associated-system starting PDB and final built PDB, plus PFOA GAFF2 MOL2 and frcmod. |

Figure 14's selected structure and provenance record are under
`02_Figure_Plotting_and_Data/01_main_text_figures/Figure_14_MD_model_and_workflow/input_data/`.
That frame is distinct from the Stage 09 start. Its paired topology and source
DCD chunk are under `01_Calculations/05_MD`; see `frame_selection.md` for the
exact replica, chunk, frame and checksums.

