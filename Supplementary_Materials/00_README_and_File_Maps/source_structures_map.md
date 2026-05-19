# Source Structures and Model-Building Map

Source structures and model-building inputs live under
`Supplementary_Materials/04_Source_Structures_and_Model_Building`.

## 01_initial_small_molecule_structures

Contents:
- `pubchem/`: source SDF files for BTMA, cholestyramine, and PFAS species.
- `processed/`: cleaned XYZ structures used as starting materials.
- `complexes/`: initial PFAS-cholestyramine complex structures.
- `PFOA-Cholestyramine_pes.xyz`: starting structure used for the PFOA PES setup.

Purpose:
Records small-molecule and fragment starting structures upstream of the DFT
model calculations.

## 02_48mer_resin_model_construction

Contents:
- `builder_code/cholestyramine_polymer5.py`
- `structures/48mer_init.sdf`
- `structures/48mer_init.pdb`
- `structures/48mer_init.xyz`
- `structures/48mer_initialopt.xyz`
- `structures/cholestyramine_multichain_topology.smi`
- `structures/cholestyramine_multichain_topology_gen3d_mmffmin.sdf`
- `structures/cholestyramine_multichain_topology_gen3d_mmffmin.xyz`

Purpose:
Contains the latest resin-builder script and the resulting 48mer model
structures. Earlier development versions of the resin builder were intentionally
not included.

## 03_MD_initial_inputs

Contents:
- `PFOA.pdb`
- `chol48_with_counterions.pdb`

Purpose:
Initial PDB inputs used in the AMBER/MD model-building workflow.

