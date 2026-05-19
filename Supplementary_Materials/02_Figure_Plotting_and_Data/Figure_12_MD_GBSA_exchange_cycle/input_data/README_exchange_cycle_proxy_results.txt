Qualitative MD/GBSA exchange-cycle proxy

Required language:
This is a qualitative MD/GBSA exchange-cycle proxy. It is not a rigorous
standard-state exchange free energy.

Exchange expression:
Delta G_exchange_proxy =
G_GBSA(R48 + PFOA + 47Cl)
- G_GBSA(R48 + 48Cl)
- G_GBSA(PFOA aq)
+ G_GBSA(Cl aq)

Endpoint verification:
- R48 + PFOA + 47Cl topology:
  work_amber/05_pfas_exchange/solvated_resin_pfoa_exchange.prmtop
  Composition: R48 1, PFO 1, Cl- 47, WAT 43893, total charge 0.0.
  Trajectory: work_amber/09_namd_pfoa_exchange/resin_pfoa_exchange_npt_1ns.dcd
  Frames used: 100, 10 ps spacing.

- R48 + 48Cl topology:
  work_amber/04_leap/solvated_resin_cl.prmtop
  Composition: R48 1, Cl- 48, WAT 20766, total charge 0.0.
  Trajectory: work_amber/07_namd_resin_only/resin_only_npt_2ns.dcd
  Frames available: 200.
  Frames used: last 100 frames, 10 ps spacing.

- PFOA aq:
  Built from existing pfoa_gaff2.mol2 and pfoa.frcmod.
  Composition: PFO 1, WAT 2736, total charge -1.0.
  No Na+ counterion was added, preserving the requested PFOA- term.
  NAMD PME therefore uses the usual uniform neutralizing background; interpret
  qualitatively.
  Ran minimization, NVT heating, and 0.5 ns NPT.
  Frames used: 50, 10 ps spacing.

- Cl aq:
  Built using standard AMBER/TIP3P ion parameters.
  Composition: Cl- 1, WAT 2096, total charge -1.0.
  No Na+ counterion was added, preserving the requested Cl- term.
  NAMD PME therefore uses the usual uniform neutralizing background; interpret
  qualitatively.
  Ran minimization, NVT heating, and 0.5 ns NPT.
  Frames used: 50, 10 ps spacing.

GBSA protocol:
- Explicit water stripped before endpoint GBSA analysis.
- AmberTools MMPBSA.py stability calculations.
- GB model igb=5.
- saltcon=0.150.
- No normal-mode entropy.
- Radii set with ante-MMPBSA.py --radii=mbondi2.

Term results, kcal/mol:
- G_GBSA(R48 + PFOA + 47Cl): -1991.546794 +/- 2.898990, n=100.
- G_GBSA(R48 + 48Cl): -2156.925874 +/- 3.281063, n=100.
- G_GBSA(PFOA aq): 115.564464 +/- 0.612138, n=50.
- G_GBSA(Cl aq): -101.987800 +/- 0.000000, n=50.

Exchange-cycle proxy:
Delta G_exchange_proxy = -52.173185 +/- 4.381026 kcal/mol.

Previous PFOA association proxy:
G(R48 + PFOA + 47Cl) - G(R48 + 47Cl) - G(PFOA)
= -23.6640 +/- 2.6985 kcal/mol.

Interpretation:
The exchange-cycle proxy favors the PFOA-associated resin state over the
chloride-form resin plus aqueous PFOA under this qualitative MD/GBSA model.
This should be interpreted as support for favorable PFOA exchange/association,
not as a rigorous standard-state exchange free energy.

Binding-site language:
Do not claim single-site binding to N9.
Use: "PFOA remained resin-associated and frequently sampled
ammonium-proximal configurations."

PFOA-associated trajectory mechanism metrics:
- nearest PFOA carboxylate oxygen to any resin ammonium N:
  mean 4.6510 A, SEM 0.0627 A, min 3.6107 A, max 6.2167 A.
- PFOA tail-resin heavy atom contacts within 4 A:
  mean 18.55, SEM 0.3418, min 10, max 27.
- chlorides within 5 A of the nearest ammonium N:
  mean 0.09, SEM 0.0288.
- chlorides within 7 A of the nearest ammonium N:
  mean 0.27, SEM 0.0468.
- chlorides within 10 A of the nearest ammonium N:
  mean 0.82, SEM 0.0783.
- waters within 5 A of the PFOA fluorinated tail:
  mean 35.69, SEM 0.2926, min 28, max 44.

Representative figure frames:
work_amber/17_exchange_cycle_proxy/05_metrics/pfoa_assoc_frame_001_resin_pfo_cl.pdb
work_amber/17_exchange_cycle_proxy/05_metrics/pfoa_assoc_frame_025_resin_pfo_cl.pdb
work_amber/17_exchange_cycle_proxy/05_metrics/pfoa_assoc_frame_050_resin_pfo_cl.pdb
work_amber/17_exchange_cycle_proxy/05_metrics/pfoa_assoc_frame_075_resin_pfo_cl.pdb
work_amber/17_exchange_cycle_proxy/05_metrics/pfoa_assoc_frame_100_resin_pfo_cl.pdb


Methods section text for manuscript
===================================

Molecular dynamics calculations were performed as a qualitative supporting
analysis for the DFT-centered study. The MD analysis was designed to test
whether explicit-solvent force-field simulations support favorable exchange of
chloride by deprotonated PFOA at a cholestyramine-like quaternary ammonium resin
environment. The reported free-energy-like value is a qualitative MD/GBSA
exchange-cycle proxy and is not a rigorous standard-state exchange free energy.
No PMF, umbrella sampling, ABF, FEP, TI, MBAR, weighted ensemble, or replica MD
calculation was performed.

Resin model and force-field construction
----------------------------------------

The resin model was a crosslinked cholestyramine-like 48mer, denoted R48. The
organic resin contained 1644 atoms and 48 quaternary ammonium groups, giving a
formal resin charge of +48. A chloride-form resin endpoint was built with 48
chloride counterions:

R48(+48) + 48 Cl(-1) = 0.

The PFOA exchange endpoint was built by replacing one chloride equivalent with
one deprotonated PFOA anion:

R48(+48) + PFOA(-1) + 47 Cl(-1) = 0.

The full 48mer connectivity was taken from the validated full-resin structure
and associated SDF/MOL2 representation. The 48mer was parameterized with GAFF2.
Whole-molecule AM1-BCC charge derivation was not used for the complete 1644-atom
+48 resin because the SQM/AM1-BCC calculation failed for the full highly charged
macromolecular model. Instead, charges were derived from capped model compounds
representing the chemically distinct resin environments. These included
internal repeat-unit, terminal, crosslink-adjacent, and crosslink-local model
compounds. AM1-BCC charges were calculated for the capped model compounds with
Antechamber/GAFF2, then mapped onto chemically equivalent atoms in the complete
48mer. The mapped full-resin charge was adjusted to give a final resin charge of
+48.000000. A full-resin GAFF2 MOL2 file with mapped read-in charges was then
processed with Antechamber in read-charge mode and checked with parmchk2. LEaP
reported no missing atom types or missing parameters for the resin-only and
PFOA-containing systems.

The deprotonated PFOA anion was parameterized with GAFF2 and AM1-BCC charges
using Antechamber. The PFOA residue name was PFO and the net charge was -1.
The parameter files used for the PFOA terms were:

work_amber/03_param/pfoa_gaff2.mol2
work_amber/03_param/pfoa.frcmod

Chloride ions were treated as standard AMBER/TIP3P ions and were not
parameterized with Antechamber.

System construction
-------------------

All solvated systems were built with LEaP using GAFF2 for organic species,
TIP3P water, and standard AMBER/TIP3P ion parameters. The chloride-form resin
endpoint contained one R48 residue, 48 chloride ions, and 20766 waters:

work_amber/04_leap/solvated_resin_cl.prmtop
work_amber/04_leap/solvated_resin_cl.inpcrd

The PFOA-containing resin endpoint contained one R48 residue, one PFOA anion,
47 chloride ions, and 43893 waters:

work_amber/05_pfas_exchange/solvated_resin_pfoa_exchange.prmtop
work_amber/05_pfas_exchange/solvated_resin_pfoa_exchange.inpcrd

For the aqueous single-anion endpoints, small TIP3P boxes were built containing
one anion and no explicit counterion. The PFOA(aq) system contained one PFOA
anion and 2736 waters:

work_amber/17_exchange_cycle_proxy/01_build_bulk/pfoa_aq.prmtop
work_amber/17_exchange_cycle_proxy/01_build_bulk/pfoa_aq.inpcrd

The Cl(aq) system contained one chloride ion and 2096 waters:

work_amber/17_exchange_cycle_proxy/01_build_bulk/cl_aq.prmtop
work_amber/17_exchange_cycle_proxy/01_build_bulk/cl_aq.inpcrd

The aqueous PFOA and chloride boxes therefore each had a net charge of -1. No
Na+ counterion was added because the exchange-cycle expression explicitly
requires the isolated aqueous anion terms, G(PFOA aq) and G(Cl aq). NAMD PME
therefore used the usual uniform neutralizing background for these charged
periodic boxes. This is one reason the exchange value is interpreted
qualitatively rather than as a rigorous standard-state free energy.

Molecular dynamics protocol
---------------------------

All MD simulations were run with NAMD3 using AMBER topology/coordinate input
mode. The NAMD executable used in this workflow was:

/home/zwa/namd/NAMD_Git-2025-12-04_Linux-x86_64-multicore-CUDA/namd3

AMBER-mode NAMD settings included:

amber yes
parmfile <AMBER prmtop>
ambercoor <AMBER inpcrd>
exclude scaled1-4
readexclusions yes
1-4scaling 0.833333333
scnb 2.0

Periodic electrostatics were treated with PME. Nonbonded interactions used a
9.0 A cutoff, 11.0 A pairlist distance, no switching, and the Lennard-Jones
long-range correction. Bonds involving hydrogen were constrained with
rigidBonds all, allowing a 2 fs timestep for the standard MD runs. Langevin
dynamics were used for temperature control at 310.15 K with a damping
coefficient of 1.0 ps^-1. NPT simulations used the Langevin piston barostat with
a target pressure of 1.01325 bar, piston period 100 fs, piston decay 50 fs, and
piston temperature 310.15 K.

No positional restraints, terminal restraints, shape restraints, fixed atoms,
extra bonds, or Colvars restraints were used in the resin-only chloride-form
endpoint, the PFOA-associated endpoint, or the aqueous-anion endpoint
simulations used for the exchange-cycle calculation. The only constraint used
was rigidBonds all. The guided-approach trajectory generated separately was not
used in the exchange-cycle proxy and is not part of this methods description.

The chloride-form resin endpoint used the existing validated 2 ns no-restraint
NPT trajectory:

work_amber/07_namd_resin_only/resin_only_npt_2ns.dcd

This trajectory contained 200 saved frames. The last 100 frames were used for
the exchange-cycle GBSA analysis, corresponding to a 10 ps frame spacing over
the final 1 ns of the 2 ns continuation.

The PFOA-associated resin endpoint used the existing 1 ns no-restraint NPT
trajectory:

work_amber/09_namd_pfoa_exchange/resin_pfoa_exchange_npt_1ns.dcd

This trajectory contained 100 saved frames and all 100 frames were used for the
exchange-cycle GBSA analysis.

For the PFOA(aq) and Cl(aq) endpoints, each system was minimized, heated under
NVT, and equilibrated for 0.5 ns under NPT. The NVT heating protocol used
velocity reassignment in three stages: 100 K for 25 ps, 200 K for 25 ps, and
310.15 K for 50 ps. The NPT production/equilibration stage was then run for
0.5 ns. Frames were saved every 10 ps, giving 50 frames for each aqueous anion
endpoint.

The final NPT conditions for PFOA(aq) were:

final temperature: 307.6767 K
TEMPAVG: 310.5207 K
final pressure: -367.8673 atm
PRESSAVG: 12.5230 atm
final volume: 84302.6832 A^3

The final NPT conditions for Cl(aq) were:

final temperature: 311.5818 K
TEMPAVG: 310.9368 K
final pressure: -231.5344 atm
PRESSAVG: 6.1605 atm
final volume: 64483.0505 A^3

The instantaneous pressures fluctuate substantially in these small periodic
boxes, so the running averages and stable completion were used as the
practical validation checks. Both aqueous-anion NPT simulations completed
without fatal errors, missing parameters, or fast-atom instability. The Cl(aq)
NPT calculation initially encountered a NAMD patch-grid shrink error during box
relaxation; this was corrected by adding margin 8.0 and rerunning the NPT stage
from the completed NVT restart.

GBSA endpoint analysis
----------------------

The exchange-cycle proxy was evaluated with AmberTools MMPBSA.py using
single-trajectory stability calculations for each endpoint. Prior to GBSA
analysis, explicit water was stripped from every trajectory with cpptraj.
The same GBSA protocol was used for all four terms. The input model used:

igb=5
saltcon=0.150

No normal-mode entropy calculation was performed. Radii were set using
ante-MMPBSA.py with --radii=mbondi2. Per-frame GBSA total energies were written
with the MMPBSA.py energy-output option and combined in the thermodynamic
cycle.

The four endpoint terms were:

G_GBSA(R48 + PFOA + 47Cl)
G_GBSA(R48 + 48Cl)
G_GBSA(PFOA aq)
G_GBSA(Cl aq)

The qualitative exchange-cycle proxy was then evaluated as:

Delta G_exchange_proxy =
G_GBSA(R48 + PFOA + 47Cl)
- G_GBSA(R48 + 48Cl)
- G_GBSA(PFOA aq)
+ G_GBSA(Cl aq)

This expression preserves the stoichiometry of the ion-exchange reaction:

R48-Cl + PFOA(aq) -> R48-PFOA + Cl(aq)

In the present notation, R48-Cl corresponds to the R48 + 48Cl endpoint, and
R48-PFOA corresponds to the R48 + PFOA + 47Cl endpoint.

The resulting endpoint means and standard errors were:

G_GBSA(R48 + PFOA + 47Cl) = -1991.546794 +/- 2.898990 kcal/mol, n = 100
G_GBSA(R48 + 48Cl)        = -2156.925874 +/- 3.281063 kcal/mol, n = 100
G_GBSA(PFOA aq)           =   115.564464 +/- 0.612138 kcal/mol, n = 50
G_GBSA(Cl aq)             =  -101.987800 +/- 0.000000 kcal/mol, n = 50

The resulting exchange-cycle proxy was:

Delta G_exchange_proxy = -52.173185 +/- 4.381026 kcal/mol

The zero standard error for the Cl(aq) term reflects the fact that, after
stripping explicit water, the endpoint contains only a single chloride ion and
the GBSA energy is effectively coordinate-independent. The uncertainty reported
for the exchange-cycle proxy was obtained by propagating the per-term standard
errors used in the cycle.

Previous association proxy
--------------------------

For comparison, a previously computed PFOA association proxy was retained in
the results summary. This earlier value used:

G(R48 + PFOA + 47Cl) - G(R48 + 47Cl) - G(PFOA)

and gave:

-23.6640 +/- 2.6985 kcal/mol

That previous association proxy is not an exchange free energy because it does
not include the chloride-form resin endpoint and the aqueous chloride term. It
is reported only as a qualitative PFOA-resin association indicator. The
stoichiometrically correct exchange-cycle proxy described above is the
preferred MD/GBSA result for the present exchange discussion.

Mechanistic trajectory metrics
------------------------------

Additional structural metrics were extracted from the 100-frame
PFOA-associated resin trajectory. These metrics were used to describe the
character of the PFOA-associated ensemble, not to define a single microscopic
exchange pathway. PFOA was not treated as bound to one unique ammonium site.
Instead, the trajectory was analyzed for proximity to any resin ammonium
nitrogen and for hydrophobic tail contact with the organic resin.

The nearest distance between either PFOA carboxylate oxygen and any resin
ammonium nitrogen had:

mean 4.6510 A
SEM 0.0627 A
minimum 3.6107 A
maximum 6.2167 A

Frames with nearest carboxylate O--ammonium N distance below selected cutoffs:

<= 4.0 A: 15 of 100 frames
<= 5.0 A: 73 of 100 frames
<= 6.0 A: 96 of 100 frames

The nearest ammonium nitrogen varied across the trajectory. The most frequently
nearest resin ammonium atoms were:

N7, atom serial 155: 55 frames
N11, atom serial 199: 22 frames
N10, atom serial 188: 17 frames
N6, atom serial 144: 3 frames
N40, atom serial 596: 3 frames

Thus, the correct qualitative description is:

PFOA remained resin-associated and frequently sampled ammonium-proximal
configurations.

The PFOA fluorinated tail also remained in contact with the resin. The number
of resin heavy atoms within 4 A of the PFOA tail had:

mean 18.55
SEM 0.3418
minimum 10
maximum 27

The local chloride background remained mobile. Counting chlorides within a
radius of the nearest ammonium nitrogen to PFOA gave:

chlorides within 5 A:  mean 0.09, SEM 0.0288
chlorides within 7 A:  mean 0.27, SEM 0.0468
chlorides within 10 A: mean 0.82, SEM 0.0783

Water exposure around the fluorinated tail was estimated with cpptraj
watershell analysis. The number of waters within 5 A of the PFOA fluorinated
tail had:

mean 35.69
SEM 0.2926
minimum 28
maximum 44

Representative stripped PDB frames were extracted for visualization of PFOA
association, ammonium proximity, tail contact, and the mobile chloride
background:

work_amber/17_exchange_cycle_proxy/05_metrics/pfoa_assoc_frame_001_resin_pfo_cl.pdb
work_amber/17_exchange_cycle_proxy/05_metrics/pfoa_assoc_frame_025_resin_pfo_cl.pdb
work_amber/17_exchange_cycle_proxy/05_metrics/pfoa_assoc_frame_050_resin_pfo_cl.pdb
work_amber/17_exchange_cycle_proxy/05_metrics/pfoa_assoc_frame_075_resin_pfo_cl.pdb
work_amber/17_exchange_cycle_proxy/05_metrics/pfoa_assoc_frame_100_resin_pfo_cl.pdb

Limitations
-----------

The exchange-cycle value reported here should be interpreted as a qualitative
MD/GBSA proxy. It is not a rigorous standard-state exchange free energy. The
calculation uses finite explicit-solvent MD sampling followed by implicit GBSA
endpoint evaluation, does not include normal-mode entropy, does not perform an
alchemical transformation, and does not calculate a reversible potential of
mean force. The charged single-anion aqueous boxes also rely on PME's uniform
neutralizing background rather than explicit sodium counterions, in order to
preserve the requested exchange-cycle stoichiometry. The result is therefore
best used as qualitative support that PFOA exchange/association with the
quaternary ammonium resin environment is favorable under the adopted force-field
model.
