from pathlib import Path

import parmed as pmd


SYSTEMS = {
    "pfoa": {
        "name": "resin_pfoa_exchange",
        "namd_dir": Path("../09_namd_pfoa_exchange"),
        "top": Path("../05_pfas_exchange/solvated_resin_pfoa_exchange.prmtop"),
        "crd": Path("../05_pfas_exchange/solvated_resin_pfoa_exchange.inpcrd"),
    },
    "pfos": {
        "name": "resin_pfos_exchange",
        "namd_dir": Path("../10_namd_pfos_exchange"),
        "top": Path("../05_pfas_exchange/solvated_resin_pfos_exchange.prmtop"),
        "crd": Path("../05_pfas_exchange/solvated_resin_pfos_exchange.inpcrd"),
    },
}


MIN_TEMPLATE = """# {title} minimization
# No resin or PFAS shape restraints are used.

amber             yes
parmfile          {top}
ambercoor         {crd}

outputname        {name}_min
binaryoutput      yes

exclude           scaled1-4
readexclusions    yes
1-4scaling        0.833333333
scnb              2.0

switching         off
cutoff            9.0
pairlistdist      11.0
LJcorrection      on

PME               yes
PMEGridSpacing    1.0

cellBasisVector1  {box_x:.8f} 0.0 0.0
cellBasisVector2  0.0 {box_y:.8f} 0.0
cellBasisVector3  0.0 0.0 {box_z:.8f}
cellOrigin        0.0 0.0 0.0

temperature       310.15

rigidBonds        all
rigidTolerance    1.0e-8
timestep          2.0

outputEnergies    100
outputTiming      100
restartfreq       1000
dcdfreq           1000
xstFreq           1000

minimize          10000
"""


NVT_TEMPLATE = """# {title} NVT heating
# No resin or PFAS shape restraints are used.

amber             yes
parmfile          {top}
ambercoor         {crd}

binCoordinates    {name}_min.restart.coor

outputname        {name}_nvt_heat
binaryoutput      yes

exclude           scaled1-4
readexclusions    yes
1-4scaling        0.833333333
scnb              2.0

switching         off
cutoff            9.0
pairlistdist      11.0
LJcorrection      on

PME               yes
PMEGridSpacing    1.0

cellBasisVector1  {box_x:.8f} 0.0 0.0
cellBasisVector2  0.0 {box_y:.8f} 0.0
cellBasisVector3  0.0 0.0 {box_z:.8f}
cellOrigin        0.0 0.0 0.0

temperature       310.15

rigidBonds        all
rigidTolerance    1.0e-8
timestep          2.0

langevin          on
langevinTemp      310.15
langevinDamping   1.0

outputEnergies    500
outputTiming      500
restartfreq       2500
dcdfreq           2500
xstFreq           2500

reinitvels         50.0
run                25000

reinitvels         100.0
run                25000

reinitvels         150.0
run                25000

reinitvels         200.0
run                25000

reinitvels         250.0
run                25000

reinitvels         310.15
run                50000
"""


NPT_TEMPLATE = """# {title} short NPT equilibration
# No resin or PFAS shape restraints are used.

amber             yes
parmfile          {top}
ambercoor         {crd}

binCoordinates    {name}_nvt_heat.restart.coor
binVelocities     {name}_nvt_heat.restart.vel
extendedSystem    {name}_nvt_heat.restart.xsc

outputname        {name}_npt_short
binaryoutput      yes

exclude           scaled1-4
readexclusions    yes
1-4scaling        0.833333333
scnb              2.0

switching         off
cutoff            9.0
pairlistdist      11.0
LJcorrection      on

PME               yes
PMEGridSpacing    1.0

# Extra patch-grid margin for the initially roomy LEaP solvent box during NPT shrinkage.
margin            8.0

rigidBonds        all
rigidTolerance    1.0e-8
timestep          2.0

langevin          on
langevinTemp      310.15
langevinDamping   1.0

langevinPiston              on
langevinPistonTarget        1.01325
langevinPistonPeriod        100.0
langevinPistonDecay         50.0
langevinPistonTemp          310.15

useGroupPressure  yes
useFlexibleCell   no
useConstantArea   no

outputEnergies    500
outputTiming      500
restartfreq       2500
dcdfreq           2500
xstFreq           2500

run               250000
"""


logs = []
for key, spec in SYSTEMS.items():
    parm = pmd.load_file(str(spec["top"]), str(spec["crd"]))
    if parm.box is None:
        raise SystemExit(f"{key}: no periodic box found")
    a, b, c, alpha, beta, gamma = parm.box
    if any(abs(x - 90.0) > 1.0e-4 for x in (alpha, beta, gamma)):
        raise SystemExit(f"{key}: non-orthorhombic box detected: {parm.box}")

    spec["namd_dir"].mkdir(parents=True, exist_ok=True)
    values = {
        "title": f"{spec['name']} chloride-exchange",
        "name": spec["name"],
        "top": str(spec["top"]),
        "crd": str(spec["crd"]),
        "box_x": a,
        "box_y": b,
        "box_z": c,
    }
    (spec["namd_dir"] / "01_minimize.conf").write_text(MIN_TEMPLATE.format(**values))
    (spec["namd_dir"] / "02_nvt_heat.conf").write_text(NVT_TEMPLATE.format(**values))
    (spec["namd_dir"] / "03_npt_short.conf").write_text(NPT_TEMPLATE.format(**values))

    box_text = (
        f"BOX_X {a:.8f}\n"
        f"BOX_Y {b:.8f}\n"
        f"BOX_Z {c:.8f}\n"
        f"ALPHA {alpha:.8f}\n"
        f"BETA {beta:.8f}\n"
        f"GAMMA {gamma:.8f}\n"
    )
    (Path("../logs") / f"{key}_exchange_box.txt").write_text(box_text)
    logs.append(f"{key}: {a:.3f} {b:.3f} {c:.3f} A")

print("\n".join(logs))
