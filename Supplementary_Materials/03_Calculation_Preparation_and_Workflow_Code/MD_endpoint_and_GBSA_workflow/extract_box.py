import parmed as pmd
from pathlib import Path

top = Path("../04_leap/solvated_resin_cl.prmtop")
crd = Path("../04_leap/solvated_resin_cl.inpcrd")

parm = pmd.load_file(str(top), str(crd))
if parm.box is None:
    raise SystemExit("ERROR: no periodic box found in solvated_resin_cl.inpcrd")

a, b, c, alpha, beta, gamma = parm.box
print(f"BOX_X {a:.8f}")
print(f"BOX_Y {b:.8f}")
print(f"BOX_Z {c:.8f}")
print(f"ALPHA {alpha:.8f}")
print(f"BETA {beta:.8f}")
print(f"GAMMA {gamma:.8f}")

if abs(alpha - 90.0) > 1e-4 or abs(beta - 90.0) > 1e-4 or abs(gamma - 90.0) > 1e-4:
    raise SystemExit("ERROR: non-orthorhombic box detected. Stop and ask user before writing NAMD config.")
