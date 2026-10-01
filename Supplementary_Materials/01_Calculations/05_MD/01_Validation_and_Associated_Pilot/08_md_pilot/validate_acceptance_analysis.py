#!/usr/bin/env python3

from pathlib import Path
import json
import math
import statistics

HERE = Path(__file__).resolve().parent
ANA = HERE / "analysis"
LOG = ANA / "cpptraj_acceptance.log"
XST = HERE / "05_acceptance_npt_1ns" / "acceptance1ns.xst"
AVG = ANA / "average_box.dat"
INPUT = HERE / "acceptance_analysis.generated.cpptraj"
EXPECTED_FRAMES = 500

FORBIDDEN = [
    "CHARMM version is >= 22 but an option other than 'shape' was specified",
    "Box is too skewed",
    "Disabling imaging due to problem with box",
    "Not using pair list due to problem with box",
    "Internal Error:",
    "Error: Could not initialize action",
    "Error: Trajectory",
    "Error: No frames",
]

SERIES = [
    "polymer_rmsd.dat",
    "polymer_rg.dat",
    "end_to_end.dat",
    "terminal1_central_mindist.dat",
    "terminal12_central_mindist.dat",
    "polymer_self_image.dat",
]


def numeric_rows(path):
    rows = []
    for line in Path(path).read_text(errors="replace").splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        vals = []
        ok = True
        for token in line.split():
            try:
                vals.append(float(token))
            except ValueError:
                ok = False
                break
        if ok and vals:
            rows.append(vals)
    return rows


def norm(v):
    return math.sqrt(sum(x * x for x in v))


def angle(u, v):
    c = sum(a * b for a, b in zip(u, v)) / (norm(u) * norm(v))
    c = max(-1.0, min(1.0, c))
    return math.degrees(math.acos(c))


def det3(a, b, c):
    return abs(
        a[0] * (b[1] * c[2] - b[2] * c[1])
        - a[1] * (b[0] * c[2] - b[2] * c[0])
        + a[2] * (b[0] * c[1] - b[1] * c[0])
    )


errors = []
log = LOG.read_text(errors="replace") if LOG.exists() else ""
if not log:
    errors.append("CPPTRAJ log is missing or empty")
for phrase in FORBIDDEN:
    if phrase in log:
        errors.append(f"CPPTRAJ log contains forbidden message: {phrase}")


if not INPUT.exists():
    errors.append("generated CPPTRAJ input is missing")
else:
    lines = [x.strip() for x in INPUT.read_text(errors="replace").splitlines() if x.strip() and not x.lstrip().startswith("#")]
    try:
        rms_i = next(i for i, x in enumerate(lines) if x.startswith("rms "))
    except StopIteration:
        errors.append("generated CPPTRAJ input has no RMS action")
        rms_i = -1
    for prefix in ("distance ", "mindist ", "minimage ", "avgbox ", "check "):
        for i, x in enumerate(lines):
            if x.startswith(prefix) and rms_i >= 0 and i > rms_i:
                errors.append(f"PBC-sensitive action occurs after RMS fitting: {x}")
    if not any(x.startswith("trajin ") and x.endswith(" shape") for x in lines):
        errors.append("generated CPPTRAJ input does not force `shape` DCD decoding")

for name in SERIES:
    path = ANA / name
    if not path.exists() or path.stat().st_size == 0:
        errors.append(f"missing or empty analysis output: {name}")
        continue
    n = len(numeric_rows(path))
    if n != EXPECTED_FRAMES:
        errors.append(f"{name}: expected {EXPECTED_FRAMES} numeric rows, found {n}")

xst_cells = []
if not XST.exists():
    errors.append("NAMD acceptance XST is missing")
else:
    for line in XST.read_text(errors="replace").splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        try:
            p = [float(x) for x in line.split()]
        except ValueError:
            continue
        if len(p) < 10:
            continue
        a, b, c = p[1:4], p[4:7], p[7:10]
        xst_cells.append({
            "a": a, "b": b, "c": c,
            "lengths": [norm(a), norm(b), norm(c)],
            "angles": [angle(b, c), angle(a, c), angle(a, b)],
            "volume": det3(a, b, c),
        })
if not xst_cells:
    errors.append("no periodic-cell records parsed from acceptance XST")

report = {"expected_cpptraj_frames": EXPECTED_FRAMES, "errors": errors}
if xst_cells:
    mean_vec = [[statistics.fmean(cell[key][i] for cell in xst_cells) for i in range(3)] for key in ("a", "b", "c")]
    mean_lengths = [statistics.fmean(cell["lengths"][i] for cell in xst_cells) for i in range(3)]
    mean_angles = [statistics.fmean(cell["angles"][i] for cell in xst_cells) for i in range(3)]
    mean_volume = statistics.fmean(cell["volume"] for cell in xst_cells)
    report["namd_xst"] = {
        "records": len(xst_cells),
        "mean_cell_vectors_A": mean_vec,
        "mean_lengths_A": mean_lengths,
        "mean_angles_deg": mean_angles,
        "mean_volume_A3": mean_volume,
    }
    for i, a in enumerate(mean_angles):
        if abs(a - 90.0) > 0.05:
            errors.append(f"NAMD XST mean cell angle {i} is {a:.6f} deg, not orthorhombic")

avg_rows = numeric_rows(AVG) if AVG.exists() else []
if len(avg_rows) != 1 or len(avg_rows[0]) < 10:
    errors.append("average_box.dat does not contain one Frame + 9-value cell row")
else:
    vals = avg_rows[0][-9:]
    cpp_mat = [vals[0:3], vals[3:6], vals[6:9]]
    report["cpptraj_avgbox_matrix_A"] = cpp_mat
    if xst_cells:
        max_delta = max(abs(cpp_mat[r][c] - mean_vec[r][c]) for r in range(3) for c in range(3))
        report["cpptraj_vs_xst_max_matrix_delta_A"] = max_delta
        if max_delta > 0.05:
            errors.append(f"CPPTRAJ average cell differs from mean NAMD XST cell by {max_delta:.6f} A (>0.05 A)")

report["errors"] = errors
report["status"] = "FAIL" if errors else "PASS"
(ANA / "box_validation.json").write_text(json.dumps(report, indent=2) + "\n")
lines = ["Stage 08 periodic-cell / CPPTRAJ validation", f"status: {report['status']}"]
if xst_cells:
    x = report["namd_xst"]
    lines.append("NAMD XST mean lengths (A): " + " x ".join(f"{v:.6f}" for v in x["mean_lengths_A"]))
    lines.append("NAMD XST mean angles (deg): " + ", ".join(f"{v:.6f}" for v in x["mean_angles_deg"]))
    lines.append(f"NAMD XST mean volume (A^3): {x['mean_volume_A3']:.6f}")
if "cpptraj_vs_xst_max_matrix_delta_A" in report:
    lines.append(f"CPPTRAJ avgbox vs XST max matrix delta (A): {report['cpptraj_vs_xst_max_matrix_delta_A']:.6f}")
lines.append("errors:")
lines.extend(["  - " + e for e in errors] or ["  - none"])
(ANA / "box_validation.txt").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
raise SystemExit(1 if errors else 0)
