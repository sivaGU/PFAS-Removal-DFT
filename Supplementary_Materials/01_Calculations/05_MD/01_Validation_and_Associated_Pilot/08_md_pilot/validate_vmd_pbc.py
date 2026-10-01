#!/usr/bin/env python3

from pathlib import Path
import json
import math
import re
import statistics

HERE = Path(__file__).resolve().parent
ANA = HERE / "analysis"
VMD_BOX = ANA / "vmd_box.dat"
VMD_LOG = ANA / "vmd_pbc.log"
XST = HERE / "05_acceptance_npt_1ns" / "acceptance1ns.xst"
CONF = HERE / "05_acceptance_npt_1ns.conf"
EXPECTED_FRAMES = 500
LEN_TOL_A = 0.05
ANG_TOL_DEG = 0.05


def fail(msg):
    raise SystemExit(f"ERROR: {msg}")


def norm(v):
    return math.sqrt(sum(x * x for x in v))


def angle(u, v):
    c = sum(a * b for a, b in zip(u, v)) / (norm(u) * norm(v))
    c = max(-1.0, min(1.0, c))
    return math.degrees(math.acos(c))


if not VMD_LOG.exists() or "VMD_PBC_STATUS PASS" not in VMD_LOG.read_text(errors="replace"):
    fail("VMD/PBCTools run did not report PASS")

m = re.search(r"^\s*DCDfreq\s+(\d+)\s*$", CONF.read_text(errors="replace"), re.M | re.I)
if not m:
    fail("Could not parse DCDfreq from acceptance configuration")
dcd_freq = int(m.group(1))

vmd = {}
for line in VMD_BOX.read_text(errors="replace").splitlines():
    if not line.strip() or line.lstrip().startswith("#"):
        continue
    p = line.split()
    if len(p) < 7:
        fail(f"Malformed VMD box row: {line}")
    f = int(p[0])
    vmd[f] = [float(x) for x in p[1:7]]
if len(vmd) != EXPECTED_FRAMES or set(vmd) != set(range(EXPECTED_FRAMES)):
    fail(f"Expected VMD cell data for frames 0-{EXPECTED_FRAMES-1}; found {len(vmd)} frames")

xst = []
for line in XST.read_text(errors="replace").splitlines():
    if not line.strip() or line.lstrip().startswith("#"):
        continue
    p = [float(x) for x in line.split()]
    if len(p) < 10:
        continue
    step = int(round(p[0]))
    a, b, c = p[1:4], p[4:7], p[7:10]
    xst.append((step, [norm(a), norm(b), norm(c), angle(b, c), angle(a, c), angle(a, b)]))
if not xst:
    fail("No XST cell records parsed")

errors = []
matched = []
for step, xb in xst:
    if step == 0:
        continue
    if step % dcd_freq != 0:
        errors.append(f"XST step {step} is not divisible by DCDfreq {dcd_freq}")
        continue

    frame = step // dcd_freq - 1
    if frame not in vmd:
        errors.append(f"XST step {step} maps to absent DCD frame {frame}")
        continue
    vb = vmd[frame]
    ld = max(abs(vb[i] - xb[i]) for i in range(3))
    ad = max(abs(vb[i] - xb[i]) for i in range(3, 6))
    matched.append((step, frame, ld, ad))
    if ld > LEN_TOL_A:
        errors.append(f"step {step}/frame {frame}: VMD DCD vs XST length delta {ld:.6f} A > {LEN_TOL_A} A")
    if ad > ANG_TOL_DEG:
        errors.append(f"step {step}/frame {frame}: VMD DCD vs XST angle delta {ad:.6f} deg > {ANG_TOL_DEG} deg")

for f, vals in vmd.items():
    if any(x <= 0 for x in vals[:3]):
        errors.append(f"VMD frame {f} has nonpositive box length: {vals[:3]}")
    if any(abs(a - 90.0) > ANG_TOL_DEG for a in vals[3:]):
        errors.append(f"VMD frame {f} is not orthorhombic within tolerance: {vals[3:]}")

report = {
    "status": "FAIL" if errors else "PASS",
    "vmd_frames": len(vmd),
    "dcd_freq_steps": dcd_freq,
    "xst_records_total": len(xst),
    "xst_records_compared": len(matched),
    "max_vmd_vs_xst_length_delta_A": max((r[2] for r in matched), default=None),
    "max_vmd_vs_xst_angle_delta_deg": max((r[3] for r in matched), default=None),
    "mean_vmd_lengths_A": [statistics.fmean(v[i] for v in vmd.values()) for i in range(3)],
    "mean_vmd_angles_deg": [statistics.fmean(v[i] for v in vmd.values()) for i in range(3, 6)],
    "errors": errors,
}
(ANA / "vmd_pbc_validation.json").write_text(json.dumps(report, indent=2) + "\n")
lines = [
    "Stage 08 VMD/PBCTools periodic-cell validation",
    f"status: {report['status']}",
    f"VMD DCD frames: {report['vmd_frames']}",
    f"XST records compared at matching DCD steps: {report['xst_records_compared']}",
    "mean VMD DCD cell lengths (A): " + " x ".join(f"{x:.6f}" for x in report["mean_vmd_lengths_A"]),
    "mean VMD DCD cell angles (deg): " + ", ".join(f"{x:.6f}" for x in report["mean_vmd_angles_deg"]),
]
if report["max_vmd_vs_xst_length_delta_A"] is not None:
    lines.append(f"max VMD DCD vs XST length delta (A): {report['max_vmd_vs_xst_length_delta_A']:.6f}")
    lines.append(f"max VMD DCD vs XST angle delta (deg): {report['max_vmd_vs_xst_angle_delta_deg']:.6f}")
lines.append("errors:")
lines.extend(["  - " + e for e in errors] or ["  - none"])
(ANA / "vmd_pbc_validation.txt").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
raise SystemExit(1 if errors else 0)
