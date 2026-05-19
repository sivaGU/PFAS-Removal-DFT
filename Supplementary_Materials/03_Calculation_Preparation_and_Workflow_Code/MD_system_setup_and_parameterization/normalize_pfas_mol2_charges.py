from pathlib import Path


TARGETS = {
    "pfoa_gaff2.mol2": -1.0,
    "pfos_gaff2.mol2": -1.0,
}


def atom_section_bounds(lines):
    start = None
    end = None
    for i, line in enumerate(lines):
        if line.startswith("@<TRIPOS>ATOM"):
            start = i + 1
            continue
        if start is not None and line.startswith("@<TRIPOS>"):
            end = i
            break
    if start is None:
        raise ValueError("missing ATOM section")
    if end is None:
        end = len(lines)
    return start, end


def parse_atom(line):
    parts = line.split()
    if len(parts) < 9:
        raise ValueError(f"cannot parse mol2 atom line: {line!r}")
    return parts


def format_atom(parts):
    return (
        f"{int(parts[0]):7d} {parts[1]:<8s}"
        f"{float(parts[2]):10.4f}{float(parts[3]):11.4f}{float(parts[4]):10.4f} "
        f"{parts[5]:<8s}{int(parts[6]):5d} {parts[7]:<8s}{float(parts[8]):11.6f}"
    )


reports = []
for filename, target in TARGETS.items():
    path = Path(filename)
    raw = path.with_name(path.stem + "_raw_bcc.mol2")
    if not raw.exists():
        raw.write_text(path.read_text())

    lines = path.read_text().splitlines()
    start, end = atom_section_bounds(lines)
    atom_rows = []
    for idx in range(start, end):
        parts = parse_atom(lines[idx])
        atom_rows.append((idx, parts))

    current = sum(float(parts[8]) for _, parts in atom_rows)
    fluorines = [(idx, parts) for idx, parts in atom_rows if parts[5].lower() == "f"]
    if not fluorines:
        raise SystemExit(f"{filename}: no fluorine atoms available for residual charge correction")

    residual = target - current
    increment = residual / len(fluorines)
    for _, parts in fluorines:
        parts[8] = f"{float(parts[8]) + increment:.6f}"

    rounded = sum(float(parts[8]) for _, parts in atom_rows)
    final_residual = target - rounded
    first_f = fluorines[0][1]
    first_f[8] = f"{float(first_f[8]) + final_residual:.6f}"

    final = sum(float(parts[8]) for _, parts in atom_rows)
    for idx, parts in atom_rows:
        lines[idx] = format_atom(parts)
    path.write_text("\n".join(lines) + "\n")

    reports.append(
        f"{filename}: raw {current:.6f}, target {target:.6f}, "
        f"corrected {final:.6f}, fluorines adjusted {len(fluorines)}"
    )

Path("../logs/pfas_charge_normalization.txt").write_text("\n".join(reports) + "\n")
print("\n".join(reports))
