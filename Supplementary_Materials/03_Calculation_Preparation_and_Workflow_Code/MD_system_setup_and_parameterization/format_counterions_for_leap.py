from pathlib import Path

inp = Path("chol48_counterions_raw.pdb")
out = Path("chol48_counterions_amber.pdb")
report = Path("../logs/counterion_format_report.txt")

records = []
counts = {}


def atom_element(line):
    elem = line[76:78].strip()
    if elem:
        return elem.upper()
    atom = line[12:16].strip().upper()
    if atom.startswith("CL"):
        return "CL"
    if atom.startswith("NA"):
        return "NA"
    if atom.startswith("K"):
        return "K"
    if atom.startswith("BR"):
        return "BR"
    if atom.startswith("I"):
        return "I"
    return atom[:2].upper()


def amber_ion_name(elem):
    if elem == "CL":
        return ("Cl-", "Cl-")
    if elem == "NA":
        return ("Na+", "Na+")
    if elem == "K":
        return ("K+", "K+")
    if elem == "BR":
        return ("Br-", "Br-")
    if elem == "I":
        return ("I-", "I-")
    return (elem, elem)


serial = 1
resid = 1
for line in inp.read_text().splitlines():
    if line[:6].strip() not in {"ATOM", "HETATM"}:
        continue
    elem = atom_element(line)
    atom, res = amber_ion_name(elem)
    counts[res] = counts.get(res, 0) + 1
    x = float(line[30:38])
    y = float(line[38:46])
    z = float(line[46:54])
    records.append(
        f"HETATM{serial:5d} {atom:^4s} {res:>3s} A{resid:4d}"
        f"    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          {elem:>2s}"
    )
    serial += 1
    resid += 1

out.write_text("\n".join(records) + "\nEND\n")
lines = ["counterion counts after AMBER renaming:"]
for k in sorted(counts):
    lines.append(f"  {k}: {counts[k]}")
report.write_text("\n".join(lines) + "\n")
print("\n".join(lines))
