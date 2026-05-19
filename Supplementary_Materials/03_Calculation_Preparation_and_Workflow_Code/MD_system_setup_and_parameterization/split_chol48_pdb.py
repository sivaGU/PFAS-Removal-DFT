from pathlib import Path

inp = Path("../01_input/chol48_with_counterions.pdb")
resin_out = Path("chol48_resin_only.pdb")
ions_out = Path("chol48_counterions_raw.pdb")
waters_out = Path("chol48_waters_raw.pdb")
report_out = Path("../logs/split_report.txt")

water_resnames = {"HOH", "WAT", "TIP3", "TP3", "SOL"}
ion_elements = {"CL", "NA", "K", "BR", "I", "F", "MG", "CA"}
ion_resnames = {
    "CL",
    "CLA",
    "CL-",
    "NA",
    "NA+",
    "K",
    "K+",
    "BR",
    "BR-",
    "I",
    "I-",
    "MG",
    "MG2",
    "CA",
    "CA2",
}


def rec(line):
    return line[:6].strip()


def resname(line):
    return line[17:20].strip().upper()


def atomname(line):
    return line[12:16].strip().upper()


def element(line):
    e = line[76:78].strip().upper()
    if e:
        return e
    a = atomname(line)
    if a.startswith("CL"):
        return "CL"
    if a.startswith("NA"):
        return "NA"
    if a.startswith("BR"):
        return "BR"
    if a.startswith("MG"):
        return "MG"
    if a.startswith("CA"):
        return "CA"
    letters = "".join(c for c in a if c.isalpha())
    return letters[:2].upper()


resin = []
ions = []
waters = []
counts = {}

for line in inp.read_text().splitlines():
    if rec(line) not in {"ATOM", "HETATM"}:
        continue
    rn = resname(line)
    el = element(line)
    counts[el] = counts.get(el, 0) + 1
    if rn in water_resnames:
        waters.append(line)
    elif rn in ion_resnames or el in ion_elements and len(atomname(line).replace("+", "").replace("-", "")) <= 2:
        ions.append(line)
    else:
        resin.append(line)

resin_out.write_text("\n".join(resin) + "\nEND\n")
ions_out.write_text("\n".join(ions) + "\nEND\n")
waters_out.write_text("\n".join(waters) + "\nEND\n")

rep = []
rep.append(f"input: {inp}")
rep.append(f"resin atom records: {len(resin)}")
rep.append(f"ion atom records: {len(ions)}")
rep.append(f"water atom records: {len(waters)}")
rep.append("element-like counts:")
for k in sorted(counts):
    rep.append(f"  {k}: {counts[k]}")
report_out.write_text("\n".join(rep) + "\n")
print("\n".join(rep))
