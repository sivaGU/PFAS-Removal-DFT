
#!/usr/bin/env python3
from pathlib import Path
import hashlib
R=Path(__file__).resolve().parent; issues=[]
for line in (R/"PACKAGE_SHA256SUMS.txt").read_text().splitlines():
    if not line.strip(): continue
    h,n=line.split(None,1); p=R/n.strip()
    if not p.is_file(): issues.append(f"missing {n.strip()}")
    elif hashlib.sha256(p.read_bytes()).hexdigest()!=h: issues.append(f"hash mismatch {n.strip()}")
print("Stage 11 static package validation: "+("FAIL" if issues else "PASS")); [print(" - "+x) for x in issues]; raise SystemExit(1 if issues else 0)
