#!/usr/bin/env python3
from pathlib import Path
import argparse,re
ap=argparse.ArgumentParser(); ap.add_argument('frcmod'); a=ap.parse_args(); text=Path(a.frcmod).read_text(errors='replace')
flags=[line for line in text.splitlines() if re.search(r'ATTN|NEED\s+REVISION|UNKNOWN|MISSING',line,re.I)]
print(f'bytes: {len(text.encode())}'); print(f'flagged_lines: {len(flags)}')
for x in flags: print(x)
raise SystemExit(2 if flags else 0)
