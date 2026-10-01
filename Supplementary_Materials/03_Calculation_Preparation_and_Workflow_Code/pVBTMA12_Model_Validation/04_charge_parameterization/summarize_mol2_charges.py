#!/usr/bin/env python3

from __future__ import annotations

import argparse
import decimal
import json
import math
from collections import Counter
from pathlib import Path


def decimal_places(token: str) -> int:
    token = token.strip().lower()
    if "e" in token:
        mantissa, exponent = token.split("e", 1)
        exp = int(exponent)
    else:
        mantissa, exp = token, 0
    if "." in mantissa:
        places = len(mantissa.split(".", 1)[1])
    else:
        places = 0
    return max(0, places - exp)


def parse_mol2(path: Path):
    lines = path.read_text(errors="replace").splitlines()
    inside = False
    rows = []
    for line in lines:
        if line.startswith("@<TRIPOS>ATOM"):
            inside = True
            continue
        if line.startswith("@<TRIPOS>") and inside:
            break
        if inside and line.strip():
            fields = line.split()
            if len(fields) < 9:
                continue
            token = fields[-1]
            try:
                charge = decimal.Decimal(token)
            except decimal.InvalidOperation:
                continue
            rows.append(
                {
                    "atom_id": int(fields[0]),
                    "atom_name": fields[1],
                    "atom_type": fields[5],
                    "charge_token": token,
                    "charge": charge,
                    "decimal_places": decimal_places(token),
                }
            )
    if not rows:
        raise SystemExit("No MOL2 atom charges parsed")
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mol2", type=Path)
    parser.add_argument("--target-charge", type=str, default="12")
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    decimal.getcontext().prec = 40
    target = decimal.Decimal(args.target_charge)
    rows = parse_mol2(args.mol2)
    charge_sum = sum((row["charge"] for row in rows), decimal.Decimal(0))
    delta = charge_sum - target


    displayed_precision_bound = sum(
        (
            decimal.Decimal("0.5")
            * (decimal.Decimal(10) ** (-row["decimal_places"]))
            for row in rows
        ),
        decimal.Decimal(0),
    )
    displayed_precision_can_explain = abs(delta) <= displayed_precision_bound
    precision_counts = Counter(row["decimal_places"] for row in rows)

    report = {
        "mol2": str(args.mol2),
        "atoms": len(rows),
        "target_charge": str(target),
        "serialized_charge_sum": str(charge_sum),
        "delta_from_target": str(delta),
        "charge_decimal_place_counts": {
            str(k): v for k, v in sorted(precision_counts.items())
        },
        "maximum_rounding_uncertainty_from_displayed_mol2_precision": str(
            displayed_precision_bound
        ),
        "displayed_mol2_serialization_rounding_can_explain_delta": (
            displayed_precision_can_explain
        ),
        "interpretation_limit": (
            "This test addresses only rounding implied by the charge precision "
            "displayed in the final MOL2. It cannot exclude earlier rounding or "
            "other numerical changes before serialization."
        ),
        "within_0p01": abs(delta) <= decimal.Decimal("0.01"),
        "within_0p001": abs(delta) <= decimal.Decimal("0.001"),
        "minimum_atomic_charge": str(min(row["charge"] for row in rows)),
        "maximum_atomic_charge": str(max(row["charge"] for row in rows)),
        "largest_absolute_charges": [
            {
                "atom_id": row["atom_id"],
                "atom_name": row["atom_name"],
                "atom_type": row["atom_type"],
                "charge": str(row["charge"]),
            }
            for row in sorted(rows, key=lambda x: abs(x["charge"]), reverse=True)[:20]
        ],
    }

    text = json.dumps(report, indent=2, sort_keys=True)
    print(text)
    if args.json_out:
        args.json_out.write_text(text + "\n")

    if not math.isfinite(float(charge_sum)):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
