#!/usr/bin/env python3
from __future__ import annotations

import csv
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GBSA = ROOT / "04_gbsa"


def total_column(path: Path) -> list[float]:
    with path.open() as handle:
        first = handle.readline()
        if first.strip().upper() != "GENERALIZED BORN:":
            handle.seek(0)
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or []
        total_name = next((f for f in fields if f.strip().upper().endswith("TOTAL")), None)
        if total_name is None:
            raise SystemExit(f"No TOTAL column found in {path}: {fields}")
        return [float(row[total_name]) for row in reader]


def stats(values: list[float]) -> tuple[float, float, int]:
    mean = sum(values) / len(values)
    if len(values) == 1:
        return mean, float("nan"), len(values)
    var = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
    sem = math.sqrt(var / len(values))
    return mean, sem, len(values)


terms = {
    "R48_PFOA_47Cl": total_column(GBSA / "r48_pfoa_47cl_gbsa_per_frame.csv"),
    "R48_48Cl": total_column(GBSA / "r48_48cl_gbsa_per_frame.csv"),
    "PFOA_aq": total_column(GBSA / "pfoa_aq_gbsa_per_frame.csv"),
    "Cl_aq": total_column(GBSA / "cl_aq_gbsa_per_frame.csv"),
}

summary = {}
for key, values in terms.items():
    summary[key] = stats(values)

exchange = []
for i, gpfoa in enumerate(terms["R48_PFOA_47Cl"]):
    gclresin = terms["R48_48Cl"][i % len(terms["R48_48Cl"])]
    gpfoaaq = terms["PFOA_aq"][i % len(terms["PFOA_aq"])]
    gclaq = terms["Cl_aq"][i % len(terms["Cl_aq"])]
    exchange.append(gpfoa - gclresin - gpfoaaq + gclaq)
summary["exchange_proxy"] = stats(exchange)

out = GBSA / "exchange_cycle_proxy_summary.csv"
with out.open("w", newline="") as handle:
    writer = csv.writer(handle)
    writer.writerow(["term", "mean_kcal_mol", "sem_kcal_mol", "n_frames"])
    for key, (mean, sem, n) in summary.items():
        writer.writerow([key, f"{mean:.6f}", f"{sem:.6f}", n])

detail = GBSA / "exchange_cycle_proxy_per_frame.csv"
with detail.open("w", newline="") as handle:
    writer = csv.writer(handle)
    writer.writerow(["frame", "exchange_proxy_kcal_mol"])
    for i, value in enumerate(exchange, start=1):
        writer.writerow([i, f"{value:.6f}"])

print(out.read_text())
