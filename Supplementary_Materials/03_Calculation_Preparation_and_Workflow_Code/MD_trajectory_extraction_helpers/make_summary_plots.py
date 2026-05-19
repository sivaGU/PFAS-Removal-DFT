#!/usr/bin/env python3
from __future__ import annotations

import csv
from pathlib import Path

try:
    import matplotlib.pyplot as plt
except Exception as exc:
    raise SystemExit(f"matplotlib is not available: {exc}")


def load_csv(path: Path):
    with path.open() as handle:
        rows = list(csv.DictReader(handle))
    return rows


def column(rows, name):
    return [float(row[name]) for row in rows]


def plot_series(bound, unbound, key, ylabel, outfile):
    plt.figure(figsize=(7, 4))
    plt.plot(column(bound, "frame"), column(bound, key), label="bound")
    plt.plot(column(unbound, "frame"), column(unbound, key), label="unbound")
    plt.xlabel("Frame")
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outfile, dpi=200)
    plt.close()


def main():
    bound = load_csv(Path("bound_complex_metrics.csv"))
    unbound = load_csv(Path("unbound_complex_metrics.csv"))
    plot_series(
        bound,
        unbound,
        "pfoa_carboxylate_site_distance_A",
        "PFOA carboxylate-site distance (A)",
        "pfoa_carboxylate_site_distance.png",
    )
    plot_series(
        bound,
        unbound,
        "nearest_chloride_site_distance_A",
        "Nearest chloride-site distance (A)",
        "nearest_chloride_site_distance.png",
    )
    plot_series(
        bound,
        unbound,
        "pfoa_tail_resin_heavy_contacts_4A",
        "Tail-resin heavy atom contacts <=4 A",
        "pfoa_tail_resin_contacts.png",
    )


if __name__ == "__main__":
    main()
