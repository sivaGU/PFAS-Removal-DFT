#!/usr/bin/env python
"""
Analyze ORCA relaxed 2D surface scan and plot PES contours/surface.
Input is the *.relaxscanact.dat table written by ORCA.
Outputs: summary to console, CSV grid, contour plot, surface plot, line-cuts plot.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

EH_TO_KCAL = 627.509474
EH_TO_KJ = 2625.49962


@dataclass
class ScanData:
    r1: np.ndarray
    r2: np.ndarray
    energy_eh: np.ndarray
    energy_rel_kcal: np.ndarray
    grid_e_rel: np.ndarray
    mesh_r1: np.ndarray
    mesh_r2: np.ndarray
    emin_eh: float
    emin_kcal: float
    r1_min: float
    r2_min: float


def load_scan(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=r"\s+", header=None, names=["r1", "r2", "E_Eh"])
    if df.shape[1] != 3:
        raise ValueError(f"Expected 3 columns in {path}, got {df.shape[1]}")
    return df


def reshape_grid(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    r1_unique = np.sort(df["r1"].unique())
    r2_unique = np.sort(df["r2"].unique())
    # Build grid with r1 along rows, r2 along columns for consistent orientation.
    grid = np.full((r1_unique.size, r2_unique.size), np.nan)
    for _, row in df.iterrows():
        i = np.searchsorted(r1_unique, row.r1)
        j = np.searchsorted(r2_unique, row.r2)
        grid[i, j] = row.E_Eh
    if np.isnan(grid).any():
        raise ValueError("Grid contains NaNs; scan data may be incomplete.")
    mesh_r2, mesh_r1 = np.meshgrid(r2_unique, r1_unique)  # note order for plotting
    return mesh_r1, mesh_r2, grid


def compute_scan(path: Path) -> ScanData:
    df = load_scan(path)
    mesh_r1, mesh_r2, grid_eh = reshape_grid(df)

    emin_eh = grid_eh.min()
    grid_rel_kcal = (grid_eh - emin_eh) * EH_TO_KCAL

    min_idx = np.unravel_index(np.argmin(grid_eh), grid_eh.shape)
    r1_unique = np.sort(df["r1"].unique())
    r2_unique = np.sort(df["r2"].unique())
    r1_min = r1_unique[min_idx[0]]
    r2_min = r2_unique[min_idx[1]]

    return ScanData(
        r1=r1_unique,
        r2=r2_unique,
        energy_eh=df["E_Eh"].values,
        energy_rel_kcal=(df["E_Eh"] - emin_eh) * EH_TO_KCAL,
        grid_e_rel=grid_rel_kcal,
        mesh_r1=mesh_r1,
        mesh_r2=mesh_r2,
        emin_eh=emin_eh,
        emin_kcal=0.0,
        r1_min=r1_min,
        r2_min=r2_min,
    )


def save_grid_csv(scan: ScanData, out_csv: Path) -> None:
    df = pd.DataFrame(scan.grid_e_rel, index=scan.r1, columns=scan.r2)
    df.index.name = "r1"
    df.to_csv(out_csv, float_format="%.6f")


def _contour_levels(max_rel: float) -> np.ndarray:
    # Make visually nice levels every ~1 kcal up to rounded max.
    upper = max(5.0, np.ceil(max_rel))
    step = 1.0 if upper <= 25 else 2.0
    return np.arange(0.0, upper + step, step)


def plot_contour(scan: ScanData, out_path: Path) -> None:
    levels = _contour_levels(np.max(scan.grid_e_rel))
    fig, ax = plt.subplots(figsize=(6.0, 4.8))
    cs = ax.contourf(scan.mesh_r1, scan.mesh_r2, scan.grid_e_rel, levels=levels, cmap="viridis")
    ax.contour(scan.mesh_r1, scan.mesh_r2, scan.grid_e_rel, levels=levels, colors="k", linewidths=0.3, alpha=0.5)
    ax.plot(scan.r1_min, scan.r2_min, "r*", markersize=10, label="Global minimum")
    cbar = fig.colorbar(cs, ax=ax, label="ΔE (kcal/mol)")
    ax.set_xlabel("r1 (Å)")
    ax.set_ylabel("r2 (Å)")
    ax.legend(loc="upper right")
    ax.set_title("PES contour (ΔE vs r1, r2)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_surface(scan: ScanData, out_path: Path) -> None:
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    fig = plt.figure(figsize=(6.5, 5.0))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(
        scan.mesh_r1,
        scan.mesh_r2,
        scan.grid_e_rel,
        cmap="viridis",
        linewidth=0,
        antialiased=True,
        rstride=1,
        cstride=1,
    )
    ax.set_xlabel("r1 (Å)")
    ax.set_ylabel("r2 (Å)")
    ax.set_zlabel("ΔE (kcal/mol)")
    fig.colorbar(surf, shrink=0.6, aspect=10, label="ΔE (kcal/mol)")
    ax.set_title("PES surface")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_linecuts(scan: ScanData, out_path: Path) -> None:
    # Lines at r1 closest to minimum and r2 closest to minimum.
    i_r1 = np.searchsorted(scan.r1, scan.r1_min)
    j_r2 = np.searchsorted(scan.r2, scan.r2_min)

    line_r2 = scan.grid_e_rel[i_r1, :]
    line_r1 = scan.grid_e_rel[:, j_r2]

    fig, ax = plt.subplots(1, 2, figsize=(9.5, 3.8), sharey=True)

    ax[0].plot(scan.r2, line_r2, "o-")
    ax[0].set_xlabel("r2 (Å)")
    ax[0].set_ylabel("ΔE (kcal/mol)")
    ax[0].set_title(f"Cut at r1 = {scan.r1_min:.3f} Å")

    ax[1].plot(scan.r1, line_r1, "o-")
    ax[1].set_xlabel("r1 (Å)")
    ax[1].set_title(f"Cut at r2 = {scan.r2_min:.3f} Å")

    fig.suptitle("1D cuts through PES")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def summarize(scan: ScanData) -> str:
    span = np.max(scan.grid_e_rel)
    return (
        f"Points: {scan.r1.size} (r1) × {scan.r2.size} (r2) = {scan.r1.size * scan.r2.size}\n"
        f"r1 range: {scan.r1.min():.3f}–{scan.r1.max():.3f} Å; r2 range: {scan.r2.min():.3f}–{scan.r2.max():.3f} Å\n"
        f"Global minimum: E = {scan.emin_eh:.9f} Eh at r1 = {scan.r1_min:.6f} Å, r2 = {scan.r2_min:.6f} Å\n"
        f"Energy span (Delta E): 0 -> {span:.2f} kcal/mol"
    )


def main() -> None:
    here = Path(__file__).resolve().parent
    default_input = here / "R4N+PFOA-_PES" / "out" / "R4N+PFOA-_0.15M.relaxscanact.dat"

    parser = argparse.ArgumentParser(description="Analyze ORCA 2D PES scan and plot results.")
    parser.add_argument("-i", "--input", type=Path, default=default_input, help="Path to relaxscanact.dat")
    parser.add_argument("-o", "--output-dir", type=Path, default=here / "plots", help="Directory to write plots/CSV")
    parser.add_argument("-t", "--tag", type=str, default=None, help="Tag/prefix for output filenames")
    args = parser.parse_args()

    input_path = args.input
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    tag = args.tag if args.tag else input_path.stem.replace(".relaxscanact", "")

    scan = compute_scan(input_path)

    print(summarize(scan))

    save_grid_csv(scan, output_dir / f"{tag}.grid.csv")
    plot_contour(scan, output_dir / f"{tag}.contour.png")
    plot_surface(scan, output_dir / f"{tag}.surface.png")
    plot_linecuts(scan, output_dir / f"{tag}.linecuts.png")


if __name__ == "__main__":
    main()
