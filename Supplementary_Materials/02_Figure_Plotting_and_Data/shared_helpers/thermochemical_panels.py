"""Thermochemical panels"""
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D
from .figure_style import BLUE, GREEN, GREY, ORANGE, PURPLE, readable_text_color
from .annotation_style import VALUE_SIZE_PT, VALUE_BOX, GRID_COLOR, GRID_WIDTH_PT, energy_grid

DATA_FILE = Path(__file__).resolve().parent.parent / "01_main_text_figures" / "Figure_05_BTMA_thermochemical_decomposition" / "input_data" / "thermochemical_decomposition.csv"
PFAS = ("FHEA", "PFHxA", "PFOA", "PFOS")
PFAS_LABELS = ("6:2\nFTCA", "PFHxA", "PFOA", "PFOS")
METHODS = ("r2SCAN-3c", "wB97X-D3")
METHOD_LABELS = (r"r$^2$SCAN-3c", r"$\omega$B97X-D3")
METHOD_COLORS = (BLUE, ORANGE)
MARKERS = ("o", "s")
SPECIES_ROWS = (
    "BTMA-PFAS complex", "BTMA-Cl complex", "Free PFAS anion", "Free Cl-",
)
SPECIES_LABELS = (
    r"BTMA$^+$P$^-$", r"BTMA$^+$Cl$^-$", r"P$^-$", r"Cl$^-$",
)
COMPONENTS = ("ZPE", "Thermal correction", "Entropy contribution")


def data_table():
    df = pd.read_csv(DATA_FILE)
    required = {"panel", "series", "category", "pfas", "value_kcal_mol"}
    if required - set(df.columns):
        raise ValueError(f"Missing columns: {sorted(required - set(df.columns))}")
    if df.duplicated(["panel", "series", "category", "pfas"]).any():
        raise ValueError("Duplicate thermochemical source record")
    if set(df.pfas) != set(PFAS):
        raise ValueError("Unexpected PFAS set in curated source")
    return df.set_index(["panel", "series", "category", "pfas"])["value_kcal_mol"]


def value(data, panel, category, pfas, series="functional_difference"):
    return float(data.loc[panel, series, category, pfas])


def panel_letter(ax, letter, x=-.12, y=1.12):
    ax.text(x, y, letter, transform=ax.transAxes, va="top", ha="right",
            fontsize=11.5, fontweight="bold", clip_on=False)


def energy_panel(ax, data, category, title, letter, limits):
    x = np.arange(4)
    pair = np.array([[value(data, "exchange", category, p, m) for p in PFAS]
                     for m in METHODS])
    for i in x:
        ax.plot([i, i], pair[:, i], color="#333333", linewidth=0.85, zorder=2)
    for values, marker, color in zip(pair, MARKERS, METHOD_COLORS):
        ax.scatter(x, values, s=23, marker=marker, color=color, zorder=3)
    for i in x:
        for rank, color in enumerate(METHOD_COLORS):
            datum = pair[rank, i]
            upper = datum >= pair[1-rank, i]
            ax.annotate(f"{datum:+.2f}", (i, datum),
                        xytext=(0, 8 if upper else -9),
                        textcoords="offset points", ha="center",
                        va="bottom" if upper else "top", color=color,
                        fontsize=VALUE_SIZE_PT, fontweight="bold",
                        bbox=VALUE_BOX, zorder=5)
    ax.set_xlim(-0.35, 3.35)
    ax.set_ylim(*limits)
    ax.set_title(title, fontsize=10.4, pad=7)
    ax.set_ylabel("Energy (kcal/mol)", fontsize=9, labelpad=7)
    ax.set_xticks(x, PFAS_LABELS, fontsize=9.2)
    ax.tick_params(axis="x", length=2.5, width=0.7, pad=3)
    ax.tick_params(axis="y", labelsize=9, length=2.5)
    energy_grid(ax)
    panel_letter(ax, letter)


def species_panel(ax, data):
    vals = np.array([[value(data, "species", s, p) for p in PFAS]
                     for s in SPECIES_ROWS])
    bound = float(np.max(np.abs(vals)))
    img = ax.imshow(vals, cmap="RdBu_r", norm=TwoSlopeNorm(vmin=-bound,
                    vcenter=0, vmax=bound), interpolation="none", aspect="auto")
    ax.set_xticks(range(4), PFAS_LABELS, fontsize=8)
    ax.set_yticks(range(4), SPECIES_LABELS, fontsize=8)
    ax.tick_params(axis="both", length=0)
    ax.set_title("Species Contributions (kcal/mol)", fontsize=10.5, pad=7)
    for row in range(4):
        for col in range(4):
            color = readable_text_color(img.cmap(img.norm(vals[row, col])))
            ax.text(col, row, f"{vals[row, col]:+.2f}", color=color, ha="center",
                    va="center", fontsize=VALUE_SIZE_PT, fontweight="bold")
    ax.set_xticks(np.arange(-.5, 4, 1), minor=True)
    ax.set_yticks(np.arange(-.5, 4, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.7)
    ax.tick_params(which="minor", bottom=False, left=False)
    panel_letter(ax, "A", x=-.13, y=1.27)
    return img


def component_panel(ax, data):
    y = np.arange(4)
    comps = [np.array([value(data, "component", c, p) for p in PFAS])
             for c in COMPONENTS]
    colors = ("#666666", GREEN, PURPLE)
    handles = []
    for offset, values, color, label in zip((-.18, 0, .18), comps, colors,
                                            ("ZPE", "Thermal", "Entropy")):
        bars = ax.barh(y + offset, values, height=.17, color=color, label=label)
        handles.append(bars[0])
    totals = np.array([value(data, "component_total", "Total", p) for p in PFAS])
    ax.scatter(totals, y, marker="D", s=25, color="black", zorder=4)
    for yi, total in zip(y, totals):
        ax.annotate(f"{total:+.2f}", (total, yi), xytext=(-5, 11),
                    textcoords="offset points", ha="right", va="bottom",
                    fontsize=VALUE_SIZE_PT, fontweight="bold", color="black",
                    bbox=VALUE_BOX, zorder=5)
    ax.set_xlim(-2.7, .75)
    ax.set_ylim(3.8, -.8)
    ax.set_yticks(y, PFAS_LABELS, fontsize=8)
    ax.tick_params(axis="x", labelsize=8)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("Energy contribution (kcal/mol)", fontsize=9, labelpad=5)
    ax.set_title("Component Contributions (kcal/mol)", fontsize=10.5, pad=7)
    energy_grid(ax, axis="x")
    ax.set_axisbelow(True)
    handles.append(Line2D([], [], marker="D", linestyle="None", color="black",
                          markersize=4.5))
    ax.legend([handles[i] for i in (0, 2, 1, 3)],
              ("ZPE", "Entropy", "Thermal", "Total"), loc="upper center",
              bbox_to_anchor=(.5, -.25), ncol=4, fontsize=9.2,
              frameon=False, columnspacing=1.5, handletextpad=.55)
    panel_letter(ax, "B", x=-.13, y=1.27)
