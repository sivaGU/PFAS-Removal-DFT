"""EDA plotting"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from PIL import Image

from .figure_style import BLUE, EXPORT_DPI, FONT, GREEN, ORANGE
from .annotation_style import VALUE_SIZE_PT, VALUE_BOX, energy_grid

PFAS_ORDER = ("FHEA", "PFHxA", "PFOA", "PFOS")
PFAS_LABELS = {"FHEA": r"6:2 FTCA$^-$", "PFHxA": r"PFHxA$^-$",
               "PFOA": r"PFOA$^-$", "PFOS": r"PFOS$^-$"}
STEP_ORDER = (("Pauli Energy", "Pauli"),
              ("Electrostatic Energy", "Elstat"),
              ("Orbital Energy", "Orb"),
              ("Delta Dispersion", "Disp"),
              ("Delta E^0(XC)", "XC"),
              ("Delta gCP correction", "gCP"),
              ("Delta CPCM Dielectric", "CPCM"))
SERIES = {
    "BTMA_r2SCAN-3c": (r"BTMA⁺, r$^2$SCAN-3c", BLUE, "o", "r²SCAN-3c"),
    "BTMA_wB97X-D3": (r"BTMA⁺, $\omega$B97X-D3", ORANGE, "s", "ωB97X-D3"),
    "DVB_BTMA_Water": (r"DVB-BTMA⁺, water", GREEN, "D", "DVB-BTMA⁺"),
}


def load_eda(path: Path, pair: tuple[str, str]) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"PFAS", "series", "Bond Energy", "Sum of listed terms",
                *(name for name, _ in STEP_ORDER)}
    if required - set(df):
        raise ValueError(f"EDA data missing columns: {sorted(required - set(df))}")
    expected = {(pfas, series) for pfas in PFAS_ORDER for series in pair}
    if set(zip(df.PFAS, df.series)) != expected or len(df) != len(expected):
        raise ValueError("Expected one row per PFAS and selected method/model")
    numeric = df[list(required - {"PFAS", "series"})].to_numpy(float)
    if not np.isfinite(numeric).all():
        raise ValueError("EDA values contain nonfinite entries")
    listed = df[[name for name, _ in STEP_ORDER]].sum(axis=1)
    if np.max(np.abs(listed - df["Sum of listed terms"])) > .001:
        raise ValueError("Printed components do not match the separately labeled sum")
    if "Preparation Energy" in df.columns:
        raise ValueError("Unvalidated Preparation Energy must not appear in plotting data")
    return df.set_index(["PFAS", "series"])


def levels(row, steps):
    return np.r_[0., np.cumsum([float(row[name]) for name, _ in steps])]


def plot_eda(data_path: Path, pair: tuple[str, str], include_gcp: bool,
             output_path: Path, dpi: int = EXPORT_DPI, legend_labels=None):
    data = load_eda(data_path, pair)
    steps = STEP_ORDER if include_gcp else tuple(s for s in STEP_ORDER if s[1] != "gCP")
    all_levels = np.concatenate([np.r_[levels(data.loc[pfas, series], steps),
                                        float(data.loc[pfas, series]["Bond Energy"])]
                                 for pfas in PFAS_ORDER for series in pair])
    lo = 10 * np.floor((float(all_levels.min()) - 12) / 10)
    hi = 10 * np.ceil((float(all_levels.max()) + 12) / 10)

    plt.rcParams.update({"font.family": FONT, "font.size": 8.6,
                         "axes.linewidth": .75, "savefig.facecolor": "white"})
    fig, axes = plt.subplots(4, 1, figsize=(6.5, 8.75), sharex=True, sharey=True)
    fig.subplots_adjust(left=.20, right=.88, top=.905, bottom=.075, hspace=.77)
    n = len(steps)
    xvals = np.arange(n + 1)
    for idx, pfas in enumerate(PFAS_ORDER):
        ax = axes[idx]
        ax.set_xlim(-.14, n + 1.36)
        ax.set_ylim(lo, hi)
        ax.set_title(PFAS_LABELS[pfas], fontsize=10.4, pad=5)
        letter = ax.text(-.075, 1.27, chr(65 + idx), transform=ax.transAxes,
                         ha="right", va="top", fontsize=11.5,
                         fontweight="bold", clip_on=False)
        energy_grid(ax)
        ax.set_ylabel("Energy (kcal/mol)", fontsize=9, labelpad=5)
        ax.tick_params(axis="y", labelsize=9.5, length=2.5, pad=3)
        ax.tick_params(axis="x", labelsize=9.2, length=2.5, pad=3)
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)

        for rank, series in enumerate(pair):
            _, color, marker, _ = SERIES[series]
            row = data.loc[pfas, series]
            y = levels(row, steps)
            x_shift = -.035 if rank == 0 else .035
            for step in range(n):
                ax.plot([xvals[step] + x_shift, xvals[step + 1] + x_shift],
                        [y[step], y[step]], color=color, linewidth=1.45)
                ax.plot([xvals[step + 1] + x_shift] * 2,
                        [y[step], y[step + 1]], color=color, linewidth=1.45)
            bond = float(row["Bond Energy"])
            closure_style = (0, (3.5, 2.6))
            ax.plot([n + x_shift, n + 1 + x_shift],
                    [y[-1], y[-1]], color=color, linewidth=1.45,
                    linestyle=closure_style)
            ax.plot([n + 1 + x_shift] * 2,
                    [y[-1], bond], color=color, linewidth=1.45,
                    linestyle=closure_style)
            final_x = n + 1.17 + (-.09 if rank == 0 else .09)
            ax.plot([n + 1 + x_shift, final_x], [bond, bond],
                    color=color, linewidth=1.45, linestyle=closure_style)
            ax.scatter([final_x], [bond], marker=marker, color=color,
                       s=27, zorder=4)
            first_higher = (float(data.loc[pfas, pair[0]]["Bond Energy"]) >=
                            float(data.loc[pfas, pair[1]]["Bond Energy"]))
            label_offset = (8 if rank == 0 else -8) * (1 if first_higher else -1)
            ax.annotate(f"{bond:+.2f}", (n + 1.30, bond),
                        xycoords="data", xytext=(3, label_offset),
                        textcoords="offset points", ha="left", va="center",
                        fontsize=VALUE_SIZE_PT, fontweight="bold", color=color,
                        bbox=VALUE_BOX, zorder=6,
                        clip_on=False)
        ax.set_xticks(np.arange(1, n + 2),
                      [short for _, short in steps] + [r"$E_{\rm bond}$"])
        ax.tick_params(axis="x", labelbottom=True)

    legend = [Line2D([], [], color=SERIES[s][1], marker=SERIES[s][2],
                     markersize=4.8, linewidth=1.3) for s in pair]
    if legend_labels is not None and len(legend_labels) != len(pair):
        raise ValueError('One legend label is required per EDA series')
    fig.legend(legend, legend_labels if legend_labels is not None else [SERIES[s][0] for s in pair],
               loc="upper center", bbox_to_anchor=(.52, .985), ncol=2,
               frameon=False, fontsize=10)

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    pt = lambda pixels: pixels * 72 / fig.dpi
    letters = [ax.texts[0].get_window_extent(renderer) for ax in axes]
    max_align = max(pt(abs(letters[0].x1 - box.x1)) for box in letters[1:])
    if max_align > .25:
        raise ValueError(f"EDA panel letter misalignment: {max_align:.2f} pt")
    marker_center_gap = pt(axes[0].transData.transform((n + 1.26, 0))[0]
                           - axes[0].transData.transform((n + 1.08, 0))[0])
    if marker_center_gap < 6.5:
        raise ValueError(f"EDA endpoint markers too close: {marker_center_gap:.1f} pt")
    print(f"EDA endpoint marker centers: {marker_center_gap:.1f} pt apart horizontally")
    for idx, (ax, box) in enumerate(zip(axes, letters)):
        xgap, ygap = pt(ax.bbox.x0 - box.x1), pt(box.y0 - ax.bbox.y1)
        if xgap < 3 or ygap < 3:
            raise ValueError(f"EDA panel {idx+1} letter too close to axis: {xgap:.1f}/{ygap:.1f} pt")
        tick_boxes = [t.get_window_extent(renderer) for t in ax.get_xticklabels() if t.get_visible()]
        tick_gap = min(pt(b.x0 - a.x1) for a, b in zip(tick_boxes, tick_boxes[1:]))
        if tick_gap < 8:
            raise ValueError(f"EDA panel {idx+1} term labels collide: {tick_gap:.1f} pt")
        labels = [t.get_window_extent(renderer) for t in ax.texts[1:]]
        label_gap = max(labels[0].y0 - labels[1].y1,
                        labels[1].y0 - labels[0].y1)
        if pt(label_gap) < 3:
            raise ValueError(f"EDA panel {idx+1} endpoint labels collide: {pt(label_gap):.1f} pt")
        if any(b.x1 > fig.bbox.x1 - 5 for b in labels):
            raise ValueError(f"EDA panel {idx+1} endpoint label exceeds canvas")
        if idx:
            previous = axes[idx-1]
            lower_text = [t.get_window_extent(renderer) for t in previous.get_xticklabels() if t.get_visible()]
            gap = pt(min(b.y0 for b in lower_text) -
                     max(ax.title.get_window_extent(renderer).y1, box.y1))
            if gap < 9:
                raise ValueError(f"EDA panels {idx}/{idx+1} text gap: {gap:.1f} pt")
        print(f"panel {chr(65+idx)}: letter {xgap:.1f} pt left, {ygap:.1f} pt above; "
              f"min x-label gap {tick_gap:.1f} pt; endpoint-label gap {pt(label_gap):.1f} pt")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(output_path.stem + ".rendering.png")
    fig.savefig(temporary, dpi=dpi, bbox_inches="tight", pad_inches=.035)
    plt.close(fig)
    with Image.open(temporary) as check:
        check.verify()
    with Image.open(temporary) as check:
        check.load()
    temporary.replace(output_path)
    print(f"Wrote {output_path}")
