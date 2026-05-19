from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# =========================
# USER SETTINGS
# =========================
ROOT = Path(__file__).resolve().parent
TXT_PATH = ROOT / "eda_sections_main_out.txt"
OUTDIR = ROOT / "eda_figures"
OUTDIR.mkdir(exist_ok=True)

FONT_FAMILY = "DejaVu Sans"
BASE_FONT_SIZE = 13
VALUE_FONT_SIZE = 14.0
FIG_DPI = 300
XTICK_FONT_SIZE = 15
XTICK_FONT_SIZE_WITH_GCP = 14

PFAS_ORDER = ["FHEA", "PFHxA", "PFOA", "PFOS"]

plt.rcParams["font.family"] = FONT_FAMILY


SERIES_STYLES = {
    "BTMA_r2SCAN-3c": {
        "label": r"BTMA r$^2$SCAN-3c",
        "color": "#1f77b4",
        "marker": "o",
    },
    "BTMA_wB97X-D3": {
        "label": r"BTMA $\omega$B97X-D3",
        "color": "#ff7f0e",
        "marker": "s",
    },
    "ExtendedMonomer_Water": {
        "label": "Extended Monomer (Water)",
        "color": "#1b9e77",
        "marker": "D",
    },
    "ExtendedMonomer_Octanol72.5": {
        "label": r"Extended Monomer (Octanol, $\epsilon = 72.5$)",
        "color": "#7570b3",
        "marker": "^",
    },
}


STEP_ORDER = [
    ("Pauli Energy", "Pauli"),
    ("Preparation Energy", "Prep"),
    ("Electrostatic Energy", "Elstat"),
    ("Orbital Energy", "Orb"),
    ("Delta Dispersion", "Disp"),
    ("Delta E^0(XC)", "XC"),
    ("Delta gCP correction", "gCP"),
    ("Delta CPCM Dielectric", "Solv"),
]

STEP_ORDER_WITHOUT_GCP = [
    step for step in STEP_ORDER if step[0] != "Delta gCP correction"
]


def parse_eda_sections(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8", errors="replace")
    chunks = [chunk for chunk in text.split("FILE: ") if chunk.strip()]
    rows = []

    for chunk in chunks:
        lines = chunk.splitlines()
        file_path = lines[0].strip()
        block = "\n".join(lines[1:])
        values = {}

        for line in lines[1:]:
            match = re.match(r"\s*(.+?)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s*$", line)
            if not match:
                continue

            key = match.group(1).strip()
            kcal = float(match.group(3))

            if key in values:
                if np.isclose(values[key], kcal, atol=1e-8):
                    continue
            values[key] = kcal

        rows.append({"file": file_path, "block": block, "values": values})

    return rows


def classify_record(file_path: str) -> dict:
    species = next((pfas for pfas in PFAS_ORDER if pfas in file_path), None)

    if "\\BTMA\\r2\\" in file_path:
        series = "BTMA_r2SCAN-3c"
    elif "\\BTMA\\wb\\" in file_path:
        series = "BTMA_wB97X-D3"
    elif "\\ExtendedMonomer\\Water\\" in file_path:
        series = "ExtendedMonomer_Water"
    elif "\\ExtendedMonomer\\Octanol_72.5\\" in file_path:
        series = "ExtendedMonomer_Octanol72.5"
    else:
        series = "Unknown"

    return {"PFAS": species, "series": series}


def build_component_table(parsed_rows: list[dict]) -> pd.DataFrame:
    table_rows = []

    for row in parsed_rows:
        meta = classify_record(row["file"])
        if meta["PFAS"] is None or meta["series"] == "Unknown":
            continue

        vals = row["values"]
        bond = vals.get("Bond Energy")
        if bond is None:
            continue

        pauli = vals.get("Pauli Energy", 0.0)
        elstat = vals.get("Electrostatic Energy", 0.0)
        orb = vals.get("Orbital Energy", 0.0)
        disp = vals.get("Delta Dispersion", 0.0)
        xc = vals.get("Delta E^0(XC)", 0.0)
        gcp = vals.get("Delta gCP correction", 0.0)
        solv = vals.get("Delta CPCM Dielectric", 0.0)

        interaction_sum = pauli + elstat + orb + disp + xc + gcp + solv
        prep = bond - interaction_sum

        table_rows.append(
            {
                "file": row["file"],
                "PFAS": meta["PFAS"],
                "series": meta["series"],
                "Bond Energy": bond,
                "Preparation Energy": prep,
                "Pauli Energy": pauli,
                "Electrostatic Energy": elstat,
                "Orbital Energy": orb,
                "Delta Dispersion": disp,
                "Delta E^0(XC)": xc,
                "Delta gCP correction": gcp,
                "Delta CPCM Dielectric": solv,
            }
        )

    df = pd.DataFrame(table_rows)
    if df.empty:
        raise RuntimeError(f"No usable EDA rows parsed from {TXT_PATH}")
    df["PFAS"] = pd.Categorical(df["PFAS"], categories=PFAS_ORDER, ordered=True)
    df = df.sort_values(["PFAS", "series"]).reset_index(drop=True)
    return df


def build_levels(record: pd.Series, step_order: list[tuple[str, str]] | None = None) -> list[float]:
    step_order = step_order or STEP_ORDER
    levels = [0.0]
    current = 0.0
    for key, _ in step_order:
        current += float(record.get(key, 0.0))
        levels.append(current)
    return levels


def draw_comparison_ladders(
    ax,
    panel_df: pd.DataFrame,
    panel_letter: str,
    pfas_label: str,
    show_xlabel: bool,
    missing_note: str | None = None,
    step_order: list[tuple[str, str]] | None = None,
) -> None:
    step_order = step_order or STEP_ORDER
    if panel_df.empty:
        ax.text(
            0.5,
            0.5,
            missing_note or "No data available for this comparison",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=BASE_FONT_SIZE,
        )
        ax.set_axis_off()
        return

    panel_df = panel_df.copy()
    panel_df["series_rank"] = panel_df["series"].map(
        {name: idx for idx, name in enumerate(SERIES_STYLES)}
    )
    panel_df = panel_df.sort_values("series_rank")
    series_names = panel_df["series"].tolist()
    n_series = len(series_names)

    all_levels = []
    for _, record in panel_df.iterrows():
        all_levels.extend(build_levels(record, step_order))

    y_min = min(all_levels)
    y_max = max(all_levels)
    y_span = max(y_max - y_min, 20.0)
    ax.set_ylim(y_min - 0.30 * y_span, y_max + 0.28 * y_span)
    ax.set_xlim(-0.25, len(step_order) + 1.55)

    base_x = np.arange(len(step_order) + 1)
    ax.axhline(0.0, linewidth=1.0, linestyle="--", color="0.45", zorder=1)

    for idx, (_, record) in enumerate(panel_df.iterrows()):
        style = SERIES_STYLES[record["series"]]
        color = style["color"]
        levels = build_levels(record, step_order)
        x_shift = 0.0

        for step_idx in range(len(step_order)):
            x0 = base_x[step_idx] + x_shift
            x1 = base_x[step_idx + 1] + x_shift
            y0 = levels[step_idx]
            y1 = levels[step_idx + 1]

            ax.hlines(y=y0, xmin=x0, xmax=x1, color="black", linewidth=1.0, alpha=0.45, zorder=2)
            ax.annotate(
                "",
                xy=(x1, y1),
                xytext=(x1, y0),
                arrowprops=dict(arrowstyle="->", color=color, lw=1.15, alpha=0.35),
                zorder=3,
            )
            ax.vlines(x=x1, ymin=y0, ymax=y1, color=color, linewidth=1.0, alpha=0.24, zorder=2)

            step_end_values = []
            for _, compare_record in panel_df.iterrows():
                step_levels = build_levels(compare_record, step_order)
                step_end_values.append(step_levels[step_idx + 1])
            threshold = float(np.median(step_end_values))
            place_above = y1 >= threshold
            y_text = y1 + (0.045 * y_span if place_above else -0.045 * y_span)
            x_text = x1
            ax.text(
                x_text,
                y_text,
                f"{y1 - y0:+.1f}",
                fontsize=VALUE_FONT_SIZE,
                color=color,
                ha="center",
                va="bottom" if place_above else "top",
                bbox=dict(facecolor="white", edgecolor="none", pad=0.15, alpha=0.85),
                clip_on=True,
                zorder=4,
            )

        final_x0 = base_x[-1] + x_shift
        final_x1 = len(step_order) + 1 + x_shift
        final_y = levels[-1]
        ax.hlines(y=final_y, xmin=final_x0, xmax=final_x1, color=color, linewidth=1.35, alpha=0.34, zorder=3)
        ax.scatter(
            final_x1,
            final_y,
            s=58,
            marker=style["marker"],
            color=color,
            zorder=4,
            label=style["label"],
        )
        final_values = [build_levels(compare_record, step_order)[-1] for _, compare_record in panel_df.iterrows()]
        final_place_above = final_y >= float(np.median(final_values))
        ax.text(
            final_x1,
            final_y + (0.055 * y_span if final_place_above else -0.055 * y_span),
            f"{record['Bond Energy']:+.1f}",
            fontsize=VALUE_FONT_SIZE,
            color=color,
            ha="center",
            va="bottom" if final_place_above else "top",
            bbox=dict(facecolor="white", edgecolor="none", pad=0.15, alpha=0.9),
            clip_on=True,
            zorder=4,
        )

    tick_positions = np.arange(1, len(step_order) + 1)
    tick_labels = [short for _, short in step_order]
    xtick_size = XTICK_FONT_SIZE_WITH_GCP if len(step_order) >= 8 else XTICK_FONT_SIZE
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, fontsize=xtick_size)
    ax.tick_params(axis="both", labelsize=BASE_FONT_SIZE - 1)
    ax.set_ylabel("Energy (kcal/mol)", fontsize=BASE_FONT_SIZE)
    ax.set_xlabel("EDA Component" if show_xlabel else "", fontsize=BASE_FONT_SIZE)
    ax.tick_params(axis="x", labelbottom=True, pad=6)
    ax.text(
        -0.07,
        1.05,
        panel_letter,
        transform=ax.transAxes,
        fontsize=18,
        fontweight="bold",
        va="top",
        ha="right",
    )
    ax.grid(axis="y", linestyle="--", alpha=0.25)

    ax.set_title(f"{pfas_label}$^-$", fontsize=BASE_FONT_SIZE + 3, fontweight="bold", pad=12)


def make_four_panel_figure(
    df: pd.DataFrame,
    series_order: list[str],
    outfile: str,
    figure_title: str,
    missing_note: str | None = None,
    pfas_order: list[str] | None = None,
    step_order: list[tuple[str, str]] | None = None,
    legend_y: float = 0.925,
    axes_top: float | None = None,
    hspace: float | None = None,
) -> None:
    step_order = step_order or STEP_ORDER
    pfas_order = pfas_order or PFAS_ORDER
    fig_height = 5.55 * len(pfas_order)
    fig, axes = plt.subplots(
        len(pfas_order),
        1,
        figsize=(12.8, fig_height),
        dpi=FIG_DPI,
    )
    axes = np.atleast_1d(axes)
    fig.suptitle(figure_title, fontsize=BASE_FONT_SIZE + 6, fontweight="bold", y=0.985)

    for idx, pfas in enumerate(pfas_order):
        ax = axes[idx]
        panel_df = df[(df["PFAS"] == pfas) & (df["series"].isin(series_order))].copy()
        draw_comparison_ladders(
            ax,
            panel_df,
            chr(ord("A") + idx),
            pfas,
            show_xlabel=idx == len(pfas_order) - 1,
            missing_note=missing_note,
            step_order=step_order,
        )

    legend_handles = [
        plt.Line2D(
            [0],
            [0],
            marker=SERIES_STYLES[series]["marker"],
            color="none",
            markerfacecolor=SERIES_STYLES[series]["color"],
            markeredgecolor=SERIES_STYLES[series]["color"],
            markersize=8,
            label=SERIES_STYLES[series]["label"],
        )
        for series in series_order
    ]
    fig.legend(
        handles=legend_handles,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, legend_y),
        ncol=len(legend_handles),
        columnspacing=1.1,
        handletextpad=0.45,
        borderaxespad=0.0,
        fontsize=BASE_FONT_SIZE,
    )
    fig.subplots_adjust(
        left=0.09,
        right=0.985,
        bottom=0.06,
        top=axes_top if axes_top is not None else (0.895 if len(pfas_order) > 2 else 0.855),
        hspace=hspace if hspace is not None else (0.54 if len(pfas_order) > 2 else 0.62),
    )

    outfile = OUTDIR / outfile
    fig.savefig(outfile, bbox_inches="tight")
    plt.close(fig)


parsed_rows = parse_eda_sections(TXT_PATH)
component_df = build_component_table(parsed_rows)
component_df.to_csv(OUTDIR / "eda_component_summary.csv", index=False)

for pfas_name in PFAS_ORDER:
    pass

make_four_panel_figure(
    component_df,
    ["BTMA_r2SCAN-3c", "BTMA_wB97X-D3"],
    "Figure_05_BTMA_energy_decomposition_analysis.png",
    "BTMA Energy Decomposition Analysis",
    legend_y=0.958,
    axes_top=0.905,
    hspace=0.34,
)

make_four_panel_figure(
    component_df,
    ["BTMA_wB97X-D3", "ExtendedMonomer_Water"],
    "Figure_09_extended_monomer_eda_across_pfas.png",
    "Extended Monomer Energy Decomposition Analysis",
    step_order=STEP_ORDER_WITHOUT_GCP,
    legend_y=0.958,
    axes_top=0.905,
    hspace=0.34,
)

make_four_panel_figure(
    component_df,
    ["ExtendedMonomer_Water", "ExtendedMonomer_Octanol72.5"],
    "Figure_10_solvent_dependence_extended_monomer_eda.png",
    "Solvent Dependence of Extended Monomer EDA",
    pfas_order=["PFOA", "PFOS"],
    step_order=STEP_ORDER_WITHOUT_GCP,
    legend_y=0.945,
    axes_top=0.872,
    hspace=0.24,
)

print("\nDerived EDA component table (kcal/mol):\n")
print(
    component_df[
        [
            "series",
            "PFAS",
            "Bond Energy",
            "Preparation Energy",
            "Pauli Energy",
            "Electrostatic Energy",
            "Orbital Energy",
            "Delta Dispersion",
            "Delta E^0(XC)",
            "Delta gCP correction",
            "Delta CPCM Dielectric",
        ]
    ].to_string(index=False, float_format=lambda x: f"{x:0.2f}")
)
