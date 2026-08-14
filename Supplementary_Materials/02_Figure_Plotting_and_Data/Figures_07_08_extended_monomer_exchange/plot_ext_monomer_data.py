from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pandas as pd

# Settings
HERE = Path(__file__).resolve().parent
INPUT_CSV = HERE / "input_data" / "ext_monomer_exchange_summary.csv"
OUTDIR = HERE / "outputs"
OUTDIR.mkdir(exist_ok=True)
STRUCTURE_FIGURE_DIR = HERE / "input_structures"

HARTREE_TO_KCAL = 627.509474
FONT_FAMILY = "DejaVu Sans"
LABEL_FONT_SIZE = 11
PFAS_ORDER = ["FHEA", "PFHxA", "PFOA", "PFOS"]

plt.rcParams["font.family"] = FONT_FAMILY

SERIES_STYLES = {
    "BTMA_wB97X-D3": {
        "label": r"BTMA $\omega$B97X-D3",
        "color": "#ff7f0e",
        "marker": "s",
    },
    "ExtendedMonomer_Water": {
        "label": r"DVB-BTMA$^{+}$ (Water)",
        "color": "#1b9e77",
        "marker": "D",
    },
    "ExtendedMonomer_Octanol72.5": {
        "label": r"DVB-BTMA$^{+}$ (Octanol, $\epsilon = 72.5$)",
        "color": "#7570b3",
        "marker": "^",
    },
}


summary_df = pd.read_csv(INPUT_CSV)
required_columns = {"series", "PFAS", "DeltaE_exchange_kcalmol", "DeltaG_exchange_kcalmol"}
missing_columns = required_columns - set(summary_df.columns)
if missing_columns:
    raise ValueError(f"{INPUT_CSV} is missing columns: {sorted(missing_columns)}")
summary_df["PFAS"] = pd.Categorical(summary_df["PFAS"], categories=PFAS_ORDER, ordered=True)
summary_df = summary_df.sort_values(["series", "PFAS"]).reset_index(drop=True)


def make_paired_plot(
    df: pd.DataFrame,
    value_col: str,
    series_left: str,
    series_right: str,
    pfas_order: list[str],
    title: str,
    ylabel: str,
    outfile: str,
    ylim: tuple[float, float],
):
    fig, ax = plt.subplots(figsize=(8.8, 4.9), dpi=300)
    label_scale = LABEL_FONT_SIZE / 8.0

    x_positions = list(range(len(pfas_order)))
    xpad_left = 0.55 + 0.10 * max(label_scale - 1.0, 0.0)
    xpad_right = 0.16 + 0.03 * max(label_scale - 1.0, 0.0)
    ax.set_xlim(-xpad_left, len(pfas_order) - xpad_right)
    ax.set_ylim(*ylim)

    lower, upper = ax.get_ylim()
    margin = (0.08 + 0.025 * max(label_scale - 1.0, 0.0)) * (upper - lower)
    zero_buffer = (0.09 + 0.02 * max(label_scale - 1.0, 0.0)) * (upper - lower)

    left_style = SERIES_STYLES[series_left]
    right_style = SERIES_STYLES[series_right]

    ax.axhline(0.0, linewidth=1.0, linestyle="--", color="0.45", zorder=1)

    for i, pfas in enumerate(pfas_order):
        sub = df[df["PFAS"] == pfas].set_index("series")
        y_left = sub.loc[series_left, value_col]
        y_right = sub.loc[series_right, value_col]

        ax.plot([i, i], [y_left, y_right], linewidth=1.8, color="black", zorder=2)

        ax.scatter(
            i,
            y_left,
            s=70,
            label=left_style["label"] if i == 0 else None,
            zorder=3,
            marker=left_style["marker"],
            color=left_style["color"],
        )
        ax.scatter(
            i,
            y_right,
            s=70,
            label=right_style["label"] if i == 0 else None,
            zorder=3,
            marker=right_style["marker"],
            color=right_style["color"],
        )

        left_above = y_left >= y_right
        right_above = not left_above

        if abs(y_left) < zero_buffer:
            left_above = y_left >= 0
        if abs(y_right) < zero_buffer:
            right_above = y_right >= 0

        if y_left > upper - margin:
            left_above = False
        if y_right > upper - margin:
            right_above = False
        if y_left < lower + margin:
            left_above = True
        if y_right < lower + margin:
            right_above = True

        dx = 14 * label_scale
        dy_primary = 12 * label_scale
        dy_secondary = 20 * label_scale

        left_dx = -dx
        right_dx = dx
        left_dy = dy_primary if left_above else -dy_primary
        right_dy = dy_secondary if right_above else -dy_secondary

        ax.annotate(
            f"{y_left:.2f}",
            xy=(i, y_left),
            xytext=(left_dx, left_dy),
            textcoords="offset points",
            fontsize=LABEL_FONT_SIZE,
            ha="right",
            va="bottom" if left_above else "top",
            clip_on=True,
            annotation_clip=True,
            arrowprops={
                "arrowstyle": "-",
                "color": left_style["color"],
                "lw": 0.9,
                "shrinkA": 0,
                "shrinkB": 6 * label_scale,
            },
        )
        ax.annotate(
            f"{y_right:.2f}",
            xy=(i, y_right),
            xytext=(right_dx, right_dy),
            textcoords="offset points",
            fontsize=LABEL_FONT_SIZE,
            ha="left",
            va="bottom" if right_above else "top",
            clip_on=True,
            annotation_clip=True,
            arrowprops={
                "arrowstyle": "-",
                "color": right_style["color"],
                "lw": 0.9,
                "shrinkA": 0,
                "shrinkB": 6 * label_scale,
            },
        )

    ax.set_xticks(x_positions)
    ax.set_xticklabels(pfas_order)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.00),
        borderaxespad=0.0,
        fontsize=max(LABEL_FONT_SIZE, 10),
    )

    fig.subplots_adjust(right=0.76)
    fig.savefig(OUTDIR / outfile, bbox_inches="tight")
    plt.close(fig)


def make_two_panel_paired_plot(
    df: pd.DataFrame,
    series_left: str,
    series_right: str,
    pfas_order: list[str],
    figure_title: str,
    panel_titles: list[str],
    outfile: str,
    ylims: list[tuple[float, float]],
    structure_paths: list[Path] | None = None,
    figure_title_fontsize: int = 16,
    panel_title_fontsize: int | None = None,
    panel_label_x: float = -0.10,
    legend_y: float = 0.925,
    legend_fontsize: int | None = None,
    legend_columnspacing: float = 2.0,
    legend_markerscale: float = 1.0,
):
    use_structure_row = structure_paths is not None
    if use_structure_row:
        if len(structure_paths) == 1:
            fig = plt.figure(figsize=(14.0, 12.6), dpi=300)
            gs = fig.add_gridspec(2, 2, height_ratios=[1.05, 1.62], hspace=0.18, wspace=0.18)
            top_axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])]
            bottom_axes = [fig.add_subplot(gs[1, :])]
            bottom_letters = ["C"]
        elif len(structure_paths) == 2:
            fig, axes = plt.subplots(
                2,
                2,
                figsize=(14.0, 12.6),
                dpi=300,
                sharex=False,
                gridspec_kw={"height_ratios": [1.05, 1.62]},
            )
            top_axes = list(axes[0])
            bottom_axes = list(axes[1])
            bottom_letters = ["C", "D"]
        else:
            raise ValueError("Expected one or two structure paths for bottom structure panels.")
    else:
        fig, top_axes = plt.subplots(1, 2, figsize=(17.0, 6.2), dpi=300, sharex=True)

    fig.suptitle(figure_title, fontsize=figure_title_fontsize, fontweight="bold", y=0.978)

    configs = [
        ("DeltaE_exchange_kcalmol", r"$\Delta E_{\mathrm{exchange}}$ (kcal/mol)", "A"),
        ("DeltaG_exchange_kcalmol", r"$\Delta G_{\mathrm{exchange}}$ (kcal/mol)", "B"),
    ]
    label_scale = LABEL_FONT_SIZE / 8.0
    x_positions = list(range(len(pfas_order)))
    left_style = SERIES_STYLES[series_left]
    right_style = SERIES_STYLES[series_right]

    for ax, (value_col, ylabel, letter), title, ylim in zip(top_axes, configs, panel_titles, ylims):
        ax.set_ylim(*ylim)
        lower, upper = ax.get_ylim()
        margin = (0.08 + 0.025 * max(label_scale - 1.0, 0.0)) * (upper - lower)
        zero_buffer = (0.09 + 0.02 * max(label_scale - 1.0, 0.0)) * (upper - lower)

        ax.axhline(0.0, linewidth=1.0, linestyle="--", color="0.45", alpha=0.65, zorder=1)

        for i, pfas in enumerate(pfas_order):
            sub = df[df["PFAS"] == pfas].set_index("series")
            y_left = sub.loc[series_left, value_col]
            y_right = sub.loc[series_right, value_col]

            ax.plot([i, i], [y_left, y_right], linewidth=1.7, color="black", alpha=0.82, zorder=2)
            ax.scatter(
                i,
                y_left,
                s=72,
                label=left_style["label"] if i == 0 else None,
                zorder=3,
                marker=left_style["marker"],
                color=left_style["color"],
            )
            ax.scatter(
                i,
                y_right,
                s=72,
                label=right_style["label"] if i == 0 else None,
                zorder=3,
                marker=right_style["marker"],
                color=right_style["color"],
            )

            left_above = y_left >= y_right
            right_above = not left_above
            if abs(y_left) < zero_buffer:
                left_above = y_left >= 0
            if abs(y_right) < zero_buffer:
                right_above = y_right >= 0
            if y_left > upper - margin:
                left_above = False
            if y_right > upper - margin:
                right_above = False
            if y_left < lower + margin:
                left_above = True
            if y_right < lower + margin:
                right_above = True

            dx = 22 * label_scale
            dy_primary = 11 * label_scale
            dy_secondary = 19 * label_scale
            left_dx, left_ha = -dx, "right"
            right_dx, right_ha = dx, "left"

            for yval, style, dx_val, dy_val, above, ha in [
                (y_left, left_style, left_dx, dy_primary if left_above else -dy_primary, left_above, left_ha),
                (y_right, right_style, right_dx, dy_secondary if right_above else -dy_secondary, right_above, right_ha),
            ]:
                ax.annotate(
                    f"{yval:.2f}",
                    xy=(i, yval),
                    xytext=(dx_val, dy_val),
                    textcoords="offset points",
                    fontsize=LABEL_FONT_SIZE,
                    ha=ha,
                    va="bottom" if above else "top",
                    clip_on=True,
                    annotation_clip=True,
                    arrowprops={
                        "arrowstyle": "-",
                        "color": style["color"],
                        "lw": 0.85,
                        "shrinkA": 0,
                        "shrinkB": 6 * label_scale,
                    },
                )

        ax.set_xticks(x_positions)
        ax.set_xticklabels(pfas_order)
        ax.set_xlim(-0.75, len(pfas_order) - 0.25)
        ax.set_ylabel(ylabel)
        if panel_title_fontsize is None:
            ax.set_title(title)
        else:
            ax.set_title(title, fontsize=panel_title_fontsize, fontweight="bold", pad=8)
        ax.grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.28)
        ax.text(
            panel_label_x,
            1.08,
            letter,
            transform=ax.transAxes,
            fontsize=17,
            fontweight="bold",
            va="top",
            ha="right",
        )

    if use_structure_row:
        for ax, letter, path in zip(bottom_axes, bottom_letters, structure_paths):
            if not path.exists():
                raise FileNotFoundError(f"Missing structure figure asset for panel {letter}: {path}")
            img = mpimg.imread(path)
            ax.imshow(img, aspect="equal")
            ax.set_axis_off()
            ax.text(
                -0.02,
                1.02,
                letter,
                transform=ax.transAxes,
                fontsize=17,
                fontweight="bold",
                va="top",
                ha="right",
            )
        handles, labels = top_axes[0].get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            frameon=False,
            loc="upper center",
            bbox_to_anchor=(0.5, legend_y),
            ncol=len(handles),
            fontsize=legend_fontsize,
            columnspacing=legend_columnspacing,
            handletextpad=0.9,
            markerscale=legend_markerscale,
        )
        fig.subplots_adjust(left=0.07, right=0.985, bottom=0.04, top=0.86, wspace=0.18, hspace=0.18)
    else:
        handles, labels = top_axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 0.905), ncol=len(handles))
        fig.subplots_adjust(left=0.075, right=0.985, bottom=0.14, top=0.76, wspace=0.22)

    fig.savefig(OUTDIR / outfile, bbox_inches="tight")
    plt.close(fig)


btma_vs_water = summary_df[summary_df["series"].isin(["BTMA_wB97X-D3", "ExtendedMonomer_Water"])].copy()
octanol_compare = summary_df[
    summary_df["series"].isin(["ExtendedMonomer_Water", "ExtendedMonomer_Octanol72.5"])
    & summary_df["PFAS"].isin(["PFOA", "PFOS"])
].copy()

make_two_panel_paired_plot(
    btma_vs_water,
    series_left="ExtendedMonomer_Water",
    series_right="BTMA_wB97X-D3",
    pfas_order=PFAS_ORDER,
    figure_title="Model Size Effects on Exchange Energetics",
    panel_titles=[
        r"$\mathbf{\Delta E}_{\mathbf{exchange}}$",
        r"$\mathbf{\Delta G}_{\mathbf{exchange}}$",
    ],
    outfile="Figure_07_model_size_exchange_energetics.png",
    ylims=[(-10.8, 1.0), (-1.2, 7.8)],
    structure_paths=[
        STRUCTURE_FIGURE_DIR / "Extended_Monomer_PFAS_r2SCAN-3c_structures_orthoscopic_no_panel_letters.png",
        STRUCTURE_FIGURE_DIR / "Extended_Monomer_PFAS_wB97X-D3_structures_orthoscopic_no_panel_letters.png",
    ],
    figure_title_fontsize=20,
    panel_title_fontsize=18,
    panel_label_x=-0.02,
    legend_y=0.94,
    legend_fontsize=14,
    legend_columnspacing=3.2,
    legend_markerscale=1.25,
)

make_two_panel_paired_plot(
    octanol_compare,
    series_left="ExtendedMonomer_Water",
    series_right="ExtendedMonomer_Octanol72.5",
    pfas_order=["PFOA", "PFOS"],
    figure_title=r"Solvent Identity Effects on DVB-BTMA$^{+}$ Exchange",
    panel_titles=[
        r"$\mathbf{\Delta E}_{\mathbf{exchange}}$",
        r"$\mathbf{\Delta G}_{\mathbf{exchange}}$",
    ],
    outfile="Figure_08_solvent_identity_exchange_energetics.png",
    ylims=[(-10.95, -8.75), (-2.35, 1.35)],
    structure_paths=[
        STRUCTURE_FIGURE_DIR / "Extended_Monomer_PFAS_Octanol_epsilon72p5_structures_orthoscopic_no_panel_letters.png",
    ],
    figure_title_fontsize=20,
    panel_title_fontsize=18,
    panel_label_x=-0.02,
    legend_y=0.94,
    legend_fontsize=14,
    legend_columnspacing=3.2,
    legend_markerscale=1.25,
)

print(f"\nSaved figures and tables to: {OUTDIR.resolve()}")
print("\nFiles created:")
for path in sorted(OUTDIR.iterdir()):
    print(" -", path.name)
