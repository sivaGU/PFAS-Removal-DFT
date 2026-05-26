from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm


# =========================
# USER SETTINGS
# =========================
OUTDIR = Path(__file__).resolve().parent / "btma_figures"
OUTDIR.mkdir(exist_ok=True)
FINAL_OUTDIR = next(
    (parent / "final_plots" for parent in Path(__file__).resolve().parents if (parent / "final_plots").is_dir()),
    None,
)

FONT_FAMILY = "DejaVu Sans"
DPI = 600

plt.rcParams.update(
    {
        "font.family": FONT_FAMILY,
        "font.size": 20,
        "axes.titlesize": 24,
        "axes.titleweight": "bold",
        "axes.labelsize": 21,
        "xtick.labelsize": 20,
        "ytick.labelsize": 20,
        "legend.fontsize": 20,
        "figure.titlesize": 28,
        "figure.titleweight": "bold",
    }
)


# =========================
# HARD-CODED DATA
# =========================
PFAS = ["FHEA", "PFHxA", "PFOA", "PFOS"]
FUNCTIONALS = ["r2SCAN-3c", "wB97X-D3"]

FUNCTIONAL_STYLE = {
    "r2SCAN-3c": {
        "label": r"r$^2$SCAN-3c",
        "color": "#1f77b4",
        "marker": "o",
    },
    "wB97X-D3": {
        "label": r"$\omega$B97X-D3",
        "color": "#ff7f0e",
        "marker": "s",
    },
}

deltaE = {
    "r2SCAN-3c": [-1.07, -2.69, -1.95, -1.04],
    "wB97X-D3": [0.18, -1.76, -1.05, -0.06],
}

deltaCorr = {
    "r2SCAN-3c": [7.06, 7.20, 7.86, 7.30],
    "wB97X-D3": [5.41, 6.30, 6.78, 6.10],
}

deltaG = {
    "r2SCAN-3c": [5.99, 4.52, 5.91, 6.26],
    "wB97X-D3": [5.58, 4.54, 5.73, 6.03],
}

species_rows = ["BTMA-PFAS complex", "BTMA-Cl complex", "Free PFAS anion", "Free Cl-"]

species_contrib = {
    "FHEA": [2.83, -2.45, -2.03, 0.00],
    "PFHxA": [3.03, -2.45, -1.48, 0.00],
    "PFOA": [3.48, -2.45, -2.11, 0.00],
    "PFOS": [4.46, -2.45, -3.21, 0.00],
}

component_rows = ["ZPE", "Thermal correction", "Entropy contribution"]

component_contrib = {
    "FHEA": [-0.18, 0.17, -1.65],
    "PFHxA": [0.07, 0.08, -1.04],
    "PFOA": [-0.13, 0.15, -1.10],
    "PFOS": [-0.13, 0.13, -1.19],
}

component_totals = {
    "FHEA": -1.65,
    "PFHxA": -0.90,
    "PFOA": -1.08,
    "PFOS": -1.20,
}


# =========================
# HELPERS
# =========================
def padded_limits(values, extra=0.22, include_zero=True):
    flat = np.asarray(values, dtype=float).ravel()
    if include_zero:
        flat = np.append(flat, 0.0)
    ymin = float(np.min(flat))
    ymax = float(np.max(flat))
    span = ymax - ymin
    if span == 0:
        span = max(abs(ymax), 1.0)
    return ymin - span * extra, ymax + span * extra


def annotate_point(ax, x, y, text, color, y_offset):
    ax.annotate(
        text,
        xy=(x, y),
        xytext=(0, y_offset),
        textcoords="offset points",
        ha="center",
        va="bottom" if y_offset > 0 else "top",
        fontsize=19,
        fontweight="bold",
        color=color,
        clip_on=True,
    )


def save_figure(fig, filename, rect=None):
    fig.tight_layout(rect=rect)
    fig.savefig(OUTDIR / filename, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


# =========================
# FIGURE 1: THERMOCHEMICAL DECOMPOSITION DUMBBELLS
# =========================
def make_thermochemical_decomposition():
    panels = [
        (
            "A. " + r"$\Delta E_{\mathrm{exchange}}$",
            deltaE,
            r"$\Delta E_{\mathrm{exchange}}$ (kcal/mol)",
        ),
        (
            "B. " + r"$\Delta(G - E_{\mathrm{el}})_{\mathrm{exchange}}$",
            deltaCorr,
            r"$\Delta(G - E_{\mathrm{el}})_{\mathrm{exchange}}$ (kcal/mol)",
        ),
        (
            "C. " + r"$\Delta G_{\mathrm{exchange}}$",
            deltaG,
            r"$\Delta G_{\mathrm{exchange}}$ (kcal/mol)",
        ),
    ]

    x = np.arange(len(PFAS))
    fig, axes = plt.subplots(1, 3, figsize=(16.8, 5.4), sharex=True)
    fig.suptitle("Thermochemical decomposition of BTMA exchange energetics", y=0.995)

    for ax, (title, dataset, ylabel) in zip(axes, panels):
        vals = np.array([dataset[functional] for functional in FUNCTIONALS])

        for i in range(len(PFAS)):
            ax.vlines(
                x=i,
                ymin=np.min(vals[:, i]),
                ymax=np.max(vals[:, i]),
                color="black",
                linewidth=1.15,
                alpha=0.55,
                zorder=1,
            )

        for functional in FUNCTIONALS:
            style = FUNCTIONAL_STYLE[functional]
            yvals = np.array(dataset[functional], dtype=float)
            ax.scatter(
                x,
                yvals,
                s=86,
                marker=style["marker"],
                color=style["color"],
                edgecolor="white",
                linewidth=0.8,
                label=style["label"],
                zorder=3,
            )

        # Place labels above the larger point and below the smaller point at each PFAS.
        for i in range(len(PFAS)):
            pair = {functional: dataset[functional][i] for functional in FUNCTIONALS}
            ordered = sorted(pair.items(), key=lambda item: item[1])
            low_functional, low_y = ordered[0]
            high_functional, high_y = ordered[-1]
            annotate_point(
                ax,
                x[i],
                low_y,
                f"{low_y:.2f}",
                FUNCTIONAL_STYLE[low_functional]["color"],
                -9,
            )
            annotate_point(
                ax,
                x[i],
                high_y,
                f"{high_y:.2f}",
                FUNCTIONAL_STYLE[high_functional]["color"],
                9,
            )

        ax.axhline(0, color="black", linewidth=1.0, linestyle="--", alpha=0.45, zorder=0)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xticks(x)
        ax.set_xticklabels(PFAS)
        ax.grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.28)
        ax.set_ylim(*padded_limits(vals, extra=0.28))
        ax.margins(x=0.12)

    axes[-1].legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False)
    save_figure(fig, "btma_thermochemical_decomposition.png", rect=[0, 0, 0.95, 0.94])


def draw_thermochemical_panel(ax, title, dataset, ylabel, letter, show_legend=False, ylim=None):
    x = np.arange(len(PFAS))
    vals = np.array([dataset[functional] for functional in FUNCTIONALS])

    for i in range(len(PFAS)):
        ax.vlines(
            x=i,
            ymin=np.min(vals[:, i]),
            ymax=np.max(vals[:, i]),
            color="black",
            linewidth=1.05,
            alpha=0.55,
            zorder=1,
        )

    for functional in FUNCTIONALS:
        style = FUNCTIONAL_STYLE[functional]
        yvals = np.array(dataset[functional], dtype=float)
        ax.scatter(
            x,
            yvals,
            s=138,
            marker=style["marker"],
            color=style["color"],
            edgecolor="white",
            linewidth=0.8,
            label=style["label"],
            zorder=3,
        )

    for i in range(len(PFAS)):
        pair = {functional: dataset[functional][i] for functional in FUNCTIONALS}
        ordered = sorted(pair.items(), key=lambda item: item[1])
        low_functional, low_y = ordered[0]
        high_functional, high_y = ordered[-1]
        annotate_point(ax, x[i], low_y, f"{low_y:.2f}", FUNCTIONAL_STYLE[low_functional]["color"], -8)
        annotate_point(ax, x[i], high_y, f"{high_y:.2f}", FUNCTIONAL_STYLE[high_functional]["color"], 8)

    ax.axhline(0, color="black", linewidth=0.9, linestyle="--", alpha=0.45, zorder=0)
    ax.set_title(title, pad=10, fontweight="bold")
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels(PFAS)
    ax.set_xlim(-0.42, len(PFAS) - 0.58)
    ax.grid(axis="y", linestyle="--", linewidth=0.65, alpha=0.28)
    ax.set_ylim(*(ylim if ylim is not None else padded_limits(vals, extra=0.30)))
    ax.text(
        -0.12,
        1.13,
        letter,
        transform=ax.transAxes,
        fontsize=31,
        fontweight="bold",
        va="top",
        ha="right",
    )
    if show_legend:
        ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False)


# =========================
# FIGURE 2: SPECIES-LEVEL CONTRIBUTIONS HEATMAP
# =========================
def make_species_contributions_heatmap():
    data = np.array([[species_contrib[pfas][i] for pfas in PFAS] for i in range(len(species_rows))])
    vmax = float(np.max(np.abs(data)))
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

    fig, ax = plt.subplots(figsize=(9.8, 6.2))
    image = ax.imshow(data, cmap="RdBu_r", norm=norm, aspect="auto")

    ax.set_title(
        "Species-level contributions to the functional dependence\n"
        "of the BTMA thermochemical correction",
        pad=34,
    )
    ax.text(
        0.5,
        1.035,
        r"$\Delta_F[Q] = \omega$B97X-D3 $-$ r$^2$SCAN-3c",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=12,
    )
    ax.set_xticks(np.arange(len(PFAS)))
    ax.set_xticklabels(PFAS)
    ax.set_yticks(np.arange(len(species_rows)))
    ax.set_yticklabels(species_rows)

    for row in range(data.shape[0]):
        for col in range(data.shape[1]):
            value = data[row, col]
            text_color = "white" if abs(value) > vmax * 0.55 else "black"
            ax.text(
                col,
                row,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=11.5,
                fontweight="bold",
                color=text_color,
            )

    ax.set_xticks(np.arange(-0.5, len(PFAS), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(species_rows), 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=1.5)
    ax.tick_params(which="minor", bottom=False, left=False)

    cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(
        r"Signed contribution to $\Delta_F[G - E_{\mathrm{el}}]_{\mathrm{exchange}}$ (kcal/mol)",
        fontsize=10.5,
    )

    save_figure(fig, "btma_species_contributions_heatmap.png")


def draw_species_heatmap(ax, letter):
    data = np.array([[species_contrib[pfas][i] for pfas in PFAS] for i in range(len(species_rows))])
    vmax = float(np.max(np.abs(data)))
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    image = ax.imshow(data, cmap="RdBu_r", norm=norm, aspect="auto")

    ax.set_title("Species-Level Contributions", pad=10)
    ax.set_xticks(np.arange(len(PFAS)))
    ax.set_xticklabels(PFAS)
    ax.set_yticks(np.arange(len(species_rows)))
    ax.set_yticklabels(species_rows)

    for row in range(data.shape[0]):
        for col in range(data.shape[1]):
            value = data[row, col]
            text_color = "white" if abs(value) > vmax * 0.55 else "black"
            ax.text(
                col,
                row,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=18,
                fontweight="bold",
                color=text_color,
            )

    ax.set_xticks(np.arange(-0.5, len(PFAS), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(species_rows), 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=1.3)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.text(
        -0.12,
        1.08,
        letter,
        transform=ax.transAxes,
        fontsize=31,
        fontweight="bold",
        va="top",
        ha="right",
    )
    return image


# =========================
# FIGURE 3: COMPONENT-LEVEL CONTRIBUTIONS
# =========================
def make_component_contributions():
    y = np.arange(len(PFAS))
    bar_height = 0.2
    offsets = np.array([-bar_height, 0.0, bar_height])
    category_colors = {
        "ZPE": "#8c8c8c",
        "Thermal correction": "#3f9b5f",
        "Entropy contribution": "#7b51a4",
    }

    fig, ax = plt.subplots(figsize=(10.5, 6.8))

    for i, component in enumerate(component_rows):
        values = np.array([component_contrib[pfas][i] for pfas in PFAS], dtype=float)
        bars = ax.barh(
            y + offsets[i],
            values,
            height=bar_height * 0.86,
            color=category_colors[component],
            edgecolor="white",
            linewidth=0.7,
            label=component,
            zorder=2,
        )

        for bar, value in zip(bars, values):
            if abs(value) < 0.06:
                continue
            x_pos = value + (0.035 if value >= 0 else -0.035)
            ax.text(
                x_pos,
                bar.get_y() + bar.get_height() / 2,
                f"{value:.2f}",
                ha="left" if value >= 0 else "right",
                va="center",
                fontsize=10.5,
                color=category_colors[component],
                fontweight="bold",
            )

    totals = np.array([component_totals[pfas] for pfas in PFAS], dtype=float)
    ax.scatter(
        totals,
        y,
        marker="D",
        s=74,
        color="black",
        label="Total",
        zorder=4,
    )

    for yi, total in zip(y, totals):
        ax.text(
            total,
            yi - 0.20,
            f"{total:.2f}",
            ha="center",
            va="center",
            fontsize=10.5,
            color="black",
            fontweight="bold",
        )

    ax.axvline(0, color="black", linewidth=1.0, linestyle="--", alpha=0.5, zorder=1)
    ax.set_yticks(y)
    ax.set_yticklabels(PFAS)
    ax.invert_yaxis()
    ax.set_xlabel(r"Contribution to $\Delta_F[G - E_{\mathrm{el}}]_{\mathrm{exchange}}$ (kcal/mol)")
    ax.set_title(
        "Component-level origin of the functional dependence\n"
        "of the BTMA thermochemical correction"
    )
    ax.text(
        0.0,
        -0.16,
        r"$\Delta_F[Q] = \omega$B97X-D3 $-$ r$^2$SCAN-3c; entropy is plotted as its signed contribution to $G$.",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=11,
    )
    ax.grid(axis="x", linestyle="--", linewidth=0.7, alpha=0.28)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False)

    all_values = []
    for pfas in PFAS:
        all_values.extend(component_contrib[pfas])
        all_values.append(component_totals[pfas])
    xmin, xmax = padded_limits(all_values, extra=0.20, include_zero=True)
    ax.set_xlim(xmin, xmax)

    save_figure(fig, "btma_component_contributions.png", rect=[0, 0.06, 0.86, 1.0])


def draw_component_contributions(ax, letter):
    y = np.arange(len(PFAS))
    bar_height = 0.2
    offsets = np.array([-bar_height, 0.0, bar_height])
    category_colors = {
        "ZPE": "#8c8c8c",
        "Thermal correction": "#3f9b5f",
        "Entropy contribution": "#7b51a4",
    }

    for i, component in enumerate(component_rows):
        values = np.array([component_contrib[pfas][i] for pfas in PFAS], dtype=float)
        ax.barh(
            y + offsets[i],
            values,
            height=bar_height * 0.86,
            color=category_colors[component],
            edgecolor="white",
            linewidth=0.7,
            label=component,
            zorder=2,
        )

    totals = np.array([component_totals[pfas] for pfas in PFAS], dtype=float)
    ax.scatter(totals, y, marker="D", s=120, color="black", label="Total", zorder=4)

    for yi, total in zip(y, totals):
        ax.text(
            total,
            yi - 0.18,
            f"{total:.2f}",
            ha="center",
            va="center",
            fontsize=18,
            color="black",
            fontweight="bold",
        )

    ax.axvline(0, color="black", linewidth=0.9, linestyle="--", alpha=0.5, zorder=1)
    ax.set_yticks(y)
    ax.set_yticklabels(PFAS)
    ax.invert_yaxis()
    ax.set_xlabel(r"Contribution to $\Delta_F[G - E_{\mathrm{el}}]_{\mathrm{exchange}}$ (kcal/mol)")
    ax.set_title("Component-Level Contributions", pad=10)
    ax.grid(axis="x", linestyle="--", linewidth=0.65, alpha=0.28)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False)

    all_values = []
    for pfas in PFAS:
        all_values.extend(component_contrib[pfas])
        all_values.append(component_totals[pfas])
    xmin, xmax = padded_limits(all_values, extra=0.20, include_zero=True)
    ax.set_xlim(xmin, xmax)
    ax.text(
        -0.10,
        1.07,
        letter,
        transform=ax.transAxes,
        fontsize=31,
        fontweight="bold",
        va="top",
        ha="right",
    )


def make_combined_thermochemical_origin():
    fig = plt.figure(figsize=(20.5, 14.2), dpi=DPI)

    ax_a = fig.add_axes([0.05, 0.60, 0.255, 0.34])
    ax_b = fig.add_axes([0.38, 0.60, 0.255, 0.34])
    ax_c = fig.add_axes([0.71, 0.60, 0.255, 0.34])
    ax_d = fig.add_axes([0.08, 0.10, 0.37, 0.39])
    ax_e = fig.add_axes([0.565, 0.10, 0.34, 0.39])

    fig.suptitle("Thermochemical Origin of Functional Dependence", y=1.035)
    draw_thermochemical_panel(
        ax_a,
        r"$\mathbf{\Delta E}_{\mathbf{exchange}}$",
        deltaE,
        "kcal/mol",
        "A",
    )
    draw_thermochemical_panel(
        ax_b,
        r"$\mathbf{\Delta}(\mathbf{G} - \mathbf{E}_{\mathbf{el}})_{\mathbf{exchange}}$",
        deltaCorr,
        "kcal/mol",
        "B",
        ylim=(4.65, 8.25),
    )
    draw_thermochemical_panel(
        ax_c,
        r"$\mathbf{\Delta G}_{\mathbf{exchange}}$",
        deltaG,
        "kcal/mol",
        "C",
        show_legend=True,
        ylim=(4.25, 6.55),
    )
    image = draw_species_heatmap(ax_d, "D")
    draw_component_contributions(ax_e, "E")

    cbar = fig.colorbar(image, ax=ax_d, fraction=0.046, pad=0.04)
    cbar.set_label(
        r"Signed contribution to $\Delta_F[G - E_{\mathrm{el}}]_{\mathrm{exchange}}$ (kcal/mol)",
        fontsize=19,
        labelpad=18,
    )
    fig.text(
        0.50,
        0.012,
        r"$\Delta_F[Q] = \omega$B97X-D3 $-$ r$^2$SCAN-3c; entropy is plotted as its signed contribution to $G$.",
        ha="center",
        va="bottom",
        fontsize=20,
    )
    figure_name = "Figure_04_thermochemical_origin_functional_dependence.png"
    fig.savefig(OUTDIR / figure_name, dpi=DPI, bbox_inches="tight")
    if FINAL_OUTDIR is not None:
        fig.savefig(FINAL_OUTDIR / figure_name, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def main():
    make_combined_thermochemical_origin()
    print(f"Saved figures to: {OUTDIR}")


if __name__ == "__main__":
    main()
