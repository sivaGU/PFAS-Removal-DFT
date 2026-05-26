from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# =========================
# USER SETTINGS
# =========================
OUTDIR = Path(__file__).resolve().parent / "btma_figures"
OUTDIR.mkdir(exist_ok=True)
FINAL_OUTDIR = Path(__file__).resolve().parents[1] / "final_plots"
FINAL_OUTDIR.mkdir(exist_ok=True)
STRUCTURE_FIGURE_DIR = FINAL_OUTDIR / "structure_figures"

HARTREE_TO_KCAL = 627.509474
FONT_FAMILY = "DejaVu Sans"
LABEL_FONT_SIZE = 11

# PFAS display order for manuscript-style figures
PFAS_ORDER = ["FHEA", "PFHxA", "PFOA", "PFOS"]

plt.rcParams["font.family"] = FONT_FAMILY

# HARD-CODED DATA
# =========================
# Small fixed dataset for the BTMA selectivity manuscript figures.
records = [
    {
        "file": "/mnt/d/Work/RZ_ORCA_PFASRemoval/01-0-Selectivity/1_Opt_0.15M/1_R4N+X-_0.15M/R4N+FHEA-_0.15M/R4N+FHEA-_0.15M.out",
        "functional": "r2SCAN-3c",
        "role": "R4N+X-",
        "species": "FHEA",
        "single_point_Eh": -2199.764389580309,
        "gibbs_Eh": -2199.47078192,
    },
    {
        "file": "/mnt/d/Work/RZ_ORCA_PFASRemoval/01-0-Selectivity/1_Opt_0.15M/1_R4N+X-_0.15M/R4N+PFHxA-_0.15M/R4N+PFHxA-_0.15M.out",
        "functional": "r2SCAN-3c",
        "role": "R4N+X-",
        "species": "PFHxA",
        "single_point_Eh": -1922.676121647777,
        "gibbs_Eh": -1922.41786628,
    },
    {
        "file": "/mnt/d/Work/RZ_ORCA_PFASRemoval/01-0-Selectivity/1_Opt_0.15M/1_R4N+X-_0.15M/R4N+PFOA-_0.15M/R4N+PFOA-_0.15M.out",
        "functional": "r2SCAN-3c",
        "role": "R4N+X-",
        "species": "PFOA",
        "single_point_Eh": -2398.242098977966,
        "gibbs_Eh": -2397.96515453,
    },
    {
        "file": "/mnt/d/Work/RZ_ORCA_PFASRemoval/01-0-Selectivity/1_Opt_0.15M/1_R4N+X-_0.15M/R4N+PFOS-_0.15M/R4N+PFOS-_0.15M.out",
        "functional": "r2SCAN-3c",
        "role": "R4N+X-",
        "species": "PFOS",
        "single_point_Eh": -3071.263984705567,
        "gibbs_Eh": -3070.97953381,
    },
    {
        "file": "/mnt/d/Work/RZ_ORCA_PFASRemoval/01-0-Selectivity/1_Opt_0.15M/2_R4N+Cl-_0.15M/R4N+Cl-_0.15M.out",
        "functional": "r2SCAN-3c",
        "role": "R4N+Cl-",
        "species": "Cl",
        "single_point_Eh": -905.517886846123,
        "gibbs_Eh": -905.31500817,
    },
    {
        "file": "/mnt/d/Work/RZ_ORCA_PFASRemoval/01-0-Selectivity/1_Opt_0.15M/3_X-_0.15M/FHEA_0.15M/FHEA_0.15M.out",
        "functional": "r2SCAN-3c",
        "role": "X-",
        "species": "FHEA",
        "single_point_Eh": -1754.590197254154,
        "gibbs_Eh": -1754.52647219,
    },
    {
        "file": "/mnt/d/Work/RZ_ORCA_PFASRemoval/01-0-Selectivity/1_Opt_0.15M/3_X-_0.15M/PFHxA_0.15M/PFHxA_0.15M.out",
        "functional": "r2SCAN-3c",
        "role": "X-",
        "species": "PFHxA",
        "single_point_Eh": -1477.499356306281,
        "gibbs_Eh": -1477.47120355,
    },
    {
        "file": "/mnt/d/Work/RZ_ORCA_PFASRemoval/01-0-Selectivity/1_Opt_0.15M/3_X-_0.15M/PFOA_0.15M/PFOA_0.15M.out",
        "functional": "r2SCAN-3c",
        "role": "X-",
        "species": "PFOA",
        "single_point_Eh": -1953.066502525915,
        "gibbs_Eh": -1953.02071754,
    },
    {
        "file": "/mnt/d/Work/RZ_ORCA_PFASRemoval/01-0-Selectivity/1_Opt_0.15M/3_X-_0.15M/PFOS_0.15M/PFOS_0.15M.out",
        "functional": "r2SCAN-3c",
        "role": "X-",
        "species": "PFOS",
        "single_point_Eh": -2626.089844333014,
        "gibbs_Eh": -2626.03564535,
    },
    {
        "file": "/mnt/d/Work/RZ_ORCA_PFASRemoval/01-0-Selectivity/1_Opt_0.15M/4_Cl-_0.15M/Cl-_0.15M.out",
        "functional": "r2SCAN-3c",
        "role": "Cl-",
        "species": "Cl",
        "single_point_Eh": -460.345405067028,
        "gibbs_Eh": -460.36114952,
    },
    {
        "file": "/mnt/d/Work/RZ_ORCA_PFASRemoval/01-0-Selectivity/2-1_Freq_wb/1_R4N+X-_wb/R4N+FHEA-_wb/R4N+FHEA-_wb.out",
        "functional": "wB97X-D3",
        "role": "R4N+X-",
        "species": "FHEA",
        "single_point_Eh": -2200.289787524607,
        "gibbs_Eh": -2199.99167612,
    },
    {
        "file": "/mnt/d/Work/RZ_ORCA_PFASRemoval/01-0-Selectivity/2-1_Freq_wb/1_R4N+X-_wb/R4N+PFHxA-_wb/R4N+PFHxA-_wb.out",
        "functional": "wB97X-D3",
        "role": "R4N+X-",
        "species": "PFHxA",
        "single_point_Eh": -1923.137309076471,
        "gibbs_Eh": -1922.87422607,
    },
    {
        "file": "/mnt/d/Work/RZ_ORCA_PFASRemoval/01-0-Selectivity/2-1_Freq_wb/1_R4N+X-_wb/R4N+PFOA-_wb/R4N+PFOA-_wb.out",
        "functional": "wB97X-D3",
        "role": "R4N+X-",
        "species": "PFOA",
        "single_point_Eh": -2398.793370784731,
        "gibbs_Eh": -2398.51089463,
    },
    {
        "file": "/mnt/d/Work/RZ_ORCA_PFASRemoval/01-0-Selectivity/2-1_Freq_wb/1_R4N+X-_wb/R4N+PFOS-_wb/R4N+PFOS-_wb.out",
        "functional": "wB97X-D3",
        "role": "R4N+X-",
        "species": "PFOS",
        "single_point_Eh": -3071.948356352467,
        "gibbs_Eh": -3071.65679623,
    },
    {
        "file": "/mnt/d/Work/RZ_ORCA_PFASRemoval/01-0-Selectivity/2-1_Freq_wb/2_R4N+Cl-_wb/R4N+Cl-_wb.out",
        "functional": "wB97X-D3",
        "role": "R4N+Cl-",
        "species": "Cl",
        "single_point_Eh": -905.731151435150,
        "gibbs_Eh": -905.52436385,
    },
    {
        "file": "/mnt/d/Work/RZ_ORCA_PFASRemoval/01-0-Selectivity/2-1_Freq_wb/3_X-_wb/FHEA_wb/FHEA_wb.out",
        "functional": "wB97X-D3",
        "role": "X-",
        "species": "FHEA",
        "single_point_Eh": -1754.944149647387,
        "gibbs_Eh": -1754.87717914,
    },
    {
        "file": "/mnt/d/Work/RZ_ORCA_PFASRemoval/01-0-Selectivity/2-1_Freq_wb/3_X-_wb/PFHxA_wb/PFHxA_wb.out",
        "functional": "wB97X-D3",
        "role": "X-",
        "species": "PFHxA",
        "single_point_Eh": -1477.788586171688,
        "gibbs_Eh": -1477.75807611,
    },
    {
        "file": "/mnt/d/Work/RZ_ORCA_PFASRemoval/01-0-Selectivity/2-1_Freq_wb/3_X-_wb/PFOA_wb/PFOA_wb.out",
        "functional": "wB97X-D3",
        "role": "X-",
        "species": "PFOA",
        "single_point_Eh": -1953.445777855603,
        "gibbs_Eh": -1953.39663489,
    },
    {
        "file": "/mnt/d/Work/RZ_ORCA_PFASRemoval/01-0-Selectivity/2-1_Freq_wb/3_X-_wb/PFOS_wb/PFOS_wb.out",
        "functional": "wB97X-D3",
        "role": "X-",
        "species": "PFOS",
        "single_point_Eh": -2626.602333802936,
        "gibbs_Eh": -2626.54301344,
    },
    {
        "file": "/mnt/d/Work/RZ_ORCA_PFASRemoval/01-0-Selectivity/2-1_Freq_wb/4_Cl-_wb/Cl-_wb.out",
        "functional": "wB97X-D3",
        "role": "Cl-",
        "species": "Cl",
        "single_point_Eh": -460.385228974217,
        "gibbs_Eh": -460.40097343,
    },
]

raw_df = pd.DataFrame(records)

if raw_df.empty:
    raise RuntimeError("No records parsed. Check TXT_PATH and file format.")

# =========================
# BUILD THERMODYNAMIC CYCLE TABLE
# =========================
# Need, for each functional and PFAS:
# Delta E_exchange = E(R4N+X-) - E(R4N+Cl-) - E(X-) + E(Cl-)
# Same for G

summary_rows = []

for functional in ["r2SCAN-3c", "wB97X-D3"]:
    sub = raw_df[raw_df["functional"] == functional].copy()

    # reference terms
    ref_r4n_cl = sub.loc[sub["role"] == "R4N+Cl-", "single_point_Eh"]
    ref_r4n_cl_g = sub.loc[sub["role"] == "R4N+Cl-", "gibbs_Eh"]
    ref_cl = sub.loc[sub["role"] == "Cl-", "single_point_Eh"]
    ref_cl_g = sub.loc[sub["role"] == "Cl-", "gibbs_Eh"]

    if ref_r4n_cl.empty or ref_cl.empty:
        raise RuntimeError(f"Missing chloride reference terms for {functional}")

    E_r4n_cl = ref_r4n_cl.iloc[0]
    G_r4n_cl = ref_r4n_cl_g.iloc[0]
    E_cl = ref_cl.iloc[0]
    G_cl = ref_cl_g.iloc[0]

    for pfas in PFAS_ORDER:
        cx = sub[(sub["role"] == "R4N+X-") & (sub["species"] == pfas)]
        fx = sub[(sub["role"] == "X-") & (sub["species"] == pfas)]

        if cx.empty or fx.empty:
            raise RuntimeError(f"Missing data for {pfas} / {functional}")

        E_r4n_x = cx["single_point_Eh"].iloc[0]
        G_r4n_x = cx["gibbs_Eh"].iloc[0]
        E_x = fx["single_point_Eh"].iloc[0]
        G_x = fx["gibbs_Eh"].iloc[0]

        delta_E_Eh = E_r4n_x - E_r4n_cl - E_x + E_cl
        delta_G_Eh = G_r4n_x - G_r4n_cl - G_x + G_cl

        summary_rows.append(
            {
                "functional": functional,
                "PFAS": pfas,
                "R4N+X-_single_point_Eh": E_r4n_x,
                "R4N+Cl-_single_point_Eh": E_r4n_cl,
                "X-_single_point_Eh": E_x,
                "Cl-_single_point_Eh": E_cl,
                "R4N+X-_gibbs_Eh": G_r4n_x,
                "R4N+Cl-_gibbs_Eh": G_r4n_cl,
                "X-_gibbs_Eh": G_x,
                "Cl-_gibbs_Eh": G_cl,
                "DeltaE_exchange_Eh": delta_E_Eh,
                "DeltaG_exchange_Eh": delta_G_Eh,
                "DeltaE_exchange_kcalmol": delta_E_Eh * HARTREE_TO_KCAL,
                "DeltaG_exchange_kcalmol": delta_G_Eh * HARTREE_TO_KCAL,
            }
        )

summary_df = pd.DataFrame(summary_rows)
summary_df["PFAS"] = pd.Categorical(summary_df["PFAS"], categories=PFAS_ORDER, ordered=True)
summary_df = summary_df.sort_values(["PFAS", "functional"]).reset_index(drop=True)

# Save machine-readable tables
raw_df.to_csv(OUTDIR / "parsed_raw_energies.csv", index=False)
summary_df.to_csv(OUTDIR / "btma_exchange_summary.csv", index=False)

# Print values to console for manuscript use
print("\nDerived BTMA exchange values (kcal/mol):\n")
print(
    summary_df[
        ["functional", "PFAS", "DeltaE_exchange_kcalmol", "DeltaG_exchange_kcalmol"]
    ].to_string(index=False, float_format=lambda x: f"{x:0.3f}")
)

# =========================
# PLOTTING HELPERS
# =========================
FUNC_COLORS = {
    "r2SCAN-3c": "#1f77b4",
    "wB97X-D3": "#ff7f0e",
}

FUNC_DISPLAY = {
    "r2SCAN-3c": r"r$^2$SCAN-3c",
    "wB97X-D3": r"$\omega$B97X-D3",
}


def make_dumbbell_plot(df, value_col, title, ylabel, outfile, ylim=None):
    """
    Paired functional comparison for each PFAS in a vertical dumbbell layout.
    """
    fig, ax = plt.subplots(figsize=(8.6, 4.8), dpi=300)
    label_scale = LABEL_FONT_SIZE / 8.0

    x_positions = list(range(len(PFAS_ORDER)))
    xpad_left = 0.55 + 0.10 * max(label_scale - 1.0, 0.0)
    xpad_right = 0.16 + 0.03 * max(label_scale - 1.0, 0.0)
    ax.set_xlim(-xpad_left, len(PFAS_ORDER) - xpad_right)

    if ylim is not None:
        ax.set_ylim(*ylim)
    lower, upper = ax.get_ylim()
    margin = (0.08 + 0.025 * max(label_scale - 1.0, 0.0)) * (upper - lower)
    zero_buffer = (0.09 + 0.02 * max(label_scale - 1.0, 0.0)) * (upper - lower)

    ax.axhline(0.0, linewidth=1.0, linestyle="--", color="0.45", zorder=1)

    for i, pfas in enumerate(PFAS_ORDER):
        sub = df[df["PFAS"] == pfas].set_index("functional")
        y1 = sub.loc["r2SCAN-3c", value_col]
        y2 = sub.loc["wB97X-D3", value_col]

        # connecting line is always black for clear paired comparison
        ax.plot([i, i], [y1, y2], linewidth=1.8, color="black", zorder=2)

        # points use consistent functional colors across all PFAS
        ax.scatter(
            i,
            y1,
            s=70,
            label=FUNC_DISPLAY["r2SCAN-3c"] if i == 0 else None,
            zorder=3,
            marker="o",
            color=FUNC_COLORS["r2SCAN-3c"],
        )
        ax.scatter(
            i,
            y2,
            s=70,
            label=FUNC_DISPLAY["wB97X-D3"] if i == 0 else None,
            zorder=3,
            marker="s",
            color=FUNC_COLORS["wB97X-D3"],
        )

        # Keep the functional staggering consistent horizontally, and force the
        # two labels for the same PFAS onto different vertical levels.
        # Preferred layout: higher value above, lower value below.
        if y1 >= y2:
            r2_above = True
            wb_above = False
        else:
            r2_above = False
            wb_above = True

        # Keep labels away from the horizontal zero reference line, but do not
        # let that preference push labels outside the plotting box.
        if abs(y1) < zero_buffer:
            r2_above = y1 >= 0
        if abs(y2) < zero_buffer:
            wb_above = y2 >= 0

        # Axis boundaries take priority over zero-line preferences.
        if y1 > upper - margin:
            r2_above = False
        if y2 > upper - margin:
            wb_above = False
        if y1 < lower + margin:
            r2_above = True
        if y2 < lower + margin:
            wb_above = True

        dx = 14 * label_scale
        dy_primary = 12 * label_scale
        dy_secondary = 20 * label_scale
        r2_dx = -dx
        wb_dx = dx
        r2_dy = dy_primary if r2_above else -dy_primary
        wb_dy = dy_secondary if wb_above else -dy_secondary

        ax.annotate(
            f"{y1:.2f}",
            xy=(i, y1),
            xytext=(r2_dx, r2_dy),
            textcoords="offset points",
            fontsize=LABEL_FONT_SIZE,
            ha="right",
            va="bottom" if r2_above else "top",
            clip_on=True,
            annotation_clip=True,
            arrowprops={
                "arrowstyle": "-",
                "color": FUNC_COLORS["r2SCAN-3c"],
                "lw": 0.9,
                "shrinkA": 0,
                "shrinkB": 6 * label_scale,
            },
        )
        ax.annotate(
            f"{y2:.2f}",
            xy=(i, y2),
            xytext=(wb_dx, wb_dy),
            textcoords="offset points",
            fontsize=LABEL_FONT_SIZE,
            ha="left",
            va="bottom" if wb_above else "top",
            clip_on=True,
            annotation_clip=True,
            arrowprops={
                "arrowstyle": "-",
                "color": FUNC_COLORS["wB97X-D3"],
                "lw": 0.9,
                "shrinkA": 0,
                "shrinkB": 6 * label_scale,
            },
        )

    ax.set_xticks(x_positions)
    ax.set_xticklabels(PFAS_ORDER)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    # Keep legend outside the axes, fixed at top-right.
    ax.legend(
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.00),
        borderaxespad=0.0,
        fontsize=max(LABEL_FONT_SIZE, 10),
    )

    fig.subplots_adjust(right=0.78)
    fig.savefig(OUTDIR / outfile, bbox_inches="tight")
    plt.close(fig)


def make_raw_components_plot(df, energy_col, title, outfile):
    """
    Supplementary-style grouped scatter plot of raw thermodynamic-cycle components
    for each PFAS and functional, split into separate figures for SP and Gibbs energies.
    """
    fig, ax = plt.subplots(figsize=(9.5, 5.2), dpi=300)

    components = ["R4N+X-", "R4N+Cl-", "X-", "Cl-"]
    component_x = {comp: i for i, comp in enumerate(components)}

    markers = {"r2SCAN-3c": "o", "wB97X-D3": "s"}
    present_functionals = list(dict.fromkeys(df["functional"].tolist()))

    for func in present_functionals:
        for pfas in PFAS_ORDER:
            subset = df[(df["functional"] == func) & (df["PFAS"] == pfas)]
            if subset.empty:
                continue
            row = subset.iloc[0]

            vals = {
                "R4N+X-": row[f"R4N+X-_{energy_col}"],
                "R4N+Cl-": row[f"R4N+Cl-_{energy_col}"],
                "X-": row[f"X-_{energy_col}"],
                "Cl-": row[f"Cl-_{energy_col}"],
            }

            # Slight offsets so points don't overlap too much
            offset = {
                "FHEA": -0.18,
                "PFHxA": -0.06,
                "PFOA": 0.06,
                "PFOS": 0.18,
            }[pfas]

            xs = [component_x[c] + offset for c in components]
            ys = [vals[c] for c in components]

            ax.plot(
                xs,
                ys,
                marker=markers[func],
                linestyle="-",
                linewidth=1.0,
                label=f"{pfas} ({func})" if component_x["R4N+X-"] == 0 else None,
                alpha=0.9,
            )

    ax.set_xticks(range(len(components)))
    ax.set_xticklabels(components)
    ax.set_ylabel("Absolute energy (Eh)")
    ax.set_title(title)

    # Clean legend: unique labels only
    handles, labels = ax.get_legend_handles_labels()
    seen = set()
    new_handles, new_labels = [], []
    for h, l in zip(handles, labels):
        if l not in seen and l:
            seen.add(l)
            new_handles.append(h)
            new_labels.append(l)
    ax.legend(new_handles, new_labels, frameon=False, fontsize=8, ncol=2)

    fig.tight_layout()
    fig.savefig(OUTDIR / outfile, bbox_inches="tight")
    plt.close(fig)


def make_combined_two_panel(df, outfile):
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(14.0, 12.6),
        dpi=300,
        gridspec_kw={"height_ratios": [1.05, 1.62]},
    )

    configs = [
        (
            "DeltaE_exchange_kcalmol",
            "A",
            r"$\mathbf{\Delta E}_{\mathbf{exchange}}$",
            r"$\Delta E_{\mathrm{exchange}}$ (kcal/mol)",
            (-3.05, 0.65),
        ),
        (
            "DeltaG_exchange_kcalmol",
            "B",
            r"$\mathbf{\Delta G}_{\mathbf{exchange}}$",
            r"$\Delta G_{\mathrm{exchange}}$ (kcal/mol)",
            (4.10, 6.55),
        ),
    ]

    fig.suptitle("BTMA Exchange Energetics", fontsize=16, fontweight="bold", y=0.975)

    for ax, (value_col, letter, title, ylabel, ylim) in zip(axes[0], configs):
        ax.set_ylim(*ylim)
        lower, upper = ax.get_ylim()
        span = upper - lower
        ax.axhline(0.0, linewidth=1.0, linestyle="--", color="0.45", alpha=0.65, zorder=1)

        for i, pfas in enumerate(PFAS_ORDER):
            sub = df[df["PFAS"] == pfas].set_index("functional")
            y1 = sub.loc["r2SCAN-3c", value_col]
            y2 = sub.loc["wB97X-D3", value_col]

            ax.plot([i, i], [y1, y2], linewidth=1.7, color="black", alpha=0.8, zorder=2)
            ax.scatter(
                i,
                y1,
                s=72,
                marker="o",
                label=FUNC_DISPLAY["r2SCAN-3c"] if i == 0 else None,
                zorder=3,
                color=FUNC_COLORS["r2SCAN-3c"],
            )
            ax.scatter(
                i,
                y2,
                s=72,
                marker="s",
                label=FUNC_DISPLAY["wB97X-D3"] if i == 0 else None,
                zorder=3,
                color=FUNC_COLORS["wB97X-D3"],
            )

            high_func, high_val = ("r2SCAN-3c", y1) if y1 >= y2 else ("wB97X-D3", y2)
            low_func, low_val = ("wB97X-D3", y2) if y1 >= y2 else ("r2SCAN-3c", y1)
            for func, val, offset in [(high_func, high_val, 0.045 * span), (low_func, low_val, -0.045 * span)]:
                va = "bottom" if offset > 0 else "top"
                if val + offset > upper:
                    offset = -0.045 * span
                    va = "top"
                if val + offset < lower:
                    offset = 0.045 * span
                    va = "bottom"
                ax.text(
                    i,
                    val + offset,
                    f"{val:.2f}",
                    ha="center",
                    va=va,
                    fontsize=LABEL_FONT_SIZE,
                    fontweight="bold",
                    color=FUNC_COLORS[func],
                    bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=0.12),
                    clip_on=True,
                    zorder=4,
                )

        ax.set_title(title, fontsize=18, fontweight="bold", pad=8)
        ax.set_ylabel(ylabel)
        ax.set_xticks(range(len(PFAS_ORDER)))
        ax.set_xticklabels(PFAS_ORDER)
        ax.set_xlim(-0.45, len(PFAS_ORDER) - 0.55)
        ax.grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.28)
        ax.text(
            -0.02,
            1.08,
            letter,
            transform=ax.transAxes,
            fontsize=17,
            fontweight="bold",
            va="top",
            ha="right",
        )

    structure_panels = [
        (
            axes[1, 0],
            "C",
            STRUCTURE_FIGURE_DIR / "BTMA_PFAS_optimized_structures_orthoscopic_no_panel_letters.png",
        ),
        (
            axes[1, 1],
            "D",
            STRUCTURE_FIGURE_DIR / "BTMA_PFAS_wB97X-D3_structures_orthoscopic_no_panel_letters.png",
        ),
    ]
    for ax, letter, path in structure_panels:
        if not path.exists():
            raise FileNotFoundError(f"Missing structure figure asset for panel {letter}: {path}")
        img = mpimg.imread(path)
        img_h, img_w = img.shape[:2]
        # Preserve native pixel ratio so structure panels are never stretched.
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

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 0.94), ncol=2)

    fig.subplots_adjust(left=0.07, right=0.985, bottom=0.04, top=0.86, wspace=0.18, hspace=0.18)
    fig.savefig(OUTDIR / outfile, bbox_inches="tight")
    fig.savefig(FINAL_OUTDIR / outfile, bbox_inches="tight")
    plt.close(fig)


# =========================
# CREATE FIGURES
# =========================

# Main-text consolidated figure
make_combined_two_panel(summary_df, "Figure_03_BTMA_exchange_energetics.png")

# Supplementary-style raw component figures
make_raw_components_plot(
    summary_df[summary_df["functional"] == "r2SCAN-3c"],
    energy_col="single_point_Eh",
    title="BTMA thermodynamic-cycle components (r2SCAN-3c single-point energies)",
    outfile="Supp_BTMA_raw_components_r2scan3c_single_point.png",
)

make_raw_components_plot(
    summary_df[summary_df["functional"] == "wB97X-D3"],
    energy_col="single_point_Eh",
    title="BTMA thermodynamic-cycle components (wB97X-D3 single-point energies)",
    outfile="Supp_BTMA_raw_components_wB97XD3_single_point.png",
)

make_raw_components_plot(
    summary_df[summary_df["functional"] == "r2SCAN-3c"],
    energy_col="gibbs_Eh",
    title="BTMA thermodynamic-cycle components (r2SCAN-3c Gibbs free energies)",
    outfile="Supp_BTMA_raw_components_r2scan3c_gibbs.png",
)

make_raw_components_plot(
    summary_df[summary_df["functional"] == "wB97X-D3"],
    energy_col="gibbs_Eh",
    title="BTMA thermodynamic-cycle components (wB97X-D3 Gibbs free energies)",
    outfile="Supp_BTMA_raw_components_wB97XD3_gibbs.png",
)

print(f"\nSaved figures and tables to: {OUTDIR.resolve()}")
print("\nFiles created:")
for p in sorted(OUTDIR.iterdir()):
    print(" -", p.name)
