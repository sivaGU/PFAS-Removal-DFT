from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import pandas as pd


HERE = Path(__file__).resolve().parent
INPUT_CSV = HERE / "input_data" / "btma_exchange_summary.csv"
STRUCTURE_DIR = HERE / "input_structures"
OUTPUT_DIR = HERE / "outputs"
OUTPUT = OUTPUT_DIR / "Figure_03_BTMA_exchange_energetics.png"

PFAS_ORDER = ["FHEA", "PFHxA", "PFOA", "PFOS"]
FUNCTIONALS = ["r2SCAN-3c", "wB97X-D3"]
FUNC_COLORS = {"r2SCAN-3c": "#1f77b4", "wB97X-D3": "#ff7f0e"}
FUNC_DISPLAY = {"r2SCAN-3c": r"r$^2$SCAN-3c", "wB97X-D3": r"$\omega$B97X-D3"}
LABEL_FONT_SIZE = 11

plt.rcParams["font.family"] = "DejaVu Sans"


def load_data() -> pd.DataFrame:
    data = pd.read_csv(INPUT_CSV)
    required = {"functional", "PFAS", "DeltaE_exchange_kcalmol", "DeltaG_exchange_kcalmol"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"{INPUT_CSV} is missing columns: {sorted(missing)}")
    if set(data["functional"]) != set(FUNCTIONALS):
        raise ValueError("The curated CSV must contain both manuscript functionals")
    data["PFAS"] = pd.Categorical(data["PFAS"], categories=PFAS_ORDER, ordered=True)
    return data.sort_values(["PFAS", "functional"]).reset_index(drop=True)


def draw_energy_panel(ax, data: pd.DataFrame, value_col: str, letter: str, title: str, ylabel: str, ylim) -> None:
    ax.set_ylim(*ylim)
    lower, upper = ax.get_ylim()
    span = upper - lower
    ax.axhline(0.0, linewidth=1.0, linestyle="--", color="0.45", alpha=0.65, zorder=1)

    for i, pfas in enumerate(PFAS_ORDER):
        values = data[data["PFAS"] == pfas].set_index("functional")[value_col]
        r2, wb = values.loc["r2SCAN-3c"], values.loc["wB97X-D3"]
        ax.plot([i, i], [r2, wb], linewidth=1.7, color="black", alpha=0.8, zorder=2)

        for functional, value, marker in (("r2SCAN-3c", r2, "o"), ("wB97X-D3", wb, "s")):
            ax.scatter(
                i,
                value,
                s=72,
                marker=marker,
                label=FUNC_DISPLAY[functional] if i == 0 else None,
                zorder=3,
                color=FUNC_COLORS[functional],
            )

        high_func, high_val = ("r2SCAN-3c", r2) if r2 >= wb else ("wB97X-D3", wb)
        low_func, low_val = ("wB97X-D3", wb) if r2 >= wb else ("r2SCAN-3c", r2)
        for functional, value, offset in (
            (high_func, high_val, 0.045 * span),
            (low_func, low_val, -0.045 * span),
        ):
            va = "bottom" if offset > 0 else "top"
            if value + offset > upper:
                offset, va = -0.045 * span, "top"
            if value + offset < lower:
                offset, va = 0.045 * span, "bottom"
            ax.text(
                i,
                value + offset,
                f"{value:.2f}",
                ha="center",
                va=va,
                fontsize=LABEL_FONT_SIZE,
                fontweight="bold",
                color=FUNC_COLORS[functional],
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 0.12},
                clip_on=True,
                zorder=4,
            )

    ax.set_title(title, fontsize=18, fontweight="bold", pad=8)
    ax.set_ylabel(ylabel)
    ax.set_xticks(range(len(PFAS_ORDER)), PFAS_ORDER)
    ax.set_xlim(-0.45, len(PFAS_ORDER) - 0.55)
    ax.grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.28)
    ax.text(-0.02, 1.08, letter, transform=ax.transAxes, fontsize=17, fontweight="bold", va="top", ha="right")


def draw_structure_panel(ax, path: Path, letter: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing structure panel: {path}")
    ax.imshow(mpimg.imread(path), aspect="equal")
    ax.set_axis_off()
    ax.text(-0.02, 1.02, letter, transform=ax.transAxes, fontsize=17, fontweight="bold", va="top", ha="right")


def make_figure() -> None:
    data = load_data()
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(14.0, 12.6),
        dpi=300,
        gridspec_kw={"height_ratios": [1.05, 1.62]},
    )
    fig.suptitle(r"BTMA$^{+}$ P$^{-}$ Exchange Energetics", fontsize=16, fontweight="bold", y=0.975)

    draw_energy_panel(
        axes[0, 0], data, "DeltaE_exchange_kcalmol", "A",
        r"$\mathbf{\Delta E}_{\mathbf{exchange}}$",
        r"$\Delta E_{\mathrm{exchange}}$ (kcal/mol)", (-3.05, 0.65),
    )
    draw_energy_panel(
        axes[0, 1], data, "DeltaG_exchange_kcalmol", "B",
        r"$\mathbf{\Delta G}_{\mathbf{exchange}}$",
        r"$\Delta G_{\mathrm{exchange}}$ (kcal/mol)", (4.10, 6.55),
    )
    draw_structure_panel(
        axes[1, 0],
        STRUCTURE_DIR / "BTMA_PFAS_optimized_structures_orthoscopic_no_panel_letters.png",
        "C",
    )
    draw_structure_panel(
        axes[1, 1],
        STRUCTURE_DIR / "BTMA_PFAS_wB97X-D3_structures_orthoscopic_no_panel_letters.png",
        "D",
    )

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 0.94), ncol=2)
    fig.subplots_adjust(left=0.07, right=0.985, bottom=0.04, top=0.86, wspace=0.18, hspace=0.18)
    OUTPUT_DIR.mkdir(exist_ok=True)
    fig.savefig(OUTPUT, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {OUTPUT}")


if __name__ == "__main__":
    make_figure()
