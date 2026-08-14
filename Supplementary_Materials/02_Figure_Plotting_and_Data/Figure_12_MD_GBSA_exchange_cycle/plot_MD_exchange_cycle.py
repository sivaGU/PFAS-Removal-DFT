from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Settings
THIS_DIR = Path(__file__).resolve().parent
GBSA_DIR = THIS_DIR / "input_data"
OUTPUT_DIR = THIS_DIR / "outputs"

SUMMARY_CSV = GBSA_DIR / "exchange_cycle_proxy_summary.csv"
PER_FRAME_CSV = GBSA_DIR / "exchange_cycle_proxy_per_frame.csv"

OUTPUT = OUTPUT_DIR / "Figure_12_MD_GBSA_exchange_cycle.png"

DPI = 600
FONT_FAMILY = "DejaVu Sans"
BASE_FONT_SIZE = 13

BOOTSTRAP_SAMPLES = 20000
BOOTSTRAP_SEED = 20260430

TERM_LABELS = {
    "R48_PFOA_47Cl": r"Ph-BTMA$_{48}^{48+}$(PFOA$^{-}$)(Cl$^{-}$)$_{47}$",
    "R48_48Cl": r"Ph-BTMA$_{48}^{48+}$(Cl$^{-}$)$_{48}$",
    "PFOA_aq": r"PFOA$^{-}$",
    "Cl_aq": r"Cl$^{-}$",
}

# Endpoint signs
ENDPOINT_SIGNS = {
    "R48_PFOA_47Cl": 1,
    "R48_48Cl": -1,
    "PFOA_aq": -1,
    "Cl_aq": 1,
}

plt.rcParams.update(
    {
        "font.family": FONT_FAMILY,
        "font.size": BASE_FONT_SIZE,
        "axes.titlesize": BASE_FONT_SIZE + 2,
        "axes.titleweight": "bold",
        "axes.labelsize": BASE_FONT_SIZE + 1,
        "xtick.labelsize": BASE_FONT_SIZE,
        "ytick.labelsize": BASE_FONT_SIZE,
        "legend.fontsize": BASE_FONT_SIZE,
        "figure.titlesize": BASE_FONT_SIZE + 5,
        "figure.titleweight": "bold",
        "axes.linewidth": 1.3,
        "xtick.major.width": 1.2,
        "ytick.major.width": 1.2,
    }
)


def require_inputs(paths: list[Path]) -> None:
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        msg = (
            "Missing required input file(s):\n"
            + "\n".join(f"  - {item}" for item in missing)
            + "\nExpected columns:\n"
            + "  exchange_cycle_proxy_summary.csv: term, mean_kcal_mol, sem_kcal_mol, n_frames\n"
            + "  exchange_cycle_proxy_per_frame.csv: frame, exchange_proxy_kcal_mol"
        )
        raise FileNotFoundError(msg)


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    require_inputs([SUMMARY_CSV, PER_FRAME_CSV])
    summary = pd.read_csv(SUMMARY_CSV)
    per_frame = pd.read_csv(PER_FRAME_CSV)

    required_summary = {"term", "mean_kcal_mol", "sem_kcal_mol", "n_frames"}
    required_per_frame = {"frame", "exchange_proxy_kcal_mol"}
    if not required_summary.issubset(summary.columns):
        raise ValueError(f"{SUMMARY_CSV} must contain columns {sorted(required_summary)}")
    if not required_per_frame.issubset(per_frame.columns):
        raise ValueError(f"{PER_FRAME_CSV} must contain columns {sorted(required_per_frame)}")

    return summary, per_frame


def bootstrap_mean_ci(values: np.ndarray) -> tuple[float, float]:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    samples = rng.choice(values, size=(BOOTSTRAP_SAMPLES, values.size), replace=True)
    means = samples.mean(axis=1)
    return tuple(np.percentile(means, [2.5, 97.5]))


def style_axis(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(length=5, width=1.2)


def add_panel_label(ax, letter: str):
    ax.text(
        -0.12,
        1.08,
        letter,
        transform=ax.transAxes,
        fontsize=BASE_FONT_SIZE + 8,
        fontweight="bold",
        va="top",
        ha="right",
    )


def draw_endpoint_contributions(ax, summary: pd.DataFrame):
    endpoint_rows = summary[summary["term"].isin(ENDPOINT_SIGNS)].copy()
    endpoint_rows["sign"] = endpoint_rows["term"].map(ENDPOINT_SIGNS)
    endpoint_rows["signed_contribution"] = endpoint_rows["sign"] * endpoint_rows["mean_kcal_mol"]
    endpoint_rows["signed_sem"] = endpoint_rows["sem_kcal_mol"].abs()
    endpoint_rows = endpoint_rows.set_index("term").loc[list(ENDPOINT_SIGNS)].reset_index()

    colors = ["#1f77b4" if value >= 0 else "#ff7f0e" for value in endpoint_rows["signed_contribution"]]
    x = np.arange(len(endpoint_rows))
    ax.bar(
        x,
        endpoint_rows["signed_contribution"],
        yerr=endpoint_rows["signed_sem"],
        color=colors,
        edgecolor="#222222",
        linewidth=1.0,
        capsize=4,
        width=0.68,
    )
    ax.axhline(0, color="#555555", lw=1.2, linestyle="--", zorder=0)
    ax.set_xticks(x)
    ax.set_xticklabels([TERM_LABELS[term] for term in endpoint_rows["term"]], rotation=18, ha="right")
    ax.set_ylabel(r"Signed contribution to $\Delta G_{\mathrm{exchange}}$ (kcal/mol)")
    ax.yaxis.set_label_coords(-0.20, 0.60)
    ax.set_title("Endpoint Contributions")
    add_panel_label(ax, "A")
    style_axis(ax)

    values = endpoint_rows["signed_contribution"].to_numpy(float)
    span = values.max() - values.min()
    y_margin = max(span * 0.12, 150.0)
    ax.set_ylim(values.min() - y_margin, values.max() + y_margin)

    pad = max(abs(endpoint_rows["signed_contribution"])) * 0.035
    for xpos, value in zip(x, endpoint_rows["signed_contribution"]):
        va = "bottom" if value >= 0 else "top"
        offset = pad if value >= 0 else -pad
        ax.text(xpos, value + offset, f"{value:+.1f}", ha="center", va=va, fontsize=BASE_FONT_SIZE - 1)


def draw_exchange_proxy(ax, summary: pd.DataFrame, per_frame: pd.DataFrame) -> tuple[float, tuple[float, float]]:
    proxy_row = summary.loc[summary["term"] == "exchange_proxy"]
    if proxy_row.empty:
        raise ValueError(f"{SUMMARY_CSV} does not contain term='exchange_proxy'")

    proxy = float(proxy_row["mean_kcal_mol"].iloc[0])
    values = per_frame["exchange_proxy_kcal_mol"].dropna().to_numpy(float)
    ci_low, ci_high = bootstrap_mean_ci(values)
    yerr = np.array([[proxy - ci_low], [ci_high - proxy]])

    ax.bar([0], [proxy], yerr=yerr, color="#4c78a8", edgecolor="#222222", linewidth=1.0, capsize=5, width=0.52)
    ax.axhline(0, color="#555555", lw=1.2, linestyle="--", zorder=0)
    ax.set_xlim(-0.75, 0.75)
    ax.set_xticks([0])
    ax.set_xticklabels(["PFOA- exchange"])
    ax.set_ylabel(r"$\Delta G_{\mathrm{exchange}}$ (kcal/mol)")
    ax.set_title(r"Total $\mathbf{\Delta G}_{\mathbf{exchange}}$")
    add_panel_label(ax, "B")
    style_axis(ax)
    ax.text(0.31, proxy * 0.90, f"{proxy:.1f}", ha="left", va="center", color="black", fontweight="bold")
    ax.text(0.10, ci_low, "*", ha="left", va="center", fontsize=BASE_FONT_SIZE + 4, fontweight="bold")
    ax.text(
        0.5,
        -0.22,
        r"* 95% confidence interval of $\Delta G_{\mathrm{exchange}}$",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=BASE_FONT_SIZE - 2,
    )
    return proxy, (ci_low, ci_high)


def make_figure() -> tuple[float, tuple[float, float]]:
    summary, per_frame = load_data()

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(13.5, 5.6), dpi=DPI)
    fig.suptitle(r"MM/GBSA Estimation of $\mathbf{\Delta G}_{\mathbf{exchange}}$ for PFOA- Exchange", y=0.98)

    draw_endpoint_contributions(ax_a, summary)
    proxy, ci = draw_exchange_proxy(ax_b, summary, per_frame)

    fig.subplots_adjust(top=0.82, bottom=0.26, wspace=0.36)
    OUTPUT_DIR.mkdir(exist_ok=True)
    fig.savefig(OUTPUT, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return proxy, ci


if __name__ == "__main__":
    proxy_value, proxy_ci = make_figure()
    print("MM/GBSA exchange-cycle figure generated.")
    print(f"Input files used:\n  - {SUMMARY_CSV}\n  - {PER_FRAME_CSV}")
    print(f"Exchange-cycle proxy: {proxy_value:.6f} kcal/mol")
    print(f"Frame-bootstrap 95% CI: [{proxy_ci[0]:.6f}, {proxy_ci[1]:.6f}] kcal/mol")
    print(f"Output:\n  - {OUTPUT}")
