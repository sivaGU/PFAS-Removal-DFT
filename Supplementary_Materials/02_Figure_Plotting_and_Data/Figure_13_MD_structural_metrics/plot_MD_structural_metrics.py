from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Settings
THIS_DIR = Path(__file__).resolve().parent
METRICS_DIR = THIS_DIR / "input_data"
OUTPUT_DIR = THIS_DIR / "outputs"

ASSOCIATION_METRICS_CSV = METRICS_DIR / "pfoa_association_mechanism_metrics.csv"
TAIL_WATER_CSV = METRICS_DIR / "pfoa_tail_waters.csv"

OUTPUT = OUTPUT_DIR / "Figure_13_MD_structural_metrics.png"

DPI = 600
FONT_FAMILY = "DejaVu Sans"
BASE_FONT_SIZE = 13
FRAME_SPACING_NS = 0.01
PFOA_AMMONIUM_PROXIMITY_THRESHOLD_A = 5.0
TAIL_RESIN_CONTACT_CUTOFF_A = 4.0
CHLORIDE_OCCUPANCY_CUTOFF_A = 5.0
TAIL_WATER_CUTOFF_A = 5.0

plt.rcParams.update(
    {
        "font.family": FONT_FAMILY,
        "font.size": BASE_FONT_SIZE,
        "axes.titlesize": BASE_FONT_SIZE + 1,
        "axes.titleweight": "bold",
        "axes.labelsize": BASE_FONT_SIZE + 1,
        "xtick.labelsize": BASE_FONT_SIZE,
        "ytick.labelsize": BASE_FONT_SIZE,
        "legend.fontsize": BASE_FONT_SIZE - 1,
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
            + "\nExpected columns/files:\n"
            + "  pfoa_association_mechanism_metrics.csv: frame, nearest_carboxylate_O_to_any_resin_N_A, "
            + "pfoa_tail_resin_heavy_contacts_4A, chlorides_within_5A_of_nearest_N\n"
            + "  pfoa_tail_waters.csv: frame, tail_waters_5A"
        )
        raise FileNotFoundError(msg)


def load_tail_waters(path: Path) -> pd.DataFrame:
    waters = pd.read_csv(path)
    required = {"frame", "tail_waters_5A"}
    if not required.issubset(waters.columns):
        raise ValueError(f"{path} must contain columns {sorted(required)}")
    return waters


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    require_inputs([ASSOCIATION_METRICS_CSV, TAIL_WATER_CSV])
    metrics = pd.read_csv(ASSOCIATION_METRICS_CSV)
    required = {
        "frame",
        "nearest_carboxylate_O_to_any_resin_N_A",
        "pfoa_tail_resin_heavy_contacts_4A",
        "chlorides_within_5A_of_nearest_N",
    }
    if not required.issubset(metrics.columns):
        raise ValueError(f"{ASSOCIATION_METRICS_CSV} must contain columns {sorted(required)}")
    waters = load_tail_waters(TAIL_WATER_CSV)
    metrics["time_ns"] = (metrics["frame"] - metrics["frame"].min()) * FRAME_SPACING_NS
    waters["time_ns"] = (waters["frame"] - waters["frame"].min()) * FRAME_SPACING_NS
    return metrics, waters


def style_axis(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(length=5, width=1.2)
    ax.xaxis.labelpad = 10
    ax.yaxis.labelpad = 10
    ax.grid(axis="y", color="#d0d0d0", linestyle="--", linewidth=0.8, alpha=0.55)


def add_panel_label(ax, letter: str):
    return


def add_aligned_panel_labels(fig, axes: list[plt.Axes], letters: list[str]):
    x_offset = 0.032
    y_offset = 0.030
    row_tops = {
        0: max(ax.get_position().y1 for ax in axes[:3]),
        1: max(ax.get_position().y1 for ax in axes[3:]),
    }
    for idx, (ax, letter) in enumerate(zip(axes, letters)):
        pos = ax.get_position()
        row = 0 if idx < 3 else 1
        fig.text(
            pos.x0 - x_offset,
            row_tops[row] + y_offset,
            letter,
            fontsize=BASE_FONT_SIZE + 8,
            fontweight="bold",
            va="top",
            ha="right",
        )


def draw_distance_timeseries(ax, metrics: pd.DataFrame):
    x = metrics["time_ns"]
    y = metrics["nearest_carboxylate_O_to_any_resin_N_A"]
    ax.plot(x, y, color="#4c78a8", lw=1.8, alpha=0.55, label="Frame value")
    ax.plot(
        x,
        y.rolling(window=5, center=True, min_periods=1).mean(),
        color="#1f4e79",
        lw=2.6,
        label="5-frame running mean",
    )
    ax.axhline(PFOA_AMMONIUM_PROXIMITY_THRESHOLD_A, color="#555555", lw=1.2, linestyle="--")
    ax.set_xlabel("Time (ns)")
    ax.set_ylabel("Minimum O...N distance (Å)")
    ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.46, 1.02), borderaxespad=0.0)
    add_panel_label(ax, "A")
    style_axis(ax)


def draw_distance_distribution(ax, metrics: pd.DataFrame):
    y = metrics["nearest_carboxylate_O_to_any_resin_N_A"]
    fraction = (y < PFOA_AMMONIUM_PROXIMITY_THRESHOLD_A).mean()
    ax.hist(y, bins=16, color="#4c78a8", edgecolor="white", linewidth=0.8)
    ax.axvline(PFOA_AMMONIUM_PROXIMITY_THRESHOLD_A, color="#555555", lw=1.2, linestyle="--")
    ax.set_xlabel("Minimum O...N distance (Å)")
    ax.set_ylabel("Frames")
    ax.text(
        0.98,
        0.92,
        f"{fraction:.0%} < {PFOA_AMMONIUM_PROXIMITY_THRESHOLD_A:.1f} Å",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=BASE_FONT_SIZE - 1,
    )
    add_panel_label(ax, "B")
    style_axis(ax)


def draw_tail_contacts(ax, metrics: pd.DataFrame):
    contacts = metrics["pfoa_tail_resin_heavy_contacts_4A"]
    bins = np.arange(contacts.min() - 0.5, contacts.max() + 1.5, 1)
    ax.hist(contacts, bins=bins, color="#59a14f", edgecolor="white", linewidth=0.8)
    ax.set_xlabel(f"Tail-resin heavy-atom contacts\nwithin {TAIL_RESIN_CONTACT_CUTOFF_A:.0f} Å")
    ax.set_ylabel("Frames")
    add_panel_label(ax, "C")
    style_axis(ax)


def draw_chloride_occupancy(ax, metrics: pd.DataFrame, panel_label: str):
    counts = metrics["chlorides_within_5A_of_nearest_N"].astype(int)
    count_table = counts.value_counts().sort_index()
    ax.bar(
        count_table.index.astype(str),
        count_table.values,
        color="#f58518",
        edgecolor="#222222",
        linewidth=1.0,
    )
    ax.set_xlabel(f"Cl$^-$ within {CHLORIDE_OCCUPANCY_CUTOFF_A:.0f} Å\nof nearest ammonium N")
    ax.set_ylabel("Frames")
    add_panel_label(ax, panel_label)
    style_axis(ax)


def draw_tail_hydration(ax, waters: pd.DataFrame, panel_label: str):
    values = waters["tail_waters_5A"]
    bins = np.arange(values.min() - 0.5, values.max() + 1.5, 1)
    ax.hist(values, bins=bins, color="#76b7b2", edgecolor="white", linewidth=0.8)
    ax.set_xlabel(f"Waters within {TAIL_WATER_CUTOFF_A:.0f} Å\nof fluorinated tail")
    ax.set_ylabel("Frames")
    add_panel_label(ax, panel_label)
    style_axis(ax)


def make_figure() -> dict:
    metrics, waters = load_data()

    fig = plt.figure(figsize=(15.8, 10.9), dpi=DPI)
    fig.suptitle(r"PFOA$^{-}$ Structural Metrics in the Ph-BTMA$_{48}^{48+}$ Trajectory", y=0.985)

    gs = fig.add_gridspec(2, 6, hspace=0.58, wspace=1.12)
    axes = [
        fig.add_subplot(gs[0, 0:2]),
        fig.add_subplot(gs[0, 2:4]),
        fig.add_subplot(gs[0, 4:6]),
        fig.add_subplot(gs[1, 0:3]),
        fig.add_subplot(gs[1, 3:6]),
    ]

    draw_distance_timeseries(axes[0], metrics)
    draw_distance_distribution(axes[1], metrics)
    draw_tail_contacts(axes[2], metrics)

    draw_tail_hydration(axes[3], waters, panel_label="D")
    draw_chloride_occupancy(axes[4], metrics, panel_label="E")

    fig.subplots_adjust(top=0.89)
    add_aligned_panel_labels(fig, axes, ["A", "B", "C", "D", "E"])

    OUTPUT_DIR.mkdir(exist_ok=True)
    fig.savefig(OUTPUT, dpi=DPI, bbox_inches="tight")
    plt.close(fig)

    return {
        "distance_fraction_below_threshold": float(
            (metrics["nearest_carboxylate_O_to_any_resin_N_A"] < PFOA_AMMONIUM_PROXIMITY_THRESHOLD_A).mean()
        ),
        "mean_distance_A": float(metrics["nearest_carboxylate_O_to_any_resin_N_A"].mean()),
        "mean_tail_contacts": float(metrics["pfoa_tail_resin_heavy_contacts_4A"].mean()),
        "mean_chlorides_5A": float(metrics["chlorides_within_5A_of_nearest_N"].mean()),
        "mean_tail_waters_5A": float(waters["tail_waters_5A"].mean()),
    }


if __name__ == "__main__":
    summary = make_figure()

    print("MD structural metrics figure generated.")
    print(f"Input files used:\n  - {ASSOCIATION_METRICS_CSV}\n  - {TAIL_WATER_CSV}")
    print(f"PFOA O...N proximity threshold: {PFOA_AMMONIUM_PROXIMITY_THRESHOLD_A:.1f} Å")
    print(f"Tail-resin contact cutoff: {TAIL_RESIN_CONTACT_CUTOFF_A:.1f} Å")
    print(f"Chloride occupancy cutoff: {CHLORIDE_OCCUPANCY_CUTOFF_A:.1f} Å")
    print(f"Tail-water cutoff: {TAIL_WATER_CUTOFF_A:.1f} Å")
    print("Summary values:")

    for key, value in summary.items():
        print(f"  {key}: {value:.4f}")

    print(f"Output:\n  - {OUTPUT}")
