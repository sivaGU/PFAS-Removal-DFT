import csv
from pathlib import Path

import matplotlib.pyplot as plt


# Settings
OUTDIR = Path(__file__).resolve().parent
DATAFILE = OUTDIR / "input_data" / "nbo_nao_orbital_levels.csv"
OUTPUT_DIR = OUTDIR / "outputs"
OUTFILE = OUTPUT_DIR / "Figure_06_NBO_analysis_pfoa_cholestyramine.png"

FONT_FAMILY = "DejaVu Sans"
DPI = 600
HARTREE_TO_EV = 27.211386245988

plt.rcParams.update(
    {
        "font.family": FONT_FAMILY,
        "font.size": 18,
        "axes.titlesize": 24,
        "axes.titleweight": "bold",
        "axes.labelsize": 22,
        "xtick.labelsize": 19,
        "ytick.labelsize": 18,
    }
)


# Data
REQUIRED_KEYS = {
    "Ch_H_1s",
    "O_2px",
    "O_2py",
    "O_2pz",
    "CH_Sigma_Star",
    "O_LP",
}
REQUIRED_COLUMNS = {
    "key",
    "label",
    "orbital_or_nbo_number",
    "occupancy",
    "energy_hartree",
    "hybridization_primary",
    "hybridization_secondary",
    "source_section",
}


def load_orbital_data(datafile=DATAFILE):
    with datafile.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing_columns = REQUIRED_COLUMNS - set(reader.fieldnames or [])
        if missing_columns:
            names = ", ".join(sorted(missing_columns))
            raise ValueError(f"Missing required CSV columns: {names}")

        records = {}
        for line_number, row in enumerate(reader, start=2):
            key = row["key"].strip()
            if not key:
                raise ValueError(f"Missing record key on CSV line {line_number}")
            if key in records:
                raise ValueError(f"Duplicate record key in CSV: {key}")

            try:
                energy_hartree = float(row["energy_hartree"])
                occupancy = float(row["occupancy"])
            except ValueError as exc:
                raise ValueError(
                    f"Invalid numeric value for {key} on CSV line {line_number}"
                ) from exc

            records[key] = {
                "E": energy_hartree * HARTREE_TO_EV,
                "Energy_Hartree": energy_hartree,
                "Occupancy": occupancy,
                "Label": row["label"].strip(),
                "Hyb_Primary": row["hybridization_primary"].strip(),
                "Hyb_Secondary": row["hybridization_secondary"].strip(),
            }

    missing_keys = REQUIRED_KEYS - records.keys()
    if missing_keys:
        names = ", ".join(sorted(missing_keys))
        raise ValueError(f"Missing required orbital records: {names}")

    return records


# Helpers
def draw_level(ax, x, y, width, color="k", style="-", lw=4):
    ax.hlines(y, x - width, x + width, color=color, lw=lw, linestyles=style, zorder=4)


def draw_arrows(ax, x, y, count, color="k"):
    scale = 0.52
    sep = 0.065
    arrow_kwargs = {
        "head_width": 0.055,
        "head_length": 0.14,
        "fc": color,
        "ec": color,
        "linewidth": 1.2,
        "length_includes_head": True,
        "zorder": 6,
    }
    if count > 0:
        ax.arrow(x - sep, y - scale / 2, 0, scale, **arrow_kwargs)
    if count > 1:
        ax.arrow(x + sep, y + scale / 2, 0, -scale, **arrow_kwargs)


def connect(ax, x1, y1, x2, y2, color="gray", style="--", alpha=0.38):
    """Draw faint orbital relationship lines behind labels and levels."""
    arrowstyle = "-"
    ax.annotate(
        "",
        xy=(x2, y2),
        xytext=(x1, y1),
        arrowprops=dict(
            arrowstyle=arrowstyle,
            color=color,
            linestyle=style,
            lw=2.0,
            alpha=alpha,
            connectionstyle="arc3,rad=0",
        ),
        zorder=1,
    )


def label_box(ax, x, y, text, color="black", ha="center", va="center", size=18, weight="bold"):
    ax.text(
        x,
        y,
        text,
        ha=ha,
        va=va,
        fontsize=size,
        fontweight=weight,
        color=color,
        zorder=10,
        bbox=dict(boxstyle="round,pad=0.28", facecolor="white", edgecolor="none", alpha=0.88),
    )


# Figure
def make_nao_to_nbo_diagram():
    data = load_orbital_data()
    fig, ax = plt.subplots(figsize=(15.5, 18))

    x_ch, x_cpx, x_pfoa = 0.35, 2.75, 5.15
    w = 0.42
    w_stag = 0.18

    connect(
        ax,
        x_pfoa - w_stag,
        data["O_2pz"]["E"],
        x_cpx + w,
        data["O_LP"]["E"],
        color="#1f77b4",
        style="--",
        alpha=0.34,
    )
    connect(
        ax,
        x_pfoa - w_stag,
        data["O_2pz"]["E"],
        x_cpx + w,
        data["CH_Sigma_Star"]["E"],
        color="#1f77b4",
        style=":",
        alpha=0.30,
    )
    connect(
        ax,
        x_ch + w,
        data["Ch_H_1s"]["E"],
        x_cpx - w,
        data["CH_Sigma_Star"]["E"],
        color="black",
        style="--",
        alpha=0.34,
    )
    connect(
        ax,
        x_ch + w,
        data["Ch_H_1s"]["E"],
        x_cpx - w,
        data["O_LP"]["E"],
        color="black",
        style=":",
        alpha=0.28,
    )

    draw_level(ax, x_ch, data["Ch_H_1s"]["E"], w, "black")
    label_box(
        ax,
        x_ch - 0.12,
        data["Ch_H_1s"]["E"] + 0.78,
        f"{data['Ch_H_1s']['Label']}\n{data['Ch_H_1s']['E']:.2f} eV",
        ha="center",
        size=18,
    )

    draw_level(ax, x_pfoa, data["O_2pz"]["E"], w_stag, "#1f77b4")
    draw_arrows(ax, x_pfoa, data["O_2pz"]["E"], 2, "#1f77b4")
    label_box(
        ax,
        x_pfoa + 0.24,
        data["O_2pz"]["E"] + 0.66,
        f"{data['O_2pz']['Label']}\n{data['O_2pz']['E']:.2f} eV",
        color="#1f77b4",
        ha="left",
        va="bottom",
        size=18,
    )

    draw_level(ax, x_pfoa + 0.42, data["O_2py"]["E"], w_stag, "#7b3294")
    draw_arrows(ax, x_pfoa + 0.42, data["O_2py"]["E"], 2, "#7b3294")
    draw_level(ax, x_pfoa - 0.42, data["O_2px"]["E"], w_stag, "#7b3294")
    draw_arrows(ax, x_pfoa - 0.42, data["O_2px"]["E"], 2, "#7b3294")
    label_box(
        ax,
        x_pfoa - 0.46,
        data["O_2px"]["E"] - 0.54,
        f"O44 2p$_x$/2p$_y$\n{data['O_2px']['E']:.2f} / {data['O_2py']['E']:.2f} eV",
        color="#7b3294",
        ha="right",
        va="top",
        size=16,
    )

    draw_level(ax, x_cpx, data["CH_Sigma_Star"]["E"], w, "#d62728")
    label_box(
        ax,
        x_cpx - 0.22,
        data["CH_Sigma_Star"]["E"] + 0.76,
        f"{data['CH_Sigma_Star']['Label']}\n{data['CH_Sigma_Star']['E']:.2f} eV",
        color="#d62728",
        ha="center",
        va="bottom",
        size=18,
    )
    label_box(
        ax,
        x_cpx + w + 0.24,
        data["CH_Sigma_Star"]["E"] - 0.22,
        f"{data['CH_Sigma_Star']['Hyb_Primary']}\n"
        f"{data['CH_Sigma_Star']['Hyb_Secondary']}",
        color="#d62728",
        ha="left",
        va="top",
        size=14,
        weight="normal",
    )

    draw_level(ax, x_cpx, data["O_LP"]["E"], w, "#1f77b4")
    draw_arrows(ax, x_cpx, data["O_LP"]["E"], 2, "#1f77b4")
    label_box(
        ax,
        x_cpx - 0.18,
        data["O_LP"]["E"] - 0.78,
        f"{data['O_LP']['Label']}\n{data['O_LP']['E']:.2f} eV",
        color="#1f77b4",
        ha="center",
        va="top",
        size=18,
    )
    label_box(
        ax,
        x_cpx + w + 0.24,
        data["O_LP"]["E"] + 0.45,
        data["O_LP"]["Hyb_Primary"],
        color="#1f77b4",
        ha="left",
        va="bottom",
        size=14,
        weight="normal",
    )

    ax.set_ylabel("")
    fig.text(
        0.09,
        0.5,
        "Energy (eV)",
        ha="center",
        va="center",
        rotation="vertical",
        fontsize=22,
        fontweight="bold",
    )
    ax.set_xticks([x_ch, x_cpx, x_pfoa])
    ax.set_xticklabels(
        ["BTMA$^{+}$\n(NAO)", "BTMA$^{+}$ + PFOA$^{-}$ Interaction\n(NBO)", "PFOA$^{-}$\n(NAO)"],
        fontweight="bold",
    )
    ax.set_xlim(-0.7, 6.2)
    ax.set_ylim(-15.6, 14.2)
    ax.set_title(r"NAO-to-NBO Interaction Diagram: BTMA$^{+}$ + PFOA$^{-}$", pad=20)
    ax.grid(axis="y", linestyle="--", linewidth=0.8, alpha=0.22)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="x", pad=14)
    ax.tick_params(axis="y", width=1.4, length=7)

    fig.tight_layout(rect=[0.075, 0.02, 1.0, 0.97])
    OUTPUT_DIR.mkdir(exist_ok=True)
    fig.savefig(OUTFILE, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure to: {OUTFILE}")


if __name__ == "__main__":
    make_nao_to_nbo_diagram()
