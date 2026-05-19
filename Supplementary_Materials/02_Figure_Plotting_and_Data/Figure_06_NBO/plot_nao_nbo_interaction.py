from pathlib import Path

import matplotlib.pyplot as plt


# =========================
# USER SETTINGS
# =========================
OUTDIR = Path(__file__).resolve().parent
OUTFILE = OUTDIR / "nao_to_nbo_pfoa_cholestyramine.png"

FONT_FAMILY = "DejaVu Sans"
DPI = 600

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


# =========================
# HARD-CODED DATA
# =========================
data = {
    "Ch_H_1s": {
        "E": 2.15,
        "Label": "Ch H(38) 1s",
    },
    "O_2pz": {
        "E": -10.85,
        "Label": r"O 2p$_z$ (n)",
    },
    "O_2px": {"E": -11.42},
    "O_2py": {"E": -11.35},
    "CH_Sigma_Star": {
        "E": 5.45,
        "Label": r"$\sigma^*$ C13-H38 (NBO 224)",
        "Hyb_C": "C13: 25.4% s, 74.4% p",
        "Hyb_H": "H38: 99.9% s, 0.1% p",
    },
    "O_LP": {
        "E": -11.05,
        "Label": r"n$_O$ (NBO 98)",
        "Hyb_O": "O: 6.2% s, 93.7% p",
    },
}


# =========================
# PLOT HELPERS
# =========================
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


# =========================
# FIGURE GENERATION
# =========================
def make_nao_to_nbo_diagram():
    fig, ax = plt.subplots(figsize=(15.5, 18))

    x_ch, x_cpx, x_pfoa = 0.35, 2.75, 5.15
    w = 0.42
    w_stag = 0.18

    # Orbital relationship lines are drawn first and kept faint so labels stay readable.
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

    # Cholestyramine fragment.
    draw_level(ax, x_ch, data["Ch_H_1s"]["E"], w, "black")
    label_box(
        ax,
        x_ch - 0.12,
        data["Ch_H_1s"]["E"] + 0.78,
        f"{data['Ch_H_1s']['Label']}\n{data['Ch_H_1s']['E']:.2f} eV",
        ha="center",
        size=18,
    )

    # PFOA fragment manifold.
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
        "O 2p$_x$/2p$_y$\n-11.42 / -11.35 eV",
        color="#7b3294",
        ha="right",
        va="top",
        size=16,
    )

    # Complex NBO levels.
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
        f"{data['CH_Sigma_Star']['Hyb_C']}\n{data['CH_Sigma_Star']['Hyb_H']}",
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
        data["O_LP"]["Hyb_O"],
        color="#1f77b4",
        ha="left",
        va="bottom",
        size=14,
        weight="normal",
    )

    # Formatting.
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
        ["Cholestyramine\n(NAO)", "PFOA-Ch Interaction\n(NBO)", "PFOA\n(NAO)"],
        fontweight="bold",
    )
    ax.set_xlim(-0.7, 6.2)
    ax.set_ylim(-15.6, 8.6)
    ax.set_title("NAO-to-NBO Interaction Diagram: PFOA-Cholestyramine", pad=20)
    ax.grid(axis="y", linestyle="--", linewidth=0.8, alpha=0.22)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="x", pad=14)
    ax.tick_params(axis="y", width=1.4, length=7)

    fig.tight_layout(rect=[0.075, 0.02, 1.0, 0.97])
    fig.savefig(OUTFILE, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure to: {OUTFILE}")


if __name__ == "__main__":
    make_nao_to_nbo_diagram()
