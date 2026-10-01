import argparse
import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from shared_helpers.annotation_style import energy_grid
from shared_helpers.figure_style import EXPORT_DPI


ROOT = Path(__file__).resolve().parent
DATA_FILE = ROOT / "input_data" / "figure_07_nbo_data.csv"
DEFAULT_OUTPUT_DIR = ROOT.parent.parent / "figure_exports"
OUTPUT_STEM = "Figure_08_NAO_NBO_DVB_BTMA_PFOA"
HARTREE_TO_EV = 27.211386245988

REQUIRED_COLUMNS = {
    "record_type", "key", "model", "representation", "label",
    "orbital_number", "occupancy", "energy_hartree",
    "hybridization_primary", "hybridization_secondary", "donor_key",
    "acceptor_key", "e2_kcal_mol", "energy_gap_hartree",
    "fock_coupling_hartree", "source_section", "source_output_line",
    "plot_annotation",
}
REQUIRED_ORBITALS = {
    "DVB_H_1s", "O_2px", "O_2py", "O_2pz", "O_LP", "CH_Sigma_Star",
}
MODEL = "DVB-BTMA+PFOA-"


def parse_args():
    parser = argparse.ArgumentParser(description="Generate the Figure 8 NAO-NBO diagram")
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
        help="directory for the generated PNG",
    )
    return parser.parse_args()


def read_records(path=DATA_FILE):
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing_columns = REQUIRED_COLUMNS - set(reader.fieldnames or [])
        if missing_columns:
            names = ", ".join(sorted(missing_columns))
            raise ValueError(f"Missing required CSV columns: {names}")
        return list(reader)


def parse_float(row, field):
    try:
        value = float(row[field])
    except ValueError as exc:
        raise ValueError(f"Invalid {field} for record {row['key']}") from exc
    if not math.isfinite(value):
        raise ValueError(f"Non-finite {field} for record {row['key']}")
    return value


def load_data():
    orbitals = {}
    interactions = []
    seen_keys = set()

    for row in read_records():
        key = row["key"].strip()
        if not key:
            raise ValueError("Every CSV record requires a key")
        if key in seen_keys:
            raise ValueError(f"Duplicate CSV record key: {key}")
        seen_keys.add(key)

        if row["model"].strip() != MODEL:
            raise ValueError(f"Unexpected model for record {key}: {row['model']}")
        if not row["label"].strip() or not row["source_section"].strip():
            raise ValueError(f"Missing label or source section for record {key}")

        record_type = row["record_type"].strip().lower()
        if record_type == "orbital":
            row["occupancy"] = parse_float(row, "occupancy")
            row["energy_hartree"] = parse_float(row, "energy_hartree")
            row["energy_ev"] = row["energy_hartree"] * HARTREE_TO_EV
            orbitals[key] = row
        elif record_type == "interaction":
            row["e2_kcal_mol"] = parse_float(row, "e2_kcal_mol")
            row["energy_gap_hartree"] = parse_float(row, "energy_gap_hartree")
            row["fock_coupling_hartree"] = parse_float(row, "fock_coupling_hartree")
            interactions.append(row)
        else:
            raise ValueError(f"Unknown record_type for {key}: {record_type}")

    missing_orbitals = REQUIRED_ORBITALS - orbitals.keys()
    extra_orbitals = orbitals.keys() - REQUIRED_ORBITALS
    if missing_orbitals or extra_orbitals:
        raise ValueError(
            "Orbital record mismatch; "
            f"missing={sorted(missing_orbitals)}, extra={sorted(extra_orbitals)}"
        )
    if len(interactions) != 1:
        raise ValueError("Exactly one donor-acceptor interaction is required")

    interaction = interactions[0]
    if interaction["donor_key"] not in orbitals or interaction["acceptor_key"] not in orbitals:
        raise ValueError("Interaction donor_key and acceptor_key must identify orbital records")
    return orbitals, interaction


def draw_level(ax, x, y, half_width, color, linewidth=2.6):
    ax.hlines(y, x - half_width, x + half_width, color=color, linewidth=linewidth, zorder=4)


def label(ax, x, y, text, color="black", ha="center", va="center", size=9):
    ax.text(
        x, y, text, color=color, fontsize=size, ha=ha, va=va, linespacing=1.25,
        zorder=8,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 1.0, "pad": 1.4},
    )


def orbital_energy_label(ax, x, y, text, size):
    ax.text(
        x, y, text, color="black", fontsize=size, ha="center", va="center",
        zorder=8,
        bbox={
            "boxstyle": "square,pad=0.18", "facecolor": "white",
            "edgecolor": "none", "linewidth": 0.0,
        },
    )


def format_p_label(orbital_label):
    atom = orbital_label.split(maxsplit=1)[0]
    axis = orbital_label.rsplit("_", maxsplit=1)[-1]
    return f"{atom} 2p$_{axis}$"


def make_figure(output_dir):
    orbitals, interaction = load_data()
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6.5, 4.85))
    x_dvb, x_complex, x_pfoa = 1.25, 4.55, 9.00
    level_width = 0.72
    p_width = 0.52

    h = orbitals["DVB_H_1s"]
    lp = orbitals[interaction["donor_key"]]
    sigma_star = orbitals[interaction["acceptor_key"]]
    p_levels = [orbitals[key] for key in ("O_2px", "O_2py", "O_2pz")]
    p_offsets = (-1.83, 0.0, 1.83)

    ax.plot(
        [x_dvb + level_width, x_complex - level_width],
        [h["energy_ev"], sigma_star["energy_ev"]],
        color="#303030", linestyle="--", linewidth=1.35, zorder=1,
    )
    p_2px_x = x_pfoa + p_offsets[0]
    ax.plot(
        [x_complex + level_width, p_2px_x - p_width],
        [lp["energy_ev"], orbitals["O_2px"]["energy_ev"]],
        color="#303030", linestyle="--", linewidth=1.35, zorder=1,
    )


    arrow_x = x_complex + 1.48
    ax.annotate(
        "", xy=(arrow_x, sigma_star["energy_ev"] - 0.25),
        xytext=(arrow_x, lp["energy_ev"] + 0.90),
        arrowprops={"arrowstyle": "-|>", "color": "#1565c0", "linewidth": 1.9},
        zorder=3,
    )
    label(
        ax, arrow_x + 0.15, (lp["energy_ev"] + sigma_star["energy_ev"]) / 2,
        f"{interaction['label']}\nE$^{{(2)}}$ = {interaction['e2_kcal_mol']:.2f} kcal/mol",
        ha="left", size=8.8,
    )

    draw_level(ax, x_dvb, h["energy_ev"], level_width, "black")
    orbital_energy_label(
        ax, x_dvb, h["energy_ev"] + 1.75,
        f"{h['label']}\n{h['energy_ev']:+.2f} eV", size=9,
    )

    nao_color = "black"
    for level, offset in zip(p_levels, p_offsets):
        level_x = x_pfoa + offset
        draw_level(ax, level_x, level["energy_ev"], p_width, nao_color, linewidth=2.4)
        orbital_energy_label(
            ax, level_x, level["energy_ev"] + 1.68,
            f"{format_p_label(level['label'])}\n{level['energy_ev']:+.2f} eV",
            size=8.5,
        )

    draw_level(ax, x_complex, lp["energy_ev"], level_width, "#0072b2")
    orbital_energy_label(
        ax, x_complex - 0.15, lp["energy_ev"] + 2.55,
        f"{lp['label']}\n{lp['energy_ev']:+.2f} eV", size=9,
    )

    draw_level(ax, x_complex, sigma_star["energy_ev"], level_width, "#c62828")
    orbital_energy_label(
        ax, x_complex - 0.15, sigma_star["energy_ev"] + 1.95,
        f"{sigma_star['label']}\n{sigma_star['energy_ev']:+.2f} eV", size=9,
    )

    ax.set_xlim(0.0, 11.8)
    ax.set_ylim(-12.0, 16.0)
    ax.set_ylabel("Orbital energy (eV)", fontsize=10)
    ax.set_xticks([x_dvb, x_complex, x_pfoa])
    ax.set_xticklabels(
        ["DVB-BTMA$^+$\nNAOs", "DVB-BTMA$^+$PFOA$^-$\nNBOs", "PFOA$^-$\nNAOs"],
        fontsize=9,
    )
    ax.set_xticks([x_pfoa + p_offsets[0], x_pfoa + p_offsets[2]], minor=True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(True)
    ax.spines["bottom"].set_color("black")
    ax.spines["bottom"].set_linewidth(1.0)
    ax.tick_params(axis="x", which="major", bottom=True, top=False,
                   direction="out", length=5, width=1.0, pad=10)
    ax.tick_params(axis="x", which="minor", bottom=True, top=False,
                   direction="out", length=5, width=1.0)
    ax.tick_params(axis="y", labelsize=9)
    energy_grid(ax)

    fig.subplots_adjust(left=0.125, right=0.985, top=0.955, bottom=0.18)
    output_file = output_dir / f"{OUTPUT_STEM}.png"
    fig.savefig(output_file, bbox_inches="tight", dpi=EXPORT_DPI)
    print(f"Wrote {output_file}")
    plt.close(fig)


if __name__ == "__main__":
    args = parse_args()
    make_figure(args.output_dir)
