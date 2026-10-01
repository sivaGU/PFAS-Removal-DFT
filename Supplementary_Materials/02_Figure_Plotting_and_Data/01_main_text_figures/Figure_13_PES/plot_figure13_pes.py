"""Figure 13 PES plots"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent.parent))
from shared_helpers.figure_style import BLUE, EXPORT_DPI, FONT
from shared_helpers.annotation_style import energy_grid

GRID = ROOT / "input_data" / "R4N+PFOA-_0.15M.grid.csv"
PROFILE = ROOT / "input_data" / "pes_mep_profile.csv"
STRUCTURE = ROOT / "structure_renders" / "lowest_sampled_grid.png"
STRUCTURE_XYZ = ROOT / "input_structures" / "lowest_sampled_grid.xyz"
DEFAULT_OUTPUT_DIR = ROOT.parent.parent / "figure_exports"
OUTPUT_NAME = "Figure_13_constrained_PES.png"


def load_data():
    grid = pd.read_csv(GRID)
    profile = pd.read_csv(PROFILE)
    r1 = grid.iloc[:, 0].to_numpy(float)
    r2 = np.array([float(value) for value in grid.columns[1:]])
    z = grid.iloc[:, 1:].to_numpy(float)
    if z.shape != (10, 10) or not np.isfinite(z).all():
        raise ValueError("Expected the complete 10 by 10 finite PES grid")
    required = {"step", "r1_A", "r2_A", "relative_energy_kcal_mol"}
    if required - set(profile.columns) or len(profile) != 16:
        raise ValueError("Expected 16 selected configurations with both coordinates")
    if not np.array_equal(profile.step.to_numpy(int), np.arange(1, 17)):
        raise ValueError("Selected configurations must be ordered 1–16")
    for row in profile.itertuples(index=False):
        i = int(np.argmin(abs(r1 - row.r1_A)))
        j = int(np.argmin(abs(r2 - row.r2_A)))
        if abs(r1[i] - row.r1_A) > .002 or abs(r2[j] - row.r2_A) > .002:
            raise ValueError(f"Configuration {row.step} is outside the grid")
        if abs(z[i, j] - row.relative_energy_kcal_mol) > .002:
            raise ValueError(f"Configuration {row.step} energy differs from the grid")
    return r1, r2, z, profile


def measured_guides():
    atoms = [line.split() for line in STRUCTURE_XYZ.read_text().splitlines()[2:]]
    if len(atoms) != 53 or atoms[25][0] != "N":
        raise ValueError("Unexpected lowest-grid-point geometry")
    if [atoms[i][0] for i in (17, 18, 52)] != ["O", "O", "Cl"]:
        raise ValueError("Unexpected contact partners in the structure render")
    nitrogen = np.array([float(x) for x in atoms[25][1:4]])
    distances = [float(np.linalg.norm(nitrogen - np.array([float(x) for x in atoms[i][1:4]])))
                 for i in (17, 18, 52)]
    if not np.allclose(distances, (3.683, 4.119, 8.520), atol=.005):
        raise ValueError(f"Source contact distances changed: {distances}")
    return distances


def panel_letters(fig, axes):
    fig.canvas.draw()
    top_y = max(axes[0].get_position().y1, axes[1].get_position().y1) + .03
    positions = ((axes[0].get_position().x0 - .035, top_y),
                 (axes[1].get_position().x0 - .035, top_y),
                 (axes[2].get_position().x0 - .035,
                  axes[2].get_position().y1 + .03))
    for letter, (x, y) in zip("ABC", positions):
        fig.text(x, y, letter, ha="left", va="bottom", fontsize=11,
                 fontweight="bold")


def verify_layout(fig, axes):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    px_per_pt = fig.dpi / 72
    upper = axes[0].get_position()
    lower = axes[2].get_position()
    gap_pt = (lower.y1 - upper.y0) * fig.bbox.height / px_per_pt
    available_pt = -gap_pt
    if available_pt < 25:
        raise ValueError(f"Only {available_pt:.1f} pt separates the plot rows")
    alignment_pt = abs(axes[0].bbox.y1 - axes[1].bbox.y1) / px_per_pt
    if alignment_pt > 2:
        raise ValueError(f"Top panels differ vertically by {alignment_pt:.1f} pt")
    top_mid = (axes[0].get_position().x0 + axes[1].get_position().x1) / 2
    bottom_mid = (axes[2].get_position().x0 + axes[2].get_position().x1) / 2
    midpoint_pt = abs(top_mid - bottom_mid) * fig.bbox.width / px_per_pt
    if midpoint_pt > 2:
        raise ValueError(f"Lower panel is {midpoint_pt:.1f} pt off the top-row midline")
    letters = [item.get_window_extent(renderer) for item in fig.texts]
    letter_offset_pt = abs(letters[0].y0 - letters[1].y0) / px_per_pt
    if letter_offset_pt > .25:
        raise ValueError(f"Panel A/B letters differ by {letter_offset_pt:.1f} pt")
    for ax, letter in zip(axes, letters):
        title = ax.title.get_window_extent(renderer)
        if letter.overlaps(title):
            raise ValueError("Panel letter overlaps its title")
    print(f"Layout check: {available_pt:.1f} pt between rows; top alignment "
          f"{alignment_pt:.1f} pt; A/B letter offset {letter_offset_pt:.1f} pt; "
          f"lower-panel midline offset {midpoint_pt:.1f} pt")


def make_figure(output_dir=DEFAULT_OUTPUT_DIR, dpi=EXPORT_DPI):
    r1, r2, z, profile = load_data()
    if not STRUCTURE.exists():
        raise FileNotFoundError(f"Missing code-rendered structure input: {STRUCTURE}")
    plt.rcParams.update({"font.family": FONT, "font.size": 9,
                         "axes.linewidth": .8, "savefig.facecolor": "white"})
    fig = plt.figure(figsize=(7.0, 7.7))
    gs = fig.add_gridspec(2, 2, left=.12, right=.94, top=.92, bottom=.09,
                          width_ratios=(1, 1.25), height_ratios=(1.1, 1),
                          hspace=.34, wspace=.38)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, :])
    ax_c.set_position([.17, ax_c.get_position().y0, .76,
                       ax_c.get_position().height])

    mesh_r1, mesh_r2 = np.meshgrid(r1, r2, indexing="ij")
    levels = np.arange(0, np.ceil(z.max() / 2) * 2 + 2, 2)
    filled = ax_a.contourf(mesh_r1, mesh_r2, z, levels=levels,
                           cmap="viridis", extend="max")
    ax_a.contour(mesh_r1, mesh_r2, z, levels=levels[::2],
                 colors="black", linewidths=.35, alpha=.48)
    i, j = np.unravel_index(np.argmin(z), z.shape)
    ax_a.scatter(r1[i], r2[j], marker="*", s=95, color="white",
                 edgecolor="black", linewidth=.8, zorder=4)
    ax_a.annotate("lowest sampled\ngrid point", (r1[i], r2[j]),
                  xytext=(6, 8), textcoords="offset points", fontsize=8,
                  color="black", ha="left", va="bottom",
                  bbox={"facecolor": "white", "edgecolor": "#555555",
                        "linewidth": .5, "pad": 2.0})
    cb = fig.colorbar(filled, ax=ax_a, fraction=.047, pad=.045)
    cb.set_label("Energy above grid minimum (kcal/mol)", fontsize=8, labelpad=13)
    cb.ax.tick_params(labelsize=8)
    ax_a.set(xlabel=r"$r_1$: PFOA carboxyl C–N distance (Å)",
             ylabel=r"$r_2$: Cl–N distance (Å)")
    ax_a.set_title("Potential Energy Surface", fontsize=10, pad=8)
    ax_a.tick_params(labelsize=8)

    ax_b.set_position([.555, gs[0, 1].get_position(fig).y0, .425,
                       gs[0, 1].get_position(fig).height])
    ax_b.set_xlim(0, 1); ax_b.set_ylim(0, 1)
    ax_b.set_xticks([])
    ax_b.set_yticks([])
    for spine in ax_b.spines.values():
        spine.set_visible(True)
        spine.set_color("#555555")
        spine.set_linewidth(.8)
    ax_b.set_title("Lowest-Energy Sampled Structure", fontsize=10, pad=8)
    inner = ax_b.inset_axes([.045, .035, .91, .695])
    with Image.open(STRUCTURE) as raw:
        im = raw.convert("RGBA")
        bounds = im.getchannel("A").getbbox()
        if bounds is None:
            raise ValueError("Empty lowest-energy structure render")
        im = im.crop(bounds)
        white = Image.new("RGBA", im.size, (255, 255, 255, 255))
        flattened = Image.alpha_composite(white, im).convert("RGB")
        inner.imshow(np.asarray(flattened), aspect="equal", interpolation="hanning")
    inner.set_anchor("C")
    inner.set_axis_off()

    box = Rectangle((.62, .793), .38, .207, transform=ax_b.transAxes,
                    facecolor="white", edgecolor="#555555", linewidth=.55,
                    clip_on=False, zorder=5)
    ax_b.add_patch(box)
    distances = measured_guides()
    guide_text = []
    for y, text, color in zip((.9655, .8965, .8275),
                              (f"O–N {distances[0]:.2f} Å",
                               f"O–N {distances[1]:.2f} Å",
                               f"N–Cl {distances[2]:.2f} Å"),
                              ("#d62728", "#1f77b4", "#2ca02c")):
        guide_text.append(ax_b.text(.81, y, text, transform=ax_b.transAxes,
                                    ha="center", va="center", fontsize=7.7,
                                    color=color, zorder=6))

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    label_box = box.get_window_extent(renderer)
    pad = fig.dpi / 72 * 1.5
    for label in guide_text:
        bounds = label.get_window_extent(renderer)
        if (bounds.x0 < label_box.x0 + pad or bounds.x1 > label_box.x1 - pad or
                bounds.y0 < label_box.y0 + pad or bounds.y1 > label_box.y1 - pad):
            raise ValueError("Figure 13 B distance label does not fit inside its box")
    if label_box.y0 - inner.bbox.y1 < fig.dpi * .08:
        raise ValueError("Figure 13 B structure needs visible clearance below its distance box")

    steps = profile.step.to_numpy(int)
    energies = profile.relative_energy_kcal_mol.to_numpy(float)
    ax_c.scatter(steps, energies, s=31, color=BLUE, zorder=3)
    for step in (1, 8, 12):
        x = step
        y = energies[step - 1]
        ax_c.scatter(x, y, s=95, facecolor="white", edgecolor=BLUE,
                     linewidth=1.7, zorder=4)
    ax_c.set(xlim=(.35, 16.65), ylim=(-2, 38),
             xlabel="Selected grid configuration index",
             ylabel="Energy above grid minimum (kcal/mol)")
    ax_c.xaxis.labelpad = 10
    ax_c.yaxis.labelpad = 12
    ax_c.set_xticks(range(1, 17))
    energy_grid(ax_c)
    ax_c.set_axisbelow(True)
    ax_c.spines[["top", "right"]].set_visible(False)
    ax_c.tick_params(labelsize=8.5)
    ax_c.set_title("Selected Grid-Point Energies", fontsize=10, pad=8)
    panel_letters(fig, (ax_a, ax_b, ax_c))
    verify_layout(fig, (ax_a, ax_b, ax_c))
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / OUTPUT_NAME
    fig.savefig(output, dpi=dpi, bbox_inches="tight", pad_inches=.05)
    plt.close(fig)
    print(f"Wrote {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dpi", type=int, default=EXPORT_DPI)
    args = parser.parse_args()
    make_figure(args.output_dir, args.dpi)
