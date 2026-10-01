"""Numerical annotation styles"""
VALUE_SIZE_PT = 9.0
VALUE_BOX = {"boxstyle": "square,pad=0.10", "facecolor": "white",
             "edgecolor": "none", "linewidth": 0.0, "alpha": 1.0}
GRID_COLOR = "#000000"
GRID_WIDTH_PT = 0.45
ZERO_COLOR = "#000000"


def energy_grid(ax, axis="y", zero=True):
    import numpy as np

    ax.grid(axis=axis, color=GRID_COLOR, linewidth=GRID_WIDTH_PT, zorder=0)
    ticks = ax.get_yticks() if axis == "y" else ax.get_xticks()
    lines = ax.get_ygridlines() if axis == "y" else ax.get_xgridlines()
    zero_tick = False
    for tick, line in zip(ticks, lines):
        if np.isclose(tick, 0.0, atol=1e-8):
            line.set_linestyle((0, (4, 3)))
            line.set_color(GRID_COLOR)
            line.set_linewidth(GRID_WIDTH_PT)
            zero_tick = True
    lo, hi = ax.get_ylim() if axis == "y" else ax.get_xlim()
    if zero and not zero_tick and min(lo, hi) < 0 < max(lo, hi):
        add_line = ax.axhline if axis == "y" else ax.axvline
        add_line(0, color=GRID_COLOR, linestyle=(0, (4, 3)),
                 linewidth=GRID_WIDTH_PT, zorder=0)
