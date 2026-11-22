"""Figure of membrane saturation (results/event_lm/knowledge_gap/membrane_saturation.json, from
experiments/membrane_saturation.py): the share of each level's entries whose charge reached the membrane's bound, for
the state reads and the threshold cells, in the full-size stack after 24,000 updates at two peak rates and in the
6-level test bed at its highest rate.

usage: python experiments/plot_saturation.py [--data JSON] [--figure-dir DIR]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from plot_dendrite import GRAY, MUTED, RESULTS, TEXT, plt, save, style

# One hue, light to dark, for the same stack at a low and a high rate; the test bed in gray as a reference.
LIGHT_VIOLET, DARK_VIOLET = "#8f73c9", "#4a2a85"
RUNS = {
    "fs-st6": ("Full size, peak rate 6e-4", LIGHT_VIOLET, "-"),
    "fs-st12": ("Full size, peak rate 1.2e-3", DARK_VIOLET, "-"),
    "bed24-lr24-s1": ("Test bed (6 levels), 2.4e-3", GRAY, "--"),
}


def series(record, run, kind):
    """The levels and percentages of one kind of population in one run."""
    levels = record["runs"][run]["by_level"]
    points = sorted((int(key.split()[1]), 100 * value) for key, value in levels.items() if key.split()[0] == kind)
    return [level for level, _ in points], [value for _, value in points]


def figure(record, output):
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.9), sharey=True)
    fig.subplots_adjust(left=0.07, right=0.9, bottom=0.15, top=0.74, wspace=0.12)
    fig.text(0.035, 0.965, "At higher rates the full-size stack's membranes saturate", size=20, weight="bold",
             color=TEXT, va="top")
    fig.text(0.036, 0.875, "Share of each level's entries whose charge reached the membrane's bound, where a neuron "
             "fires its largest count and learns nothing through its charge.\nFirst 32,768 validation bytes as 16 "
             "streams; full size after 24,000 updates. The pool that lifts the ceiling removes the bound.",
             size=10, color=MUTED, va="top")
    for ax, (kind, title) in zip(axes, (("reads", "State reads"), ("cells", "Threshold cells"))):
        style(ax, grid_axis="y")
        ax.set_title(title, loc="left", color=TEXT, size=12, weight="bold", pad=10)
        for run, (label, color, line) in RUNS.items():
            if run not in record["runs"]:
                continue
            levels, values = series(record, run, kind)
            ax.plot(levels, values, line, color=color, linewidth=2, marker="o", markersize=5, label=label)
            # The test bed ends mid-axis, where the full size's lines pass: among the cells its label sits above its
            # last point, clear of the line that climbs past it there.
            above = levels[-1] < 10 and kind == "cells"
            ax.annotate(f"{values[-1]:.1f}%", (levels[-1], values[-1]), textcoords="offset points",
                        xytext=(0, 10) if above else (7, 0), ha="center" if above else "left", va="center",
                        color=TEXT, size=9)
        ax.set_xticks(range(11))
        ax.set_xlabel("Level, from the bottom of the stack", color=MUTED, labelpad=8)
        ax.set_xlim(-0.4, 11.2)
    axes[0].set_ylabel("Entries at the bound, %", color=MUTED, labelpad=8)
    axes[0].legend(frameon=False, fontsize=9, labelcolor=TEXT, loc="upper left")
    save(fig, output, "event_lm_membrane_saturation.png")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--data", type=Path, default=RESULTS / "event_lm" / "knowledge_gap" / "membrane_saturation.json")
    parser.add_argument("--figure-dir", type=Path, default=RESULTS / "figures")
    args = parser.parse_args()
    figure(json.loads(args.data.read_text()), args.figure_dir)


if __name__ == "__main__":
    main()
