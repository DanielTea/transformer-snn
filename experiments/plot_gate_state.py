"""Figure of the gate measure (results/event_lm/knowledge_gap/proportional_gate_check.json, the amendment made with
experiments/gate_state.py): how the open gates of each gated cell split by the share of the value dendrite's charge
that they let in, and how each cell scored against plain cells under the same pool after 2,250 updates.

usage: python experiments/plot_gate_state.py [--data JSON] [--figure-dir DIR]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from matplotlib.patches import Patch

from plot_dendrite import MUTED, PAPER, RESULTS, TEXT, plt, save, style

# One hue, light to dark, for the share that an open gate lets in: up to a quarter, a half, three quarters, more.
QUARTERS = (("up to \u00bc", "#e6def5", TEXT), ("\u00bc to \u00bd", "#bba8e2", TEXT),
            ("\u00bd to \u00be", "#8f73c9", PAPER), ("more than \u00be", "#4a2a85", PAPER))
# Run, the gate's law, its pool, and the plain run that it is scored against.
ROWS = (
    ("lift-r16i", "Proportional, 16 thresholds wide", "pool lifts the ceiling", "lift-base"),
    ("r16i", "Proportional, 16 thresholds wide", "bounded pool", None),
    ("si", "Shunt, k / (k + 4)", "bounded pool", None),
    ("lift-si", "Shunt, k / (k + 4)", "pool lifts the ceiling", "lift-base"),
    ("lift-r4i", "Proportional, 4 thresholds wide", "pool lifts the ceiling", "lift-base"),
    ("r4i", "Proportional, 4 thresholds wide", "bounded pool", None),
)
AT = "2,250 updates"


def spread(run):
    """The mean over the levels of the share of open gates in each quarter, in percent."""
    levels = [row["open_gates_by_quarter_share"] for row in run["cells_by_level"].values()]
    return [100 * sum(level[quarter] for level in levels) / len(levels) for quarter in range(4)]


def outcome(record, name, base):
    """How the run scored against plain cells under its pool at the check's end."""
    score = record["runs"][name]["curve"][AT]["averaged weights on streams"]
    plain = (record["runs"][base]["curve"][AT]["averaged weights on streams"] if base
             else record["references"]["rd-base-24k-s3, plain, bounded"][AT])
    behind = score - plain
    if behind >= 0.2:
        return f"diverged: {score:.2f} bpb"
    if abs(behind) < 0.01:
        return f"level: {abs(behind):.3f} {'behind' if behind > 0 else 'ahead'}"
    return f"{abs(behind):.3f} {'behind' if behind > 0 else 'ahead'}"


def figure(record, output):
    measured = record["amendment"]["diagnosis at 2,250 updates on streams"]
    fig, ax = plt.subplots(figsize=(12.4, 5.5))
    fig.subplots_adjust(left=0.265, right=0.8, bottom=0.13, top=0.68)
    fig.text(0.035, 0.965, "The gate that keeps pace stays graded; the gates that lose are switches", size=20,
             weight="bold", color=TEXT, va="top")
    fig.text(0.036, 0.885, "Open gates of each gated cell, split by the share of the value dendrite's charge that "
             "they let in: mean of the six levels, on 65,536 validation bytes as streams,\nafter 2,250 updates on the "
             "test bed (one seed). Right: validation bits per byte against plain cells under the same pool.",
             size=10, color=MUTED, va="top")
    style(ax, grid_axis="x")
    for row, (name, law, pool, base) in enumerate(ROWS):
        left = 0.0
        for (_, color, ink), value in zip(QUARTERS, spread(measured[name])):
            ax.barh(row, value, left=left, height=0.56, color=color, edgecolor=PAPER, linewidth=2)
            if value >= 6:
                ax.text(left + value / 2, row, f"{value:.0f}%", ha="center", va="center", color=ink, size=9.5)
            left += value
        ax.text(-2.5, row - 0.13, law, ha="right", va="center", color=TEXT, size=10.5)
        ax.text(-2.5, row + 0.17, pool, ha="right", va="center", color=MUTED, size=9)
        ax.text(103, row, outcome(record, name, base), ha="left", va="center", color=TEXT, size=10.5,
                weight="bold" if row == 0 else "normal")
    ax.text(103, -0.78, "Against plain cells", ha="left", va="center", color=MUTED, size=9.5)
    ax.set_ylim(len(ROWS) - 0.45, -0.55)
    ax.set_yticks([])
    ax.set_xlim(0, 100)
    ax.set_xticks(range(0, 101, 25), [f"{tick}%" for tick in range(0, 101, 25)])
    ax.set_xlabel("Share of the cell's open gates", color=MUTED, labelpad=8)
    ax.legend(handles=[Patch(facecolor=color, edgecolor=PAPER, label=label) for label, color, _ in QUARTERS],
              title="Share of the value that an open gate lets in", title_fontsize=9.5, alignment="left", frameon=False, fontsize=9.5,
              labelcolor=TEXT, ncols=4, loc="lower left", bbox_to_anchor=(-0.008, 1.0), columnspacing=1.6,
              handlelength=1.3, handleheight=1.0)
    ax.get_legend().get_title().set_color(MUTED)
    save(fig, output, "event_lm_gate_state.png")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--data", type=Path,
                        default=RESULTS / "event_lm" / "knowledge_gap" / "proportional_gate_check.json")
    parser.add_argument("--figure-dir", type=Path, default=RESULTS / "figures")
    args = parser.parse_args()
    figure(json.loads(args.data.read_text()), args.figure_dir)


if __name__ == "__main__":
    main()
