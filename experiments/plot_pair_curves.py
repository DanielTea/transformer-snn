"""Figure of the full-size pairs along training (results/event_lm/knowledge_gap/pair_curves.json): the stack's
validation reading minus its Transformer's at every reading, against the updates on a doubling scale, for v17's pair
at a shared rate and v19's pair at each model's best rate.

usage: python experiments/plot_pair_curves.py [--figure-dir DIR]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from plot_dendrite import MUTED, RESULTS, TEXT, plt, save, style

DARK_VIOLET, LIGHT_VIOLET = "#4a2a85", "#8f73c9"
RECORD = RESULTS / "event_lm" / "knowledge_gap" / "pair_curves.json"
PAIRS = (("l2048-know-st", "tf2048", "Both at 3e-4 (v17)", LIGHT_VIOLET),
         ("fsp-st", "fsp-tf", "Each at its best rate (v19)", DARK_VIOLET))


def gaps(record):
    runs = json.loads(Path(record).read_text())["runs"]
    found = []
    for stack, transformer, label, color in PAIRS:
        steps = np.array(runs[stack]["steps"])
        assert steps.tolist() == runs[transformer]["steps"]
        gap = np.array(runs[stack]["validation_bpb"]) - np.array(runs[transformer]["validation_bpb"])
        late = steps >= 12000
        slope = np.polyfit(np.log2(steps[late]), gap[late], 1)[0]
        found.append((label, color, steps, gap, slope))
    return found


def figure(found, output):
    fig, ax = plt.subplots(figsize=(12.4, 5.8))
    fig.subplots_adjust(left=0.085, right=0.76, bottom=0.13, top=0.74)
    fig.text(0.035, 0.965, "The stack leads early and falls behind as training goes on", size=19.5, weight="bold",
             color=TEXT, va="top")
    fig.text(0.036, 0.885, "Full size, one seed a run: the spiking stack's validation reading minus its Transformer's "
             "at every 1,500 updates, 16 windows a reading in the training\ncontext (noisy, and not the release's stream "
             "scores). Above the line the Transformer is ahead. The updates double from left to right.",
             size=10, color=MUTED, va="top", linespacing=1.45)
    style(ax)
    ax.axhline(0, color=TEXT, linewidth=1.0, zorder=2)
    # The readings from 6,000 updates on: before that both models are far from trained.
    found = [(label, color, steps[steps >= 6000], gap[steps >= 6000], slope) for label, color, steps, gap, slope in found]
    # End labels spread apart where the lines end close together.
    places, last = {}, None
    for label, *_, gap, _slope in sorted(found, key=lambda row: row[3][-1]):
        place = gap[-1] if last is None else max(gap[-1], last + 0.022)
        places[label], last = place, place
    for label, color, steps, gap, slope in found:
        ax.plot(steps, gap, color=color, linewidth=2.2, marker="o", markersize=3.5, zorder=3)
        ax.annotate(f"{label}\n{gap[-1]:+.3f} at 90,000 updates,\n{slope:+.3f} a doubling from 12,000", (steps[-1], gap[-1]),
                    xytext=(steps[-1] * 1.12, places[label]), textcoords="data", ha="left", va="center", color=TEXT,
                    size=9.5, annotation_clip=False,
                    arrowprops=dict(arrowstyle="-", color=color, linewidth=0.8, shrinkA=0, shrinkB=3))
    ax.set_xscale("log", base=2)
    ticks = [6000, 12000, 24000, 48000, 90000]
    ax.set_xticks(ticks, [f"{tick:,}" for tick in ticks])
    ax.minorticks_off()
    ax.set_xlabel("Updates (4,096 training bytes each)", color=MUTED, labelpad=8)
    ax.set_ylabel("Stack minus Transformer, bits per byte", color=MUTED, labelpad=8)
    save(fig, output, "event_lm_pair_curves.png")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--figure-dir", type=Path, default=RESULTS / "figures")
    args = parser.parse_args()
    figure(gaps(RECORD), args.figure_dir)


if __name__ == "__main__":
    main()
