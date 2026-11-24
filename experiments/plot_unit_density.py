"""Figure of how many units carry a model's signal at each peak learning rate
(results/event_lm/knowledge_gap/unit_density_by_rate.json): left, the share of a level's cells that fire at a byte and
of a block's units whose gate is open; right, the share that carries nine tenths of what the level or block passes on.
The stack and the Transformer, on the test bed and at full size, each after 24,000 updates.

usage: python experiments/plot_unit_density.py [--data JSON] [--figure-dir DIR]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from plot_dendrite import MUTED, ORANGE, PAPER, RESULTS, TEXT, VIOLET, plt, save, style

FULL = "full size, 24,000 updates on windows of 512 bytes, one seed"
BED = "test bed, 24,000 updates"
SEEDS = ("seed 20001", "seed 20002")
# Each panel's measure: its title, and the record's entry for the stack and for the Transformer.
MEASURES = (
    ("In use at a byte: cells that fire, units whose gate is open", "cells firing at a byte", "units open at a byte"),
    ("Carrying nine tenths of what a level or a block passes on", "cells sending nine tenths of a level's spikes",
     "units passing nine tenths of a block's output"),
)
# Each model's rate with the lowest loss after 24,000 updates: (stack, Transformer). The stack's at full size is the
# plain stack's; under the lifting pool 6e-4 and 1.2e-3 are level.
BEST = {FULL: ("6e-4", "1.2e-3"), BED: ("1.2e-3", "2.4e-3")}


def curves(record):
    """{size: {model: {measure: [(rate, share)]}}}: the median level or block, on the test bed the mean of two seeds."""
    result = record["result"]
    found = {}
    for size in (FULL, BED):
        models = {"stack": {}, "Transformer": {}}
        for rate, entry in result[size].items():
            seeds = [entry[seed] for seed in SEEDS] if size == BED else [entry]
            for model, lines in models.items():
                for _, cells, units in MEASURES:
                    key = cells if model == "stack" else units
                    lines.setdefault(key, []).append(
                        (float(rate), sum(seed[model][key][0] for seed in seeds) / len(seeds)))
        found[size] = models
    return found


def figure(record, output):
    fig, panels = plt.subplots(1, 2, figsize=(12.4, 5.5), sharey=True)
    fig.subplots_adjust(left=0.065, right=0.975, bottom=0.2, top=0.705, wspace=0.07)
    fig.text(0.035, 0.965, "A higher learning rate thins the stack's code, not the Transformer's", size=20,
             weight="bold", color=TEXT, va="top")
    fig.text(0.036, 0.885, "The median level of the stack and the median block of the Transformer after 24,000 "
             "updates, on validation bytes. A cell is in use where it fires a spike,\na unit where the drive of its "
             "gate is above zero. A closed unit still passes a share, since its gate is smooth; a silent cell passes "
             "nothing.\nA number marks each model at its best rate.", size=10, color=MUTED, va="top",
             linespacing=1.35)
    found = curves(record)
    rates = sorted({rate for size in found.values() for rate, _ in size["stack"][MEASURES[0][1]]})
    names = {1.5e-4: "1.5e-4", 3e-4: "3e-4", 6e-4: "6e-4", 1.2e-3: "1.2e-3", 2.4e-3: "2.4e-3"}
    series = (("Transformer", ORANGE, "s"), ("stack", VIOLET, "o"))
    labels = {"Transformer": "Transformer", "stack": "Spiking stack"}
    sizes = ((FULL, "-", "full size"), (BED, (0, (4, 2.5)), "test bed"))
    for ax, (title, cells, units) in zip(panels, MEASURES):
        style(ax, grid_axis="y")
        ax.set_title(title, loc="left", size=11, weight="bold", color=TEXT, pad=10)
        ax.set_xscale("log", base=2)
        ax.set_xticks(rates, [names[rate] for rate in rates])
        ax.minorticks_off()
        ax.set_xlim(rates[0] / 1.25, rates[-1] * 1.25)
        ax.set_ylim(0, 0.62)
        ax.set_yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6], ["0%", "10%", "20%", "30%", "40%", "50%", "60%"])
        ax.set_xlabel("Peak learning rate", color=MUTED, labelpad=8)
        for model, color, marker in series:
            key = cells if model == "stack" else units
            for size, dashes, _ in sizes:
                points = sorted(found[size][model][key])
                xs, ys = [x for x, _ in points], [y for _, y in points]
                solid = size == FULL
                ax.plot(xs, ys, color=color, linewidth=2, linestyle=dashes, marker=marker, markersize=7,
                        markerfacecolor=color if solid else PAPER, markeredgecolor=color, markeredgewidth=1.6,
                        zorder=3)
                # Each model's share at its own best rate, in ink beside the mark, on the side where no other
                # line passes.
                best = float(BEST[size][0 if model == "stack" else 1])
                share = dict(points)[best]
                above = (model == "Transformer") != solid if ax is panels[0] else model == "Transformer" or not solid
                ax.annotate(f"{share:.0%}", (best, share), textcoords="offset points",
                            xytext=(0, 9 if above else -15), ha="center", color=TEXT, size=9, weight="bold")
    panels[0].set_ylabel("Share of a level's cells or a block's units", color=MUTED, labelpad=8)
    handles = [plt.Line2D([], [], color=color, linewidth=2, linestyle=dashes, marker=marker, markersize=7,
                          markerfacecolor=color if size == FULL else PAPER, markeredgecolor=color,
                          markeredgewidth=1.6, label=f"{labels[model]}, {name}")
               for model, color, marker in series for size, dashes, name in sizes]
    fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.06, 0.012), ncol=4, frameon=False, fontsize=9.5,
               labelcolor=TEXT, handlelength=3.2, columnspacing=2.2)
    save(fig, output, "event_lm_unit_density.png")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--data", type=Path,
                        default=RESULTS / "event_lm" / "knowledge_gap" / "unit_density_by_rate.json")
    parser.add_argument("--figure-dir", type=Path, default=RESULTS / "figures")
    args = parser.parse_args()
    figure(json.loads(args.data.read_text()), args.figure_dir)


if __name__ == "__main__":
    main()
