"""Figure of the float unit diagnostic (results/event_lm/knowledge_gap/plain_units_with_data.json): the test bed's
Transformer with plain or gated feed-forward units, after 6,000 and 24,000 updates on two seeds, each unit's bits per
byte minus those of the reference's own gated units. A diagnostic in floats, never a path.

usage: python experiments/plot_plain_units.py [--figure-dir DIR]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from plot_dendrite import MUTED, ORANGE, RESULTS, TEXT, plt, save, style

DARK_VIOLET, LIGHT_VIOLET = "#4a2a85", "#8f73c9"
RECORD = RESULTS / "event_lm" / "knowledge_gap" / "plain_units_with_data.json"
UNITS = (("relu", "Plain, shut below zero (the stack's cell in floats)", DARK_VIOLET, "o"),
         ("gelu", "Plain, opening smoothly", LIGHT_VIOLET, "D"),
         ("reglu", "Gated: shut below zero, multiplying above", ORANGE, "s"))


def gaps(record):
    result = json.loads(Path(record).read_text())["result"]
    table = next(value for key, value in result.items() if key.startswith("gaps to swiglu"))
    return {unit: {seed: [table[f"{unit} {seed}"][length]["gap"] for length in ("6k", "24k")] for seed in ("s1", "s2")}
            for unit, *_ in UNITS}


def figure(found, output):
    fig, ax = plt.subplots(figsize=(12.4, 6.2))
    fig.subplots_adjust(left=0.085, right=0.62, bottom=0.12, top=0.74)
    fig.text(0.035, 0.965, "In floats, the plain hard unit falls behind as data grows",
             size=18.5, weight="bold", color=TEXT, va="top")
    fig.text(0.036, 0.885, "The test bed's Transformer (4.8M parameters) at a peak rate of 1.2e-3 with three kinds of "
             "feed-forward unit, two seeds each: bits per byte on 2M validation bytes\nminus those of its own gated "
             "units (SwiGLU) at the same length and seed. Thin markers are seeds, thick lines their mean. A "
             "diagnostic in floats, never a path.", size=10, color=MUTED, va="top", linespacing=1.45)
    style(ax)
    ax.axhline(0, color=TEXT, linewidth=1.0, zorder=2)
    means = {unit: [(found[unit]["s1"][i] + found[unit]["s2"][i]) / 2 for i in range(2)] for unit, *_ in UNITS}
    # Labels at the right end, spread apart where the lines end close together.
    places, last = {}, None
    for unit in sorted(means, key=lambda name: means[name][1]):
        place = means[unit][1] if last is None else max(means[unit][1], last + 0.0055)
        places[unit], last = place, place
    for unit, label, color, marker in UNITS:
        seeds = found[unit]
        for values in seeds.values():
            ax.plot([0, 1], values, color=color, linewidth=1.0, alpha=0.45, marker=marker, markersize=5, zorder=3)
        mean = [(seeds["s1"][i] + seeds["s2"][i]) / 2 for i in range(2)]
        ax.plot([0, 1], mean, color=color, linewidth=2.6, marker=marker, markersize=9, zorder=4,
                markeredgecolor="white", markeredgewidth=1.4)
        growth = mean[1] - mean[0]
        ax.annotate(f"{label}\n{mean[1]:+.4f} after 24,000, {growth:+.4f} with the data", (1, mean[1]),
                    xytext=(1.06, places[unit]), textcoords="data", ha="left", va="center", color=TEXT, size=9.5,
                    annotation_clip=False, arrowprops=dict(arrowstyle="-", color=color, linewidth=0.8,
                                                           shrinkA=0, shrinkB=4))
    ax.set_xlim(-0.12, 1.08)
    ax.set_xticks([0, 1], ["6,000 updates", "24,000 updates"])
    ax.tick_params(axis="x", labelsize=10.5, labelcolor=TEXT)
    ax.grid(axis="x", visible=False)
    ax.set_ylabel("Bits per byte against the gated reference (lower is better)", color=MUTED, labelpad=8)
    save(fig, output, "event_lm_plain_units.png")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--figure-dir", type=Path, default=RESULTS / "figures")
    args = parser.parse_args()
    figure(gaps(RECORD), args.figure_dir)


if __name__ == "__main__":
    main()
