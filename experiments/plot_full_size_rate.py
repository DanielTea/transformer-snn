"""Figure of the full-size learning-rate pilots (results/event_lm/knowledge_gap/full_size_rate_pilot.json and
full_size_lifted_ceiling_pilot.json): the full-size spiking stack, the same stack with the pool that lifts the ceiling,
and the full-size Transformer, each trained for 24,000 updates at a peak rate, scored on the first 2,000,000 validation
bytes as 16 unbroken streams. Left, each model's loss against its peak rate; right, the stack's gap to the Transformer
at the same rate.

usage: python experiments/plot_full_size_rate.py [--figure-dir DIR]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from plot_dendrite import GRAY, MUTED, ORANGE, RESULTS, TEXT, plt, save, style

LIGHT_VIOLET, DARK_VIOLET = "#8f73c9", "#4a2a85"
RECORDS = RESULTS / "event_lm" / "knowledge_gap"


def scores():
    """{model: {rate: bits per byte}} from the two pilots' result blocks."""
    rate_pilot = json.loads((RECORDS / "full_size_rate_pilot.json").read_text())["result"]["bpb"]
    lifted = json.loads((RECORDS / "full_size_lifted_ceiling_pilot.json").read_text())["result"]["bpb"]["lifted"]
    rate = lambda name: float(name)  # noqa: E731
    return {"stack": {rate(k): v for k, v in rate_pilot["stack"].items()},
            "lifted": {rate(k): v for k, v in lifted.items()},
            "transformer": {rate(k): v for k, v in rate_pilot["transformer"].items()}}


def figure(points, output):
    fig, (losses, gaps) = plt.subplots(1, 2, figsize=(12.4, 4.8), width_ratios=(1.2, 1))
    fig.subplots_adjust(left=0.07, right=0.975, bottom=0.17, top=0.76, wspace=0.26)
    fig.text(0.035, 0.965, "At full size, lifting the ceiling stops the stack breaking at a high rate", size=20,
             weight="bold", color=TEXT, va="top")
    fig.text(0.036, 0.875, "Full size (34.8M parameters), 24,000 updates of 8 windows of 512 bytes, cosine schedule, "
             "one seed; scored on 2M validation bytes as 16 unbroken streams.\nThe pool that lifts the ceiling "
             "divides the charge before the membrane's bound, so no neuron saturates.", size=10, color=MUTED, va="top")
    rates = sorted(points["stack"])
    lines = (("transformer", "Transformer (34.75M)", ORANGE, "-"), ("stack", "Spiking stack (34.81M)", LIGHT_VIOLET, "-"),
             ("lifted", "Spiking stack, lifted ceiling", DARK_VIOLET, ":"))
    # Each model's best score sits where no other line passes: the plain stack's and the Transformer's above their
    # points, the lifted stack's below.
    offsets = {"transformer": (0, 10), "stack": (0, 10), "lifted": (0, -15)}
    names = {3e-4: "3e-4", 6e-4: "6e-4", 1.2e-3: "1.2e-3"}
    for ax in (losses, gaps):
        style(ax, grid_axis="both")
        ax.set_xscale("log", base=2)
        ax.set_xticks(rates, [names.get(rate, f"{rate:g}") for rate in rates])
        ax.set_xlim(rates[0] / 1.35, rates[-1] * 1.35)
        ax.set_xlabel("Peak learning rate", color=MUTED, labelpad=8)
    for model, label, color, line in lines:
        xs = sorted(points[model])
        ys = [points[model][x] for x in xs]
        losses.plot(xs, ys, "o" + line, color=color, linewidth=2, markersize=6, label=label)
        best = min(xs, key=points[model].get)
        # The records hold five decimals; printing four of them would round twice.
        losses.annotate(f"{points[model][best]:.5f}", (best, points[model][best]), textcoords="offset points",
                        xytext=offsets[model], ha="center", color=TEXT, size=9)
    losses.set_ylabel("Bits per byte", color=MUTED, labelpad=8)
    losses.legend(frameon=False, fontsize=9, labelcolor=TEXT, loc="upper center")
    for model, label, color, line in lines[1:]:
        xs = sorted(set(points[model]) & set(points["transformer"]))
        gap = [points[model][x] - points["transformer"][x] for x in xs]
        gaps.plot(xs, gap, "o" + line, color=color, linewidth=2, markersize=6, label=label)
        for x, value in zip(xs, gap):
            gaps.annotate(f"{value:+.3f}", (x, value), textcoords="offset points", xytext=(0, 9), ha="center",
                          color=TEXT, size=9)
    best_gap = min(points["stack"].values()) - min(points["transformer"].values())
    lifted_gap = min(points["lifted"].values()) - min(points["transformer"].values())
    gaps.axhline(best_gap, color=GRAY, linewidth=0.8, linestyle="--")
    gaps.annotate(f"each at its best rate: plain {best_gap:+.3f}, lifted {lifted_gap:+.3f}", (rates[0], best_gap),
                  textcoords="offset points", xytext=(0, 6), color=MUTED, size=8.5)
    gaps.set_ylabel("Stack minus Transformer, bits per byte", color=MUTED, labelpad=8)
    gaps.legend(frameon=False, fontsize=9, labelcolor=TEXT, loc="upper left")
    save(fig, output, "event_lm_full_size_rate.png")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--figure-dir", type=Path, default=RESULTS / "figures")
    args = parser.parse_args()
    figure(scores(), args.figure_dir)


if __name__ == "__main__":
    main()
