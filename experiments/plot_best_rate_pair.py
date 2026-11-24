"""Figure of the full-size pair (results/event_lm/knowledge_gap/full_size_best_rate_pair.json): the spiking stack
and its Transformer after 90,000 updates, both at a peak rate of 3e-4 as v17 trained them, and each at the rate
where it trains best at full size. Two lines a split, one for each model, from the old rate to the best one.

usage: python experiments/plot_best_rate_pair.py [--figure-dir DIR] [--record JSON]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from plot_dendrite import MUTED, ORANGE, RESULTS, TEXT, plt, save, style

DARK_VIOLET = "#4a2a85"
RECORD = RESULTS / "event_lm" / "knowledge_gap" / "full_size_best_rate_pair.json"
STACK = "the stack (fsp-st), with fast output synapses on all levels at 2^24, at the chosen reach"
TRANSFORMER = "the Transformer (fsp-tf), windows of 2,048 bytes"
BEFORE = "both at 3e-4 (v17: l2048-know-st and tf2048, the same readings)"


def scores(record):
    """For each split, the two models' bits per byte at 3e-4 and at their best rates."""
    result = json.loads(Path(record).read_text())["result"]
    before = result[BEFORE]
    return {split: {"stack": (before["stack"][split], result[STACK][split]),
                    "transformer": (before["transformer"][split], result[TRANSFORMER]["frozen"][split])}
            for split in ("test", "validation")}


def bracket(ax, x, low, high, text, side):
    """A thin bracket between two points at one rate, with the distance beside it."""
    ax.plot([x, x], [low, high], color=MUTED, linewidth=0.9, zorder=2)
    for end in (low, high):
        ax.plot([x - 0.012, x + 0.012], [end, end], color=MUTED, linewidth=0.9, zorder=2)
    ax.annotate(text, (x, (low + high) / 2), textcoords="offset points", xytext=(7 * side, 0),
                ha="left" if side > 0 else "right", va="center", color=TEXT, size=10, weight="bold")


def figure(found, output):
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.9), sharey=True)
    fig.subplots_adjust(left=0.075, right=0.975, bottom=0.115, top=0.72, wspace=0.07)
    test = found["test"]
    gains = {name: test[name][0] - test[name][1] for name in test}
    fig.text(0.035, 0.965, "At full length the best learning rate pays the Transformer, not the stack", size=19.5,
             weight="bold", color=TEXT, va="top")
    fig.text(0.036, 0.888, "Bits per byte on enwik8 after 90,000 updates from random weights at full size (the stack "
             "36.3M parameters, the Transformer 34.8M), one seed a run.\nThe stack reads each split as 16 unbroken "
             "streams with fast output synapses; the Transformer reads windows of 2,048 bytes. Lower is better.\n"
             f"On the test split the Transformer gains {gains['transformer']:.3f} from its best rate, 1.2e-3, and "
             f"the stack {gains['stack']:.3f} from its own, 6e-4.", size=10, color=MUTED, va="top", linespacing=1.45)
    for ax, split in zip(axes, ("test", "validation")):
        style(ax)
        ax.set_title(f"enwik8 {split}", loc="left", size=11.5, weight="bold", color=TEXT, pad=10)
        stack, transformer = found[split]["stack"], found[split]["transformer"]
        for name, pair, color, marker in (("Spiking stack", stack, DARK_VIOLET, "o"),
                                          ("Transformer", transformer, ORANGE, "s")):
            ax.plot([0, 1], pair, color=color, linewidth=2.2, marker=marker, markersize=8, zorder=3,
                    markeredgecolor="white", markeredgewidth=1.5)
            ax.annotate(f"{pair[0]:.4f}", (0, pair[0]), textcoords="offset points", xytext=(-12, 0), ha="right",
                        va="center", color=TEXT, size=10)
            ax.annotate(f"{name}  {pair[1]:.4f}", (1, pair[1]), textcoords="offset points", xytext=(12, 0),
                        ha="left", va="center", color=TEXT, size=10.5, weight="bold")
        bracket(ax, 0.09, transformer[0], stack[0], f"{stack[0] - transformer[0]:.3f} apart", 1)
        bracket(ax, 0.91, transformer[1], stack[1], f"{stack[1] - transformer[1]:.3f} apart", -1)
        ax.set_xlim(-0.34, 1.78)
        ax.set_xticks([0, 1], ["Both at 3e-4", "Each at its best rate"])
        ax.tick_params(axis="x", labelsize=10.5, labelcolor=TEXT)
        ax.grid(axis="x", visible=False)
    low = min(pair[1] for split in found.values() for pair in split.values())
    high = max(pair[0] for split in found.values() for pair in split.values())
    axes[0].set_ylim(low - 0.012, high + 0.012)
    axes[0].set_ylabel("Bits per byte", color=MUTED, labelpad=8)
    save(fig, output, "event_lm_best_rate_pair.png")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--figure-dir", type=Path, default=RESULTS / "figures")
    parser.add_argument("--record", type=Path, default=RECORD)
    args = parser.parse_args()
    figure(scores(args.record), args.figure_dir)


if __name__ == "__main__":
    main()
