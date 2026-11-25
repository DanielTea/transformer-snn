"""Figure of the looped stack at full length (results/event_lm/knowledge_gap/full_size_loop_run.json): the plain
stack of v19 and the looped stack of v20 after 90,000 updates, beside their Transformer, on both splits.

usage: python experiments/plot_loop_full_length.py [--figure-dir DIR] [--record JSON]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from plot_dendrite import MUTED, ORANGE, RESULTS, TEXT, plt, save, style

DARK_VIOLET = "#4a2a85"
RECORD = RESULTS / "event_lm" / "knowledge_gap" / "full_size_loop_run.json"


def scores(record):
    """For each split, the plain stack's, the looped stack's and the Transformer's bits per byte."""
    found = json.loads(Path(record).read_text())["result"]["scores"]
    loop, plain, transformer = found["fsp-loop"], found["fsp-st"], found["fsp-tf"]
    return {
        "test": {"plain": plain["taught test (reach 8,192, fast output synapses on its 11 levels)"],
                 "loop": loop["taught test (reach 8,192, fast output synapses at 2^24 on all 18 passes)"],
                 "transformer": transformer["frozen test (windows of 2,048)"]},
        "validation": {"plain": plain["taught validation"], "loop": loop["taught validation"],
                       "transformer": transformer["frozen validation"]},
    }


def bracket(ax, x, low, high, text, side):
    """A thin bracket between the stack and the Transformer at one stack, with the distance beside it."""
    ax.plot([x, x], [low, high], color=MUTED, linewidth=0.9, zorder=2)
    for end in (low, high):
        ax.plot([x - 0.012, x + 0.012], [end, end], color=MUTED, linewidth=0.9, zorder=2)
    ax.annotate(text, (x, (low + high) / 2), textcoords="offset points", xytext=(7 * side, 0),
                ha="left" if side > 0 else "right", va="center", color=TEXT, size=10, weight="bold")


def figure(found, output):
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.9), sharey=True)
    fig.subplots_adjust(left=0.075, right=0.975, bottom=0.115, top=0.72, wspace=0.07)
    test = found["test"]
    fig.text(0.035, 0.965, "At full length the loop holds and narrows the gap to the Transformer", size=19.5,
             weight="bold", color=TEXT, va="top")
    fig.text(0.036, 0.888, "Bits per byte on enwik8 after 90,000 updates from random weights at full size (the stacks "
             "36.3M parameters, the Transformer 34.8M), one seed a run.\nThe stacks read each split as 16 unbroken "
             "streams with fast output synapses; the Transformer reads windows of 2,048 bytes. Lower is better.\n"
             f"The looped stack runs levels 2 to 8 twice with the same synapses: {test['plain'] - test['loop']:.4f} "
             "better on test, for 1.63 times the pulses.", size=10, color=MUTED, va="top", linespacing=1.45)
    for ax, split in zip(axes, ("test", "validation")):
        style(ax)
        ax.set_title(f"enwik8 {split}", loc="left", size=11.5, weight="bold", color=TEXT, pad=10)
        row = found[split]
        stack = (row["plain"], row["loop"])
        ax.plot([0, 1], stack, color=DARK_VIOLET, linewidth=2.2, marker="o", markersize=8, zorder=3,
                markeredgecolor="white", markeredgewidth=1.5)
        ax.plot([0, 1], [row["transformer"]] * 2, color=ORANGE, linewidth=2.2, marker="s", markersize=8, zorder=3,
                markeredgecolor="white", markeredgewidth=1.5)
        ax.annotate(f"{stack[0]:.4f}", (0, stack[0]), textcoords="offset points", xytext=(-12, 0), ha="right",
                    va="center", color=TEXT, size=10)
        ax.annotate(f"{row['transformer']:.4f}", (0, row["transformer"]), textcoords="offset points",
                    xytext=(-12, 0), ha="right", va="center", color=TEXT, size=10)
        ax.annotate(f"Spiking stack  {stack[1]:.4f}", (1, stack[1]), textcoords="offset points", xytext=(12, 0),
                    ha="left", va="center", color=TEXT, size=10.5, weight="bold")
        ax.annotate(f"Transformer  {row['transformer']:.4f}", (1, row["transformer"]), textcoords="offset points",
                    xytext=(12, 0), ha="left", va="center", color=TEXT, size=10.5, weight="bold")
        bracket(ax, 0.09, row["transformer"], stack[0], f"{stack[0] - row['transformer']:.3f} apart", 1)
        bracket(ax, 0.91, row["transformer"], stack[1], f"{stack[1] - row['transformer']:.3f} apart", -1)
        ax.set_xlim(-0.34, 1.78)
        ax.set_xticks([0, 1], ["Plain stack (v19)", "Looped stack (v20)"])
        ax.tick_params(axis="x", labelsize=10.5, labelcolor=TEXT)
        ax.grid(axis="x", visible=False)
    low = min(row["transformer"] for row in found.values())
    high = max(row["plain"] for row in found.values())
    axes[0].set_ylim(low - 0.012, high + 0.012)
    axes[0].set_ylabel("Bits per byte", color=MUTED, labelpad=8)
    save(fig, output, "event_lm_loop_full_length.png")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--figure-dir", type=Path, default=RESULTS / "figures")
    parser.add_argument("--record", type=Path, default=RECORD)
    args = parser.parse_args()
    figure(scores(args.record), args.figure_dir)


if __name__ == "__main__":
    main()
