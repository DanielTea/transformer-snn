"""Figures for long streams: how much the fast output synapses gain against how much of a stream they have read."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from plot_dendrite import GRID, MUTED, TEXT, plt, save, style
from plot_read_clocks import SLATE, end_labels
from plot_state_reads import VIOLET, tall_heading

# The stack of a release is violet, the Transformer orange and enwik8 slate. Whole books are the darker violet of
# the reach shades of v15.
WHOLE_VIOLET = "#3b2c9c"


def middle(row):
    """The middle of a bin on the doubling axis: the geometric mean of its edges, and 4 KB for the first bin."""
    return math.sqrt(row["from_kb"] * row["to_kb"]) if row["from_kb"] else row["to_kb"] / 2


def plot_reading(series, output, title, subtitle):
    """``series`` holds (record, label in the record, label shown, colour, with quartiles) for every line."""
    fig, ax = plt.subplots(figsize=(10.4, 6.0))
    fig.subplots_adjust(left=0.09, right=0.8, bottom=0.18, top=0.8)
    tall_heading(fig, title, subtitle)
    style(ax)
    labels = []
    for path, name, shown, color, band in series:
        record = json.loads(Path(path).read_text())
        rows = [row for row in next(item for item in record["series"] if item["label"] == name)["bins"] if row]
        x = [middle(row) for row in rows]
        gains = [row["gain"] for row in rows]
        if band:
            ax.fill_between(x, [row["quartiles"][0] for row in rows], [row["quartiles"][1] for row in rows],
                            color=color, alpha=0.12, linewidth=0, zorder=1)
        ax.plot(x, gains, color=color, linewidth=2, zorder=3, marker="o", markersize=5, markeredgecolor="white",
                markeredgewidth=0.9)
        labels.append((x[-1], gains[-1], shown, color))
    ax.set_xscale("log", base=2)
    ax.set_xlim(2.5, None)
    ticks = [4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
    top = ax.get_xlim()[1]
    ax.set_xticks([tick for tick in ticks if tick <= top], [f"{tick:,}" for tick in ticks if tick <= top])
    ax.minorticks_off()
    ax.set_ylim(0, None)
    ax.axhline(0, color=TEXT, linewidth=0.8)
    ax.grid(axis="x", color=GRID, linewidth=0.75)
    end_labels(ax, labels)
    ax.set_xlabel("Kilobytes of the stream read before, on a doubling scale", color=MUTED, labelpad=8)
    ax.set_ylabel("Bits per byte that the fast synapses gain", color=MUTED, labelpad=8)
    fig.text(0.09, 0.03, "Each point pools the pieces that follow that much reading in their stream. The band holds "
             "the middle half of the streams' own gains.", color=MUTED, size=8.5)
    save(fig, output, "event_lm_long_streams_reading")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--books", type=Path, required=True, help="reading curve of the PG-19 books")
    parser.add_argument("--enwik8", type=Path, required=True, help="reading curve of the enwik8 test split")
    args = parser.parse_args()
    plot_reading([(args.books, "Stack, PG-19 in parts", "Stack, PG-19 books", VIOLET, True),
                  (args.enwik8, "Stack, enwik8 test", "Stack, enwik8 test", SLATE, False)],
                 args.output, "Fast synapses keep learning as they read a book",
                 "the released stack of v15 · fast synapses start at zero in every stream · PG-19 books in parts "
                 "of 256 KB · enwik8 test")


if __name__ == "__main__":
    main()
