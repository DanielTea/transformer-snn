"""Figure of where the full-size distance lies (results/event_lm/knowledge_gap/full_size_spikes_late.json): the
spiking stack, a float stand-in whose reads and cells answer in real numbers, and the Transformer, each after 24,000
updates at 3e-4. The stand-in is a diagnostic and never a path: the part of the distance that it closes is what the
spikes of reads and cells cost, the part that it leaves is the layout's.

usage: python experiments/plot_distance_split.py [--figure-dir DIR]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from plot_dendrite import GRAY, MUTED, ORANGE, RESULTS, TEXT, plt, save, style

DARK_VIOLET = "#4a2a85"
RECORD = RESULTS / "event_lm" / "knowledge_gap" / "full_size_spikes_late.json"


def scores(record):
    bpb = json.loads(Path(record).read_text())["result"]["bpb"]
    return bpb["fs-st3"], bpb["fs-float3"], bpb["fs-tf3"]


def bracket(ax, low, high, y, text, note):
    """A bracket over a stretch of the axis, with its length above and what it is below."""
    ax.plot([low, high], [y, y], color=MUTED, linewidth=1.0, zorder=2)
    for end in (low, high):
        ax.plot([end, end], [y - 0.05, y + 0.05], color=MUTED, linewidth=1.0, zorder=2)
    ax.annotate(text, ((low + high) / 2, y), textcoords="offset points", xytext=(0, 9), ha="center", va="bottom",
                color=TEXT, size=12, weight="bold")
    ax.annotate(note, ((low + high) / 2, y), textcoords="offset points", xytext=(0, -10), ha="center", va="top",
                color=MUTED, size=9.5)


def figure(found, output):
    stack, stand_in, transformer = found
    fig, ax = plt.subplots(figsize=(12.4, 4.6))
    fig.subplots_adjust(left=0.05, right=0.97, bottom=0.17, top=0.66)
    share = (stand_in - transformer) / (stack - transformer)
    fig.text(0.035, 0.955, f"{share:.0%} of the distance to the Transformer is the layout, not the spikes", size=19.5,
             weight="bold", color=TEXT, va="top")
    fig.text(0.036, 0.865, "Full size, 24,000 updates at 3e-4, one seed a run: bits per byte on 2M validation bytes, "
             "the stack as 16 unbroken streams. The stand-in replaces the stack's racing\nreads and threshold cells "
             "with softmax reads and real-valued cells; its stream still fires whole spikes. It is a diagnostic, never "
             "a path. Lower is better.", size=10, color=MUTED, va="top", linespacing=1.45)
    style(ax, grid_axis="x")
    for value, label, color, marker, size in ((stack, "Spiking stack", DARK_VIOLET, "o", 11),
                                              (stand_in, "Float stand-in", GRAY, "D", 9),
                                              (transformer, "Transformer", ORANGE, "s", 10)):
        ax.plot([value], [0], marker=marker, markersize=size, color=color, markeredgecolor="white",
                markeredgewidth=1.6, zorder=4, linestyle="none")
        ax.annotate(f"{label}\n{value:.4f}", (value, 0), textcoords="offset points", xytext=(0, -16), ha="center",
                    va="top", color=TEXT, size=10.5, weight="bold" if color != GRAY else "normal")
    ax.axhline(0, color=GRAY, linewidth=0.8, zorder=1)
    bracket(ax, stand_in, stack, 0.55, f"{stack - stand_in:.3f}", "spikes in reads and cells")
    bracket(ax, transformer, stand_in, 0.55, f"{stand_in - transformer:.3f}", "the layout")
    ax.set_xlim(transformer - 0.006, stack + 0.006)
    ax.set_ylim(-1.0, 1.0)
    ax.set_yticks([])
    ax.invert_xaxis()
    ax.set_xlabel("Bits per byte (better to the right)", color=MUTED, labelpad=8)
    save(fig, output, "event_lm_distance_split.png")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--figure-dir", type=Path, default=RESULTS / "figures")
    args = parser.parse_args()
    figure(scores(RECORD), args.figure_dir)


if __name__ == "__main__":
    main()
