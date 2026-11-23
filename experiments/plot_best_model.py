"""Figure of the best spiking stack so far (l2048-know-st): what it is, how it scores against v16 and the Transformer
of its size, and where it wins and loses, split by what each byte's context offers."""

from __future__ import annotations

import argparse
from pathlib import Path

from plot_dendrite import GRID, MUTED, ORANGE, PAPER, RESULTS, TEXT, plt, read_result, save, style
from plot_knowledge_architecture import GREEN_FILL, SMALL, VIOLET_FILL, arrow, box, canvas, label, note
from plot_state_reads import GREEN, VIOLET, tall_heading

LIGHT_VIOLET = "#b7aef0"
# enwik8 bits per byte: (name, colour, test, validation). The stacks read 8,192 cells back with v15's fast synapses on
# all levels; the Transformer reads windows of 2,048 bytes, 16 scored after 2,032 of history.
SCORES = (("Transformer, 34.75M", ORANGE, 1.10669, 1.13033),
          ("v16 stack, 34.81M", LIGHT_VIOLET, 1.11994, 1.15138),
          ("Best stack now, 36.26M", VIOLET, 1.11702, 1.14963))
SPLITS = (("Test", "event_lm/knowledge_gap/gap_by_piece_best_test.json"),
          ("Validation", "event_lm/knowledge_gap/gap_by_piece_best_validation.json"))
KINDS = (("copyable near", "Copyable from the last 2 KB"), ("copyable far only", "Copyable only 2 to 8 KB back"),
         ("new", "New context"))


def architecture(ax):
    canvas(ax)
    label(ax, 3, 98.5, "The stack", size=12)
    box(ax, 10, 89.5, 80, 6, "Byte in", ["one of 256 values, one after another"])
    arrow(ax, (50, 89.5), (50, 86.6))
    box(ax, 10, 78.4, 80, 8.2, "Input synapses", ["256 values → 512 channels of spike counts"])
    arrow(ax, (50, 78.4), (50, 75.6))
    box(ax, 3, 25.8, 94, 49.8, None, dashed=True, face=PAPER)
    note(ax, 6, 73.4, "11 levels on one stream", ha="left")
    box(ax, 7, 50.2, 86, 21.0, "State read: memory", [], face=VIOLET_FILL, edge=VIOLET, title_color=VIOLET, shift=7.0)
    note(ax, 50, 64.5, "8 heads race for 256 spikes over the tape of past cells:")
    note(ax, 50, 61.4, "2,048 cells back in training, 8,192 when reading")
    note(ax, 50, 58.3, "ages that fade by learned slopes; tags hear the last bytes too")
    box(ax, 11, 51.6, 78, 4.4, None, face=GREEN_FILL, edge=GREEN)
    label(ax, 50, 53.8, "+ 128 knowledge cells a head: learned cells and a place for 'nothing fits'",
          color=GREEN, size=SMALL - 0.3, ha="center")
    arrow(ax, (50, 50.2), (50, 47.4))
    box(ax, 7, 28.6, 86, 18.8, "Threshold cells: knowledge", [], face=VIOLET_FILL, edge=VIOLET, title_color=VIOLET,
        shift=5.4)
    note(ax, 50, 39.2, "1,188 cells a level, on 1,024 fibres:")
    note(ax, 50, 36.1, "every channel as it is now, and once more 1 to 3 bytes ago")
    note(ax, 50, 32.0, "integer spikes, a budget of 311,000 pulses a byte")
    arrow(ax, (50, 25.8), (50, 22.8))
    box(ax, 10, 12.4, 80, 10.4, "Output neurons", ["512 channels → 256 byte values",
                                                    "fast synapses learn from every byte read"])
    arrow(ax, (50, 12.4), (50, 9.4))
    note(ax, 50, 7.4, "spikes of evidence for the next byte")
    box(ax, 10, 0.6, 80, 4.0, None, face=GREEN_FILL, edge=GREEN)
    label(ax, 50, 2.6, "trained on streams of 20 KB, every state carried from byte to byte", color=GREEN,
          size=SMALL - 0.3, ha="center")


def scores(ax):
    style(ax, grid_axis="y")
    width = 0.24
    for split in range(2):
        for place, (name, color, test, validation) in enumerate(SCORES):
            value = (test, validation)[split]
            x = split + (place - 1) * (width + 0.03)
            ax.bar(x, value, width=width, color=color, zorder=2)
            ax.annotate(f"{value:.4f}", (x, value), xytext=(0, 4), textcoords="offset points", ha="center",
                        va="bottom", size=8.8, color=TEXT)
        gap = SCORES[2][2 + split] - SCORES[0][2 + split]
        ax.annotate(f"gap {gap:+.4f}", (split, 1.168), ha="center", va="bottom", size=9.5, color=TEXT, weight="bold")
    ax.set_xticks((0, 1), ("enwik8 test", "enwik8 validation"), color=TEXT, size=10)
    ax.set_ylim(1.09, 1.18)
    ax.set_ylabel("Bits per byte (lower is better)", color=MUTED, labelpad=8)
    handles = [plt.Rectangle((0, 0), 1, 1, color=color) for _, color, _, _ in SCORES]
    ax.legend(handles, [name for name, *_ in SCORES], frameon=False, labelcolor=TEXT, fontsize=9, loc="upper left",
              bbox_to_anchor=(0.0, 1.02), ncol=3)
    ax.set_title("Scores, 4,999,999 bytes a split", loc="left", size=11, color=TEXT, weight="bold", pad=22)


def gaps(ax):
    style(ax, grid_axis="x")
    height = 0.34
    for split, (title, path) in enumerate(SPLITS):
        rows = read_result(path)["by_what_the_context_offers"]
        for place, (key, _) in enumerate(KINDS):
            row = rows[key]
            gap = row["gap_bits_per_byte"]
            y = place + (split - 0.5) * (height + 0.04)
            ax.barh(y, gap, height=height, color=VIOLET if gap < 0 else ORANGE, alpha=1.0 if split == 0 else 0.55,
                    zorder=2)
            side, align = (5, "left") if gap >= 0 else (-5, "right")
            ax.annotate(f"{title.lower()} {gap:+.3f} ({row['share_of_bytes']:.0%})", (gap, y), xytext=(side, 0),
                        textcoords="offset points", ha=align, va="center", size=8.4, color=TEXT)
    ax.axvline(0, color=TEXT, linewidth=0.9, zorder=4)
    ax.set_yticks(range(len(KINDS)), [name for _, name in KINDS], color=TEXT, size=9.5)
    ax.invert_yaxis()
    ax.set_xlim(-0.62, 0.22)
    ax.set_xlabel("Bits per byte, best stack minus Transformer (violet: the stack is better)", color=MUTED,
                  labelpad=8)
    ax.set_title("Where it wins and loses (share of bytes in brackets)", loc="left", size=11, color=TEXT,
                 weight="bold", pad=10)


def plot(output):
    fig = plt.figure(figsize=(16.5, 10.4))
    tall_heading(fig, "The best spiking stack so far: 0.010 behind the Transformer",
                 "l2048-know-st, one seed: the v16 stack forked at 24,000 updates with 128 knowledge cells a head, "
                 "trained on streams to 90,000 updates (368.6M training bytes)")
    architecture(fig.add_axes((0.02, 0.06, 0.40, 0.80)))
    scores(fig.add_axes((0.50, 0.53, 0.47, 0.31)))
    gaps(fig.add_axes((0.62, 0.135, 0.35, 0.29)))
    fig.text(0.47, 0.012, "The best stack has 4.2% more parameters than the Transformer, the knowledge cells; by the "
             "stack's depth scaling\nthat is worth about 0.002. It copies far better, from near and far, and loses "
             "0.057 to 0.070 bits a byte\non new contexts, three quarters of the bytes: the gap is knowledge.",
             size=8.8, color=MUTED, va="bottom")
    save(fig, output, "event_lm_best_model.png")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=RESULTS / "figures")
    args = parser.parse_args()
    plot(args.output)


if __name__ == "__main__":
    main()
