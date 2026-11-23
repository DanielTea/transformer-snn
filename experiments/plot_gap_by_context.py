"""Figure of where the stack's gap to the Transformer lies: the paired bits of the v16 headline pair split by what
the eight bytes before every target offer (experiments/gap_by_piece.py), on both splits of enwik8."""

from __future__ import annotations

import argparse
from pathlib import Path

from plot_dendrite import GRID, MUTED, ORANGE, RESULTS, TEXT, plt, read_result, save, style
from plot_state_reads import VIOLET, tall_heading

RECORDS = (("Test split", "event_lm/knowledge_gap/gap_by_piece_test.json"),
           ("Validation split", "event_lm/knowledge_gap/gap_by_piece_validation.json"))
LABELS = {"copyable near": "Copyable from the last 2 KB",
          "copyable far only": "Copyable only from 2 to 8 KB back",
          "seen, other follower": "Seen recently, another byte followed",
          "new": "New context"}


def plot(output):
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 6.0), sharey=True)
    fig.subplots_adjust(left=0.235, right=0.975, bottom=0.25, top=0.7, wspace=0.1)
    tall_heading(fig, "The stack out-remembers the Transformer; its gap is knowledge",
                 "v16 stack (34.81M parameters, reach 8,192, fast synapses) minus the Transformer (34.75M, windows of "
                 "2,048), bits per byte,\nby what the eight bytes before a target offer")
    names = list(LABELS)
    for ax, (title, path) in zip(axes, RECORDS):
        record = read_result(path)
        rows = record["by_what_the_context_offers"]
        style(ax, grid_axis="x")
        for place, name in enumerate(names):
            row = rows[name]
            gap, (low, high) = row["gap_bits_per_byte"], row["gap_95"]
            color = VIOLET if gap < 0 else ORANGE
            ax.barh(place, gap, height=0.56, color=color, zorder=2)
            ax.plot((low, high), (place, place), color=TEXT, linewidth=1.1, zorder=3)
            side, align = (6, "left") if gap >= 0 else (-6, "right")
            ax.annotate(f"{gap:+.3f}", (high if gap >= 0 else low, place), xytext=(side, 0),
                        textcoords="offset points", ha=align, va="center", size=9.5, color=TEXT)
            ax.text(1.0, place, f"{row['share_of_bytes']:.0%}", transform=ax.get_yaxis_transform(), ha="right",
                    va="center", size=9.5, color=MUTED)
        ax.axvline(0, color=TEXT, linewidth=0.9, zorder=4)
        ax.set_title(f"{title}: whole gap {record['gap']:+.4f}", loc="left", size=11, color=TEXT, weight="bold",
                     pad=10)
        ax.set_xlim(-0.47, 0.25)
        ax.text(1.0, len(names) - 0.35, "of bytes", transform=ax.get_yaxis_transform(), ha="right", va="center",
                size=8.5, color=MUTED)
        ax.set_xlabel("Bits per byte, stack minus Transformer", color=MUTED, labelpad=8)
    axes[0].set_yticks(range(len(names)), [LABELS[name] for name in names], color=TEXT, size=10)
    fig.text(0.036, 0.035, "Violet: the stack predicts better; orange: the Transformer does. Every piece of 256 bytes "
             "holds a mix of kinds, so the gap of a kind is the least-squares fit over 19,488 pieces\n(4,988,928 "
             "targets a split), with 95% intervals from resampled pieces. The stack copies better from near and far; "
             "on new contexts, three quarters of the bytes,\nit is 0.057 to 0.066 behind: more than the whole gap.",
             size=8.8, color=MUTED, va="bottom")
    save(fig, output, "event_lm_gap_by_context.png")


STAGES = (("Test bed\n6,000 updates\n24.6M training bytes", "event_lm/knowledge_gap/gap_by_piece_bed_6k.json"),
          ("Test bed\n24,000 updates\n98.3M training bytes", "event_lm/knowledge_gap/gap_by_piece_bed_24k.json"),
          ("Full size, v16\n90,000 updates\n368.6M training bytes", "event_lm/knowledge_gap/gap_by_piece_validation.json"))


def plot_with_data(output):
    fig, ax = plt.subplots(figsize=(11.2, 6.4))
    fig.subplots_adjust(left=0.09, right=0.97, bottom=0.27, top=0.72)
    tall_heading(fig, "With data the stack learns to copy first, knowledge slowly",
                 "Stack minus Transformer, bits per byte on the enwik8 validation split, by what the eight bytes before "
                 "a target offer; the\nTransformer reads windows of its training length, the stack the same bytes as "
                 "unbroken streams")
    style(ax, grid_axis="y")
    width = 0.3
    for place, (label, path) in enumerate(STAGES):
        record = read_result(path)
        rows = record["by_what_the_context_offers"]
        for offset, (name, color) in zip((-width / 2 - 0.02, width / 2 + 0.02),
                                         (("copyable near", VIOLET), ("new", ORANGE))):
            row = rows[name]
            gap, (low, high) = row["gap_bits_per_byte"], row["gap_95"]
            ax.bar(place + offset, gap, width=width, color=color, zorder=2)
            ax.plot((place + offset, place + offset), (low, high), color=TEXT, linewidth=1.1, zorder=3)
            ax.annotate(f"{gap:+.3f}", (place + offset, high if gap >= 0 else low), xytext=(0, 5 if gap >= 0 else -5),
                        textcoords="offset points", ha="center", va="bottom" if gap >= 0 else "top", size=9.5,
                        color=TEXT)
        ax.annotate(f"whole gap {record['gap']:+.3f}", (place, 0.255), ha="center", va="bottom", size=10,
                    color=TEXT, weight="bold")
    ax.axhline(0, color=TEXT, linewidth=0.9, zorder=4)
    ax.set_xticks(range(len(STAGES)), [label for label, _ in STAGES], color=TEXT, size=10)
    ax.set_ylim(-0.2, 0.29)
    ax.set_ylabel("Bits per byte, stack minus Transformer", color=MUTED, labelpad=8)
    handles = [plt.Rectangle((0, 0), 1, 1, color=VIOLET), plt.Rectangle((0, 0), 1, 1, color=ORANGE)]
    ax.legend(handles, ["Copyable from what both models reach (the last 240 or 2,032 bytes)",
                        "New context: the eight bytes before did not run in the last 8 KB (about 3/4 of the bytes)"],
              frameon=False, labelcolor=TEXT, fontsize=9.5, loc="lower left", bbox_to_anchor=(0.0, 0.0))
    fig.text(0.036, 0.03, "Test bed: 4.86M-parameter stack and 4.79M-parameter Transformer, reach 256, trained in "
             "windows on the server's GPU, seed 20001. Full size: 34.81M and 34.75M,\nthe stack with its fast synapses "
             "and a reach of 8,192. Gaps per kind are least-squares fits over pieces of 512 (test bed) or 256 bytes "
             "(full size),\nwith 95% intervals from resampled pieces. One seed each.",
             size=8.8, color=MUTED, va="bottom")
    save(fig, output, "event_lm_gap_with_data.png")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=RESULTS / "figures")
    args = parser.parse_args()
    plot(args.output)
    plot_with_data(args.output)


if __name__ == "__main__":
    main()
