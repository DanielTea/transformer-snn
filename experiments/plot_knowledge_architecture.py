"""Figure of the spiking stack with knowledge cells, at the size of the v16 model: the stack, one level in detail, one
head's race at one byte with its equations, and where the parameters sit. Violet is kept from v16, green is new."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

from plot_dendrite import MUTED, ORANGE, PAPER, RESULTS, TEXT, plt, save
from plot_state_reads import GREEN, VIOLET, tall_heading

VIOLET_FILL, GREEN_FILL, GRAY_FILL, GRAY_EDGE, ORANGE_FILL = "#ece9fb", "#e1f3ec", "#f2f4f7", "#9aa6b6", "#fbeee2"
SMALL = 8.8


def box(ax, x, y, width, height, title, lines=(), *, face=GRAY_FILL, edge=GRAY_EDGE, title_color=TEXT,
        dashed=False, size=10.5, line_size=SMALL, shift=0.0):
    ax.add_patch(FancyBboxPatch((x, y), width, height, boxstyle="round,pad=0,rounding_size=1.2", facecolor=face,
                                edgecolor=edge, linewidth=1.0, linestyle=(0, (4, 3)) if dashed else "solid"))
    rows = [title, *lines] if title else list(lines)
    step = 3.1 if height < 12 else 3.4
    top = y + height / 2 + step * (len(rows) - 1) / 2 + shift
    for row, text in enumerate(rows):
        bold = bool(title) and row == 0
        ax.text(x + width / 2, top - step * row, text, ha="center", va="center", size=size if bold else line_size,
                weight="bold" if bold else "normal", color=title_color if bold else MUTED)


def label(ax, x, y, text, *, color=TEXT, size=10.5, weight="bold", ha="left"):
    ax.text(x, y, text, ha=ha, va="center", size=size, weight=weight, color=color)


def note(ax, x, y, text, *, color=MUTED, size=SMALL, ha="center"):
    label(ax, x, y, text, color=color, size=size, weight="normal", ha=ha)


def arrow(ax, start, end, color=MUTED, width=1.1):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=10, color=color, linewidth=width,
                                 shrinkA=0, shrinkB=0))


def canvas(ax):
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")


def stack(ax):
    """The whole stack, from a byte in to the evidence for the next byte."""
    canvas(ax)
    label(ax, 4, 98, "The stack", size=12)
    box(ax, 10, 87.5, 80, 6.5, "Byte in", ["one of 256 values at every step"])
    arrow(ax, (50, 87.5), (50, 84))
    box(ax, 10, 74, 80, 10, "Input synapses", ["256 values → 512 channels", "131,072 synapses"])
    arrow(ax, (50, 74), (50, 70.6))
    box(ax, 4, 26, 92, 44.6, None, dashed=True, face=PAPER)
    note(ax, 7, 67.8, "11 levels on one stream", ha="left")
    note(ax, 7, 64.6, "512 channels, counts of −512 to 512", ha="left")
    box(ax, 9, 41, 82, 20.5, "State read", ["8 heads of 64 channels", "each races for 256 spikes"], face=VIOLET_FILL,
        edge=VIOLET, title_color=VIOLET, shift=2.6)
    box(ax, 15, 42.4, 70, 4.2, None, face=GREEN_FILL, edge=GREEN)
    label(ax, 50, 44.5, "+ 128 knowledge cells a head", color=GREEN, size=SMALL, ha="center")
    arrow(ax, (50, 41), (50, 38.2))
    box(ax, 9, 28.2, 82, 10, "Threshold cells", ["1,188 cells on 1,024 fibres"], face=VIOLET_FILL, edge=VIOLET,
        title_color=VIOLET)
    arrow(ax, (50, 26), (50, 22.6))
    box(ax, 10, 11.6, 80, 11, "Output neurons", ["512 channels → 256 byte values", "fast synapses learn while reading"])
    arrow(ax, (50, 11.6), (50, 8))
    note(ax, 50, 5.4, "spikes of evidence for every next byte")
    note(ax, 50, 2.2, "a budget of 310,000 pulses a byte")


def level(ax):
    """One level: the state read, its neurons, its tape and its knowledge cells, then the threshold cells."""
    canvas(ax)
    label(ax, 2, 98, "One level, in detail", size=12)
    box(ax, 1, 33, 98, 62, None, face=VIOLET_FILL, edge=VIOLET)
    label(ax, 3, 92.6, "State read: 8 heads of 64 channels", color=VIOLET)
    note(ax, 3, 89.4, "Every neuron fires up to 128 pulses a byte, and a pool of inhibition raises its threshold with "
         "the stream's activity.", size=8.2, ha="left")
    box(ax, 4, 80.5, 92, 6, None, face="white", edge=GRAY_EDGE)
    note(ax, 50, 83.5, "The stream now (512 channels), and late fibres: the same channels 1 to 3 bytes ago (512)",
         color=TEXT)
    for x in (17, 50, 83):
        arrow(ax, (x, 80.5), (x, 77.3))
    for x, name, heard in ((4, "Probe neurons", "512, hear the stream now"), (37, "Tag neurons", "512, hear now and late"),
                           (70, "Payload neurons", "512, hear the stream now")):
        box(ax, x, 69.5, 26, 7.8, name, [heard], face="white", edge=VIOLET, title_color=VIOLET, size=9.6)
    box(ax, 14, 53.5, 34, 10.1, "Tape of earlier cells", ["their tags and payloads,", "2,048 back in training, 8,192 read"],
        face="white", edge=VIOLET, title_color=VIOLET, size=9.6, line_size=8.2)
    box(ax, 52, 53.5, 44, 10.1, "Knowledge cells (new)", ["128 a head, each with a learned tag, payload, offset",
                                                          "1,024 a level, 132,096 synapses"],
        face=GREEN_FILL, edge=GREEN, title_color=GREEN, size=9.6, line_size=8.2)
    arrow(ax, (9, 69.5), (9, 50.6))
    arrow(ax, (44, 69.5), (40, 63.6))
    arrow(ax, (80, 69.5), (46, 63.6))
    arrow(ax, (31, 53.5), (36, 50.6))
    arrow(ax, (74, 53.5), (64, 50.6))
    box(ax, 4, 44, 92, 6.6, None, face=ORANGE_FILL, edge=ORANGE)
    label(ax, 50, 47.3, "Every head's race for 256 spikes: its probe against every tag it reaches", size=9.6,
          color=ORANGE, ha="center")
    arrow(ax, (50, 44), (50, 39.6))
    box(ax, 4, 35.2, 92, 4.4, None, face="white", edge=VIOLET)
    note(ax, 50, 37.4, "Read neurons: the winners' mean payload, through 512 synapses each, added to the stream",
         color=TEXT)
    arrow(ax, (50, 33), (50, 29.6))
    box(ax, 1, 2, 98, 27.6, None, face=VIOLET_FILL, edge=VIOLET)
    label(ax, 3, 27.2, "Threshold cells", color=VIOLET)
    box(ax, 4, 19, 92, 5.2, None, face="white", edge=GRAY_EDGE)
    note(ax, 50, 21.6, "1,024 fibres: every channel of the stream now, and once more 1 to 3 bytes ago", color=TEXT)
    arrow(ax, (50, 19), (50, 16.8))
    box(ax, 4, 9.6, 92, 7.2, "1,188 threshold cells", ["each weighs all 1,024 fibres and has a learned rest below "
                                                        "its threshold; up to 128 spikes a byte"],
        face="white", edge=VIOLET, title_color=VIOLET, size=9.6, line_size=8.2)
    arrow(ax, (50, 9.6), (50, 7.8))
    box(ax, 4, 3.2, 92, 4.6, None, face="white", edge=GRAY_EDGE)
    note(ax, 50, 5.5, "Send synapses: every cell to all 512 channels, added to the stream", color=TEXT)


def race(ax):
    """One head's race at one byte, and the equations it follows."""
    canvas(ax)
    label(ax, 2, 98, "One head's race at one byte", size=12)
    stream_x = np.linspace(5, 58, 12)
    known_x = np.linspace(68, 95, 6)
    width = 3.1
    for place, x in enumerate(stream_x):
        ax.add_patch(Rectangle((x - width / 2, 76), width, 6, facecolor=VIOLET, edgecolor=VIOLET, linewidth=0.8,
                               alpha=0.22 + 0.78 * place / (len(stream_x) - 1)))
    for x in known_x:
        ax.add_patch(Rectangle((x - width / 2, 76), width, 6, facecolor=GREEN, edgecolor=GREEN, linewidth=0.8))
    label(ax, 31.5, 92.5, "Cells of earlier bytes", color=VIOLET, size=10, ha="center")
    note(ax, 31.5, 89.2, "written by the stream, fading with age")
    label(ax, 81.5, 92.5, "Knowledge cells", color=GREEN, size=10, ha="center")
    note(ax, 81.5, 89.2, "written by training, ageless")
    note(ax, stream_x[0], 84.6, "oldest", size=8.2)
    note(ax, stream_x[-1], 84.6, "this byte", size=8.2)
    won = {0: 0.3, 3: 2.6, 5: 0.5, 8: 1.2, 10: 1.9, 11: 3.2}
    known_won = {1: 2.8, 3: 1.4, 5: 0.6}
    probe = (50, 66.4)
    for xs, spikes in ((stream_x, won), (known_x, known_won)):
        for place, count in spikes.items():
            ax.plot([probe[0], xs[place]], [probe[1], 76], color=ORANGE, linewidth=count * 1.2,
                    alpha=0.3 + 0.2 * count, solid_capstyle="round")
    box(ax, 35, 60, 30, 6.4, "Probe", ["the present byte's pulses"], face=ORANGE_FILL, edge=ORANGE,
        title_color=ORANGE, size=9.6)
    note(ax, 50, 56.6, "Line width: the spikes a cell wins of the head's 256")
    box(ax, 2, 1, 96, 52, None, face="white", edge=GRAY_EDGE)
    rows = (("Charge of a cell of an earlier byte", "probe · tag × gain + the offset of its age", VIOLET),
            ("", "learned for ages below 512; beyond, the head's slope × log₂(age / 511)", VIOLET),
            ("Charge of a knowledge cell", "probe · tag × gain + its own learned offset", GREEN),
            ("Rate", "256 × e^charge / the sum of e^charge over every cell it reaches", TEXT),
            ("", "summed over the earlier cells by age, then the knowledge cells", TEXT),
            ("Spikes", "the whole part of rate + phase, with phases spread over the cells", TEXT),
            ("Read", "the sum of spikes × payload / the sum of spikes, in whole pulses", TEXT))
    y = 50
    for name, text, color in rows:
        if name:
            y -= 1.0
            label(ax, 5, y, name, color=color, size=9.4)
            y -= 3.3
        label(ax, 5, y, text, size=8.6, weight="normal", color=TEXT)
        y -= 3.1
    note(ax, 5, 3.6, "A stream cut anywhere fires the same spikes.", size=8.6, ha="left")


def parameters(ax):
    """Where the 34.81M parameters of the full-size stack sit, and what knowledge cells change."""
    canvas(ax)
    columns = (2, 36, 52, 62)
    heads = ("Part of the full-size stack", "Synapses", "Share", "With knowledge cells")
    rows = (("Threshold cells: fibres in, rests, synapses out", "20,085,516", "57.7%",
             "kept, or 86 of the 1,188 cells a level give way to hold the size", VIOLET),
            ("State reads: probe, tag, payload and read neurons, offsets", "14,463,240", "41.5%", "kept", VIOLET),
            ("Input and output synapses", "262,144", "0.8%", "kept", VIOLET),
            ("Knowledge cells", "1,453,056", "+4.2%", "new: 128 a head, 1,024 a level, 11,264 in all", GREEN))
    label(ax, columns[0], 88, heads[0], size=9.4)
    label(ax, columns[1] + 6, 88, heads[1], size=9.4, ha="right")
    label(ax, columns[2] + 4, 88, heads[2], size=9.4, ha="right")
    label(ax, columns[3], 88, heads[3], size=9.4)
    ax.plot([1, 99], [78, 78], color=GRAY_EDGE, linewidth=0.8)
    for row, (part, count, share, change, color) in enumerate(rows):
        y = 68 - 19 * row
        ax.add_patch(Rectangle((0.6, y - 2.6), 0.6, 5.2, facecolor=color, edgecolor=color))
        note(ax, columns[0], y, part, color=TEXT, ha="left")
        note(ax, columns[1] + 6, y, count, color=TEXT, ha="right")
        note(ax, columns[2] + 4, y, share, color=TEXT, ha="right")
        note(ax, columns[3], y, change, color=GREEN if color == GREEN else MUTED, ha="left")


def plot(output):
    fig = plt.figure(figsize=(18, 12.4))
    tall_heading(fig, "The spiking stack with knowledge cells",
                 "The v16 model at full size, 34.81M parameters, with every part it keeps in violet and the one new "
                 "part in green. Each head of each state read gains 128 knowledge\ncells: learned tags, payloads and "
                 "offsets that race beside the cells of earlier bytes for the same 256 spikes. Nothing else changes.")
    stack(fig.add_axes((0.012, 0.2, 0.2, 0.69)))
    level(fig.add_axes((0.225, 0.2, 0.43, 0.69)))
    race(fig.add_axes((0.67, 0.2, 0.32, 0.69)))
    parameters(fig.add_axes((0.012, 0.035, 0.976, 0.15)))
    legend = ((VIOLET, "kept from v16"), (GREEN, "new"), (ORANGE, "the race"))
    x = 0.68
    for color, name in legend:
        fig.patches.append(Rectangle((x, 0.012), 0.009, 0.014, transform=fig.transFigure, facecolor=color,
                                     edgecolor=color))
        fig.text(x + 0.013, 0.019, name, size=9.2, color=TEXT, va="center")
        x += 0.013 + 0.0058 * len(name) + 0.025
    save(fig, output, "event_lm_knowledge_cells.png")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=RESULTS / "figures")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    plot(args.output)


if __name__ == "__main__":
    main()
