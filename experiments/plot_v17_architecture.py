"""Figure of the v17 stack as it will probably look: the full-size stack with what the v17 tests established, what the
tests running now decide, and what was left out; what every part does, by the kind of byte it helps; and which parts
add up."""

from __future__ import annotations

import argparse
from pathlib import Path

from plot_dendrite import MUTED, PAPER, RESULTS, TEXT, plt, save
from plot_knowledge_architecture import GRAY_EDGE, GRAY_FILL, GREEN_FILL, SMALL, VIOLET_FILL, arrow, box, canvas, label, note
from plot_state_reads import GREEN, VIOLET, tall_heading

AMBER, AMBER_FILL = "#a86f00", "#fdf3dc"


def stack(ax):
    """The full-size stack, a byte in at the top and the evidence for the next byte at the bottom."""
    canvas(ax)
    label(ax, 3, 98.5, "The stack, 34.8M parameters", size=12)
    box(ax, 10, 89, 80, 6, "Byte in", ["one of 256 values, one after another, without end"])
    arrow(ax, (50, 89), (50, 86))
    box(ax, 10, 77.5, 80, 8.5, "Input synapses", ["256 values → 512 channels of spike counts"])
    arrow(ax, (50, 77.5), (50, 74.6))
    box(ax, 3, 24.6, 94, 50, None, dashed=True, face=PAPER)
    note(ax, 6, 72.4, "11 levels on one stream, every count a whole number", ha="left")
    # Memory: the state read.
    box(ax, 7, 50.5, 86, 19.6, "State read: memory, as a hippocampus", [], face=VIOLET_FILL, edge=VIOLET,
        title_color=VIOLET, shift=6.3)
    note(ax, 50, 63.6, "8 heads race for spikes over the tape of past cells:")
    note(ax, 50, 60.6, "2,048 cells back in training, 8,192 when reading; ages that fade by learned slopes")
    box(ax, 11, 52.2, 78, 5.0, None, face=AMBER_FILL, edge=AMBER, dashed=True)
    label(ax, 50, 54.7, "+ one sink cell a head: the answer that nothing here fits", color=AMBER, size=SMALL,
          ha="center")
    arrow(ax, (50, 50.5), (50, 47.6))
    # Knowledge: the threshold cells.
    box(ax, 7, 27.2, 86, 20.4, "Threshold cells: knowledge, as a cortex", [], face=VIOLET_FILL, edge=VIOLET,
        title_color=VIOLET, shift=6.5)
    note(ax, 50, 41.0, "1,188 cells a level, each hearing every channel now and a byte or more ago")
    box(ax, 11, 34.0, 78, 4.6, None, face=AMBER_FILL, edge=AMBER, dashed=True)
    label(ax, 50, 36.3, "myelinated fibres: every fibre learns its delay, 0 to 31 bytes", color=AMBER, size=SMALL,
          ha="center")
    box(ax, 11, 28.6, 78, 4.6, None, face=AMBER_FILL, edge=AMBER, dashed=True)
    label(ax, 50, 30.9, "or a cerebellar sheet: 4× the cells, each on a quarter of the stream", color=AMBER,
          size=SMALL, ha="center")
    arrow(ax, (50, 24.6), (50, 21.6))
    box(ax, 10, 10.8, 80, 10.8, "Output neurons", ["512 channels → 256 byte values",
                                                    "fast synapses learn from every byte read"])
    arrow(ax, (50, 10.8), (50, 7.6))
    note(ax, 50, 5.4, "spikes of evidence for the next byte")
    box(ax, 10, 0.4, 80, 3.4, None, face=GREEN_FILL, edge=GREEN)
    label(ax, 50, 2.1, "trained on streams, every state carried from byte to byte", color=GREEN, size=SMALL,
          ha="center")


ROWS = (
    # part, the bytes it helps, what was measured, status (colour).
    ("Delayed fibres (v13)", "new contexts", "with streams, closed the bed's new-context gap", "in", VIOLET),
    ("Learned forgetting, reach 8,192 (v16)", "copies from far", "0.32 bits a byte ahead of the Transformer there",
     "in", VIOLET),
    ("Fast synapses (v15)", "the bytes just read", "0.004 at full size", "in", VIOLET),
    ("Training on streams", "all", "0.029 over long streams on the test bed", "in, the rule", GREEN),
    ("One sink cell a head", "copies", "0.096 on copyable bytes, 0.005 in all (one seed)", "4 seeds running",
     AMBER),
    ("Myelinated fibres", "new contexts (aim)", "built, exact and causal; 1,500 updates in", "screens running",
     AMBER),
    ("Cerebellar sheet", "new contexts (aim)", "equal parameters, 4,064 cells a level", "screens running", AMBER),
    ("Trained plasticity", "the bytes just read", "0.002 at full size on short streams", "scoring now", AMBER),
    ("More data", "new contexts", "their gap shrinks as the data to the -0.44", "next window", AMBER),
)


def parts(ax):
    """What every part does: the kind of byte it helps, what was measured and where it stands."""
    canvas(ax)
    label(ax, 1, 97, "What every part does, by the kind of byte it helps", size=12)
    columns = (1, 33, 52, 87)
    heads = ("Part", "Helps", "Measured", "Status")
    top = 89
    for x, head in zip(columns, heads):
        label(ax, x, top, head, color=MUTED, size=SMALL)
    ax.plot((1, 99), (top - 3.2, top - 3.2), color=GRAY_EDGE, linewidth=0.8)
    step = 9.3
    for row, (part, helps, measured, status, color) in enumerate(ROWS):
        y = top - 8.5 - step * row
        label(ax, columns[0], y, part, color=TEXT, size=SMALL)
        label(ax, columns[1], y, helps, color=MUTED, size=SMALL, weight="normal")
        label(ax, columns[2], y, measured, color=MUTED, size=SMALL - 0.4, weight="normal")
        label(ax, columns[3], y, status, color=color, size=SMALL)


def adds_up(ax):
    """Which parts add up, and why the probable v17 is the combination it is."""
    canvas(ax)
    label(ax, 1, 95, "What adds up", size=12)
    lines = (
        ("Parts that help different kinds of bytes add up; parts that help the same kind compete.", TEXT, "bold"),
        ("Knowledge cells and trained plasticity both work on what was just read: together they lose 0.011.",
         MUTED, "normal"),
        ("Knowledge cells keep only 0.005 of their 0.009 beside fast synapses, which also copy.", MUTED, "normal"),
        ("So v17 takes one part for each kind of byte: a sink for copies, learned delays or the sheet for new",
         TEXT, "normal"),
        ("contexts, fast synapses for the bytes just read, and training on streams for all of them.", TEXT,
         "normal"),
        ("Left out after their tests: kept cells, keyed fast synapses, eight fixed delays, narrower reads,", MUTED,
         "normal"),
        ("branching dendrites, apical bursts, racing experts, and more knowledge cells than one a head.", MUTED,
         "normal"),
        ("Added parameters: about 11,000 for the delays and 11,000 for the sinks, 0.07% of the stack.", MUTED,
         "normal"),
    )
    for row, (text, color, weight) in enumerate(lines):
        label(ax, 1, 82 - 10.6 * row, text, color=color, size=SMALL + 0.4 if weight == "bold" else SMALL,
              weight=weight)


def plot(output):
    fig = plt.figure(figsize=(16.5, 10.6))
    tall_heading(fig, "ShuntFormer v17, as it will probably look",
                 "Memory like a hippocampus, knowledge like a cortex, spiking and streaming throughout. Violet: "
                 "kept from v16. Green: settled by the v17 tests. Amber, dashed: decided by tests running now.")
    stack(fig.add_axes((0.02, 0.03, 0.42, 0.82)))
    parts(fig.add_axes((0.47, 0.40, 0.52, 0.46)))
    adds_up(fig.add_axes((0.47, 0.04, 0.52, 0.33)))
    save(fig, output, "event_lm_v17_architecture.png")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=RESULTS / "figures")
    args = parser.parse_args()
    plot(args.output)


if __name__ == "__main__":
    main()
