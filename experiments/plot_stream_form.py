"""Figures and tables for the stream form: the test, the circuit, the training runs, the reach of the reads and the
synapses that a stream teaches."""

from __future__ import annotations

import argparse
import textwrap
from pathlib import Path

from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, Rectangle
from matplotlib.ticker import MultipleLocator

from plot_dendrite import GRID, MUTED, ORANGE, RESULTS, TEXT, plt, read_result, save, style
from plot_read_clocks import GREEN, PINK, SLATE, end_labels
from plot_state_reads import GREEN_FILL, LIGHT, VIOLET, VIOLET_FILL, arrow, box, tall_heading

STUDY = "event_lm/stream_form"
# The Transformer is orange, the stack of this release violet, its window-trained parent green and float stand-ins
# of the stack pink. The four pass the checks for colour-blind readers in pairs.
# Arms in arms.json: the stack trained on windows of 512 bytes, its fork on windows of 1,024 and the reference.
WINDOWS, LONG, TRANSFORMER = "fp16-d4-rd4-3e-4", "fp16-d4-rd4-w1024-3e-4", "tf-rot-3e-4"
# One hue from light to dark for the cells that a read reaches back.
REACH_SHADES = {512: "#e9e6fa", 768: "#d3cdf4", 1024: "#b6adec", 2048: "#8f83e2", 4096: "#6250d6", 8192: "#3b2c9c"}
# The same hue for the four strengths of the fast synapses, the weakest lightest.
STRENGTH_SHADES = ("#3b2c9c", "#6250d6", "#9c91e6", "#c5bef1")
# The steps from the Transformer to the spiking stack: arm, what the step changed, and the colour of its kind.
LADDER = (
    ("tf-rot-1e-3", "Transformer", ORANGE),
    ("tf-relu-rot-1e-3", "with plain units in place of gated ones", ORANGE),
    ("twin-rot-1e-3", "and an offset per distance in place of rotary phases", ORANGE),
    ("twin-untied-rot-1e-3", "and output weights of its own", ORANGE),
    ("real-ln-free-1e-3", "The stack in floats, with LayerNorm", PINK),
    ("ln-free-1e-3", "with whole spikes and synapses on the 8-bit grid", PINK),
    ("ln-1e-3", "and the budget of 310,000 pulses a byte", PINK),
    ("fp16-1e-3", "The spiking stack: a pool of inhibition in place of LayerNorm", VIOLET),
)
LADDER_STEP = 9000
# The groups of the test figure and their rows: label, the record of the test split and of the validation split, the
# rate of a Transformer's fast synapses where a record holds several, and the colour. The Transformer along the
# streams of the stacks comes first in each group: it is the group's reference.
TAUGHT_RATE = 0.07
TEST = (
    ("Weights as trained", (
        ("Transformer, 496 bytes of history", "test_transformer_taught_history496",
         "validation_transformer_taught_history496", 0.0, ORANGE),
        ("448 bytes of history", "test_transformer_taught", "validation_transformer_taught_fine", 0.0, ORANGE),
        ("256 bytes of history", "test_transformer_history256", None, None, ORANGE),
        ("Stack on long windows, as a stream", "test_long_windows_stream", "validation_long_windows_stream", None,
         VIOLET),
        ("with a reach of 4,096 cells after levels 5 to 7", "test_long_windows_reach",
         "validation_long_windows_reach", None, VIOLET),
        ("Stack on windows of 512 bytes, as a stream", "test_windows_stream", "validation_windows_stream", None,
         GREEN),
        ("with a reach of 2,048 cells after levels 4 to 7", "test_windows_reach", "validation_windows_reach", None,
         GREEN),
    )),
    ("With output synapses that the stream teaches", (
        ("Transformer, 496 bytes of history", "test_transformer_taught_history496",
         "validation_transformer_taught_history496", TAUGHT_RATE, ORANGE),
        ("448 bytes of history", "test_transformer_taught", "validation_transformer_taught_fine", TAUGHT_RATE,
         ORANGE),
        ("Stack on long windows, with its longer reach", "test_long_windows_taught",
         "validation_long_windows_taught", None, VIOLET),
        ("Stack on windows of 512 bytes, with its longer reach", "test_windows_taught",
         "validation_windows_taught", None, GREEN),
    )),
)
# Title, what the family asked, and its arms in arms.json.
FAMILIES = (
    ("The stack in the Transformer's layout",
     "The first stacks: a read and a layer of threshold cells at each of eleven levels, with the coarse pool of the "
     "decoders. `stack-r64` is the stack command as it stands, and the others change the address rate, the phases "
     "of a read or the learning rate, or give the cells delayed fibres.",
     ("stack-a", "stack-r64", "stack-d4", "stack-d4-r64", "stack-p16-r64", "stack-lr1e3", "stack-3e-4",
      "stack-1e-3")),
    ("Reference Transformers",
     "The Transformer of the standard shape under the rotated optimizer at two learning rates. The run at 0.0003 "
     "is the reference of this release. `transformer-std-rot` is the same configuration, begun in the study of "
     "v14 and continued here from its saved state at 7,500 updates on another machine: the two runs are 0.002 "
     "apart at 9,000 updates.",
     ("tf-rot-3e-4", "tf-rot-1e-3", "transformer-std-rot")),
    ("Transformers with parts of the stack",
     "Controls in floats: Transformers that take over the stack's plain units, its age offsets or its untied output, "
     "one at a time and together, and two with other units. None is a spiking network.",
     ("tf-twin", "tf-relu-rot-1e-3", "tf-gelu-rot-1e-3", "tf-rglu-rot-1e-3", "tf-age-rot-1e-3", "twin-rot-1e-3",
      "twin-untied-rot-1e-3")),
    ("Float stand-ins of the stack",
     "Stacks that hear a LayerNorm or an exact division of the stream, count in real numbers, or have no pulse "
     "budget. They place the distance to the Transformer and are no spike mechanisms. Their switches are those of "
     "`experiments/stack/diagnosis_build.py`.",
     ("ln-1e-3", "nl1-1e-3", "ln-2e-3", "ln-3e-4", "ln-free-1e-3", "real-ln-1e-3", "real-ln-free-1e-3",
      "real-ln-free-noceil-1e-3", "fp16-lnout-1e-3")),
    ("The fine pool",
     "The pool of inhibition that divides: thresholds in quanta sixteen times finer and a pool cell that fires once "
     "for every 64 pulses of the stream. Then turned probes, a wider learning slope for the cells, and a ceiling "
     "that the pool lifts, as a run of its own and as a fork beside its unchanged parent.",
     ("fp16-1e-3", "fp16-3e-4", "fp16-1e-3-s2", "fp16-turn-1e-3", "fp16-wide0-1e-3", "fp16-wide-1e-3",
      "fp16-lift-1e-3", "fp16-lift-fork-1e-3")),
    ("Gated cells and a veto",
     "Cells with a gate dendrite, which is the spike form of the Transformer's gated unit, in several variants, and "
     "cells with a dendrite that closes them. None learned as well as plain cells.",
     ("stack-tg", "stack-tgp", "fpg-1e-3", "fpg-free-1e-3", "fpg-3e-4", "fpg-d4-1e-3", "fpg-1e-3-s2", "fpg-u-1e-3",
      "fpg-wide-1e-3", "fp16-veto-1e-3")),
    ("Delayed fibres",
     "Cells that hear every channel again after one to three bytes, and tags that hear the bytes after their own. "
     "`fp16-d4-rd4-3e-4` is the stack that the stream form starts from. It trained on windows of 512 bytes to the "
     "end.",
     ("fp16-d4-1e-3", "fp16-d4-3e-4", "fp16-d4-rd4-1e-3", "fp16-d4-rd4p-1e-3", "fp16-d4-rd4-3e-4")),
    ("Streams and long windows",
     "Forks of `fp16-d4-rd4-3e-4`: training on streams in stretches of 20 and 60 windows, in groups of streams "
     "that take turns, the same with a longer reach, and training on windows of 1,024 bytes, which is the released "
     "stack. Their window validation reads windows of their own length, so the last column is left empty where "
     "the windows are not those of the Transformer.",
     ("fp16-d4-rd4-s20-3e-4", "fp16-d4-rd4-s60-3e-4", "fp16-d4-rd4-g15-3e-4", "fp16-d4-rd4-g15-r1024-3e-4",
      "fp16-d4-rd4-w1024-3e-4")),
)
# The flags of the fine pool and of the private build that preceded them, which the arms page names in words.
FINE_POOL = ("--pool-fine 16 --pool-rest 4 --pool-rate 32 --output-pool 128",
             "SNN_FINE=16 SNN_POOL=4,32 SNN_POOL_OUT=1")
CURVE_STEPS = (3000, 6000, 9000, 12000, 18000, 24000, 42000, 60000, 90000)
# What the fast synapses hear in the float probes: row of fast_synapses.json and its label.
HEARD = (
    ("stream", "The stream into the output"),
    ("raw", "Cells of the last level, as they fire"),
    ("pool", "less the mean of all cells at that byte"),
    ("cells", "less each cell's own running mean"),
    ("cells3", "Cells of the last three levels, less their means"),
)


def minus(value, digits=4):
    """A signed difference with a true minus sign."""
    return f"{value:+.{digits}f}".replace("-", "−")


def arms():
    """Every arm of the study by its name."""
    return {run["name"]: run for run in read_result(f"{STUDY}/arms.json")["arms"]}


def curve(run, key="averaged_val_bpb"):
    """A validation loss of a run by its updates."""
    return {point["step"]: point[key] for point in run["curve"] if key in point}


def plot_ladder(output):
    runs = arms()
    rows = [(label, curve(runs[name])[LADDER_STEP], color) for name, label, color in LADDER]
    fig, ax = plt.subplots(figsize=(11.6, 5.6))
    fig.subplots_adjust(left=0.45, right=0.93, bottom=0.2, top=0.76)
    tall_heading(fig, "From the Transformer to spikes, one change at a time",
                 f"Validation loss of the running weight average at {LADDER_STEP:,} of 90,000 updates · every "
                 "row one run at a learning rate of 0.001 · one seed")
    style(ax, grid_axis="x")
    for index, (label, loss, color) in enumerate(rows):
        if index:
            before = rows[index - 1][1]
            ax.plot([before, before], [index - 1, index], color=LIGHT, linewidth=1.4, zorder=1)
            ax.plot([before, loss], [index, index], color=LIGHT, linewidth=2.2, zorder=1)
        ax.scatter(loss, index, s=64, color=color, zorder=2, edgecolor="white", linewidth=1)
        text = f"{loss:.4f}" if not index else f"{loss:.4f}   {minus(loss - rows[index - 1][1])}"
        ax.annotate(text, (max(loss, rows[index - 1][1] if index else loss), index), xytext=(10, 0),
                    textcoords="offset points", ha="left", va="center", size=9.5, color=TEXT)
    ax.set_yticks(range(len(rows)), [label for label, *_ in rows])
    ax.tick_params(axis="y", labelcolor=TEXT, labelsize=10)
    ax.set_ylim(len(rows) - 0.4, -0.6)
    ax.set_xlim(rows[0][1] - 0.006, rows[-1][1] + 0.016)
    ax.set_xlabel("Bits per byte", color=MUTED, labelpad=8)
    for y, text, color in ((0.155, "Transformers", ORANGE), (0.125, "float stand-ins of the stack", PINK),
                           (0.095, "the spiking stack", VIOLET)):
        fig.add_artist(Line2D([0.048], [y + 0.008], marker="o", markersize=7, color=color, linestyle="",
                              transform=fig.transFigure))
        fig.text(0.06, y, text, color=TEXT, size=9)
    fig.text(0.27, 0.035, "Beside each score: the change from the row above. The last step is taken at this rate "
             "only: here the membranes of the\nspiking stack run into their bound, and at a rate of 0.0003, where "
             "they do not, the pool is ahead of LayerNorm.", color=MUTED, size=9, linespacing=1.5)
    save(fig, output, "event_lm_stream_form_ladder.png")


def reach_by_level(setting, trained):
    """The cells that the read after each of the eleven levels reaches back under a setting like 4-7@2048."""
    reach = [trained] * 11
    if setting != "as trained":
        for part in setting.split("+"):
            levels, cells = part.split("@")
            first, _, last = levels.partition("-")
            for level in range(int(first), int(last or first) + 1):
                reach[level] = int(cells)
    return reach


def reach_panel(ax_map, ax_gain, scan, title):
    base = next(row["bpb"] for row in scan["rows"] if row["reach"] == "as trained")
    rows = sorted(scan["rows"], key=lambda row: row["bpb"])
    for index, row in enumerate(rows):
        for level, cells in enumerate(reach_by_level(row["reach"], scan["trained_reach"])):
            ax_map.add_patch(Rectangle((level - 0.44, index - 0.36), 0.88, 0.72, color=REACH_SHADES[cells],
                                       linewidth=0))
        gain = row["bpb"] - base
        chosen = row["reach"] == scan.get("chosen")
        ax_gain.plot([0, gain], [index, index], color=LIGHT, linewidth=2.2, zorder=1)
        ax_gain.scatter(gain, index, s=46, color=VIOLET if gain < 0 else SLATE, zorder=2, edgecolor="white",
                        linewidth=1)
        ax_gain.annotate(("chosen  " if chosen else "") + ("as trained" if not gain else minus(gain)),
                         (min(gain, 0), index), xytext=(-9, 0), textcoords="offset points", ha="right", va="center",
                         size=9, color=TEXT, weight="bold" if chosen else "normal")
    ax_map.set_xlim(-0.6, 10.6)
    ax_map.set_xticks(range(11))
    ax_map.set_xlabel("Level of the stack", color=MUTED, labelpad=6)
    ax_map.set_title(title, fontsize=10, loc="left", color=TEXT, pad=10)
    low = min(row["bpb"] for row in rows) - base
    high = max(row["bpb"] for row in rows) - base
    ax_gain.set_xlim(low - 0.019, max(high, 0) + 0.002)
    ax_gain.xaxis.set_major_locator(MultipleLocator(0.01))
    ax_gain.set_xticks([tick for tick in ax_gain.get_xticks() if low - 0.004 <= tick <= max(high, 0) + 0.001])
    ax_gain.axvline(0, color=TEXT, linewidth=0.8)
    ax_gain.set_xlabel("Bits per byte against the reach of training", color=MUTED, labelpad=6)
    for ax in (ax_map, ax_gain):
        ax.set_ylim(len(rows) - 0.4, -0.6)
        ax.set_yticks([])
        for name in ("top", "right", "left", "bottom"):
            ax.spines[name].set_visible(False)
        ax.tick_params(length=0, labelcolor=MUTED, pad=5, labelsize=9)
    ax_gain.grid(axis="x", color=GRID, linewidth=0.75)
    ax_gain.set_axisbelow(True)


def plot_reach(output):
    scans = {(scan["arm"], scan["updates"]): scan for scan in read_result(f"{STUDY}/reach_by_level.json")["scans"]}
    fig = plt.figure(figsize=(12.6, 6.6))
    tall_heading(fig, "A longer reach pays at the middle levels of the stack",
                 "First 1,000,000 enwik8 validation bytes as 16 unbroken streams · weights saved during training "
                 "· a longer reach adds no synapse")
    panels = ((0.04, scans[("windows", 27000)], "Trained on windows of 512 bytes, at 27,000 updates"),
              (0.53, scans[("long-windows", 31500)], "Trained on windows of 1,024 bytes, at 31,500 updates"))
    for left, scan, title in panels:
        ax_map = fig.add_axes((left, 0.25, 0.19, 0.5))
        ax_gain = fig.add_axes((left + 0.2, 0.25, 0.235, 0.5))
        reach_panel(ax_map, ax_gain, scan, title)
    x = 0.04
    fig.text(x, 0.118, "Cells that the read after a level reaches back", color=MUTED, size=9)
    for cells, shade in REACH_SHADES.items():
        fig.patches.append(Rectangle((x, 0.068), 0.017, 0.03, transform=fig.transFigure, color=shade, linewidth=0))
        fig.text(x + 0.022, 0.083, f"{cells:,}", color=TEXT, size=9, va="center")
        x += 0.062
    fig.text(0.44, 0.052, "Each row is one scoring of the same weights, the best on top. Cells beyond the window "
             "of training take\nthe age offset of its oldest cell. The rows marked chosen set the reach of the "
             "final scorings, before\nany final weights were scored.", color=MUTED, size=9, linespacing=1.5)
    save(fig, output, "event_lm_stream_form_reach.png")


def plot_teaching(output):
    record = read_result(f"{STUDY}/fast_synapses.json")
    course = record["course_along_the_stream"]
    fig, (left, right) = plt.subplots(1, 2, figsize=(12.6, 5.8), width_ratios=(1.05, 1))
    fig.subplots_adjust(left=0.08, right=0.975, bottom=0.27, top=0.73, wspace=0.66)
    tall_heading(fig, "Synapses that the arriving bytes teach gain most early in a stream",
                 "enwik8 validation bytes as unbroken streams · the fast output synapses start at zero in "
                 "every stream · weights saved during training")
    style(left)
    ends = [0] + course["scored_bytes_of_a_stream_at_each_part_end"]
    middles = [(before + after) / 2000 for before, after in zip(ends, ends[1:])]
    labels = []
    for row, shade in zip(course["rows"], STRENGTH_SHADES):
        gains = row["over_plain_by_part"]
        left.plot(middles, gains, color=shade, linewidth=2, zorder=3, marker="o", markersize=4.5,
                  markeredgecolor="white", markeredgewidth=0.8)
        power = row["threshold"].bit_length() - 1
        labels.append((middles[-1], gains[-1], f"2$^{{{power}}}$  {minus(row['over_plain'])}", shade))
    left.axhline(0, color=TEXT, linewidth=0.8)
    left.set_xlim(0, 160)
    left.set_xticks(range(0, 121, 20))
    left.set_ylim(-0.0145, 0.001)
    end_labels(left, labels)
    left.set_xlabel("Bytes of the stream heard so far, in thousands", color=MUTED, labelpad=8)
    left.set_ylabel("Bits per byte against the frozen stack", color=MUTED, labelpad=8)
    left.set_title("By the place in a stream of 125,000 bytes, at four thresholds", fontsize=10, loc="left",
                   color=TEXT)
    style(right, grid_axis="x")
    probes = record["float_probes"]["windows_at_21000_updates_last_weights"]["rows"]
    rows = []
    for heard, label in HEARD:
        rows.append((label, min(row["over_plain"] for row in probes if row["heard"] == heard)))
    adam = min(row["over_plain"] for row in record["whole_stack_learning"]["rows"])
    rows.append(("All synapses of the stack learn by Adam", adam))
    for index, (label, gain) in enumerate(rows):
        color = SLATE if index == len(rows) - 1 else VIOLET
        right.plot([0, gain], [index, index], color=LIGHT, linewidth=2.2, zorder=1)
        right.scatter(gain, index, s=52, color=color, zorder=2, edgecolor="white", linewidth=1)
        right.annotate(minus(gain), (gain, index), xytext=(-9, 0), textcoords="offset points", ha="right",
                       va="center", size=9.5, color=TEXT)
    right.set_yticks(range(len(rows)), [label for label, _ in rows])
    right.tick_params(axis="y", labelcolor=TEXT, labelsize=9.5)
    right.set_ylim(len(rows) - 0.4, -0.6)
    right.set_xlim(-0.0135, 0.0005)
    right.axvline(0, color=TEXT, linewidth=0.8)
    right.set_xlabel("Bits per byte against the frozen stack", color=MUTED, labelpad=8)
    right.set_title("By what the fast synapses hear, the rule in floats", fontsize=10, loc="left", color=TEXT)
    fig.text(0.035, 0.03, "Left: the long-window stack at 31,500 updates with the reach of the release and fast "
             "synapses from the cells of all eleven levels, in whole numbers. Beside each line: the units of\n"
             "charge to one fast spike, so a lower number is a stronger synapse, and the gain over the whole "
             "stream. Right: the window-trained stack at 21,000 updates on 300,000 bytes, each row at\nits best "
             "rate; the last row is a control on 400,000 bytes.", color=MUTED, size=9, linespacing=1.5)
    save(fig, output, "event_lm_stream_form_teaching.png")


def shade_between(dark, light, share):
    """The colour a share of the way from a dark to a light one."""
    mixed = [round(int(dark[at:at + 2], 16) * (1 - share) + int(light[at:at + 2], 16) * share) for at in (1, 3, 5)]
    return "#" + "".join(f"{part:02x}" for part in mixed)


def plot_circuit(output):
    fig = plt.figure(figsize=(14.2, 8.4))
    tall_heading(fig, "The stack as it reads a stream",
                 "One stream of 512 channels of signed spike counts \u00b7 eleven levels, each a state read and then "
                 "a layer of threshold cells \u00b7 34,810,812 parameters")
    ax = fig.add_axes([0.02, 0.03, 0.96, 0.83])
    ax.set_xlim(0, 144)
    ax.set_ylim(0, 66)
    ax.axis("off")
    stream = 50
    ax.plot([9, 131], [stream, stream], color=TEXT, linewidth=3.4, solid_capstyle="round", zorder=1)
    box(ax, 0.5, stream - 3.2, 8.5, 6.4, "byte", size=10, weight="bold", face="white", edge=TEXT)
    ax.text(4.75, stream - 4.6, "its synapses\nstart the stream", ha="center", va="top", size=8, color=MUTED,
            linespacing=1.3)
    centers = [17.2 + 10.4 * level for level in range(11)]
    for level, center in enumerate(centers):
        # A level is a read below the stream and then the cells above it.
        ax.add_patch(FancyBboxPatch((center - 4.9, stream - 12.6), 9.8, 25.2, boxstyle="round,pad=0,rounding_size=0.8",
                                    facecolor="#f5f7fa", edgecolor="#e3e8ef", linewidth=1, zorder=0))
        ax.text(center, stream + 14.1, f"level {level}" if level in (0, 10) else str(level), ha="center",
                va="bottom", size=8.5, color=MUTED)
        box(ax, center - 4.3, stream - 11.4, 4.6, 5.6, "read", size=8, face=VIOLET_FILL, edge=VIOLET, rounding=0.5)
        arrow(ax, (center - 2.9, stream - 0.4), (center - 2.9, stream - 5.7), color=VIOLET, width=1, head=6)
        arrow(ax, (center - 1.1, stream - 5.8), (center - 1.1, stream - 0.5), color=VIOLET, width=1, head=6)
        box(ax, center - 0.3, stream + 5.8, 4.6, 5.6, "cells", size=8, face=GREEN_FILL, edge=GREEN, rounding=0.5)
        arrow(ax, (center + 1.1, stream + 0.4), (center + 1.1, stream + 5.7), color=GREEN, width=1, head=6)
        arrow(ax, (center + 2.9, stream + 5.8), (center + 2.9, stream + 0.5), color=GREEN, width=1, head=6)
        # The fast synapses of the output hear the cells of every level.
        ax.plot([center + 2, center + 2], [stream + 11.5, stream + 13.2], color=TEXT, linewidth=0.9,
                linestyle=(0, (2, 1.6)), zorder=2)
    ax.plot([centers[0] + 2, 137.4], [stream + 13.2, stream + 13.2], color=TEXT, linewidth=0.9,
            linestyle=(0, (2, 1.6)), zorder=2)
    arrow(ax, (137.4, stream + 13.2), (137.4, stream + 6.7), dashed=True, width=0.9, head=7)
    box(ax, 132.4, stream - 6.5, 11.1, 13, "256\noutput\nneurons", size=9, weight="bold", face="white", edge=TEXT)
    arrow(ax, (129.4, stream), (132.3, stream), width=2)
    ax.text(143.5, stream + 14.4, "fast synapses that\nthe stream teaches", ha="right", va="bottom", size=8.5,
            color=MUTED, linespacing=1.3)

    # A read in a stream: the cells that earlier bytes left, the window of training and the reach beyond it.
    ax.text(0.5, 31.6, "A read in a stream", size=11, weight="bold", color=TEXT, va="center")
    ax.text(0.5, 28.9, "8 heads, each a race for 256 spikes among\nthe cells that earlier bytes left", size=9,
            color=MUTED, va="top", linespacing=1.35)
    tape, first, width = 14.5, 1.2, 1.5
    beyond, window = 14, 14
    oldest = "#e6e2f9"
    for index in range(beyond + window):
        inside = index >= beyond
        # The shade is the age offset: one for every cell of the window, and its oldest for all cells beyond.
        shade = shade_between(VIOLET, oldest, (beyond + window - 1 - index) / (window - 1)) if inside else oldest
        ax.add_patch(Rectangle((first + index * width, tape - 1.5), width - 0.3, 3, facecolor=shade,
                               edgecolor=VIOLET if inside else "#b6adec", linewidth=0.6, zorder=3))
    now = first + (beyond + window) * width + 1.8
    ax.add_patch(plt.Circle((now, tape), 1.5, facecolor="white", edgecolor=VIOLET, linewidth=1.6, zorder=3))
    ax.text(now + 2.3, tape, "the byte\nnow", ha="left", va="center", size=8.5, color=TEXT, linespacing=1.3)
    for index, spikes, label, bend in ((22, 171, "171", 0.5), (16, 62, "62", 0.42), (5, 23, "23 spikes", 0.3)):
        x = first + index * width + (width - 0.3) / 2
        arrow(ax, (now - 0.2, tape + 1.6), (x, tape + 1.75), color=VIOLET, width=0.7 + 0.012 * spikes, bend=bend,
              head=7)
        ax.text(x - 0.9, tape + 2.2, label, ha="right", va="bottom", size=8, color=TEXT)
    split = first + beyond * width - 0.15
    ax.plot([split, split], [tape - 2.3, tape + 2.3], color=TEXT, linewidth=0.9, zorder=4)
    for start, end, label in ((split + 0.5, first + (beyond + window) * width - 0.4,
                               "the window of training:\n512 cells, a learned\nage offset for each"),
                              (first, split - 0.5, "beyond it, as far as the\nreach is set: the offset of\nthe "
                               "window's oldest cell")):
        ax.plot([start, start, end, end], [tape - 2.5, tape - 3.3, tape - 3.3, tape - 2.5], color=MUTED,
                linewidth=0.8)
        ax.text((start + end) / 2, tape - 4.1, label, ha="center", va="top", size=8.5, color=MUTED,
                linespacing=1.35)
    ax.text(0.5, 1.7, "A cell is what the stream held at one byte: a payload, and a\ntag that later bytes "
            "match their probes against. The tag\nalso hears the stream 1 to 3 bytes after its own.", ha="left",
            va="center", size=8.5, color=MUTED, linespacing=1.4)

    # A layer of cells: fibres of four delays, threshold cells and the pool cell that divides.
    left = 55
    ax.text(left, 31.6, "A layer of cells and its pool", size=11, weight="bold", color=TEXT, va="center")
    ax.text(left, 28.9, "1,188 threshold cells that hear every\nchannel now and 1 to 3 bytes ago", size=9,
            color=MUTED, va="top", linespacing=1.35)
    box(ax, left, 6.6, 6.8, 16.4, "512\nstream\nchannels", size=7.5, face="white", edge=TEXT)
    delays = (("now", 21), ("1 byte ago", 17), ("2 bytes ago", 13), ("3 bytes ago", 9))
    for label, y in delays:
        ax.plot([left + 6.9, left + 18.5], [y, y], color=GREEN, linewidth=1.4, zorder=2)
        ax.text(left + 18.5, y + 0.5, label, ha="right", va="bottom", size=7.5, color=MUTED)
    cells = [(left + 27, 21.6 - 3.3 * index) for index in range(5)]
    for index, (x, y) in enumerate(cells):
        fires = index in (1, 3)
        for _, source in delays:
            ax.plot([left + 18.5, x - 1.3], [source, y], color=GREEN if fires else "#b9dccf",
                    linewidth=0.7 if fires else 0.5, zorder=1)
        ax.add_patch(plt.Circle((x, y), 1.3, facecolor=GREEN if fires else "white", edgecolor=GREEN, linewidth=1.2,
                                zorder=3))
    pool = (left + 27, 3.4)
    ax.add_patch(plt.Circle(pool, 1.6, facecolor="white", edgecolor=TEXT, linewidth=1.2, zorder=3))
    ax.text(pool[0], pool[1], "P", ha="center", va="center", size=8, color=TEXT, weight="bold", zorder=4)
    ax.plot([left + 3.4, left + 3.4, pool[0] - 1.7], [6.5, pool[1], pool[1]], color=TEXT, linewidth=0.9, zorder=2)
    ax.plot([pool[0] + 1.7, left + 31.1, left + 31.1], [pool[1], pool[1], 21.6], color=TEXT, linewidth=0.9, zorder=2)
    for x, y in cells:
        ax.plot([left + 31.1, x + 1.9], [y, y], color=TEXT, linewidth=0.9, zorder=2)
        ax.add_patch(plt.Circle((x + 1.75, y), 0.42, facecolor=TEXT, edgecolor=TEXT, zorder=4))
    ax.text(left + 32.6, 12.5, "the pool cell P fires\nonce for every 64\npulses of the stream,\nand each of its "
            "spikes\nraises every threshold\nby a sixteenth of\nits rest", ha="left", va="center", size=8,
            color=MUTED, linespacing=1.35)

    # Fast output synapses: silent at the start of a stream, taught by the byte that arrives.
    left = 104
    ax.text(left, 31.6, "Fast output synapses", size=11, weight="bold", color=TEXT, va="center")
    ax.text(left, 28.9, "from the cells of all levels, empty\nwhen a stream starts", size=9, color=MUTED, va="top",
            linespacing=1.35)
    sources = [(left + 2, 20.8 - 4.1 * index) for index in range(4)]
    targets = [(left + 17.5, 21.6 - 3.2 * index) for index in range(5)]
    taught = 2
    for index, (x, y) in enumerate(sources):
        active = index in (0, 2)
        ax.add_patch(plt.Circle((x, y), 1.3, facecolor=GREEN if active else "white", edgecolor=GREEN, linewidth=1.2,
                                zorder=3))
        for number, (tx, ty) in enumerate(targets):
            grows = active and number == taught
            ax.plot([x + 1.4, tx - 1.4], [y, ty], color=TEXT if grows else "#aeb8c6", linewidth=1.5 if grows else 0.6,
                    linestyle="-" if grows else (0, (2, 1.6)), zorder=1)
    for number, (tx, ty) in enumerate(targets):
        ax.add_patch(plt.Circle((tx, ty), 1.3, facecolor=TEXT if number == taught else "white", edgecolor=TEXT,
                                linewidth=1.2, zorder=3))
    ax.text(left + 2, 4.9, "cells", ha="center", va="top", size=8, color=MUTED)
    ax.text(left + 17.5, 4.9, "output\nneurons", ha="center", va="top", size=8, color=MUTED, linespacing=1.3)
    box(ax, left + 23.6, targets[taught][1] - 2.3, 6.4, 4.6, "byte", size=8.5, weight="bold", face="white",
        edge=TEXT, rounding=0.5)
    arrow(ax, (left + 23.5, targets[taught][1]), (left + 19.1, targets[taught][1]), width=1.2, head=7)
    ax.text(left + 21.4, 22.6, "the byte that arrives\ngives its neuron\n256 teaching spikes", ha="left",
            va="center", size=8, color=MUTED, linespacing=1.35)
    ax.text(left + 21.4, 7.4, "the synapses from\nthe cells that fired\ngrow by the spikes\nthe neuron missed",
            ha="left", va="center", size=8, color=MUTED, linespacing=1.35)
    save(fig, output, "event_lm_stream_form_circuit.png")


def plot_training(output):
    runs = arms()
    reference = curve(runs[TRANSFORMER])
    older = {run["name"]: run for run in read_result("event_lm/read_clocks/arms.json")["arms"]}
    # The two stacks without delayed fibres share a colour: the coarse pool is dashed.
    series = (("Decoder of v14", curve(older["tag8-a8"]), SLATE, "-"),
              ("Stack, coarse pool", curve(runs["stack-3e-4"]), PINK, (0, (4, 2.5))),
              ("Stack, fine pool", curve(runs["fp16-3e-4"]), PINK, "-"),
              ("Stack with delayed fibres, on windows of 512 bytes", curve(runs[WINDOWS]), GREEN, "-"))
    fig, (left, right) = plt.subplots(1, 2, figsize=(12.6, 5.6))
    # The right margin leaves room for the end labels of the second panel.
    fig.subplots_adjust(left=0.065, right=0.92, bottom=0.25, top=0.74, wspace=0.3)
    tall_heading(fig, "In windows the stack follows the Transformer at a steady distance",
                 "Validation loss of the running weight average \u00b7 16 batches of 8 windows of 512 enwik8 "
                 "validation bytes from an empty state \u00b7 one seed")
    for ax in (left, right):
        style(ax)
        ax.set_xlabel("Updates of 4,096 targets, in thousands", color=MUTED, labelpad=8)
        ax.set_xlim(0, 99)
        ax.set_xticks(range(0, 91, 15))
    thousands = [step / 1000 for step in sorted(reference)]
    left.plot(thousands, [reference[step] for step in sorted(reference)], color=ORANGE, linewidth=2, zorder=3,
              label="Transformer")
    ends = []
    for label, losses, color, dashes in series:
        points = sorted((step, loss) for step, loss in losses.items() if step in reference)
        steps = [step / 1000 for step, _ in points]
        left.plot(steps, [loss for _, loss in points], color=color, linewidth=2, linestyle=dashes, zorder=3,
                  label=label)
        gaps = [loss - reference[step] for step, loss in points]
        right.plot(steps, gaps, color=color, linewidth=2, linestyle=dashes, zorder=3)
        right.scatter(steps[-1], gaps[-1], s=22, color=color, zorder=4)
        ends.append((steps[-1], gaps[-1], f"{minus(gaps[-1], 3)} at {steps[-1]:g}k", color))
    right.set_ylim(-0.13, 0.11)
    # The pink and the green line end close together: a wider gap keeps their values off the lines.
    end_labels(right, ends, gap=24)
    left.set_ylabel("Bits per byte", color=MUTED, labelpad=8)
    left.set_ylim(1.2, 1.85)
    left.set_title("Validation loss", fontsize=10, loc="left", color=TEXT)
    right.axhline(0, color=TEXT, linewidth=0.8)
    right.set_ylabel("Bits per byte above the Transformer", color=MUTED, labelpad=8)
    right.set_title("Loss minus the Transformer's at the same update", fontsize=10, loc="left", color=TEXT)
    handles, names = left.get_legend_handles_labels()
    left.legend(handles, names, frameon=False, labelcolor=TEXT, fontsize=9, loc="upper center",
                bbox_to_anchor=(1.12, -0.2), ncol=5, handlelength=2.2, columnspacing=1.6)
    fig.text(0.035, 0.03, "All runs share the data order, the schedule, the rotated optimizer and a learning rate "
             "of 0.0003, and have 34.75M to 34.82M parameters. The stacks with the coarse and\nthe fine pool "
             "were stopped where their lines end. The released stack leaves the green run at 24,000 updates for "
             "windows of 1,024 bytes, which this validation does not read.", color=MUTED, size=9, linespacing=1.5)
    save(fig, output, "event_lm_stream_form_training.png")


def scored(name, rate=None):
    """Bits per byte and parameters of a score record: of its one checkpoint, or of a Transformer at one rate of
    its fast output synapses."""
    record = read_result(f"{STUDY}/scores/{name}.json")
    if rate is None:
        rows = record["checkpoints"]
        if len(rows) > 1:
            # A record may hold other checkpoints beside the averaged weights that this study released.
            rows = [row for row in rows if row["checkpoint"].startswith("checkpoints/stream_form/")
                    and row["checkpoint"].endswith(".averaged")]
        (row,) = rows
        return row["bpb"], row["parameters"]
    (row,) = [row for row in record["fast_output_synapses"] if row["rate"] == rate]
    return row["bpb"], record["parameters"]


def test_groups(split=0):
    """The groups of the test figure for the test split (0) or the validation split (1) as (title, rows of label,
    bits per byte and colour). A score that was not measured is None."""
    def score(names, rate):
        name = names[split]
        if name is None or not (RESULTS / f"{STUDY}/scores/{name}.json").is_file():
            return None
        return scored(name, rate)[0]

    return [(title, [(label, score(names, rate), color) for label, *names, rate, color in rows])
            for title, rows in TEST]


def plot_test(output):
    # One panel for each held-out split of enwik8. Every group is measured against the Transformer at its best
    # window, which is its first row.
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 6.9), sharey=True)
    fig.subplots_adjust(left=0.322, right=0.975, bottom=0.2, top=0.74, wspace=0.07)
    tall_heading(fig, "As a stream the spiking stack is level with the Transformer",
                 "enwik8, 4,999,999 targets a split \u00b7 the stacks as 16 unbroken streams, the Transformer in "
                 "warmed windows along the same streams \u00b7 368.64M training targets each")
    for split, (ax, name) in enumerate(zip(axes, ("Test split", "Validation split"))):
        style(ax, grid_axis="x")
        ax.set_title(name, loc="left", size=11, color=TEXT, weight="bold", pad=10)
        place, ticks, scores = 0, [], []
        for title, rows in test_groups(split):
            reference = rows[0][1]
            if split == 0:
                ax.text(-0.455, place - 0.05, title, transform=ax.get_yaxis_transform(), ha="left", va="center",
                        size=9.5, color=TEXT, weight="bold")
            ax.plot([reference, reference], [place + 0.55, place + len(rows) + 0.45], color=ORANGE, linewidth=1,
                    linestyle=(0, (3, 3)), zorder=1)
            for label, score, color in rows:
                place += 1
                ticks.append((place, label))
                if score is None:
                    ax.annotate("not measured", (reference, place), xytext=(10, 0), textcoords="offset points",
                                ha="left", va="center", size=9, color=MUTED)
                    continue
                scores.append(score)
                ax.plot([reference, score], [place, place], color=LIGHT, linewidth=2.2, zorder=1)
                ax.scatter(score, place, s=64, color=color, zorder=2, edgecolor="white", linewidth=1)
                ax.annotate(f"{score:.4f}", (max(score, reference), place), xytext=(10, 0),
                            textcoords="offset points", ha="left", va="center", size=9.5, color=TEXT, weight="bold")
                if score != reference:
                    ax.annotate(minus(score - reference), (max(score, reference), place), xytext=(52, 0),
                                textcoords="offset points", ha="left", va="center", size=9, color=MUTED)
            place += 1.6
        ax.set_yticks([at for at, _ in ticks], [label for _, label in ticks])
        ax.tick_params(axis="y", labelcolor=TEXT, labelsize=9.5)
        ax.set_ylim(place - 1.1, -0.7)
        ax.set_xlim(min(scores) - 0.006, max(scores) + 0.034)
        ax.set_xlabel("Bits per byte", color=MUTED, labelpad=8)
    fig.text(0.035, 0.03, "Beside each score: the difference to the Transformer at its best window, the first row of "
             "its group (the dashed line). A Transformer carries no state, so it reads a stream in windows that "
             "hold the\nbytes of history named here before the scored ones. The stack on long windows trained on "
             "windows of 1,024 bytes from 24,000 updates; the Transformer and the other stack trained on windows "
             "of\n512 bytes. 34.75M and 34.81M parameters. One seed per row.", color=MUTED, size=9,
             linespacing=1.5)
    save(fig, output, "event_lm_stream_form.png")


def prose(text):
    """A paragraph wrapped like the other pages of the repository."""
    return textwrap.fill(text, width=65, break_long_words=False, break_on_hyphens=False)


def command(flags):
    """The lines of a command's flags, broken between flags only."""
    return textwrap.wrap(flags, 76, break_long_words=False, break_on_hyphens=False)


def changes(run):
    """What an arm adds to the command of its kind, with the fine pool named in words."""
    flags, build = run["flags"], run.get("diagnosis_build", "").split()
    parts = []
    if FINE_POOL[0] in flags or set(FINE_POOL[1].split()) <= set(build):
        parts.append("the fine pool")
        flags = flags.replace(FINE_POOL[0], "")
        build = [switch for switch in build if switch not in FINE_POOL[1].split()]
    if flags.split():
        parts.append(f"`{' '.join(flags.split())}`")
    if build:
        parts.append(f"build `{' '.join(build)}`")
    if "continues" in run:
        parts.append(f"continues `{run['continues']['arm']}` from {run['continues']['from_update']:,}")
    return ", ".join(parts) or "the command as it stands"


def own_windows(run):
    """Whether an arm's window validation reads windows of another length than the Transformer's."""
    parts = run["flags"].split()
    return "--ctx" in parts and parts[parts.index("--ctx") + 1] != "512"


def arms_page():
    """docs/STREAM_FORM_ARMS.md: every arm of the study by family, with what it changed and where it ended."""
    record = read_result(f"{STUDY}/arms.json")
    runs = arms()
    listed = [name for _, _, names in FAMILIES for name in names]
    missing = set(runs) ^ set(listed)
    if missing or len(listed) != len(set(listed)):
        raise ValueError(f"arms and families do not match: {sorted(missing)}")
    reference = curve(runs[TRANSFORMER])
    unscored = sum(1 for run in record["arms"] if not run["curve"])
    commands = record["commands"]
    lines = ["# Every training arm of the stream-form study", "",
             prose(f"The study of 19 to 20 November 2026 trained {len(record['arms'])} arms at full size. Each row "
                   "is one arm with one seed. Validation is 16 batches of 8 windows of 512 enwik8 validation bytes "
                   "from an empty state and scores the running weight average. A stack arm is the stack command "
                   "below with what its row names added, where a flag given again replaces the value of the "
                   "command. A Transformer arm is the Transformer command with its additions. The last column is "
                   f"an arm's loss minus that of the reference Transformer `{TRANSFORMER}` at the same update."),
             "",
             prose("Most arms were pilots that were stopped when they had answered their question, so their last "
                   "update says where they were stopped and not where they would have ended. An arm that continues "
                   "another took over its saved state at the update named, and its curve starts there. A fork "
                   "and its parent share everything up to that update, which makes their difference more reliable "
                   "than a difference between runs. "
                   + (f"{unscored} arms were stopped before their first evaluation and have no loss."
                      if unscored != 1 else "One arm was stopped before its first evaluation and has no loss.")),
             "",
             prose("The flags do not record the code. Flags were added while the study ran, and every one of them "
                   "is in the code of this release. Arms that name a build ran on a private build whose switches "
                   "`experiments/stack/diagnosis_build.py` reproduces. The fine pool is `--pool-fine 16 "
                   "--pool-rest 4 --pool-rate 32 --output-pool 128`, and the arms that started before the pool "
                   "was in the package ran it as a build."),
             "",
             "The stack command:", "", "```text", *command(commands["stack"]), "```", "",
             "The Transformer command:", "", "```text", *command(commands["transformer"]), "```", "",
             "Both end with:", "", "```text", *command(commands["shared"]), "```", "",
             "[Study page](STREAM_FORM.md) \u00b7",
             "[All flags and curves](../results/event_lm/stream_form/arms.json)", ""]
    for title, question, names in FAMILIES:
        lines += [f"## {title}", "", prose(question), "",
                  "| Arm | Started, UTC | On | Parameters | Adds | Updates | Averaged | Above the Transformer |",
                  "|---|---|---|---:|---|---:|---:|---:|"]
        for name in names:
            run = runs[name]
            mine = curve(run)
            started = f"{run['started_utc'][8:10]} {run['started_utc'][11:16]}"
            start = f"| `{name}` | {started} | {run['machine']} | {run['parameters']:,} | {changes(run)} |"
            if not mine:
                lines.append(f"{start} | | |")
                continue
            last = max(mine)
            # A run is compared at its last update that the Transformer was evaluated at too.
            shared = max((step for step in mine if step in reference), default=None)
            against = ""
            if shared is not None and name != TRANSFORMER and not own_windows(run):
                against = minus(mine[shared] - reference[shared]) + ("" if shared == last else f" at {shared:,}")
            lines.append(f"{start} {last:,} | {mine[last]:.4f} | {against} |")
        lines.append("")
    return "\n".join(lines)


def tables():
    """The tables of docs/STREAM_FORM.md that are computed from the records."""
    runs = arms()
    older = {run["name"]: run for run in read_result("event_lm/read_clocks/arms.json")["arms"]}
    columns = [("Transformer", curve(runs[TRANSFORMER])), ("Decoder of v14", curve(older["tag8-a8"])),
               ("Stack, coarse pool", curve(runs["stack-3e-4"])), ("Stack, fine pool", curve(runs["fp16-3e-4"])),
               ("Stack with delayed fibres", curve(runs[WINDOWS]))]
    print("averaged validation in windows\n| Updates | " + " | ".join(label for label, _ in columns) + " |\n|---:|"
          + "---:|" * len(columns))
    for step in CURVE_STEPS:
        print(f"| {step:,} | " + " | ".join(f"{losses[step]:.4f}" if step in losses else "" for _, losses in columns)
              + " |")
    print(f"\nthe ladder at {LADDER_STEP:,} updates\n| Run | Averaged validation | Step |\n|---|---:|---:|")
    before = None
    for name, label, _ in LADDER:
        loss = curve(runs[name])[LADDER_STEP]
        print(f"| {label} | {loss:.4f} | {'' if before is None else minus(loss - before)} |")
        before = loss
    print("\naveraged validation on streams\n| Updates | " + " | ".join(
        f"`{name}`" for name in (WINDOWS, *FAMILIES[-1][2])) + " |\n|---:|" + "---:|" * (1 + len(FAMILIES[-1][2])))
    streams = {name: curve(runs[name], "averaged_stream_val_bpb") for name in (WINDOWS, *FAMILIES[-1][2])}
    for step in sorted(set().union(*streams.values())):
        print(f"| {step:,} | " + " | ".join(f"{losses[step]:.4f}" if step in losses else ""
                                            for losses in streams.values()) + " |")
    for split, name in enumerate(("test", "validation")):
        print(f"\nenwik8 {name}\n| Row | Bits per byte | Against the group's Transformer |\n|---|---:|---:|")
        for title, rows in test_groups(split):
            reference = rows[0][1]
            print(f"| **{title}** | | |")
            for label, score, _ in rows:
                if score is not None:
                    print(f"| {label} | {score:.4f} | {'' if score == reference else minus(score - reference)} |")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tables", action="store_true", help="print the tables of docs/STREAM_FORM.md instead")
    parser.add_argument("--arms", action="store_true", help="print docs/STREAM_FORM_ARMS.md instead")
    parser.add_argument("--only", nargs="+", help="names of the figures to draw, all by default")
    parser.add_argument("--output", type=Path, default=RESULTS / "figures", help="directory for the figures")
    args = parser.parse_args()
    if args.tables or args.arms:
        tables() if args.tables else print(arms_page())
        return
    figures = {"test": plot_test, "training": plot_training, "circuit": plot_circuit, "ladder": plot_ladder,
               "reach": plot_reach, "teaching": plot_teaching}
    for name in args.only or figures:
        figures[name](args.output)


if __name__ == "__main__":
    main()
