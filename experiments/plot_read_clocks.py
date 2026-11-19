"""Figures for the read clocks: where heads look, how the reads' addresses moved, and the clocks themselves."""

from __future__ import annotations

import argparse
import textwrap
from pathlib import Path

from matplotlib.colors import LinearSegmentedColormap
from matplotlib.transforms import blended_transform_factory

from plot_dendrite import GRAY, GRID, MUTED, ORANGE, PAPER, RESULTS, TEXT, np, plt, read_result, save, style
from plot_state_reads import LIGHT, VIOLET, end_label, tall_heading

from common import cosine_lr

STUDY = "event_lm/read_clocks"
# The v13 decoder is slate. Its successors and the Transformers pass the checks for colour-blind readers in pairs.
GREEN, PINK, SLATE = "#0f8f6c", "#e0559b", "#4a5568"
REACH = ("self", "1", "2-3", "4-7", "8-15", "16-63", "64+")
# Arm and label of the rows of the made-up language, the reads alone first.
LANGUAGE_ARMS = (
    ("read, ages in bytes", "Ages in bytes"),
    ("read, word clock", "Clock at word starts"),
    ("read, word clock, own ages", "Clock at word starts, heads start apart"),
    ("read, clock at word ends", "Clock at word ends"),
    ("read, clock at word ends, own ages", "Clock at word ends, heads start apart"),
    ("read, clock cells", "Learned clock cells"),
    ("read, clock cells, own ages", "Learned clock cells, heads start apart"),
    ("read, ages in bytes, then cells", "Ages in bytes"),
    ("read, word clock, then cells", "Clock at word starts"),
    ("read, clock at word ends, then cells", "Clock at word ends"),
    ("read, clock cells, then cells", "Learned clock cells"),
)
LINE = "the old man saw the ca"
# Offsets by age in beats of the second head of the read after block 2, in the decoder with the clock at word
# ends at 18,000 updates (read_addresses_clock_at_word_ends.txt), for the sketch.
SKETCH_HEAD = (0.69, 1.88, 1.14, 0.63, 0.16, -0.29)
AGE_SHADES = ("#8f83e2", "#a69ce9", "#bcb4ef", "#d0caf4", "#e2def8", "#f0eefb")
# Label and colour of the decoders that the paired figures show, by their checkpoint in the records.
SHOWN = {"pilot-race64-d8-c350-dl4/model.pt.averaged": ("Spike decoder of v13", SLATE),
         "tag8-a8/model.pt.averaged": ("With clock cells, v14", VIOLET)}
# Where a tapped stream lies in its network's depth, from 0 at the input to 1 at the top.
DEPTH = {"block 2, before the first read": 3 / 8 - 0.035, "read after block 2": 3 / 8 + 0.035,
         "read after block 4": 5 / 8, "read after block 6": 7 / 8, "where the output reads": 1.0}
# The rows of the gap figure: heading of a group, then label and row name in the record of that group.
# Name in arms.json, label and colour of the three decoders of full size, and the flags they share.
RUNS = (("tag8-a8", "Clock cells", VIOLET), ("rate8", "Address rate", PINK), ("end8-a8", "Clock at word ends", GREEN))
BASE = "end8-a8"
OLD, TRANSFORMER, STANDARD = "v13 decoder", "Transformer, rotated optimizer", "transformer-std"
CURVE_STEPS = (3000, 6000, 9000, 12000, 18000, 24000, 42000, 60000, 90000)
# Label, study, score record, checkpoint and colour of the rows of the test figure, the reference first.
TEST = (
    ("Transformer, 256 wide", "event_lm/state_reads", "test_v12_decoders",
     "checkpoints/scratch-study/transformer_s20001.pt", ORANGE),
    ("Transformer of the standard shape", STUDY, "test_transformer_standard",
     "checkpoints/read_clocks/transformer-std/model.pt.averaged", ORANGE),
    ("Clock cells, v14", STUDY, "test_clock_cells", "checkpoints/read_clocks/tag8-a8/model.pt.averaged", VIOLET),
    ("Address rate", STUDY, "test_address_rate", "checkpoints/read_clocks/rate8/model.pt.averaged", PINK),
    ("Spike decoder of v13", "event_lm/state_reads", "test_spike_reads_and_cells",
     "checkpoints/state_reads/pilot-race64-d8-c350-dl4/model.pt.averaged", SLATE),
)
# Title, what the family asked, and its arms in arms.json.
FAMILIES = (
    ("The three decoders",
     "The address rate alone, with the clock at word ends, and with a learned clock cell for every head. "
     "`end8-a8-laptop` is the second of these trained again on another machine to 12,000 updates.",
     ("rate8", "end8-a8", "tag8-a8", "end8-a8-laptop")),
    ("Read heads and clocks",
     "More heads, more reads, marked cells, a faster clock cell, heads that start sharp on one word each, and an "
     "address rate of 64 in place of 8.",
     ("end8-a8-h8", "r7h8", "end8-a8-m", "tag8-a8-m", "tag8-a8-c8", "end8-a8-s8", "rate64", "end64-m")),
    ("Beat fibres for the race cells",
     "The beat fibres of v13 at full size: cells that hear the stream of the last words, with the clock at word "
     "starts and at word ends.",
     ("word-sum", "gend3-sum", "gend3-mean")),
    ("Clusters",
     "More and smaller clusters, each on fibres of its own, cell layers right after the reads, and fibres that "
     "reach eight bytes back.",
     ("c32n175", "c64n87", "c32f128", "c64f64", "c350n16", "c700n8", "g0246", "dl8")),
    ("Races within a cluster",
     "The cells of a cluster race in equal groups, so that more cells answer at a byte. The last arm raises the "
     "pulse budget by the spikes that its races add.",
     ("r7b9", "r25b3", "r50b2", "r175b1", "r50b8", "r50b8p")),
    ("Blocks against cells",
     "Thinner blocks with more cells at the same size, and more cells at a larger size.",
     ("h122c624", "h61c762", "big-n700", "big-c32")),
    ("What a cluster hears and reaches",
     "Wider fans with fewer cells at the same size, and clusters that hear more channels than their axons reach "
     "or the reverse.",
     ("f512c175", "f1024c88", "f1408c63", "f1408k32", "h1408r256", "h256r1408")),
    ("How cells learn and answer",
     "Racing cells that learn from a softer race than they fire in, cells that answer with a smooth function of "
     "their own charge and do not spike, and threshold cells that learn below their threshold. The last four arms "
     "give the cells relay neurons that hear the whole stream, so that a cell weighs a learned view of all of it. "
     "In two of them the cells also hear the relays as they were at the last letter of each of the last three "
     "words.",
     ("soft2", "soft4", "softcells", "slope", "slope-relay", "soft-relay", "slope-relay-last", "relay-last")),
    ("Transformers of the standard shape",
     "The reference Transformer is 256 wide with 6 layers and an MLP of 42 times its width. These have the usual "
     "shape at the same size: 512 wide, 11 layers, 8 heads, an MLP of 4 times the width.",
     ("transformer-std", "transformer-std-rot")),
)
GAP_GROUPS = (
    ("All targets", "words_in_view.json", (
        ("All", "all"), ("Copyable", "copyable (match >= 4)"), ("Not copyable", "not copyable"))),
    ("First letters of words, by the words in view", "words_in_view.json", (
        ("None", "first letters, 0 words in view"), ("1", "first letters, 1 words in view"),
        ("2 to 3", "first letters, 2-3 words in view"), ("4 to 9", "first letters, 4-9 words in view"),
        ("10 to 39", "first letters, 10-39 words in view"), ("40 and more", "first letters, 40+ words in view"))),
    ("Later letters, by the words in view", "words_in_view.json", (
        ("1", "later letters, 1 words in view"), ("2 to 3", "later letters, 2-3 words in view"),
        ("4 to 9", "later letters, 4-9 words in view"), ("10 to 39", "later letters, 10-39 words in view"),
        ("40 and more", "later letters, 40+ words in view"))),
    ("Later letters that cannot be copied, by what the window showed", "later_letters.json", (
        ("The word's beginning never began a word", "beginning never began an earlier word"),
        ("It did, never followed by this letter", "beginning seen, never with this letter"),
        ("It did, followed by this letter", "beginning seen with this letter"))),
)


def spread(values, gap):
    """The values moved apart until neighbours lie at least a gap apart, each as near its place as that allows."""
    order = sorted(range(len(values)), key=values.__getitem__)
    placed = [values[index] for index in order]
    for _ in range(200):
        moved = False
        for index in range(1, len(placed)):
            short = gap - (placed[index] - placed[index - 1])
            if short > 1e-9:
                placed[index - 1] -= short / 2
                placed[index] += short / 2
                moved = True
        if not moved:
            break
    return [value for _, value in sorted(zip(order, placed))]


def end_labels(ax, labels, *, side="right", gap=11.5):
    """Write a value beside the end of each line, and move the values apart where they would overprint.

    labels holds x, y, text and the line's colour per line. A value that had to move keeps a thin line to its
    point. The axis limits must be final when this is called.
    """
    low, high = ax.get_ylim()
    per_point = (high - low) / (ax.get_position().height * ax.figure.get_figheight() * 72)
    placed = spread([y for _, y, _, _ in labels], gap * per_point)
    sign = 1 if side == "right" else -1
    for (x, y, text, color), at in zip(labels, placed):
        lift = (at - y) / per_point
        moved = abs(lift) > 1
        if moved:
            ax.annotate("", (x, y), xytext=(sign * 13, lift), textcoords="offset points",
                        arrowprops={"arrowstyle": "-", "color": color, "linewidth": 0.9, "shrinkA": 0, "shrinkB": 4})
        ax.annotate(text, (x, y), xytext=(sign * (15 if moved else 7), lift), textcoords="offset points",
                    ha="left" if side == "right" else "right", va="center", size=9, color=TEXT)


def ramp(color):
    return LinearSegmentedColormap.from_list("ramp", ["#ffffff", color])


def head_panel(ax, side, record, color, title, unit, *, labels=True):
    """One model's heads at first letters: a row per head, the share of its read by the bytes back."""
    rows = [row for row in record["rows"] if row["where"] == "first letter"]
    layers = sorted({row["layer"] for row in rows})
    shares = np.array([[row["bytes_back"][name] for name in REACH] for row in rows])
    cells = np.array([row["cells"] for row in rows])
    ends = np.array([row["ends"] for row in rows])
    # A gap between the layers keeps their heads together.
    place = np.array([index + 0.6 * layers.index(row["layer"]) for index, row in enumerate(rows)])
    for y, line in zip(place, shares):
        for x, share in enumerate(line):
            ax.add_patch(plt.Rectangle((x - 0.47, y - 0.44), 0.94, 0.88, color=ramp(color)(min(1.0, share / 0.5)),
                                       linewidth=0))
            if share >= 0.2:
                ax.text(x, y, f"{share:.2f}".lstrip("0"), ha="center", va="center", size=7.2,
                        color=PAPER if share >= 0.3 else TEXT)
    ax.set_xlim(-0.6, len(REACH) - 0.4)
    ax.set_ylim(place[-1] + 0.7, -0.7)
    ax.set_xticks(range(len(REACH)), REACH if labels else [""] * len(REACH))
    ax.set_yticks([place[[row["layer"] == layer for row in rows]].mean() for layer in layers],
                  [f"{unit} {layer}" for layer in layers])
    ax.tick_params(length=0, labelcolor=MUTED, labelsize=8.5, pad=5)
    for name in ("top", "right", "left", "bottom"):
        ax.spines[name].set_visible(False)
    ax.set_title(title, loc="left", size=11, weight="bold", color=TEXT, pad=8)
    side.barh(place, cells, height=0.72, color=color, linewidth=0)
    for y, count, share in zip(place, cells, ends):
        side.text(count * 1.12, y, f"{count:.0f}", va="center", size=7.4, color=MUTED)
        if share >= 0.3:
            # A head that sits on the first bytes after earlier words.
            side.plot(0.72, y, marker="D", color=TEXT, markersize=3.6, clip_on=False)
    side.set_xscale("log")
    side.set_xlim(1, 900)
    side.set_ylim(*ax.get_ylim())
    side.set_yticks([])
    side.set_xticks([1, 10, 100], ["1", "10", "100"] if labels else [""] * 3)
    side.minorticks_off()
    side.tick_params(length=0, labelcolor=MUTED, labelsize=8.5, pad=5)
    if labels:
        ax.set_xlabel("bytes back", color=MUTED, size=9)
        side.set_xlabel("cells that share a read", color=MUTED, size=9)
    side.grid(axis="x", color=GRID, linewidth=0.75)
    side.set_axisbelow(True)
    for name in ("top", "right", "left", "bottom"):
        side.spines[name].set_visible(False)


def plot_heads(output):
    transformer = read_result(f"{STUDY}/head_use_transformer.json")
    before = read_result(f"{STUDY}/head_use_decoder_v13.json")
    after = read_result(f"{STUDY}/head_use_decoder_v14.json")
    fig = plt.figure(figsize=(11.8, 8.2))
    grid = fig.add_gridspec(2, 5, width_ratios=(1.75, 0.8, 0.42, 1.75, 0.8), left=0.075, right=0.975, bottom=0.08,
                            top=0.81, wspace=0.08, hspace=0.2)
    tall_heading(fig, "Where heads look at a word's first letter",
                 "Share of every head's read by the distance to the cell it reads. A diamond marks a head with at "
                 "least 30% on first bytes\nafter earlier words (15% of all bytes). 24 validation windows, bytes 128 "
                 "to 511; all three models at 90,000 updates.")
    head_panel(fig.add_subplot(grid[:, 0]), fig.add_subplot(grid[:, 1]), transformer, ORANGE,
               "Transformer: 24 attention heads", "layer")
    head_panel(fig.add_subplot(grid[0, 3]), fig.add_subplot(grid[0, 4]), before, SLATE,
               "Spike decoder of v13: 12 read heads", "read after block", labels=False)
    head_panel(fig.add_subplot(grid[1, 3]), fig.add_subplot(grid[1, 4]), after, VIOLET,
               "With clock cells, v14: 12 read heads", "read after block")
    save(fig, output, "event_lm_read_clocks_heads.png")


def address_curves(name):
    """Updates and, per read, the mean over its heads of the offset of near cells over far ones."""
    record = read_result(f"{STUDY}/{name}")
    reads = sorted({head["read_after"] for head in record["checkpoints"][0]["heads"]})
    updates = [point["updates"] for point in record["checkpoints"]]
    near = {read: [float(np.mean([head["near"] for head in point["heads"] if head["read_after"] == read]))
                   for point in record["checkpoints"]] for read in reads}
    return updates, near


def summed_rate(updates, total=90000, warmup=1000, rate=3e-4):
    """The learning rates of the schedule summed up to each of ``updates``."""
    summed = np.cumsum([cosine_lr(step, total, warmup, rate) for step in range(total)])
    return [float(summed[update - 1]) for update in updates]


def plot_addresses(output):
    old_updates, old = address_curves("read_addresses_decoder_v13.json")
    new_updates, new = address_curves("read_addresses_address_rate.json")
    fig, (left, right) = plt.subplots(1, 2, figsize=(12.6, 5.6))
    fig.subplots_adjust(left=0.065, right=0.9, bottom=0.2, top=0.73, wspace=0.34)
    tall_heading(fig, "How far a head's address moves is set by the step size",
                 "Offset of a read's eight newest cells over its 64 oldest, mean over its four heads · one unit "
                 "multiplies a cell's rate in the race by e")
    for ax in (left, right):
        style(ax)
    shades = dict(zip(sorted(old), (0.95, 0.7, 0.45)))
    travelled = summed_rate(old_updates)
    for read, values in old.items():
        left.plot(travelled, values, color=SLATE, alpha=shades[read], linewidth=2, zorder=3)
        end_label(left, travelled[-1], values[-1], f"read after block {read}", TEXT)
    slope = old[min(old)][-1] / travelled[-1]
    left.plot([0, travelled[-1]], [0, slope * travelled[-1]], color=TEXT, linewidth=0.9, linestyle=(0, (3, 3)), zorder=2)
    left.annotate(f"{slope:.2f} per unit of\nsummed learning rate", (0.52 * travelled[-1], 0.52 * slope * travelled[-1]),
                  xytext=(-10, 14), textcoords="offset points", ha="right", va="bottom", size=9, color=MUTED,
                  linespacing=1.35)
    left.set_xlabel("Learning rates summed over the updates so far", color=MUTED, labelpad=8)
    left.set_ylabel("Offset of near cells over far ones", color=MUTED, labelpad=8)
    left.set_title("v13: the offsets grow with the summed learning rate to the last update", fontsize=10,
                   loc="left", color=TEXT)
    left.set_xlim(0, travelled[-1] * 1.02)
    left.set_ylim(0, None)
    for read, values in old.items():
        right.plot([update / 1000 for update in old_updates], values, color=SLATE, alpha=shades[read], linewidth=2,
                   zorder=2)
    end_label(right, old_updates[-1] / 1000, old[min(old)][-1], "v13", TEXT)
    for read, values in new.items():
        right.plot([update / 1000 for update in new_updates], values, color=PINK, alpha=shades[read], linewidth=2,
                   zorder=3)
        end_label(right, new_updates[-1] / 1000, values[-1], f"block {read}", TEXT)
    right.set_xlabel("Updates of 4,096 targets, in thousands", color=MUTED, labelpad=8)
    right.set_xlim(0, 99)
    right.set_xticks(range(0, 91, 15))
    right.set_ylim(0, None)
    right.set_title("The same offsets with an address rate of 8 (pink)", fontsize=10, loc="left", color=TEXT)
    fig.text(0.035, 0.04, "Slate: the v13 decoder. Pink: the same decoder with the address rate, which multiplies "
             "the stored offsets and gains of a read and changes nothing else. Darker lines are earlier reads.\n"
             "The weight decay pulls a stored offset back in proportion to its size, so the rate also raises the "
             "size at which pull and push balance.", color=MUTED, size=9, linespacing=1.5)
    save(fig, output, "event_lm_read_clocks_addresses.png")


def language_rows():
    """Bits per first letter of every read arm for both seeds, and the two bounds of the made-up language."""
    records = [read_result(f"{STUDY}/word_time_reads_s{seed}.json") for seed in (0, 1)]
    bounds = {(record["first_letter_bits_without_the_word_before_last"],
               record["first_letter_bits_with_the_word_before_last"]) for record in records}
    if len(bounds) != 1:
        raise ValueError("the two seeds must share one language")
    scores = [{arm["arm"]: arm["first_letter_bits"] for arm in record["arms"]} for record in records]
    return [(label, [seed[arm] for seed in scores]) for arm, label in LANGUAGE_ARMS], bounds.pop(), records[0]["settings"]


def beat_ages(line):
    """Every cell's age in beats seen from the last byte, with a beat at the first byte after each word."""
    beats = [index > 0 and line[index] == " " and line[index - 1] != " " for index in range(len(line))]
    started = np.cumsum(beats)
    return started[-1] - started, beats


def plot_language(output):
    rows, (without, with_word), settings = language_rows()
    fig = plt.figure(figsize=(13.6, 6.6))
    tall_heading(fig, "A read head counts time in beats",
                 f"Left: how a head addresses a cell. Right: a made-up language where a word depends on the word "
                 f"before last · bits per first letter after {settings['steps']:,} updates · two seeds")
    sketch = fig.add_axes([0.03, 0.1, 0.43, 0.66])
    sketch.set_xlim(0, 44)
    sketch.set_ylim(0, 34)
    sketch.axis("off")
    ages, beats = beat_ages(LINE)
    left, width, tall = 1.0, 1.8, 2.6

    def strip(y, fills, outlined=()):
        for index, letter in enumerate(LINE):
            sketch.add_patch(plt.Rectangle((left + width * index, y), width, tall, facecolor=fills[index],
                                           edgecolor=GRID, linewidth=0.8, zorder=1))
            sketch.text(left + width * (index + 0.5), y + tall / 2, "·" if letter == " " else letter, ha="center",
                        va="center", size=10.5, color=TEXT, family="DejaVu Sans Mono", zorder=3)
        for index in outlined:
            sketch.add_patch(plt.Rectangle((left + width * index, y), width, tall, facecolor="none", edgecolor=TEXT,
                                           linewidth=1.5, zorder=2))
        end = left + width * len(LINE)
        sketch.annotate("", (end + 2.3, y + tall / 2), (end + 0.3, y + tall / 2),
                        arrowprops={"arrowstyle": "-|>", "color": MUTED, "linewidth": 1})
        sketch.text(end + 2.6, y + tall / 2, "?", ha="left", va="center", size=12, color=TEXT, weight="bold")

    sketch.text(left, 32.6, "Age in bytes", size=10, color=TEXT, weight="bold", va="center")
    sketch.text(left, 30.6, "Every earlier byte is a cell. A head has one learned offset per age in bytes.",
                size=9, color=MUTED, va="center")
    last = len(LINE) - 1
    byte_shades = [AGE_SHADES[min(len(AGE_SHADES) - 1, (last - index) // 4)] for index in range(len(LINE))]
    strip(25.4, byte_shades)
    for back in (0, 4, 8, 12, 16, 20):
        sketch.text(left + width * (last - back + 0.5), 24.6, "now" if back == 0 else str(back), ha="center",
                    va="top", size=8, color=MUTED)
    sketch.text(left, 19.0, "Age in beats of a clock", size=10, color=TEXT, weight="bold", va="center")
    sketch.text(left, 16.3, "A beat at the first byte after a word opens a stretch. The head has one more\n"
                "learned offset per age in beats, whatever the lengths of the words.", size=9, color=MUTED,
                va="center", linespacing=1.45)
    base = 4.6
    strip(base, [AGE_SHADES[min(len(AGE_SHADES) - 1, int(age))] for age in ages],
          outlined=[index for index, beat in enumerate(beats) if beat])
    scale, floor = 1.25, base + tall + 2.2
    sketch.plot([left, left + width * len(LINE)], [floor, floor], color=GRID, linewidth=0.9, zorder=1)
    for age in range(int(ages.max()) + 1):
        cells = np.flatnonzero(ages == age)
        first, after = left + width * cells[0], left + width * (cells[-1] + 1)
        offset = SKETCH_HEAD[age]
        sketch.add_patch(plt.Rectangle((first + 0.25, floor), after - first - 0.5, scale * offset,
                                       facecolor=VIOLET if offset > 0 else GRAY, linewidth=0, zorder=2))
        sketch.text((first + after) / 2, floor + scale * offset + (0.35 if offset > 0 else -0.35),
                    f"{offset:+.1f}".replace("-", "\u2212"), ha="center", va="bottom" if offset > 0 else "top",
                    size=8, color=TEXT)
        sketch.text((first + after) / 2, base - 0.6, f"age {age}", ha="center", va="top", size=8, color=MUTED)
    for index, beat in enumerate(beats):
        if beat:
            sketch.plot([left + width * (index + 0.5)] * 2, [base + tall + 0.15, base + tall + 1.0], color=TEXT,
                        linewidth=1.3, zorder=3)
    sketch.text(left, 1.4, "Bars: the offsets of one head of the full-size decoder with the clock at word ends, "
                "at 18,000 updates.\nOutlined: the first byte after a word, where that clock beats.", size=8.5,
                color=MUTED, va="center", linespacing=1.45)
    ax = fig.add_axes([0.705, 0.17, 0.265, 0.57])
    style(ax, grid_axis="x")
    place = [index + (0.9 if index >= 7 else 0) for index in range(len(rows))]
    for y, (label, values), (arm, _) in zip(place, rows, LANGUAGE_ARMS):
        ax.plot([min(values), max(values)], [y, y], color=LIGHT, linewidth=2.2, zorder=1)
        ax.scatter(values, [y] * len(values), s=46, color=VIOLET if "clock cells" in arm else GRAY, zorder=2,
                   edgecolor="white", linewidth=0.8)
        ax.annotate(" · ".join(f"{value:.2f}" for value in values), (max(values), y), xytext=(8, 0),
                    textcoords="offset points", ha="left", va="center", size=8.5, color=TEXT)
    for bound, label in ((without, "without the word\nbefore last"), (with_word, "with it")):
        ax.axvline(bound, color=TEXT, linewidth=0.9, linestyle=(0, (3, 3)), zorder=1)
        ax.annotate(f"{label}\n{bound:.2f}", (bound, -0.8), ha="center", va="bottom", size=8.5, color=MUTED,
                    linespacing=1.35)
    ax.set_yticks(place, [label for label, _ in rows])
    ax.tick_params(axis="y", labelcolor=TEXT, labelsize=9.5)
    for y, title in ((-0.05, "A read alone"), (7.85, "A read, then race cells")):
        ax.text(-0.02, y - 0.62, title, transform=ax.get_yaxis_transform(), ha="right", va="center", size=9,
                color=MUTED, style="italic")
    ax.set_ylim(place[-1] + 0.6, -1.0)
    ax.set_xlim(2.4, 3.95)
    ax.set_xticks([2.5, 3.0, 3.5])
    ax.set_xlabel("Bits per first letter of a word", color=MUTED, labelpad=8)
    fig.text(0.035, 0.028, "24 made-up words; half of the words are chosen by the word before last. One state "
             "read of four heads on a bare byte embedding. Violet: every head has a learned clock cell.\nBeside "
             "each row: seed 0 · seed 1.", color=MUTED, size=9, linespacing=1.5)
    save(fig, output, "event_lm_read_clocks_language.png")


def probe_depth(stream):
    """A tapped stream's place in the depth of its network."""
    if stream in DEPTH:
        return DEPTH[stream]
    block, _, depth = stream.removeprefix("block ").partition(" of ")
    return int(block) / int(depth)


def plot_words(output):
    record = read_result(f"{STUDY}/word_probe.json")
    transformer = record["checkpoints"][0]
    decoders = [(SHOWN[checkpoint["checkpoint"]], checkpoint) for checkpoint in record["checkpoints"][1:]
                if checkpoint["checkpoint"] in SHOWN]
    fig, (left, right) = plt.subplots(1, 2, figsize=(12.6, 5.95))
    fig.subplots_adjust(left=0.065, right=0.975, bottom=0.325, top=0.75, wspace=0.2)
    tall_heading(fig, "The decoder's stream holds the earlier words",
                 f"Bits that a linear readout of a stream removes from the uncertainty about an earlier word · "
                 f"{record['bytes']:,} bytes inside words of {record['windows']:,} validation windows")
    for ax in (left, right):
        style(ax)
    back = range(1, 6)

    def best(checkpoint):
        """The stream of a network that knows the last word best."""
        return max(checkpoint["streams"], key=lambda row: row["word"][1])

    # Every network is drawn at the stream that knows the last word best, and the Transformer at its top as well.
    series = [(f"Transformer, {best(transformer)['stream']}", ORANGE, "-", best(transformer)["word"]),
              ("Transformer, top", ORANGE, (0, (4, 2.5)), transformer["streams"][-1]["word"])]
    series += [(f"{label}, {best(decoder)['stream']}", color, "-", best(decoder)["word"])
               for (label, color), decoder in decoders]
    for label, color, dashes, bits in series:
        left.plot(back, bits[1:], color=color, linewidth=2, linestyle=dashes, marker="o", markersize=5, zorder=3,
                  label=label)
    left.legend(frameon=False, labelcolor=TEXT, fontsize=9, loc="upper center", bbox_to_anchor=(1.1, -0.19),
                ncol=2, handlelength=2.6, columnspacing=3)
    left.set_xlim(0.45, 5.3)
    left.set_xticks(list(back))
    left.set_ylim(0, 3.1)
    end_labels(left, [(1, bits[1], f"{bits[1]:.2f}", color) for _, color, _, bits in series], side="left")
    left.set_xlabel("Words back from the word in progress", color=MUTED, labelpad=8)
    left.set_ylabel("Bits about the word", color=MUTED, labelpad=8)
    left.set_title("By the distance in words, at the stream that knows the last word best", fontsize=10,
                   loc="left", color=TEXT)
    curves = [("Transformer", ORANGE, transformer)] + [(label, color, decoder) for (label, color), decoder in decoders]
    ends = []
    for label, color, checkpoint in curves:
        depth = [probe_depth(row["stream"]) for row in checkpoint["streams"]]
        bits = [row["word"][2] for row in checkpoint["streams"]]
        right.plot(depth, bits, color=color, linewidth=2, marker="o", markersize=5, zorder=3)
        ends.append((depth[-1], bits[-1], f"{bits[-1]:.2f}", color))
        peak = int(np.argmax(bits))
        if peak != len(bits) - 1:
            end_label(right, depth[peak], bits[peak], f"{bits[peak]:.2f}", TEXT, dx=0, dy=9, ha="center")
    for read in (3 / 8, 5 / 8, 7 / 8):
        right.axvline(read, color=GRID, linewidth=1.1, zorder=1)
    right.annotate("state reads", (7 / 8, 0.03), xycoords=("data", "axes fraction"), xytext=(-5, 0),
                   textcoords="offset points", ha="right", va="bottom", size=8.5, color=MUTED)
    right.set_xlim(0, 1.09)
    right.set_xticks([0, 0.25, 0.5, 0.75, 1], ["input", "", "half way", "", "top"])
    right.set_ylim(0, 1.7)
    end_labels(right, ends)
    right.set_xlabel("Depth of the tapped stream in its network", color=MUTED, labelpad=8)
    right.set_ylabel("Bits about the word two back", color=MUTED, labelpad=8)
    right.set_title("Along the depth: the word two back", fontsize=10, loc="left", color=TEXT)
    fig.text(0.035, 0.03, "The readout names the word among the 255 most frequent words and one class for all "
             "others, at bytes inside a word with at least six earlier words in view.\nIt is fitted on half of "
             "the windows and scored on a quarter it never saw. Readouts are measuring tools and no part of a "
             "model.\nThe decoder's stream has 1,408 channels and the Transformer's 256, so the decoder's "
             "readout has more inputs to fit.", color=MUTED, size=9, linespacing=1.5)
    save(fig, output, "event_lm_read_clocks_words.png")


def gap_rows(name):
    """Share of the targets and, per decoder, its loss over the Transformer's with the interval, by row name."""
    record = read_result(f"{STUDY}/{name}")
    return {row["targets"].strip(): (row["share"], row["over_reference"]) for row in record["rows"]}, record


def plot_gap(output):
    records = {name: gap_rows(name) for name in {group[1] for group in GAP_GROUPS}}
    first = next(iter(records.values()))[1]
    # The place of every shown decoder among the differences of a row, which leave the reference out.
    shown = [(index - 1, SHOWN[name]) for index, name in enumerate(first["checkpoints"]) if name in SHOWN]
    decoders, windows = len(shown), first["windows"]
    fig, ax = plt.subplots(figsize=(11.8, 9.0))
    fig.subplots_adjust(left=0.36, right=0.9, bottom=0.155, top=0.85)
    beside = blended_transform_factory(fig.transFigure, ax.transData)
    tall_heading(fig, "The Transformer's lead begins with the fourth word in view",
                 f"Loss of a spike decoder minus the Transformer's, in bits per target of a kind · {windows} "
                 "validation windows of 512 bytes · 95% intervals over windows")
    style(ax, grid_axis="x")
    place, ticks, labels = 0.0, [], []
    spread = 0.17 if decoders > 1 else 0.0
    for heading_, name, rows in GAP_GROUPS:
        ax.text(0.036, place - 0.1, heading_, transform=beside, ha="left", va="center", size=9.5, color=TEXT,
                weight="bold")
        place += 0.95
        for label, key in rows:
            share, gaps = records[name][0][key]
            for index, (column, (_, color)) in enumerate(shown):
                gap = gaps[column]
                y = place + (index - (decoders - 1) / 2) * 2 * spread
                ax.plot([gap["low"], gap["high"]], [y, y], color=color, linewidth=2.2, alpha=0.45, zorder=2,
                        solid_capstyle="round")
                ax.scatter([gap["mean"]], [y], s=42, color=color, zorder=3, edgecolor="white", linewidth=0.8)
            ax.annotate(f"{100 * share:.1f}%" if share >= 0.0005 else "<0.1%", (1.0, place),
                        xycoords=("axes fraction", "data"), xytext=(8, 0), textcoords="offset points", ha="left",
                        va="center", size=8.5, color=MUTED)
            ticks.append(place)
            labels.append(label)
            place += 1.0
        place += 0.55
    ax.axvline(0, color=TEXT, linewidth=1.0, zorder=1)
    ax.set_yticks(ticks, labels)
    ax.tick_params(axis="y", labelcolor=TEXT, labelsize=9.5)
    ax.set_ylim(place - 1.0, -1.0)
    ax.set_xlabel("Bits per target over the Transformer · left of the line the decoder is ahead", color=MUTED,
                  labelpad=8)
    ax.annotate("share of\nall targets", (1.0, -0.55), xycoords=("axes fraction", "data"), xytext=(8, 0),
                textcoords="offset points", ha="left", va="center", size=8.5, color=MUTED, linespacing=1.3)
    for index, (_, (label, color)) in enumerate(shown):
        fig.text(0.36 + 0.21 * index, 0.872, "\u25cf", color=color, size=11, va="center")
        fig.text(0.375 + 0.21 * index, 0.872, label, color=TEXT, size=9.5, va="center")
    fig.text(0.035, 0.025, "A word is a run of letters, digits and bytes above 127. The words in view are those "
             "that started in the window before the target.\nA target is copyable when the 4 or more bytes "
             "before it occurred earlier in the window and the byte that followed them there is the target.\n"
             "All networks have 34.8M parameters and 90,000 updates.", color=MUTED, size=9, linespacing=1.5)
    save(fig, output, "event_lm_read_clocks_gap.png")


def arms():
    """Every arm and reference curve of the study by its name."""
    record = read_result(f"{STUDY}/arms.json")
    return {run["name"]: run for run in record["arms"] + record["references"]}


def curve(run):
    """The validation loss of a run's weight average by its updates."""
    return {point["step"]: point["averaged_val_bpb"] for point in run["curve"] if "averaged_val_bpb" in point}


def test_rows():
    """Label, bits per byte, difference to the first row with its interval, colour and parameters of every scored
    checkpoint."""
    rows = []
    for label, study, record, checkpoint, color in TEST:
        scores = {row["checkpoint"]: row for row in read_result(f"{study}/scores/{record}.json")["checkpoints"]}
        row = scores[checkpoint]
        rows.append((label, row["bpb"], row.get("bpb_over_first_whole_windows"), row.get("bpb_over_first_paired_95"),
                     color, row["parameters"]))
    return rows


def plot_test(output):
    rows = test_rows()
    reference = rows[0][1]
    fig, ax = plt.subplots(figsize=(11.6, 5.2))
    fig.subplots_adjust(left=0.29, right=0.965, bottom=0.215, top=0.75)
    tall_heading(fig, "Faster addresses gain 0.018 bits; Transformers stay ahead",
                 "enwik8 test \u00b7 4,999,999 targets \u00b7 windows of 256 unscored and 256 scored bytes from an "
                 "empty state \u00b7 368.64M training targets each")
    style(ax, grid_axis="x")
    ax.axvline(reference, color=ORANGE, linewidth=1, linestyle=(0, (3, 3)), zorder=1)
    for index, (_, score, gap, _, color, _) in enumerate(rows):
        ax.plot([reference, score], [index, index], color=LIGHT, linewidth=2.2, zorder=1)
        ax.scatter(score, index, s=64, color=color, zorder=2, edgecolor="white", linewidth=1)
        ax.annotate(f"{score:.4f}", (max(score, reference), index), xytext=(10, 0), textcoords="offset points",
                    ha="left", va="center", size=9.5, color=TEXT, weight="bold")
        if gap is not None:
            ax.annotate(f"{gap:+.4f}".replace("-", "\u2212"), (max(score, reference), index), xytext=(52, 0),
                        textcoords="offset points", ha="left", va="center", size=9, color=MUTED)
    ax.set_yticks(range(len(rows)), [label for label, *_ in rows])
    ax.tick_params(axis="y", labelcolor=TEXT, labelsize=10)
    ax.set_ylim(len(rows) - 0.4, -0.6)
    low, high = min(score for _, score, *_ in rows), max(score for _, score, *_ in rows)
    ax.set_xlim(low - 0.012, high + 0.03)
    ax.set_xlabel("Bits per byte", color=MUTED, labelpad=8)
    fig.text(0.035, 0.03, "Beside each score: the difference to the 256-wide Transformer over whole windows (the "
             "dashed line), which is the reference of v13. The decoders and the\nTransformer of the standard "
             "shape are scored at their running weight average. 34.75M to 34.85M parameters. One seed per row.",
             color=MUTED, size=9, linespacing=1.5)
    save(fig, output, "event_lm_read_clocks.png")


def plot_training(output):
    runs = arms()
    reference = curve(runs[TRANSFORMER])
    fig, (left, right) = plt.subplots(1, 2, figsize=(12.6, 5.6))
    fig.subplots_adjust(left=0.065, right=0.95, bottom=0.27, top=0.74, wspace=0.3)
    tall_heading(fig, "An address rate brings the decoder closer to the Transformer",
                 "Validation loss of the running weight average · 16 batches of 8 windows of 512 enwik8 validation "
                 "bytes · one seed")
    for ax in (left, right):
        style(ax)
        ax.set_xlabel("Updates of 4,096 targets, in thousands", color=MUTED, labelpad=8)
        ax.set_xlim(0, 99)
        ax.set_xticks(range(0, 91, 15))
    series = [(OLD, "Spike decoder of v13", SLATE, "-"), *((name, label, color, "-") for name, label, color in RUNS),
              (STANDARD, "Transformer of the standard shape", ORANGE, (0, (4, 2.5))),
              (TRANSFORMER, "Transformer", ORANGE, "-")]
    ends = []
    right.set_ylim(-0.21, 0.1)
    for name, label, color, dashes in series:
        # The reference Transformer was evaluated every 3,000 updates and the other runs twice as often.
        points = sorted((step, loss) for step, loss in curve(runs[name]).items() if step in reference)
        thousands = [step / 1000 for step, _ in points]
        losses = [loss for _, loss in points]
        gaps = [loss - reference[step] for step, loss in points]
        left.plot(thousands, losses, color=color, linewidth=2, linestyle=dashes, zorder=3, label=label)
        if name == TRANSFORMER:
            continue
        right.plot(thousands, gaps, color=color, linewidth=2, linestyle=dashes, zorder=3)
        right.scatter(thousands[-1], gaps[-1], s=22, color=color, zorder=4)
        value = f"{gaps[-1]:+.3f}".replace("-", "\u2212")
        if points[-1][0] == 90000:
            ends.append((thousands[-1], gaps[-1], value, color))
        else:
            # A run that stopped early is labelled between the bundle of lines and zero, where the panel is empty.
            right.plot([thousands[-1]] * 2, [gaps[-1] - 0.007, 0.037], color=color, linewidth=0.9, zorder=2)
            right.text(thousands[-1] - 3, 0.033, f"{value} at {thousands[-1]:.0f}k, where this run stopped",
                       ha="left", va="top", size=9, color=TEXT)
    end_labels(right, ends)
    left.set_ylabel("Bits per byte", color=MUTED, labelpad=8)
    left.set_ylim(1.2, 1.85)
    left.set_title("Validation loss", fontsize=10, loc="left", color=TEXT)
    right.axhline(0, color=TEXT, linewidth=0.8)
    right.set_ylabel("Bits per byte above the Transformer", color=MUTED, labelpad=8)
    right.set_title("Loss minus the Transformer's at the same update", fontsize=10, loc="left", color=TEXT)
    handles, names = left.get_legend_handles_labels()
    left.legend(handles, names, frameon=False, labelcolor=TEXT, fontsize=9, loc="upper center",
                bbox_to_anchor=(1.12, -0.2), ncol=3, handlelength=2.6)
    fig.text(0.035, 0.03, "The Transformer is 256 wide with 6 layers and the one of the standard shape 512 wide with "
             "11, both of the decoders' size. The decoders and the Transformer are trained with the rotated\n"
             "optimizer and the Transformer of the standard shape with AdamW. All runs share the data order and "
             "the schedule and are compared here every 3,000 updates.", color=MUTED, size=9, linespacing=1.5)
    save(fig, output, "event_lm_read_clocks_training.png")


def flags(run):
    """A run's trainer flags as a mapping of each flag to its values."""
    parts, found = run["flags"].split(), {}
    for index, part in enumerate(parts):
        if part.startswith("--"):
            values = []
            for value in parts[index + 1:]:
                if value.startswith("--"):
                    break
                values.append(value)
            found[part] = " ".join(values)
    return found


def changes(run, base):
    """The flags in which a run differs from the base of the page."""
    if run["model"] == "transformer":
        return "a Transformer under " + ("the rotated optimizer" if "--rotated-adam" in flags(run) else "AdamW")
    mine, theirs = flags(run), flags(base)
    added = [f"`{flag} {value}`".replace(" `", "`") for flag, value in mine.items() if theirs.get(flag) != value]
    dropped = [f"no `{flag}`" for flag in theirs if flag not in mine]
    return ", ".join(added + dropped) or "the base of this page"


def prose(text):
    """A paragraph wrapped like the other pages of the repository."""
    return textwrap.fill(text, width=65, break_long_words=False, break_on_hyphens=False)


def arms_page():
    """docs/READ_CLOCKS_ARMS.md: every arm of the study by family, with what it changed and where it ended."""
    record = read_result(f"{STUDY}/arms.json")
    runs = arms()
    listed = {name for _, _, names in FAMILIES for name in names}
    missing = {run["name"] for run in record["arms"]} ^ listed
    if missing:
        raise ValueError(f"arms and families do not match: {sorted(missing)}")
    base, reference = runs[BASE], curve(runs[TRANSFORMER])
    ahead = curve(base)
    unscored = sum(1 for run in record["arms"] if not run["curve"])
    lines = ["# Every training arm of the read-clock study", "",
             prose(f"The study of 18 to 19 November 2026 trained {len(record['arms'])} arms. Each row is one arm "
                   "with one seed. Validation is 16 batches of 8 windows of 512 enwik8 validation bytes and scores "
                   "the running weight average. Every decoder arm is the release command of v13 with flags added; "
                   f"the flags are given as they differ from those of `{BASE}`, which are "
                   f"`{base['flags']}`. The last column is an arm's loss minus that of `{BASE}` at the same "
                   "update, and for a Transformer minus that of the reference Transformer."),
             "",
             prose("Most pilots were told to stop at 12,000 updates and were stopped by hand before that when they "
                   "fell behind. The study later found that the first 6,000 updates are the wrong place to judge "
                   "an arm: there every decoder leads the Transformer, and the Transformer's lead forms after "
                   f"that. So an arm that was stopped early was judged on its early learning alone. {unscored} "
                   "arms were stopped before their first evaluation at 1,500 updates and have no loss."),
             "",
             prose("The flags do not record the code. Flags were added during the study, and every one of them is "
                   "in the code of this release. Two more pilots ran on the laptop before the first arm of this "
                   "list, with a clock cell that heard the whole stream. The second left the range of exact sums "
                   "near 1,500 updates and neither wrote a loss, and the clock cell of the release hears its "
                   "read's tags through synapses that saturate instead."),
             "[Study page](READ_CLOCKS.md) ·",
             "[All flags and curves](../results/event_lm/read_clocks/arms.json)", ""]
    for title, question, names in FAMILIES:
        lines += [f"## {title}", "", prose(question), "",
                  "| Arm | Started, UTC | Parameters | Differs in | Updates | Averaged | Against the base |",
                  "|---|---|---:|---|---:|---:|---:|"]
        for name in names:
            run = runs[name]
            mine = curve(run)
            started = f"{run['started_utc'][8:10]} {run['started_utc'][11:16]}"
            if not mine:
                lines.append(f"| `{name}` | {started} | {run['parameters']:,} | {changes(run, base)} | | | |")
                continue
            last = max(mine)
            other = reference if run["model"] == "transformer" else ahead
            # A run is compared at its last update that the other run was evaluated at too.
            shared = max((step for step in mine if step in other), default=None)
            against = ""
            if shared is not None and name != BASE:
                against = f"{mine[shared] - other[shared]:+.4f}".replace("-", "\u2212")
                against += "" if shared == last else f" at {shared:,}"
            lines.append(f"| `{name}` | {started} | {run['parameters']:,} | {changes(run, base)} | {last:,} | "
                         f"{mine[last]:.4f} | {against} |")
        lines.append("")
    return "\n".join(lines)


def tables():
    """The tables of docs/READ_CLOCKS.md that are computed from the records."""
    runs = arms()
    columns = [(TRANSFORMER, "Transformer"), (STANDARD, "Transformer, standard shape"), (OLD, "v13 decoder"),
               *((name, label) for name, label, _ in RUNS)]
    print("averaged validation\n| Updates | " + " | ".join(label for _, label in columns) + " |\n|---:|"
          + "---:|" * len(columns))
    curves = {name: curve(runs[name]) for name, _ in columns}
    for step in CURVE_STEPS:
        print(f"| {step:,} | " + " | ".join(f"{curves[name][step]:.4f}" if step in curves[name] else ""
                                            for name, _ in columns) + " |")
    rows, (without, with_word), _ = language_rows()
    print(f"\nmade-up language: {without:.4f} without the word before last, {with_word:.4f} with it\n"
          "| Read | Seed 0 | Seed 1 |\n|---|---:|---:|")
    for label, values in rows:
        print(f"| {label} | " + " | ".join(f"{value:.3f}" for value in values) + " |")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tables", action="store_true", help="print the tables of docs/READ_CLOCKS.md instead")
    parser.add_argument("--arms", action="store_true", help="print docs/READ_CLOCKS_ARMS.md instead")
    parser.add_argument("--only", nargs="+", help="names of the figures to draw, all by default")
    parser.add_argument("--output", type=Path, default=RESULTS / "figures", help="directory for the figures")
    args = parser.parse_args()
    if args.tables or args.arms:
        tables() if args.tables else print(arms_page())
        return
    figures = {"test": plot_test, "training": plot_training, "heads": plot_heads, "addresses": plot_addresses,
               "language": plot_language, "words": plot_words, "gap": plot_gap}
    for name in args.only or figures:
        figures[name](args.output)


if __name__ == "__main__":
    main()
