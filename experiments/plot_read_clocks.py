"""Figures for the read clocks: where heads look, how the reads' addresses moved, and the clocks themselves."""

from __future__ import annotations

import argparse

from matplotlib.colors import LinearSegmentedColormap

from plot_dendrite import GRAY, GRID, MUTED, ORANGE, PAPER, RESULTS, TEXT, heading, np, plt, read_result, save, style
from plot_state_reads import LIGHT, VIOLET, tall_heading

STUDY = "event_lm/read_clocks"
GREEN = "#0f8f6c"
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
# Offsets by age in beats of one head of the pilot decoder, for the sketch.
SKETCH_HEAD = (0.40, 2.55, 1.54, 0.77, 0.17, -0.27)
AGE_SHADES = ("#8f83e2", "#a69ce9", "#bcb4ef", "#d0caf4", "#e2def8", "#f0eefb")


def ramp(color):
    return LinearSegmentedColormap.from_list("ramp", ["#ffffff", color])


def head_panel(ax, side, record, color, title, unit):
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
    ax.set_xticks(range(len(REACH)), REACH)
    ax.set_yticks([place[[row["layer"] == layer for row in rows]].mean() for layer in layers],
                  [f"{unit} {layer}" for layer in layers])
    ax.tick_params(length=0, labelcolor=MUTED, labelsize=8.5, pad=5)
    ax.set_xlabel("bytes back", color=MUTED, size=9)
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
    side.set_xticks([1, 10, 100], ["1", "10", "100"])
    side.tick_params(length=0, labelcolor=MUTED, labelsize=8.5, pad=5)
    side.set_xlabel("cells that share a read", color=MUTED, size=9)
    side.grid(axis="x", color=GRID, linewidth=0.75)
    side.set_axisbelow(True)
    for name in ("top", "right", "left", "bottom"):
        side.spines[name].set_visible(False)


def plot_heads(output):
    transformer = read_result(f"{STUDY}/head_use_transformer.json")
    decoder = read_result(f"{STUDY}/head_use_decoder_v13.json")
    fig = plt.figure(figsize=(11.4, 7.6))
    grid = fig.add_gridspec(1, 5, width_ratios=(1.75, 0.8, 0.34, 1.75, 0.8), left=0.075, right=0.975, bottom=0.085,
                            top=0.77, wspace=0.08)
    heading(fig, "Where heads look at a word's first letter",
            "Share of every head's read by the distance to the cell it reads. A diamond marks a head with at least "
            "30% on first bytes\nafter earlier words (15% of all bytes). 24 validation windows, bytes 128 to 511; "
            "both models at 90,000 updates.")
    head_panel(fig.add_subplot(grid[0]), fig.add_subplot(grid[1]), transformer, ORANGE,
               "Transformer: 24 attention heads", "layer")
    axis, side = fig.add_subplot(grid[3]), fig.add_subplot(grid[4])
    head_panel(axis, side, decoder, GREEN, "Spike decoder of v13: 12 read heads", "read after block")
    # The decoder has half as many heads: keep its rows as tall as the Transformer's.
    low, high = fig.axes[0].get_ylim()
    for ax in (axis, side):
        ax.set_ylim(low, high)
    save(fig, output, "event_lm_read_clocks_heads.png")


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
    sketch.text(left, 19.0, "Age in beats of the head's clock cell", size=10, color=TEXT, weight="bold", va="center")
    sketch.text(left, 16.3, "A beat at the first byte after a word opens a stretch. The head has one more\n"
                "learned offset per age in beats, and one for the cell that a beat marked.", size=9, color=MUTED,
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
    sketch.text(left, 1.4, "Bars: the offsets of one head of the full-size decoder at 6,000 updates. Outlined: "
                "marked cells,\nwhere the stream has just read a whole word.", size=8.5, color=MUTED, va="center",
                linespacing=1.45)
    ax = fig.add_axes([0.705, 0.17, 0.265, 0.57])
    style(ax, grid_axis="x")
    place = [index + (0.9 if index >= 7 else 0) for index in range(len(rows))]
    for y, (label, values), (arm, _) in zip(place, rows, LANGUAGE_ARMS):
        ax.plot([min(values), max(values)], [y, y], color=LIGHT, linewidth=2.2, zorder=1)
        ax.scatter(values, [y] * len(values), s=46, color=GREEN if "clock cells" in arm else GRAY, zorder=2,
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
             "read of four heads on a bare byte embedding. Green: every head has a learned clock cell.\nBeside "
             "each row: seed 0 · seed 1.", color=MUTED, size=9, linespacing=1.5)
    save(fig, output, "event_lm_read_clocks_language.png")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", nargs="+", help="names of the figures to draw, all by default")
    args = parser.parse_args()
    figures = {"heads": plot_heads, "language": plot_language}
    output = RESULTS / "figures"
    for name in args.only or figures:
        figures[name](output)
    _ = GRAY


if __name__ == "__main__":
    main()
