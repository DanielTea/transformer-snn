"""Figures for the read clocks: where heads look, how the reads' addresses moved, and the clocks themselves."""

from __future__ import annotations

import argparse

from matplotlib.colors import LinearSegmentedColormap

from plot_dendrite import GRAY, GRID, MUTED, ORANGE, PAPER, RESULTS, TEXT, heading, np, plt, read_result, save

STUDY = "event_lm/read_clocks"
GREEN = "#0f8f6c"
REACH = ("self", "1", "2-3", "4-7", "8-15", "16-63", "64+")


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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", nargs="+", help="names of the figures to draw, all by default")
    args = parser.parse_args()
    figures = {"heads": plot_heads}
    output = RESULTS / "figures"
    for name in args.only or figures:
        figures[name](output)
    _ = GRAY


if __name__ == "__main__":
    main()
