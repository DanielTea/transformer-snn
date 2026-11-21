"""Figures of the long-streams study (v16): the stack and the Transformer at equal training context, the gain of fast
output synapses against how much of a stream they have read, the two models along their training at two depths, and
the trained-plasticity test bed."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from matplotlib.lines import Line2D

from plot_dendrite import GRID, MUTED, ORANGE, RESULTS, TEXT, plt, read_result, save, style
from plot_read_clocks import GREEN, PINK, SLATE, end_labels
from plot_state_reads import LIGHT, VIOLET, tall_heading

STUDY = "event_lm/long_streams"
V15 = "event_lm/stream_form/scores"
PLASTIC = "event_lm/trained_plasticity/arms.json"
# The training windows of the comparison at equal context and, for each, the records of the stack as a stream at the
# reach it trained with and of the Transformer in windows of that many bytes, 16 scored after the rest as history.
EQUAL = (
    (512, f"{V15}/test_windows_stream", f"{V15}/validation_windows_stream",
     f"{V15}/test_transformer_taught_history496", f"{V15}/validation_transformer_taught_history496"),
    (1024, f"{V15}/test_long_windows_stream", f"{V15}/validation_long_windows_stream",
     f"{STUDY}/scores/tf1024.taught-test-h1008", f"{STUDY}/scores/tf1024.taught-validation-h1008"),
    (2048, f"{STUDY}/scores/l2048-flat.stream-test", f"{STUDY}/scores/l2048-flat.stream-validation",
     f"{STUDY}/scores/tf2048.taught-test-h2032", f"{STUDY}/scores/tf2048.taught-validation-h2032"),
)
# Settings of the stack read further than it trained, with no new training: (window, label, test, validation).
LONGER = (
    (1024, "reach 4,096 at levels 5 to 7 (v15)", f"{V15}/test_long_windows_reach", f"{V15}/validation_long_windows_reach"),
    (1024, "reach 16,384, forgetting slope -1", f"{STUDY}/scores/released.forget-test",
     f"{STUDY}/scores/released.forget-validation"),
    (2048, "reach 8,192 at levels 4 to 7", f"{STUDY}/scores/l2048-flat.reach-test",
     f"{STUDY}/scores/l2048-flat.reach-validation"),
)


WHOLE_VIOLET = "#3b2c9c"
# Reading curves in reading_curve.json: key, colour and label. Whole books, once scored, are the darker violet.
READING = (("pg19", VIOLET, "16 PG-19 books, parts of 256 KB"), ("pg19_whole", WHOLE_VIOLET, "16 PG-19 books, whole"),
           ("enwik8_test", SLATE, "enwik8 test, 16 streams"))


def middle(row):
    """The middle of a bin on the doubling axis: the geometric mean of its edges, and 4 KB for the first bin."""
    return float(np.sqrt(row["from_kb"] * row["to_kb"])) if row["from_kb"] else row["to_kb"] / 2


def bpb(name, rate=0.0):
    """Bits per byte of a score record: of its checkpoint, or of a Transformer without fast output synapses."""
    record = read_result(f"{name}.json")
    if "fast_output_synapses" in record:
        (row,) = [row for row in record["fast_output_synapses"] if row["rate"] == rate]
        return row["bpb"]
    (row,) = record["checkpoints"]
    return row["bpb"]


def plot_equal(output):
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 6.2), sharey=False)
    fig.subplots_adjust(left=0.07, right=0.86, bottom=0.27, top=0.76, wspace=0.34)
    tall_heading(fig, "At equal context the stack stays about 0.035 behind",
                 "enwik8, 4,999,999 targets a split · the stack as 16 unbroken streams reaching as far as it "
                 "trained, the Transformer in windows of its training length · one seed each")
    for split, (ax, title) in enumerate(zip(axes, ("Test split", "Validation split"))):
        style(ax)
        ax.set_title(title, loc="left", size=11, color=TEXT, weight="bold", pad=10)
        windows = [row[0] for row in EQUAL]
        stack = [bpb(row[1 + split]) for row in EQUAL]
        transformer = [bpb(row[3 + split]) for row in EQUAL]
        ax.plot(windows, transformer, color=ORANGE, linewidth=2, marker="o", markersize=6, zorder=3)
        ax.plot(windows, stack, color=VIOLET, linewidth=2, marker="o", markersize=6, zorder=3)
        for window, ours, theirs in zip(windows, stack, transformer):
            ax.annotate(f"+{ours - theirs:.3f}", (window, (ours + theirs) / 2), xytext=(8, 0),
                        textcoords="offset points", ha="left", va="center", size=9, color=MUTED)
        for window, label, *names in LONGER:
            ax.scatter(window, bpb(names[split]), s=46, facecolor="white", edgecolor=VIOLET, linewidth=1.4, zorder=4)
        ax.set_xscale("log", base=2)
        ax.set_xticks(windows, [f"{window:,}" for window in windows])
        ax.minorticks_off()
        ax.set_xlim(420, 2500)
        ax.set_xlabel("Training window and reach, bytes", color=MUTED, labelpad=8)
        ax.set_ylabel("Bits per byte", color=MUTED, labelpad=8)
        end_labels(ax, [(windows[-1], stack[-1], f"Stack {stack[-1]:.4f}", VIOLET),
                        (windows[-1], transformer[-1], f"Transformer {transformer[-1]:.4f}", ORANGE)], gap=13)
    handles = [Line2D([], [], color=VIOLET, linewidth=2, marker="o", label="Stack, as a stream"),
               Line2D([], [], color=ORANGE, linewidth=2, marker="o", label="Transformer, in windows"),
               Line2D([], [], color=VIOLET, linestyle="none", marker="o", markerfacecolor="white",
                      label="Stack read further than it trained")]
    axes[0].legend(handles=handles, frameon=False, labelcolor=TEXT, fontsize=9, loc="upper center",
                   bbox_to_anchor=(1.12, -0.16), ncol=3, handlelength=2.2, columnspacing=1.8)
    fig.text(0.035, 0.03, "Beside each pair: the stack's distance to the Transformer. Each doubling of the window "
             "gives both models about half the gain of the last one.\nThe runs on windows of 1,024 and 2,048 bytes "
             "trained on windows of 512 bytes for their first 24,000 of 90,000 updates; those on 512 bytes trained "
             "on them\nthroughout. 34.75M and 34.81M parameters, 368.64M training targets each.", color=MUTED,
             size=9, linespacing=1.5)
    save(fig, output, "event_lm_long_streams.png")


def plot_reading(output):
    record = read_result(f"{STUDY}/reading_curve.json")
    fig, ax = plt.subplots(figsize=(12.6, 5.8))
    fig.subplots_adjust(left=0.08, right=0.8, bottom=0.2, top=0.76)
    tall_heading(fig, "On books the gain grows as the square root of the text read",
                 "Gain of the fast output synapses of the released v15 stack, pooled over the pieces that follow "
                 "the same amount of reading · one seed")
    style(ax)
    ends = []
    for key, color, label in READING:
        if key not in record:
            continue
        (series,) = record[key]["series"]
        # A stretch of reading that few streams reach is left out: its points would be single streams.
        bins = [part for part in series["bins"] if part["streams"] >= 8]
        middles = np.array([middle(part) for part in bins])
        gain = np.array([part["gain"] for part in bins])
        low = np.array([part["quartiles"][0] for part in bins])
        high = np.array([part["quartiles"][1] for part in bins])
        ax.fill_between(middles, low, high, color=color, alpha=0.12, linewidth=0)
        ax.plot(middles, gain, color=color, linewidth=2, marker="o", markersize=5, markeredgecolor="white",
                markeredgewidth=0.9, zorder=3)
        ends.append((middles[-1], gain[-1], label, color))
        if key == "pg19":
            slope, offset = np.polyfit(np.log(middles), np.log(gain), 1)
            line = np.geomspace(middles[0], middles[-1], 50)
            ax.plot(line, np.exp(offset) * line ** slope, color=VIOLET, linewidth=1, linestyle=(0, (3, 3)))
            ax.annotate(f"gain ∝ KB read^{slope:.2f}", (line[25], np.exp(offset) * line[25] ** slope),
                        xytext=(-12, 14), textcoords="offset points", ha="right", size=9.5, color=VIOLET)
    end_labels(ax, ends, gap=14)
    ax.set_xscale("log", base=2)
    ticks = [4, 8, 16, 32, 64, 128, 256, 512]
    ax.set_xticks(ticks, [str(tick) for tick in ticks])
    ax.minorticks_off()
    ax.set_xlabel("KB of the stream read before the piece", color=MUTED, labelpad=8)
    ax.set_ylabel("Bits per byte gained", color=MUTED, labelpad=8)
    ax.axhline(0, color=TEXT, linewidth=0.8)
    fig.text(0.035, 0.03, "Points: the gain over all pieces of a stretch of reading; bands: the middle half of the "
             "streams. The reads of the stack reach at most 4,096 bytes back, so the rise comes\nfrom the fast "
             "synapses. A Wikipedia stretch changes its subject every few kilobytes, and the gain stays level; a "
             "book keeps its names and words, and the gain keeps growing.", color=MUTED, size=9, linespacing=1.5)
    save(fig, output, "event_lm_long_streams_reading.png")


def curve(runs, name):
    return {step: value for step, value in runs[name]["val_bpb_by_update"]}


def plot_scaling(output):
    runs = read_result(f"{STUDY}/curves.json")["runs"]
    fig, (left, right) = plt.subplots(1, 2, figsize=(12.6, 6.0))
    fig.subplots_adjust(left=0.07, right=0.86, bottom=0.27, top=0.76, wspace=0.28)
    tall_heading(fig, "The stack learns faster at first, the Transformer for longer",
                 "Validation loss in windows of the training length from an empty state, current weights, logged "
                 "by the trainer every 1,500 updates · one seed per run")
    pairs = (("Depth 11, 34.8M parameters", ("p-forget", "l2048-forget"), ("tp", "tf2048"), VIOLET, "-"),
             ("Depth 22, 69.4M parameters", ("d22-forget", "l2048-d22"), ("d22-tp", "tf2048-d22"), PINK,
              (0, (5, 3))))
    ends = []
    for label, stacks, transformers, color, dashes in pairs:
        stack = curve(runs, stacks[0]) | curve(runs, stacks[1])
        transformer = curve(runs, transformers[0]) | curve(runs, transformers[1])
        steps = [step for step in sorted(stack) if step in transformer and step >= 1500]
        thousands = [step / 1000 for step in steps]
        left.plot(thousands, [stack[step] for step in steps], color=color, linewidth=2, linestyle=dashes)
        left.plot([step / 1000 for step in sorted(transformer)], [transformer[step] for step in sorted(transformer)],
                  color=ORANGE, linewidth=2 if color == VIOLET else 1.4, linestyle=dashes)
        gap = [stack[step] - transformer[step] for step in steps]
        right.plot(thousands, gap, color=color, linewidth=2, linestyle=dashes)
        right.scatter(thousands[-1], gap[-1], s=22, color=color, zorder=4)
        ends.append((thousands[-1], gap[-1], f"{label.split(',')[0]}: +{gap[-1]:.3f} at {thousands[-1]:g}k", color))
    for ax in (left, right):
        style(ax)
        ax.axvline(24, color=MUTED, linewidth=0.8, linestyle=(0, (2, 3)))
        ax.set_xlabel("Updates of 4,096 targets, in thousands", color=MUTED, labelpad=8)
        ax.set_xlim(0, 92)
    left.text(24.8, 1.92, "windows of 512 → 2,048 bytes", size=8.5, color=MUTED)
    left.set_ylim(1.08, 2.0)
    left.set_ylabel("Bits per byte", color=MUTED, labelpad=8)
    left.set_title("Validation loss", fontsize=10, loc="left", color=TEXT)
    right.axhline(0, color=TEXT, linewidth=0.8)
    right.set_ylim(-0.1, 0.075)
    right.set_ylabel("Stack minus Transformer, bits per byte", color=MUTED, labelpad=8)
    right.set_title("The stack's loss less the Transformer's at the same update", fontsize=10, loc="left",
                    color=TEXT)
    end_labels(right, ends, gap=16)
    handles = [Line2D([], [], color=VIOLET, linewidth=2, label="Stack, depth 11"),
               Line2D([], [], color=PINK, linewidth=2, linestyle=(0, (5, 3)), label="Stack, depth 22"),
               Line2D([], [], color=ORANGE, linewidth=2, label="Transformer, depth 11"),
               Line2D([], [], color=ORANGE, linewidth=1.4, linestyle=(0, (5, 3)), label="Transformer, depth 22")]
    left.legend(handles=handles, frameon=False, labelcolor=TEXT, fontsize=9, loc="upper center",
                bbox_to_anchor=(1.12, -0.16), ncol=4, handlelength=2.4, columnspacing=1.6)
    fig.text(0.035, 0.03, "Doubling the depth gains the Transformer about 0.04 bits per byte at equal updates and "
             "the stack about 0.02. Fitted over 3,000 to 24,000 updates, loss = E + A t^-α gives α of "
             "0.23 to 0.31 for\nthe stacks and 0.74 to 0.85 for the Transformers. The stack leads by up to 0.25 at "
             "first, below this scale. Runs still training end where their lines end.", color=MUTED, size=9,
             linespacing=1.5)
    save(fig, output, "event_lm_long_streams_scaling.png")


def plot_plasticity(output):
    arms = read_result(PLASTIC)["results"]["arms"]
    fig, (left, right) = plt.subplots(1, 2, figsize=(12.8, 6.2), gridspec_kw={"width_ratios": [1, 1]})
    fig.subplots_adjust(left=0.17, right=0.97, bottom=0.19, top=0.76, wspace=0.66)
    tall_heading(fig, "A rule in the loop makes better keys for the rule given after",
                 "Test bed: 6 levels, 4.86M parameters, 6,000 updates · enwik8 validation as 16 unbroken streams · "
                 "one seed per arm")
    own = (("B16", "Rule B, chunks of 16"), ("B64", "Rule B, chunks of 64"), ("e8", "Rule A, decay of 2^8"),
           ("e10", "Rule A, decay of 2^10"), ("f", "Rule A, shuffled (control)"), ("B16s", "Rule B 16, shuffled (control)"),
           ("N", "N, normalized"), ("carry", "Rule A on streams"), ("d", "Rule A, all levels"), ("c", "Rule A"))
    cells = (("e10", "Rule A, decay of 2^10"), ("B16", "Rule B, chunks of 16"), ("e8", "Rule A, decay of 2^8"),
             ("B64", "Rule B, chunks of 64"), ("c", "Rule A"), ("carry", "Rule A on streams"),
             ("d", "Rule A, all levels"), ("f", "Rule A, shuffled (control)"))
    for ax, rows, key, title, limits in (
            (left, own, "own_rule", "Each scored with its own rule", (-0.16, 0.035)),
            (right, cells, "v15_rule_levels_0_5_at_2^22", "Its cells read with v15's rule", (-0.002, 0.0175))):
        style(ax, grid_axis="x")
        ax.set_title(title, loc="left", size=11, color=TEXT, weight="bold", pad=10)
        for place, (arm, label) in enumerate(rows):
            gain, low, high = arms[arm][key]["gain_over_b"]
            color = SLATE if arm in ("f", "B16s") else VIOLET
            shown = max(gain, limits[0])
            ax.barh(place, shown, height=0.62, color=color if gain > limits[0] else LIGHT, zorder=2)
            if gain > limits[0]:
                ax.plot([low, high], [place, place], color=TEXT, linewidth=1, zorder=3)
            text = f"{gain:+.4f}" if gain > limits[0] else f"{gain:+.1f}, off the scale"
            ax.annotate(text, (max(shown, high if gain > limits[0] else shown, 0), place), xytext=(6, 0),
                        textcoords="offset points", ha="left", va="center", size=9, color=TEXT)
        ax.set_yticks(range(len(rows)), [label for _, label in rows])
        ax.tick_params(axis="y", labelcolor=TEXT, labelsize=9.5)
        ax.invert_yaxis()
        ax.axvline(0, color=TEXT, linewidth=0.8)
        ax.set_xlim(*limits)
        ticks = (-0.15, -0.1, -0.05, 0.0) if ax is left else (0.0, 0.005, 0.01, 0.015)
        ax.set_xticks(ticks, [f"{tick:+.3f}".replace("+0.000", "0") for tick in ticks])
        ax.set_xlabel("Bits per byte better than b", color=MUTED, labelpad=8)
    right.axvline(0.010, color=GREEN, linewidth=1, linestyle=(0, (3, 3)))
    right.annotate("the go/no-go's bar", (0.010, len(cells) - 0.4), xytext=(4, 0), textcoords="offset points",
                   size=8.5, color=GREEN, va="center")
    fig.text(0.035, 0.03, "b scores 1.8141. Left: no rule trained in the loop beats v15's rule given after training. "
             "Right: the same stacks with their own rule off and v15's rule on all six levels; cells trained with a "
             "rule\nwith real memory take about 0.18 bits per byte from v15's rule against 0.137 for untrained or "
             "control cells. Lines: paired 95% intervals over 10,016 pieces of 512 bytes.", color=MUTED, size=9,
             linespacing=1.5)
    save(fig, output, "event_lm_trained_plasticity.png")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=RESULTS / "figures")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    plot_equal(args.output)
    plot_reading(args.output)
    plot_scaling(args.output)
    plot_plasticity(args.output)


if __name__ == "__main__":
    main()
