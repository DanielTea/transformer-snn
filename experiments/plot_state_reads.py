"""Figures and tables for the state reads, the race cells and their delayed fibres."""

from __future__ import annotations

import argparse
import textwrap

from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from plot_dendrite import GRAY, GRID, MUTED, ORANGE, RESULTS, TEXT, heading, plt, read_result, save, style

STUDY = "event_lm/state_reads"
GREEN, VIOLET, YELLOW, LIGHT = "#0f8f6c", "#6250d6", "#e0a000", "#c9d2de"
# The arms that every figure follows, by their names in arms.json.
TRANSFORMER, CELLS, READS, HYBRID = "full-transformer-rot", "pilot-race64-d8-c350-dl4", "full-rate-rot", "full-soft-a8-h316-rot"
EARLY_TRANSFORMER, PLAIN_CELLS = "early-transformer-rot", "full-rate-race64-rot"
NAMES = {TRANSFORMER: "Transformer", CELLS: "Spike reads and race cells", READS: "Spike reads",
         HYBRID: "Soft reads (hybrid)", PLAIN_CELLS: "Race cells without delays"}
COLORS = {TRANSFORMER: ORANGE, CELLS: GREEN, READS: VIOLET, HYBRID: YELLOW, PLAIN_CELLS: GRAY}
# Label, score record, checkpoint and colour of the rows of the test figure, the reference first.
TEST = (
    ("Transformer, AdamW", "test_v12_decoders", "checkpoints/scratch-study/transformer_s20001.pt", ORANGE),
    ("Transformer, rotated optimizer", "test_transformer_rotated",
     "checkpoints/state_reads/full-transformer-rot/model.pt.averaged", ORANGE),
    ("Spike reads and race cells on delayed fibres", "test_spike_reads_and_cells",
     "checkpoints/state_reads/pilot-race64-d8-c350-dl4/model.pt.averaged", GREEN),
    ("Spike reads", "test_spike_reads", "checkpoints/state_reads/full-rate-rot/model.pt.averaged", VIOLET),
    ("Soft reads (softmax hybrid), AdamW", "test_soft_reads", "checkpoints/state_reads/full-soft-d13w256/model.pt",
     YELLOW),
    ("v12 decoder with read heads", "test_v12_decoders", "checkpoints/v12-scratch/heads_s20001.pt", GRAY),
    ("v12 plain decoder", "test_v12_decoders", "checkpoints/scratch-study/clock_s20001.pt", GRAY),
)
COPYABLE, OTHER = "copyable (match >= 4)", "not copyable or match < 4"
KINDS = ("lower-case letter", "capital letter", "digit", "space", "newline", "punctuation", "markup sign",
         "byte of a longer character")
PLACES = ("0", "1", "2", "3-4", "5-7", "8+")
# Key, title, detail and the arms named in the panel as (arm, label) of every family of arms.
FAMILIES = (
    ("first", "Read designs", "AdamW · 12,000-update schedule",
     (("plain", "no reads"), ("transformer", "Transformer"), ("reads-d13w256", "soft reads"))),
    ("adamw", "Longer runs under AdamW", "90,000-update schedule",
     (("full-rate-b256-r4", "spike reads"), ("full-soft-d13w256", "soft reads"))),
    ("forks", "Forks of the soft-read run at 12,000", "step size, weight average, rotated optimizer",
     (("fork12k-avg-lr3e4", "AdamW, averaged"), ("fork12k-rot-lr3e4", "rotated optimizer"))),
    ("rotated", "Rotated optimizer, no cells", "with a running weight average",
     ((READS, "spike reads"), (HYBRID, "soft reads"))),
    ("threshold", "Threshold cells", "cells that fire past a threshold",
     (("full-rate-gran256-free-rot", "no price on spikes"),)),
    ("race", "Race cells", "a cluster shares a budget of spikes",
     (("pilot-race64-d8-c350", "8 blocks, 350 cells"), (PLAIN_CELLS, "10 blocks, 212 cells"))),
    ("delays", "Delayed fibres and what followed", "fibres that bring the last bytes",
     (("pilot-race64-d8-g7x200-dl4", "cells after every block"), (CELLS, "the release"))),
)
BEAT_ARMS = (("delays 0-3", "Delayed fibres, 0 to 3 bytes"), ("delays 0-7", "Delayed fibres, 0 to 7 bytes"),
             ("every-byte clock", "Beat fibres, a beat at every byte"), ("word clock, last", "Word clock, last count"),
             ("word clock, mean", "Word clock, mean count"), ("word clock, sum", "Word clock, summed count"),
             ("clock cells, last", "Learned clocks, last count"), ("clock cells, mean", "Learned clocks, mean count"),
             ("clock cells, sum", "Learned clocks, summed count"))
SOFT_FILL, GREEN_FILL, VIOLET_FILL = "#eef2f7", "#def3ec", "#e9e6fb"
# The arm that the settings of a family's arms are compared with, and what the family asked.
BASES = {"first": "reads-d13w256", "adamw": "full-soft-d13w256", "forks": "fork12k-avg-lr3e4", "rotated": READS,
         "threshold": "full-rate-gran256-rot", "race": PLAIN_CELLS, "delays": CELLS}
QUESTIONS = {
    "first": "Does a read of the stream's own earlier states help, in which shape, and can spikes carry it? "
             "Budget of 12,000 updates under AdamW, raw weights.",
    "adamw": "The best read designs on the 90,000-update schedule, still under AdamW.",
    "forks": "Why does a smaller step help so much in the middle of a run? Forks of the soft-read run at 12,000 "
             "updates (one of the spike-read run at 18,000) with other step sizes, a running weight average and the "
             "rotated optimizer. A fork's updates count from its start.",
    "rotated": "The Transformer and the read decoders from scratch under the rotated optimizer, and other shapes "
               "of the decoder at the same size.",
    "threshold": "Many cheap cells on sparse fixed wiring that fire past a threshold.",
    "race": "The cells of a cluster race for a fixed budget of spikes instead.",
    "delays": "Fibres that bring each channel as it was up to 3 bytes ago, and changes on top of them.",
}
# Settings that say nothing about a model or its training, and settings where 0 or an empty list means absent.
UNTOLD = {"data", "eval_batches", "eval_bs", "eval_every", "final_eval_batches", "save_every", "load_weights",
          "validation_only", "seed", "ctx", "bs", "training_mixture", "teacher_checkpoint"}
ABSENT_AT_ZERO = {"granule_relays", "granule_traces", "granule_beats", "lr_scale", "rotate_stacks"}
# Flags that were added during the study with the value the code had before them.
AS_BEFORE = {"read_address_rate": 1.0, "read_synapse_gain": 1.0, "adam_beta2": 0.99}
CURVE_STEPS = (3000, 6000, 9000, 12000, 24000, 42000, 60000, 90000)


def arms():
    return {arm["name"]: arm for arm in read_result(f"{STUDY}/arms.json")["arms"]}


def curve(arm, averaged=True):
    """Updates and validation loss of an arm: its running weight average where it kept one, else its last weights."""
    key = "averaged_val_bpb" if averaged and "averaged_val_bpb" in arm["curve"][-1] else "val_bpb"
    start = arm.get("initialized_from", {}).get("updates", 0)
    return [start + point["step"] for point in arm["curve"]], [point[key] for point in arm["curve"]]


def test_rows():
    """Label, bits per byte, difference to the first row with its interval, and colour of every scored checkpoint."""
    rows = []
    for label, record, checkpoint, color in TEST:
        scores = {row["checkpoint"]: row for row in read_result(f"{STUDY}/scores/{record}.json")["checkpoints"]}
        row = scores[checkpoint]
        rows.append((label, row["bpb"], row.get("bpb_over_first_whole_windows"), row.get("bpb_over_first_paired_95"),
                     color, row["parameters"]))
    return rows


def split_scores(weights="averaged", windows=512):
    """Bits per target of every snapshot in the split tables, as {(arm, updates, cells cut): {targets: bits}}."""
    scores = {}
    for table in read_result(f"{STUDY}/probes/splits.json")["tables"]:
        if table["validation_windows"] != windows:
            continue
        for index, model in enumerate(table["models"]):
            if model["weights"] != weights:
                continue
            key = (model["checkpoint"].split("/")[0], model["updates"], bool(model.get("cells_cut")))
            scores.setdefault(key, {}).update({row["targets"]: row["bpb"][index] for row in table["rows"]})
    return scores


def kind_rows(updates=84000):
    """Share, bits of the Transformer and gaps of the two decoders for every kind of target at one snapshot."""
    for table in read_result(f"{STUDY}/probes/splits.json")["tables"]:
        names = [(model["checkpoint"].split("/")[0], model["updates"]) for model in table["models"]]
        rows = {row["targets"]: row for row in table["rows"]}
        if names == [(TRANSFORMER, updates), (READS, updates), (CELLS, updates)] and KINDS[0] in rows:
            return rows
    raise ValueError(f"no split by kind at {updates} updates")


def context_sections():
    record = read_result(f"{STUDY}/probes/context_split.json")
    if [model["checkpoint"].split("/")[0] for model in record["models"]] != [TRANSFORMER, CELLS, READS]:
        raise ValueError("the context split must hold the Transformer, the cells and the reads, in this order")
    return {section["title"]: section["rows"] for section in record["sections"]}


def tall_heading(fig, title, subtitle):
    """The heading of plot_dendrite at its usual distances on a figure of another height."""
    height = fig.get_figheight()
    fig.text(0.035, 1 - 0.19 / height, title, size=21, weight="bold", color=TEXT, va="top")
    fig.text(0.036, 1 - 0.71 / height, subtitle, size=10.5, color=MUTED, va="top")


def end_label(ax, x, y, text, color, *, dx=7, dy=0, ha="left", va="center", weight="normal"):
    ax.annotate(text, (x, y), xytext=(dx, dy), textcoords="offset points", ha=ha, va=va, size=9, color=color,
                weight=weight)


def plot_test(output):
    rows = test_rows()
    reference = rows[0][1]
    fig, ax = plt.subplots(figsize=(11.6, 5.6))
    fig.subplots_adjust(left=0.335, right=0.965, bottom=0.2, top=0.765)
    heading(fig, "Spike reads and race cells against the Transformer",
            "enwik8 test · 4,999,999 targets · windows of 256 unscored and 256 scored bytes from an empty state · "
            "368.64M training targets each")
    style(ax, grid_axis="x")
    ax.axvline(reference, color=ORANGE, linewidth=1, linestyle=(0, (3, 3)), zorder=1)
    for index, (_, score, gap, _, color, _) in enumerate(rows):
        ax.plot([reference, score], [index, index], color=LIGHT, linewidth=2.2, zorder=1)
        ax.scatter(score, index, s=64, color=color, zorder=2, edgecolor="white", linewidth=1)
        ax.annotate(f"{score:.4f}", (max(score, reference), index), xytext=(10, 0), textcoords="offset points",
                    ha="left", va="center", size=9.5, color=TEXT, weight="bold")
        if gap is not None:
            ax.annotate(f"{gap:+.4f}".replace("-", "−"), (max(score, reference), index), xytext=(52, 0), textcoords="offset points",
                        ha="left", va="center", size=9, color=MUTED)
    ax.set_yticks(range(len(rows)), [label for label, *_ in rows])
    ax.tick_params(axis="y", labelcolor=TEXT, labelsize=10)
    ax.invert_yaxis()
    ax.set_ylim(len(rows) - 0.4, -0.6)
    ax.set_xlim(1.15, 1.72)
    ax.set_xlabel("Bits per byte", color=MUTED, labelpad=8)
    fig.text(0.035, 0.03, "Beside each score: the difference to the AdamW Transformer over whole windows (the dashed "
             "line). Rows of the rotated optimizer are scored at their running weight average.\nSoft reads weigh the "
             "cells with a softmax and are no spike circuit. 34.6M to 34.8M parameters. One seed per row.",
             color=MUTED, size=9, linespacing=1.5)
    save(fig, output, "event_lm_state_reads.png")


def plot_training(output):
    runs = arms()
    fig, (left, right) = plt.subplots(1, 2, figsize=(12.6, 5.4))
    fig.subplots_adjust(left=0.065, right=0.95, bottom=0.27, top=0.73, wspace=0.3)
    heading(fig, "The decoders lead early, then the Transformer pulls away",
            "Validation loss of the running weight average · 16 batches of 8 windows of 512 enwik8 validation bytes · "
            "rotated optimizer · one seed")
    steps, reference = curve(runs[TRANSFORMER])
    baseline = dict(zip(steps, reference, strict=True))
    for ax in (left, right):
        style(ax)
        ax.set_xlabel("Updates of 4,096 targets, in thousands", color=MUTED, labelpad=8)
        ax.set_xlim(0, 99)
        ax.set_xticks(range(0, 91, 15))
    for name in (HYBRID, READS, CELLS, TRANSFORMER):
        # The Transformer was evaluated every 3,000 updates, some decoders more often.
        points = [(step / 1000, value) for step, value in zip(*curve(runs[name]), strict=True) if step in baseline]
        thousands, values = zip(*points, strict=True)
        gaps = [value - baseline[round(1000 * step)] for step, value in points]
        stopped = "" if thousands[-1] == 90 else f" at {thousands[-1]:.0f}k"
        for ax, series, text in ((left, values, f"{values[-1]:.3f}{stopped}"), (right, gaps, f"{gaps[-1]:+.3f}{stopped}")):
            if ax is right and name == TRANSFORMER:
                continue
            ax.plot(thousands, series, color=COLORS[name], linewidth=2, zorder=2 if name == HYBRID else 3,
                    label=NAMES[name])
            ax.scatter(thousands[-1], series[-1], s=22, color=COLORS[name], zorder=4)
            above = bool(stopped) and ax is left
            end_label(ax, thousands[-1], series[-1], text, TEXT, dx=0 if above else 7, dy=8 if above else 0,
                      ha="center" if above else "left", va="bottom" if above else "center")
    left.set_ylabel("Bits per byte", color=MUTED, labelpad=8)
    left.set_title("Validation loss", fontsize=10, loc="left", color=TEXT)
    right.axhline(0, color=TEXT, linewidth=0.8)
    right.set_ylabel("Bits per byte above the Transformer", color=MUTED, labelpad=8)
    right.set_title("Loss minus the Transformer's at the same update", fontsize=10, loc="left", color=TEXT)
    right.annotate("decoder ahead", (30, -0.012), ha="left", va="top", size=9, color=MUTED)
    right.annotate("Transformer ahead", (30, 0.012), ha="left", va="bottom", size=9, color=MUTED)
    handles, names = left.get_legend_handles_labels()
    left.legend(handles[::-1], names[::-1], frameon=False, labelcolor=TEXT, fontsize=9, loc="upper center",
                bbox_to_anchor=(1.12, -0.2), ncol=4)
    fig.text(0.035, 0.035, "All four runs: 34.6M to 34.8M parameters, the same data order and schedule, evaluated here "
             "every 3,000 updates. The soft-read run was stopped at 66,000 updates.", color=MUTED, size=9)
    save(fig, output, "event_lm_state_reads_training.png")


def plot_context(output):
    sections = context_sections()
    panels = (("All targets by the bytes in view", "All targets", "Bytes in view before the target"),
              ("First letter of a word by the words that ended in view", "First letter of a word",
               "Whole words in view before it"))
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 5.4), width_ratios=(1.45, 1))
    fig.subplots_adjust(left=0.085, right=0.975, bottom=0.3, top=0.73, wspace=0.17)
    heading(fig, "The gap opens when earlier words come into view",
            "enwik8 validation · final averaged weights · loss minus the Transformer's on the same targets · "
            "bars span 1.96 standard errors over targets")
    for ax, (title, label, axis) in zip(axes, panels, strict=True):
        rows = sections[title]
        style(ax)
        places = range(len(rows))
        for column, name, shift in ((1, READS, -0.09), (0, CELLS, 0.09)):
            values = [row["gap"][column] for row in rows]
            errors = [1.96 * row["standard_error"][column] for row in rows]
            ax.errorbar([place + shift for place in places], values, yerr=errors, color=COLORS[name], linewidth=2,
                        marker="o", markersize=5, elinewidth=1, capsize=0, label=NAMES[name])
        ax.axhline(0, color=TEXT, linewidth=0.8)
        ticks = [row["targets"].split(" ")[-1].replace("-", "–") for row in rows]
        ax.set_xticks(places, ticks, rotation=40 if len(rows) > 12 else 0, ha="right" if len(rows) > 12 else "center",
                      rotation_mode="anchor")
        ax.tick_params(axis="x", labelsize=9)
        ax.set_xlabel(axis, color=MUTED, labelpad=6)
        ax.set_title(label, fontsize=10, loc="left", color=TEXT)
    axes[0].set_ylabel("Bits per byte above the Transformer", color=MUTED, labelpad=8)
    axes[0].legend(frameon=False, labelcolor=TEXT, fontsize=9, loc="upper left", bbox_to_anchor=(0.0, 1.0))
    fig.text(0.035, 0.035, "Windows start from an empty state, so a target with n bytes in view is the n-th byte of "
             "its window. A word is a run of letters. The two panels have their own scales.", color=MUTED, size=9)
    save(fig, output, "event_lm_state_reads_context.png")


def plot_split(output):
    scores = split_scores()
    series = ((TRANSFORMER, ((EARLY_TRANSFORMER, 3000), (EARLY_TRANSFORMER, 6000), (EARLY_TRANSFORMER, 12000),
                             (TRANSFORMER, 24000), (TRANSFORMER, 42000), (TRANSFORMER, 84000))),
              (PLAIN_CELLS, tuple((PLAIN_CELLS, updates) for updates in (3000, 6000, 12000, 24000))),
              (READS, tuple((READS, updates) for updates in (12000, 24000, 42000, 84000))),
              (CELLS, tuple((CELLS, updates) for updates in (12000, 24000, 42000, 84000))))
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 5.4))
    fig.subplots_adjust(left=0.065, right=0.975, bottom=0.29, top=0.73, wspace=0.17)
    heading(fig, "Copying comes early to the decoders. The rest is where they fall behind",
            "enwik8 validation · 512 windows of 512 bytes · running weight averages · "
            "bits per target of each kind over training")
    panels = ((COPYABLE, "Copyable targets · 15.6% of all"), (OTHER, "All other targets · 84.4% of all"))
    for ax, (targets, title) in zip(axes, panels, strict=True):
        style(ax)
        for name, points in series:
            updates = [step / 1000 for _, step in points]
            values = [scores[(arm, step, False)][targets] for arm, step in points]
            ax.plot(updates, values, color=COLORS[name], linewidth=2, marker="o", markersize=5, label=NAMES[name],
                    zorder=2 if name == PLAIN_CELLS else 3)
            if targets == OTHER and name != PLAIN_CELLS:
                end_label(ax, updates[-1], values[-1], f"{values[-1]:.3f}", TEXT, dx=8)
            elif name in (TRANSFORMER, CELLS):
                end_label(ax, updates[-1], values[-1], f"{values[-1]:.3f}", TEXT, dx=0,
                          dy=9 if name == TRANSFORMER else -10, ha="center")
            if name in (TRANSFORMER, PLAIN_CELLS):
                beside = name == TRANSFORMER
                end_label(ax, updates[0], values[0], f"{values[0]:.3f}", TEXT, dx=8 if beside else 0,
                          dy=(4 if targets == OTHER else 0) if beside else (-10 if targets == OTHER else 9),
                          ha="left" if beside else "center")
        ax.set_xscale("log")
        ax.set_xticks([3, 6, 12, 24, 42, 84], ["3", "6", "12", "24", "42", "84"])
        ax.minorticks_off()
        ax.set_xlim(2.6, 112 if targets == OTHER else 97)
        ax.set_xlabel("Updates, in thousands", color=MUTED, labelpad=6)
        ax.set_title(title, fontsize=10, loc="left", color=TEXT)
    axes[0].set_ylabel("Bits per target", color=MUTED, labelpad=8)
    axes[0].legend(frameon=False, labelcolor=TEXT, fontsize=9, loc="upper center", bbox_to_anchor=(1.08, -0.2), ncol=4)
    fig.text(0.035, 0.03, "A target is copyable when the 4 or more bytes before it occurred earlier in the window and "
             "the byte that followed them there is the target.\nThe Transformer's first three points are from a second "
             "run of it that kept early snapshots.", color=MUTED, size=9, linespacing=1.5)
    save(fig, output, "event_lm_state_reads_split.png")


def plot_kinds(output):
    rows = kind_rows()
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 5.4), width_ratios=(1.15, 1))
    fig.subplots_adjust(left=0.215, right=0.975, bottom=0.27, top=0.73, wspace=0.27)
    heading(fig, "Most of the gap is on letters, largest early in a word",
            "enwik8 validation · 512 windows of 512 bytes · 84,000 updates · running weight averages · "
            "loss minus the Transformer's")
    arms_ = ((READS, 0), (CELLS, 1))
    width = 0.36
    left, right = axes
    style(left, grid_axis="x")
    for name, column in arms_:
        places = [index + (column - 0.5) * width for index in range(len(KINDS))]
        values = [rows[kind]["gap_times_share"][column] for kind in KINDS]
        left.barh(places, values, width * 0.92, color=COLORS[name], label=NAMES[name])
        for place, value in zip(places, values, strict=True):
            left.annotate(f"{value:+.4f}", (max(value, 0), place), xytext=(4, 0), textcoords="offset points",
                          ha="left", va="center", size=8, color=TEXT)
    left.set_yticks(range(len(KINDS)), [f"{kind} · {100 * rows[kind]['share']:.1f}%" for kind in KINDS])
    left.tick_params(axis="y", labelcolor=TEXT)
    left.invert_yaxis()
    left.set_xlim(0, 0.09)
    left.set_xlabel("Bits per byte of the whole gap", color=MUTED, labelpad=6)
    left.set_title("The gap by kind of target", fontsize=10, loc="left", color=TEXT)
    style(right)
    for name, column in arms_:
        places = [index + (column - 0.5) * width for index in range(len(PLACES))]
        values = [rows[f"letter after {place} letters"]["gap"][column] for place in PLACES]
        right.bar(places, values, width * 0.92, color=COLORS[name], label=NAMES[name])
        for place, value in zip(places, values, strict=True):
            right.annotate(f"{value:.3f}", (place, value), xytext=(0, 3), textcoords="offset points", ha="center",
                           size=8, color=TEXT)
    right.set_xticks(range(len(PLACES)), [place.replace("-", "–") for place in PLACES])
    right.set_ylim(0, 0.17)
    right.set_xlabel("Letters of the word before the target", color=MUTED, labelpad=6)
    right.set_ylabel("Bits per letter above the Transformer", color=MUTED, labelpad=8)
    right.set_title("A letter's gap by its place in the word", fontsize=10, loc="left", color=TEXT)
    left.legend(frameon=False, labelcolor=TEXT, fontsize=9, loc="upper center", bbox_to_anchor=(0.72, -0.17), ncol=2)
    total = rows["all"]["gap_times_share"]
    fig.text(0.035, 0.03, f"The whole gap at this snapshot is {total[0]:+.4f} bits per byte for the spike reads and "
             f"{total[1]:+.4f} with the race cells. The left bars add up to it, apart from the few other bytes.\n"
             "Beside each kind: its share of all targets. A word is a run of letters.", color=MUTED, size=9,
             linespacing=1.5)
    save(fig, output, "event_lm_state_reads_kinds.png")


def family(arm):
    """The family of the study that an arm belongs to."""
    settings, parent = arm["settings"], arm.get("initialized_from")
    if parent:
        return "race" if parent["arm"] == PLAIN_CELLS else "forks"
    if arm["model"] == "snn" and settings.get("granule_after"):
        if settings.get("granule_budget") is None:
            return "threshold"
        return "delays" if (settings.get("granule_delays") or 1) > 1 else "race"
    if settings.get("rotated_adam"):
        return "rotated"
    return "first" if arm["schedule"]["steps"] == 12000 else "adamw"


def design_color(arm):
    """Orange Transformers, yellow soft reads, violet spike reads, green cells on delayed fibres, gray the rest."""
    settings = arm["settings"]
    if arm["model"] == "transformer":
        return ORANGE
    if settings.get("granule_after"):
        return GREEN if settings.get("granule_budget") and (settings.get("granule_delays") or 1) > 1 else GRAY
    if not settings.get("read_after") or settings.get("read_mode") == "age":
        return GRAY
    return YELLOW if settings["read_mode"] == "soft" else VIOLET


def plot_iterations(output):
    runs = arms()
    fig, axes = plt.subplots(2, 4, figsize=(15.4, 8.6), sharey=True)
    fig.subplots_adjust(left=0.05, right=0.985, bottom=0.115, top=0.815, wspace=0.07, hspace=0.42)
    tall_heading(fig, f"The night's {len(runs)} training arms",
            "Validation loss over updates · 16 batches of 8 windows of 512 enwik8 validation bytes · the running weight "
            "average where an arm kept one, else its last weights · one seed")
    references = {name: curve(runs[name]) for name in (TRANSFORMER, CELLS)}
    parent = runs["full-soft-d13w256"]
    for ax, (key, title, detail, named) in zip(axes.flat, FAMILIES, strict=False):
        style(ax)
        members = [arm for arm in runs.values() if family(arm) == key]
        after_fork = key == "forks"
        if after_fork:
            # A fork is drawn over the updates since its start, beside the run it left.
            steps, values = curve(parent)
            ax.plot([(step - 12000) / 1000 for step in steps if step > 12000],
                    [value for step, value in zip(steps, values, strict=True) if step > 12000], color=YELLOW,
                    linewidth=1.3, linestyle=(0, (1, 1.6)), zorder=2)
        else:
            for name, (steps, values) in references.items():
                ax.plot([step / 1000 for step in steps], values, color=COLORS[name], linewidth=1.3,
                        linestyle=(0, (1, 1.6)), zorder=2)
        labels = {arm: (label, 0.93 - 0.09 * place) for place, (arm, label) in enumerate(named)}
        for arm in members:
            steps, values = curve(arm)
            if after_fork:
                steps = [step - arm["initialized_from"]["updates"] for step in steps]
            thousands = [step / 1000 for step in steps]
            bold = arm["name"] in labels
            ax.plot(thousands, values, color=design_color(arm), linewidth=2.3 if bold else 1.0, zorder=4 if bold else 3,
                    marker="o" if len(steps) == 1 else None, markersize=3)
            if bold:
                label, height = labels[arm["name"]]
                ax.scatter(thousands[-1], values[-1], s=16, color=design_color(arm), zorder=5)
                ax.annotate(f"{label} {values[-1]:.3f}", (thousands[-1], values[-1]), xytext=(0.985, height),
                            textcoords="axes fraction", ha="right", va="center", size=8.5, color=TEXT,
                            arrowprops={"arrowstyle": "-", "color": MUTED, "linewidth": 0.6, "shrinkA": 1,
                                        "shrinkB": 3})
        ax.set_xscale("log")
        ax.set_xlim(0.45, 100)
        ax.set_xticks([1, 3, 10, 30, 90], ["1", "3", "10", "30", "90"])
        ax.minorticks_off()
        ax.set_ylim(1.25, 2.2)
        ax.tick_params(labelsize=9)
        ax.set_title(f"{title}\n", fontsize=10.5, loc="left", color=TEXT, weight="bold", pad=2)
        ax.annotate(f"{detail} · {len(members)} arms", (0, 1), xycoords="axes fraction", xytext=(0, 5),
                    textcoords="offset points", size=8.5, color=MUTED, va="bottom")
        ax.set_xlabel("Updates after the fork, in thousands" if after_fork else "Updates, in thousands", color=MUTED,
                      labelpad=4, size=9)
    for ax in axes[:, 0]:
        ax.set_ylabel("Validation bits per byte", color=MUTED, labelpad=6, size=9)
    key = axes.flat[len(FAMILIES)]
    key.axis("off")
    rows = ((ORANGE, "-", "Transformer"), (YELLOW, "-", "Soft reads (softmax hybrid)"), (VIOLET, "-", "Spike reads"),
            (GREEN, "-", "Race cells on delayed fibres"), (GRAY, "-", "No reads, or cells without delays"),
            (ORANGE, ":", "Reference: Transformer, rotated optimizer"), (GREEN, ":", "Reference: the release decoder"),
            (YELLOW, ":", "Forks panel: the run the forks left"))
    for index, (color, line, label) in enumerate(rows):
        y = 0.97 - 0.088 * index - (0.05 if index > 4 else 0)
        key.plot([0.02, 0.16], [y, y], color=color, linewidth=1.3 if line == ":" else 2.3,
                 linestyle=(0, (1, 1.6)) if line == ":" else "-", transform=key.transAxes, clip_on=False)
        key.text(0.21, y, label, transform=key.transAxes, va="center", size=9, color=TEXT)
    key.text(0.02, 0.0, "Thick lines are the arms named in a panel,\nwith their last validation loss.",
             transform=key.transAxes, va="bottom", size=9, color=MUTED, linespacing=1.5)
    fig.text(0.035, 0.028, "Most arms were stopped as soon as they fell behind the best arm so far, so the lines differ "
             "in length. Every arm with its settings and its whole curve: results/event_lm/state_reads/arms.json.",
             color=MUTED, size=9)
    save(fig, output, "event_lm_state_reads_iterations.png")


def cell_records():
    record = read_result(f"{STUDY}/probes/cells.json")
    final = next(row for row in record["delays"] if row["checkpoint"].startswith(CELLS + "/") and row["updates"] == 90000)
    cuts = {}
    for row in record["ablations"]:
        cuts.setdefault(row["checkpoint"].split("/")[0], []).append((row["updates"], row["without_any_cells"]))
    # The race cells at 12,000 updates were cut in a split table.
    scores = split_scores()
    whole, cut = (scores[(PLAIN_CELLS, 12000, state)]["all"] for state in (False, True))
    cuts[PLAIN_CELLS].append((12000, cut - whole))
    return final, cuts


def plot_cells(output):
    final, cuts = cell_records()
    fig, axes = plt.subplots(1, 3, figsize=(13.6, 5.4), width_ratios=(0.8, 1, 1.15))
    fig.subplots_adjust(left=0.06, right=0.9, bottom=0.24, top=0.72, wspace=0.42)
    heading(fig, "What the race cells hear, and why they stay awake",
            "enwik8 validation · left and middle: the release decoder at its averaged weights, 128 windows · "
            "right: cost of cutting every cell's output synapses")
    left, middle, right = axes
    delays = sorted(final["without_fibres_of_delay"], key=int)
    style(left)
    costs = [final["without_fibres_of_delay"][delay] for delay in delays]
    left.bar(range(len(delays)), costs, 0.62, color=GREEN)
    for place, cost in enumerate(costs):
        left.annotate(f"+{cost:.2f}", (place, cost), xytext=(0, 4), textcoords="offset points", ha="center", size=9,
                      color=TEXT)
    left.set_xticks(range(len(delays)), ["now", "1", "2", "3"])
    left.set_xlabel("Bytes that the fibres delay", color=MUTED, labelpad=6)
    left.set_ylabel("Bits per byte lost", color=MUTED, labelpad=8)
    left.set_ylim(0, 1.85)
    left.set_title("Silencing the fibres of one delay", fontsize=10, loc="left", color=TEXT)
    style(middle)
    shades = ("#9fd9c5", "#57b89a", "#0f8f6c", "#0a5c46")
    layers = sorted(final["mean_ticks_by_delay"], key=int)
    for layer, shade in zip(layers, shades, strict=True):
        ticks = final["mean_ticks_by_delay"][layer]
        middle.plot(range(len(ticks)), ticks, color=shade, linewidth=2, marker="o", markersize=5)
        end_label(middle, len(ticks) - 1, ticks[-1], f"after block {layer}", TEXT)
    middle.set_xticks(range(len(delays)), ["now", "1", "2", "3"])
    middle.set_xlim(-0.25, 4.25)
    middle.set_xlabel("Bytes that the fibres delay", color=MUTED, labelpad=6)
    middle.set_ylabel("Mean synapse strength, weight ticks", color=MUTED, labelpad=8)
    middle.set_title("How strongly the cells listen", fontsize=10, loc="left", color=TEXT)
    style(right)
    threshold = ("full-rate-gran256-free-rot", "pilot-gran256-nobudget", "full-rate-gran256-rot", "pilot-gran256-rs")
    for name in threshold:
        updates, costs = zip(*sorted(cuts[name]), strict=True)
        right.plot([step / 1000 for step in updates], costs, color="#9aa7b8", linewidth=1.4, marker="o", markersize=6,
                   markerfacecolor="white", markeredgewidth=1.6)
    updates, costs = zip(*sorted(cuts[PLAIN_CELLS]), strict=True)
    right.plot([step / 1000 for step in updates], costs, color=COLORS[PLAIN_CELLS], linewidth=2.4, marker="o",
               markersize=6.5)
    end_label(right, updates[-1] / 1000, costs[-1], f"race cells\n+{costs[-1]:.2f}", TEXT)
    last = min(cuts["full-rate-gran256-rot"])
    end_label(right, last[0] / 1000, last[1], f"threshold cells\n+{last[1]:.3f}", TEXT)
    right.set_yscale("log")
    right.set_ylim(0.006, 2.2)
    right.set_yticks([0.01, 0.03, 0.1, 0.3, 1], ["0.01", "0.03", "0.1", "0.3", "1"])
    right.set_xscale("log")
    right.set_xticks([1.5, 3, 4.5, 12], ["1.5", "3", "4.5", "12"])
    right.minorticks_off()
    right.set_xlim(1.3, 15)
    right.set_xlabel("Updates, in thousands", color=MUTED, labelpad=6)
    right.set_ylabel("Bits per byte lost without the cells", color=MUTED, labelpad=8)
    right.set_title("Cells that training keeps using", fontsize=10, loc="left", color=TEXT)
    fig.text(0.035, 0.03, f"Left: the release decoder scores {final['bpb']:.4f} bits per byte on these windows. "
             "Right: four runs with threshold cells (open marks), which training silenced within 4,500 updates,\nand "
             "the first run with race cells (filled), whose fibres have no delays yet. Logarithmic scale on the right.",
             color=MUTED, size=9, linespacing=1.5)
    save(fig, output, "event_lm_state_reads_cells.png")


def beat_rows():
    """Bits per first letter of every arm for both seeds, and the two bounds of the made-up language."""
    records = [read_result(f"event_lm/beat_fibres/word_time_s{seed}.json") for seed in (0, 1)]
    bounds = {(record["first_letter_bits_without_the_word_before_last"],
               record["first_letter_bits_with_the_word_before_last"]) for record in records}
    if len(bounds) != 1:
        raise ValueError("the two seeds must share one language")
    scores = [{arm["arm"]: arm["first_letter_bits"] for arm in record["arms"]} for record in records]
    return [(label, [seed[arm] for seed in scores]) for arm, label in BEAT_ARMS], bounds.pop(), records[0]["settings"]


def plot_beats(output):
    rows, (without, with_word), settings = beat_rows()
    fig = plt.figure(figsize=(13.6, 6.0))
    tall_heading(fig, "Word time on a made-up language",
            f"A word depends on the word before last · bits per first letter of a word after {settings['steps']:,} "
            "updates · two seeds per arm · tested on a laptop CPU, not yet at full size")
    sketch = fig.add_axes([0.03, 0.13, 0.42, 0.61])
    sketch.set_xlim(0, 42)
    sketch.set_ylim(0, 30)
    sketch.axis("off")
    line = "the old man saw the ca"
    spans = ((0, 4), (4, 8), (8, 12), (12, 16), (16, 20), (20, 22))

    def strip(y, fills):
        for index, letter in enumerate(line):
            sketch.add_patch(plt.Rectangle((1 + 1.8 * index, y), 1.8, 2.6, facecolor=fills.get(index, "white"),
                                           edgecolor=GRID, linewidth=0.8))
            sketch.text(1.9 + 1.8 * index, y + 1.3, "·" if letter == " " else letter, ha="center", va="center",
                        size=10.5, color=TEXT, family="DejaVu Sans Mono")
        sketch.annotate("", (1 + 1.8 * len(line) + 2.3, y + 1.3), (1 + 1.8 * len(line) + 0.3, y + 1.3),
                        arrowprops={"arrowstyle": "-|>", "color": MUTED, "linewidth": 1})
        sketch.text(1 + 1.8 * len(line) + 2.6, y + 1.3, "?", ha="left", va="center", size=12, color=TEXT, weight="bold")

    sketch.text(1, 28.6, "Delayed fibres count bytes", size=10, color=TEXT, weight="bold", va="center")
    sketch.text(1, 26.6, "A cell hears the last four bytes side by side.", size=9, color=MUTED, va="center")
    strip(21.6, {index: GREEN_FILL for index in range(len(line) - 4, len(line))})
    for offset, delay in enumerate(("3", "2", "1", "now")):
        sketch.text(1.9 + 1.8 * (len(line) - 4 + offset), 20.7, delay, ha="center", va="top", size=8, color=MUTED)
    sketch.text(1, 14.6, "Beat fibres count beats of a clock", size=10, color=TEXT, weight="bold", va="center")
    sketch.text(1, 12.6, "A beat opens a stretch. The synapse keeps the stretch's summed\ncount, and at a beat the "
                "kept packets move one place down.", size=9, color=MUTED, va="center", linespacing=1.45)
    shades = {4: "#d9d6f5", 3: "#c3bdf0", 2: "#ada4ea", 1: "#978be5", 0: "#def3ec"}
    held = {}
    for age, (first, last) in enumerate(reversed(spans)):
        if age < 4:
            held.update({index: shades[age] for index in range(first, last)})
    strip(5.6, held)
    for age, (first, last) in enumerate(reversed(spans)):
        if age < 4:
            middle = 1 + 1.8 * (first + last) / 2
            sketch.plot([1.15 + 1.8 * first, 0.85 + 1.8 * last], [5.0, 5.0], color=VIOLET if age else GREEN, linewidth=2)
            sketch.text(middle, 4.2, "so far" if age == 0 else f"held {age}", ha="center", va="top", size=8,
                        color=MUTED)
        if first:
            sketch.plot([1 + 1.8 * first] * 2, [8.2, 9.4], color=TEXT, linewidth=1.2)
    sketch.text(1 + 1.8 * 4, 9.9, "beats of a word clock", ha="left", va="bottom", size=8, color=MUTED)
    ax = fig.add_axes([0.685, 0.2, 0.285, 0.52])
    style(ax, grid_axis="x")
    for index, (_, values) in enumerate(rows):
        ax.plot([min(values), max(values)], [index, index], color=LIGHT, linewidth=2.2, zorder=1)
        ax.scatter(values, [index] * len(values), s=46, color=GREEN if index >= 6 else GRAY, zorder=2,
                   edgecolor="white", linewidth=0.8)
        ax.annotate(" · ".join(f"{value:.2f}" for value in values), (max(values), index), xytext=(8, 0),
                    textcoords="offset points", ha="left", va="center", size=8.5, color=TEXT)
    for bound, label in ((without, "without the word\nbefore last"), (with_word, "with it")):
        ax.axvline(bound, color=TEXT, linewidth=0.9, linestyle=(0, (3, 3)), zorder=1)
        ax.annotate(f"{label}\n{bound:.2f}", (bound, -0.75), ha="center", va="bottom", size=8.5, color=MUTED,
                    linespacing=1.35)
    ax.set_yticks(range(len(rows)), [label for label, _ in rows])
    ax.tick_params(axis="y", labelcolor=TEXT, labelsize=9.5)
    ax.set_ylim(len(rows) - 0.5, -0.8)
    ax.set_xlim(2.35, 4.05)
    ax.set_xticks([2.5, 3.0, 3.5])
    ax.set_xlabel("Bits per first letter of a word", color=MUTED, labelpad=8)
    fig.text(0.035, 0.03, "24 made-up words, each chosen by the word before last. A single granule layer on a bare "
             "byte embedding. Green: every beat fibre has a learned clock cell of its own.\nBeside each row: seed 0 · "
             "seed 1. The full-size decoder with beat fibres has not been trained yet.", color=MUTED, size=9,
             linespacing=1.5)
    save(fig, output, "event_lm_beat_fibres.png")


def box(ax, x, y, width, height, label, *, face=SOFT_FILL, edge=GRAY, size=9, color=TEXT, weight="normal", rounding=0.6):
    ax.add_patch(FancyBboxPatch((x, y), width, height, boxstyle=f"round,pad=0,rounding_size={rounding}", facecolor=face,
                                edgecolor=edge, linewidth=1.2, zorder=3))
    if label:
        ax.text(x + width / 2, y + height / 2, label, ha="center", va="center", size=size, color=color, weight=weight,
                zorder=4, linespacing=1.3)


def arrow(ax, start, end, *, color=TEXT, width=1.2, bend=0.0, dashed=False, head=9):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=head, color=color, linewidth=width,
                                 connectionstyle=f"arc3,rad={bend}", linestyle=(0, (3, 2.4)) if dashed else "-",
                                 shrinkA=0, shrinkB=0, zorder=2))


def plot_circuit(output):
    fig = plt.figure(figsize=(14.2, 6.5))
    tall_heading(fig, "The decoder of this release",
                 "One stream of 1,408 channels of signed spike counts · 8 blocks · race cells after blocks 0, 1, 3 and "
                 "5 · state reads after blocks 2, 4 and 6 · 34,819,544 parameters")
    ax = fig.add_axes([0.02, 0.04, 0.96, 0.78])
    ax.set_xlim(0, 144)
    ax.set_ylim(0, 50)
    ax.axis("off")
    stream = 27
    ax.plot([8, 131], [stream, stream], color=TEXT, linewidth=3.4, solid_capstyle="round", zorder=1)
    box(ax, 0.5, stream - 3.2, 8.5, 6.4, "byte", size=10, weight="bold", face="white", edge=TEXT)
    ax.text(4.75, stream - 4.6, "its synapses\nstart the stream", ha="center", va="top", size=8, color=MUTED,
            linespacing=1.3)
    centers = [16 + 14 * index for index in range(8)]
    for index, center in enumerate(centers):
        clocked = index == 7
        box(ax, center - 3.3, stream - 3.6, 6.6, 7.2, f"block\n{index}", size=9.5, weight="bold",
            face="#fff3e0" if clocked else SOFT_FILL, edge=TEXT if clocked else GRAY)
    for index in (0, 1, 3, 5):
        middle = centers[index] + 7
        box(ax, middle - 5.2, stream + 9.5, 10.4, 6.4, "race cells\n16 × 350", size=9, face=GREEN_FILL, edge=GREEN)
        arrow(ax, (middle - 2.2, stream + 0.4), (middle - 2.2, stream + 9.4), color=GREEN)
        arrow(ax, (middle + 2.2, stream + 9.4), (middle + 2.2, stream + 0.5), color=GREEN)
    for index in (2, 4, 6):
        # The read after the last byte-rate block returns before the forecast neurons tap the stream.
        middle = centers[index] + 7 - (1.4 if index == 6 else 0)
        box(ax, middle - 5.2, stream - 15.9, 10.4, 6.4, "state read\n4 heads", size=9, face=VIOLET_FILL, edge=VIOLET)
        arrow(ax, (middle - 1.6, stream - 0.4), (middle - 1.6, stream - 9.4), color=VIOLET)
        arrow(ax, (middle + 1.6, stream - 9.4), (middle + 1.6, stream - 0.5), color=VIOLET)
        # Tags and payloads that the same place of the stream left at earlier bytes.
        arrow(ax, (middle - 15.5, stream - 12.7), (middle - 5.4, stream - 12.7), color=VIOLET, dashed=True)
    ax.text(centers[2] - 9, stream - 12.7, "what this place of the stream\nheld at the last 512 bytes", ha="right",
            va="center", size=8.5, color=MUTED, linespacing=1.35)
    ax.text(centers[0] + 7, stream + 17.3, "each cluster hears 256 channels\nas they were 0 to 3 bytes ago",
            ha="center", va="bottom", size=8.5, color=MUTED, linespacing=1.35)
    tap = centers[6] + 9.6
    box(ax, tap - 4.4, stream + 9.5, 10.8, 6.4, "forecast\nneurons", size=9, face="white", edge=TEXT)
    arrow(ax, (tap - 1.6, stream + 0.4), (tap - 1.6, stream + 9.4))
    arrow(ax, (centers[7], stream + 9.4), (centers[7], stream + 3.8), dashed=True)
    ax.text(tap + 7.2, stream + 12.7, "their surprise is the\nclock of the last block", ha="left", va="center",
            size=8.5, color=MUTED, linespacing=1.35)
    box(ax, 121.5, stream - 3.6, 9.2, 7.2, "prediction\nmemory", size=9, face="white", edge=TEXT)
    box(ax, 133.6, stream - 6.5, 9.9, 13, "256\nvocabulary\nneurons", size=9, weight="bold", face="white", edge=TEXT)
    arrow(ax, (130.8, stream), (133.5, stream), width=2)
    ax.text(0.5, 3.6, "Block: a population of integrate-and-fire neurons that reads the stream through delayed, "
            "compact synapses and adds a correction to it.\nRace cells and state reads add to the stream in the same "
            "way: through a firing population whose output synapses start at zero.", ha="left", va="center", size=9,
            color=MUTED, linespacing=1.5)
    save(fig, output, "event_lm_state_reads_circuit.png")


def plot_mechanisms(output):
    fig = plt.figure(figsize=(14.2, 7.2))
    tall_heading(fig, "Two races for a budget of spikes",
            "Left: a cluster of race cells on delayed fibres · right: a state read · in both, inhibition ends the race "
            "after a fixed number of spikes, so the spike counts are a softmax in whole spikes")
    left = fig.add_axes([0.02, 0.07, 0.47, 0.72])
    right = fig.add_axes([0.51, 0.07, 0.47, 0.72])
    for ax in (left, right):
        ax.set_xlim(0, 70)
        ax.set_ylim(0, 52)
        ax.axis("off")
    left.text(0, 50.5, "A cluster of race cells", size=11, weight="bold", color=TEXT, va="center")
    left.text(0, 47.6, "16 clusters in a layer, 350 cells in a cluster", size=9, color=MUTED, va="center")
    box(left, 1, 8, 6.5, 33, "256 of\nthe\n1,408\nstream\nchannels", size=8.5, face="white", edge=TEXT)
    delays = (("now", 36.5), ("1 byte ago", 28.5), ("2 bytes ago", 20.5), ("3 bytes ago", 12.5))
    for index, (label, y) in enumerate(delays):
        left.plot([7.6, 27], [y, y], color=GREEN, linewidth=1.6, zorder=2)
        if index:
            box(left, 12, y - 1.9, 6.4, 3.8, f"delay {index}", size=8, face=GREEN_FILL, edge=GREEN, rounding=0.4)
        left.text(27, y + 1.5, label, ha="right", va="bottom", size=8, color=MUTED)
    left.text(17.5, 43.3, "about 64 fibres of each delay", ha="center", va="center", size=8.5, color=MUTED)
    cells = [(38, 40 - 4.4 * index) for index in range(8)]
    spikes = (0, 49, 0, 0, 11, 0, 4, 0)
    for (x, y), count in zip(cells, spikes, strict=True):
        for _, source in delays:
            left.plot([27, x - 1.5], [source, y], color=GREEN if count else "#b9dccf", linewidth=0.7 if count else 0.5,
                      zorder=1, alpha=0.9)
        left.add_patch(plt.Circle((x, y), 1.5, facecolor=GREEN if count else "white", edgecolor=GREEN, linewidth=1.3,
                                  zorder=3))
        if count:
            left.plot([x + 1.6, 53.3], [y, y], color=GREEN, linewidth=0.6 + count / 14, zorder=2)
            left.text(x + 3.2, y + 0.9, f"{count} spikes", ha="left", va="bottom", size=8, color=TEXT)
    left.text(38, 4.5, "350 cells, each with its\nown synapses and its\nown resting charge", ha="center", va="center",
              size=8.5, color=MUTED, linespacing=1.35)
    left.text(38, 43.6, "⋮", ha="center", va="center", size=12, color=MUTED)
    box(left, 53.5, 8, 6.5, 33, "256\nother\nstream\nchannels", size=8.5, face="white", edge=TEXT)
    left.add_patch(plt.Circle((45, 46.4), 1.9, facecolor="white", edgecolor=TEXT, linewidth=1.3, zorder=3))
    left.text(45, 46.4, "–", ha="center", va="center", size=12, color=TEXT, zorder=4)
    arrow(left, (39.6, 41.2), (43.2, 45.6), color=TEXT, width=0.9, head=7)
    left.text(47.6, 46.4, "the pool's inhibition ends the\nrace after 64 spikes per byte", ha="left", va="center",
              size=8.5, color=MUTED, linespacing=1.35)
    left.text(61.5, 24.5, "the spikes\njoin the\nstream\nthrough a\nfiring\npopulation", ha="left", va="center", size=8.5,
              color=MUTED, linespacing=1.35)
    right.text(0, 50.5, "A state read", size=11, weight="bold", color=TEXT, va="center")
    right.text(0, 47.6, "4 heads, each a race over the last 512 bytes", size=9, color=MUTED, va="center")
    places = [5 + 6.6 * index for index in range(9)]
    charges = (0.4, 0.7, 4.6, 0.5, 1.6, 0.3, 0.9, 2.4)
    counts = (0, 0, 197, 0, 14, 0, 0, 45)
    right.plot([1, 66], [14, 14], color=TEXT, linewidth=3, solid_capstyle="round", zorder=1)
    right.text(1, 10.6, "the stream at one place between two blocks, byte after byte", ha="left", va="center", size=8.5,
               color=MUTED)
    for index, x in enumerate(places):
        present = index == len(places) - 1
        box(right, x - 2.2, 12, 4.4, 4, "now" if present else "", size=8, face="white" if present else SOFT_FILL,
            edge=TEXT if present else GRAY, rounding=0.4)
        if present:
            continue
        box(right, x - 2.2, 19, 2.1, 2.6, "", face=VIOLET_FILL, edge=VIOLET, rounding=0.3)
        box(right, x + 0.1, 19, 2.1, 2.6, "", face="white", edge=VIOLET, rounding=0.3)
        right.plot([x, x], [16.1, 18.9], color=VIOLET, linewidth=0.8, zorder=1)
        right.add_patch(plt.Rectangle((x - 1.6, 26), 3.2, 2.1 * charges[index], facecolor=VIOLET if counts[index] else
                                      "#cfc9f3", edgecolor="none", zorder=2))
        if counts[index]:
            right.text(x, 26.5 + 2.1 * charges[index], f"{counts[index]}", ha="center", va="bottom", size=8.5,
                       color=TEXT)
    right.text(places[0] - 2.2, 23.3, "every earlier byte left a tag (filled) and a payload", ha="left", va="center",
               size=8.5, color=MUTED)
    arrow(right, (places[-1], 16.2), (places[-1], 43.6), color=VIOLET)
    arrow(right, (places[-1], 43.6), (places[0] - 2.2, 43.6), color=VIOLET, dashed=True)
    right.text(places[-1] + 1.2, 45.2, "the present byte sends a probe", ha="right", va="bottom", size=8.5,
               color=MUTED)
    right.text(places[0] - 2.2, 40.4, "a cell's charge: probe pulses that meet its tag's pulses, plus an offset for "
               "its age", ha="left", va="center", size=8.5, color=MUTED)
    right.text(places[2] + 3, 33.5, "its spikes, of a budget of 256", ha="left", va="center", size=8.5, color=MUTED)
    box(right, 49, 1.2, 17, 5.6, "read neurons count the\nmean payload in pulses", size=8.5, face=VIOLET_FILL,
        edge=VIOLET)
    arrow(right, (places[-1], 6.9), (places[-1], 11.9), color=VIOLET)
    right.text(47.5, 4, "payload pulses of the cells that fired", ha="right", va="center", size=8.5, color=MUTED)
    fig.text(0.035, 0.028, "The numbers of spikes are examples. A cell's rate is its softmax share of the budget, and "
             "evenly spread starting phases turn the rates into whole spikes.", color=MUTED, size=9)
    save(fig, output, "event_lm_state_reads_mechanisms.png")


def told(settings):
    """The settings of an arm that describe its model and training, with absent ones left out."""
    told = {key: value for key, value in settings.items()
            if key not in UNTOLD and value is not None and not (key in ABSENT_AT_ZERO and not value)
            and AS_BEFORE.get(key) != value}
    if told.get("read_mode") != "race":
        # Only the spike read has signed pulses.
        told.pop("read_signed", None)
    return told


def changes(arm, base):
    """The settings in which an arm differs from the base of its family, as trainer flags."""
    if arm["model"] == "transformer":
        settings = arm["settings"]
        optimizer = "the rotated optimizer" if settings.get("rotated_adam") else "AdamW"
        return (f"Transformer of {settings['depth']} layers and width {settings['dim']} under {optimizer}, "
                f"schedule of {arm['schedule']['steps']:,} updates")
    mine, theirs = ({**told(run["settings"]), "steps": run["schedule"]["steps"]} for run in (arm, base))
    different = [f"`{key.replace('_', '-')}={mine.get(key, 'unset')}`" for key in sorted(set(mine) | set(theirs))
                 if mine.get(key) != theirs.get(key)]
    return " ".join(different) or "the base of this family"


def prose(text):
    """A paragraph wrapped like the other pages of the repository."""
    return textwrap.fill(text, width=65, break_long_words=False, break_on_hyphens=False)


def arms_page():
    """docs/STATE_READS_ARMS.md: every arm of the night by family, with what it changed and where it ended."""
    record = read_result(f"{STUDY}/arms.json")
    runs = {arm["name"]: arm for arm in record["arms"]}
    lines = ["# Every training arm of the state-read study", "",
             prose(f"The night of 17 to 18 November 2026 trained {len(runs)} arms. Each row is one arm with one "
                   "seed. The last column is the validation loss of the running weight average where an arm kept "
                   "one. Validation is 16 batches of 8 windows of 512 enwik8 validation bytes, so two arms that "
                   "differ by less than about 0.03 bits per byte before 12,000 updates are not told apart. Most "
                   "arms were stopped as soon as they fell behind the best arm so far."),
             "",
             prose("The settings do not record the code. The spike read changed form twice in the first hours: "
                   "arms named `race`, `race2` and `race3` ran the three forms in turn, and only the third is in "
                   "the code of this release. Flags were added during the night, and an arm that started before a "
                   "flag existed has no value for it. So `read-signed=True` on a spike-read arm means that its "
                   "code had the flag, and the first arm `race-d13w256` ran with positive pulses only. A cell arm "
                   "without `granule-price` ran before that flag existed and had a price on its cells' own "
                   "spikes. `fork12k-avg-save` has the settings of `fork12k-avg-lr3e4` and was started to keep "
                   "both the last and the averaged weights."),
             "[Study page](STATE_READS.md) ·",
             "[All settings and curves](../results/event_lm/state_reads/arms.json)", "",
             "![Validation loss of every arm by family](../results/figures/event_lm_state_reads_iterations.png)", ""]
    for key, title, _, _ in FAMILIES:
        members = [arm for arm in runs.values() if family(arm) == key]
        base = runs[BASES[key]]
        lines += [f"## {title}", "", prose(QUESTIONS[key]), "",
                  prose(f"Settings are given as the trainer flags that differ from `{base['name']}`."), "",
                  "| Arm | Started, UTC | Parameters | Differs in | Updates | Last weights | Averaged |",
                  "|---|---|---:|---|---:|---:|---:|"]
        for arm in members:
            last = arm["curve"][-1]
            parent = arm.get("initialized_from")
            origin = f" From `{parent['arm']}` at {parent['updates']:,}." if parent else ""
            averaged = f"{last['averaged_val_bpb']:.4f}" if "averaged_val_bpb" in last else ""
            lines.append(f"| `{arm['name']}` | {arm['started_utc'][8:10]} {arm['started_utc'][11:16]} | "
                         f"{arm['parameters']:,} | {changes(arm, base)}.{origin} | {last['step']:,} | "
                         f"{last['val_bpb']:.4f} | {averaged} |")
        lines.append("")
    stopped = record["stopped_before_the_first_evaluation"]
    lines += ["## Stopped before the first evaluation", "",
              prose("Three more arms were started and stopped before they wrote a validation loss: "
                    + ", ".join(f"`{arm['name']}`" for arm in stopped) + "."), ""]
    return "\n".join(lines)


def tables():
    print("test\n| Model | Parameters | enwik8 test | Over the AdamW Transformer | Paired 95% interval |\n"
          "|---|---:|---:|---:|---:|")
    for label, score, gap, interval, _, parameters in test_rows():
        over = "" if gap is None else f"{gap:+.4f}"
        span = "" if interval is None else f"{interval[0]:+.4f} to {interval[1]:+.4f}"
        print(f"| {label} | {parameters:,} | {score:.4f} | {over} | {span} |")
    runs = arms()
    print("\naveraged validation\n| Updates | " + " | ".join(NAMES[name] for name in (TRANSFORMER, HYBRID, READS, CELLS))
          + " |\n|---:|---:|---:|---:|---:|")
    curves = {name: dict(zip(*curve(runs[name]), strict=True)) for name in (TRANSFORMER, HYBRID, READS, CELLS)}
    for step in CURVE_STEPS:
        print(f"| {step:,} | " + " | ".join(f"{curves[name][step]:.4f}" if step in curves[name] else ""
                                            for name in (TRANSFORMER, HYBRID, READS, CELLS)) + " |")
    scores = split_scores()
    columns = ((TRANSFORMER, EARLY_TRANSFORMER), (PLAIN_CELLS,), (READS,), (CELLS,))
    for targets in ("all", COPYABLE, OTHER):
        print(f"\nsplit: {targets}\n| Updates | Transformer | Race cells without delays | Spike reads | "
              "Spike reads and race cells |\n|---:|---:|---:|---:|---:|")
        for step in (3000, 6000, 12000, 24000, 42000, 84000):
            cells = []
            for names in columns:
                found = [scores[(name, step, False)][targets] for name in names
                         if targets in scores.get((name, step, False), ())]
                cells.append(f"{found[0]:.4f}" if found else "")
            print(f"| {step:,} | " + " | ".join(cells) + " |")
    rows = kind_rows()
    print("\nkinds at 84,000\n| Target | Share | Transformer | Spike reads, gap | With race cells, gap | "
          "Of the whole gap |\n|---|---:|---:|---:|---:|---:|")
    for kind in ("all", *KINDS, *(f"letter after {place} letters" for place in PLACES)):
        row = rows[kind]
        gaps = row.get("gap") or row["gap_times_share"]
        print(f"| {kind} | {100 * row['share']:.1f}% | {row['bpb'][0]:.4f} | {gaps[0]:+.4f} | {gaps[1]:+.4f} | "
              f"{row['gap_times_share'][1]:+.4f} |")
    for title, rows in context_sections().items():
        print(f"\ncontext: {title}\n| Targets | Count | Transformer | With race cells, gap | Spike reads, gap |\n"
              "|---|---:|---:|---:|---:|")
        for row in rows:
            print(f"| {row['targets']} | {row['count']:,} | {row['bpb'][0]:.4f} | "
                  f"{row['gap'][0]:+.4f} ±{row['standard_error'][0]:.4f} | "
                  f"{row['gap'][1]:+.4f} ±{row['standard_error'][1]:.4f} |")
    final, cuts = cell_records()
    print("\ncells\n| Delay | Cost of silencing | " + " | ".join(f"Ticks after block {layer}" for layer in
                                                                 sorted(final["mean_ticks_by_delay"], key=int))
          + " |\n|---:|---:|---:|---:|---:|---:|")
    for delay in sorted(final["without_fibres_of_delay"], key=int):
        print(f"| {delay} | {final['without_fibres_of_delay'][delay]:+.4f} | "
              + " | ".join(f"{final['mean_ticks_by_delay'][layer][int(delay)]:.2f}"
                           for layer in sorted(final["mean_ticks_by_delay"], key=int)) + " |")
    print("\ncuts\n| Arm | Updates | Cost of cutting all cells |\n|---|---:|---:|")
    for name, points in cuts.items():
        for updates, cost in sorted(points):
            print(f"| {name} | {updates:,} | {cost:+.4f} |")
    counts = read_result(f"{STUDY}/probes/count_tables.json")
    print("\ncount tables\n| Table | Group | Transformer | With race cells | Spike reads |\n|---|---|---:|---:|---:|")
    for row in counts["rows"]:
        print(f"| {row['table']} | {row['group']} | " + " | ".join(f"{value:.4f}" for value in row["bits_saved"]) + " |")
    grid = read_result(f"{STUDY}/probes/weight_grid.json")
    print("\nweight grid\n| Mean magnitude, ticks | Cost | Rounded to zero |\n|---:|---:|---:|")
    for row in grid["grids"]:
        print(f"| {row['mean_magnitude_ticks']} | {row['bpb_over_unrounded']:+.4f} | "
              f"{100 * row['share_rounded_to_zero']:.0f}% |")
    rows, (without, with_word), _ = beat_rows()
    print(f"\nbeat fibres: {without:.4f} without the word before last, {with_word:.4f} with it\n"
          "| Arm | Seed 0 | Seed 1 |\n|---|---:|---:|")
    for label, values in rows:
        print(f"| {label} | " + " | ".join(f"{value:.4f}" for value in values) + " |")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tables", action="store_true", help="print the tables of docs/STATE_READS.md instead")
    parser.add_argument("--arms", action="store_true", help="print docs/STATE_READS_ARMS.md instead")
    parser.add_argument("--only", nargs="+", help="names of the figures to draw, all by default")
    args = parser.parse_args()
    if args.tables or args.arms:
        tables() if args.tables else print(arms_page())
        return
    output = RESULTS / "figures"
    figures = {"test": plot_test, "training": plot_training, "context": plot_context, "split": plot_split,
               "kinds": plot_kinds, "iterations": plot_iterations, "cells": plot_cells, "beats": plot_beats,
               "circuit": plot_circuit, "mechanisms": plot_mechanisms}
    for name in args.only or figures:
        figures[name](output)


if __name__ == "__main__":
    main()
