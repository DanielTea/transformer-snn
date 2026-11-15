"""Build the Dendrite report figures from the checked-in benchmark JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402
from matplotlib.ticker import MultipleLocator, PercentFormatter  # noqa: E402
import numpy as np  # noqa: E402


RESULTS = Path(__file__).resolve().parents[1] / "results"
PAPER, TEXT, MUTED, GRID = "#ffffff", "#15243a", "#46566b", "#dce3ec"
BLUE, VIOLET, ORANGE, GRAY = "#0072b2", "#77519d", "#b64d00", "#687a92"


def read_result(path):
    return json.loads((RESULTS / path).read_text(encoding="utf-8"))


def style(ax, *, grid_axis="y"):
    ax.set_facecolor(PAPER)
    for name in ("top", "right", "left", "bottom"):
        ax.spines[name].set_visible(False)
    ax.tick_params(length=0, color=MUTED, labelcolor=MUTED, pad=7)
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.75)
    ax.set_axisbelow(True)


def heading(fig, title, subtitle):
    fig.text(0.035, 0.965, title, size=21, weight="bold", color=TEXT, va="top")
    fig.text(0.036, 0.868, subtitle, size=10.5, color=MUTED, va="top")


def save(fig, output, name):
    fig.savefig(output / name, dpi=170, facecolor=PAPER, metadata={"Software": "Dendrite benchmark plots"})
    plt.close(fig)


def recall_distance(output):
    gaps = [0, 4, 16, 64, 128]
    seed_colors = (
        ("Dendrite event", "dendrite_event_ilif4", BLUE, "o", "-"),
        ("RoPE Transformer", "transformer_ann0", ORANGE, "s", "--"),
    )
    fig, (matched, anchored) = plt.subplots(1, 2, figsize=(10.7, 4.1), width_ratios=(1.36, 1))
    fig.subplots_adjust(left=0.085, right=0.97, bottom=0.24, top=0.735, wspace=0.24)
    heading(fig, "Recall with intervening distractors", "Training gaps were 0–4 tokens in both evaluations.")
    for ax in (matched, anchored):
        style(ax)
        ax.set_ylim(0, 110)
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.yaxis.set_major_formatter(PercentFormatter(100, decimals=0))
        ax.set_xlabel("Distractors between key and value", color=MUTED, labelpad=9)
    matched.set_title("Matched test   /   3 seeds · context 524", loc="left", size=11, weight="bold", color=TEXT)
    matched.set_ylabel("Query accuracy", color=MUTED)
    x = np.arange(len(gaps))
    matched.axvspan(-0.3, 1.25, color="#eaf2f8", zorder=0)
    for label, file_prefix, color, marker, linestyle in seed_colors:
        paths = [f"recall_distance/{file_prefix}_T1_s{seed}_gap_v1_P4_G0-4.json" for seed in range(3)]
        runs = [read_result(path) for path in paths]
        scores = 100 * np.array([[run["final_acc_by_gap"][str(gap)] for gap in gaps] for run in runs])
        matched.fill_between(x, scores.min(axis=0), scores.max(axis=0), color=color, alpha=0.14)
        matched.plot(x, scores.mean(axis=0), color=color, marker=marker, linewidth=2, linestyle=linestyle, label=label)
        final_label = f"{scores.mean(axis=0)[-1]:.2f}%"
        matched.annotate(final_label, (x[-1], scores.mean(axis=0)[-1]), xytext=(-1, 10),
                         textcoords="offset points", ha="right", color=color, fontsize=9, weight="bold")
    matched.set_xticks(x, [f"{gap:,}" for gap in gaps])
    matched.legend(loc="center", frameon=False, labelcolor=TEXT, fontsize=9)

    result = read_result("recall_distance/dendrite_event_s0_anchored_extrapolation.json")
    points = sorted((int(gap), score) for gap, score in result["accuracy_by_gap"].items())
    x_anchored = np.arange(len(points))
    anchored.set_title("Anchored probe   /   1 seed · context 8,204", loc="left", size=11, weight="bold", color=TEXT)
    anchored.plot(x_anchored, [100 * score for _, score in points], linestyle="none",
                  marker="D", color=BLUE, markersize=6)
    anchored.set_xticks(x_anchored, [f"{gap:,}" for gap, _ in points])
    query_count = result["samples_per_gap"] * result["queries"]
    worst_correct = round(min(score for _, score in points) * query_count)
    qualifier = "at every tested gap" if worst_correct == query_count else "at the lowest scoring gap"
    anchored.text(0.5, 0.55, f"{worst_correct:,} / {query_count:,} correct\n{qualifier}",
                  ha="center", va="center", transform=anchored.transAxes, size=12, weight="bold", color=BLUE)
    fig.text(0.036, 0.07, "Shading marks training gaps. Bands show seed min–max. "
             "The anchored probe holds values and queries in place.", color=MUTED, size=9)
    save(fig, output, "dendrite_recall_distance.png")


def average_bpb(entries, key):
    return sum(entry[key] * entry["target_bytes"] for entry in entries) / sum(entry["target_bytes"] for entry in entries)


def successor_loss(output):
    frozen = read_result("lm_successor/scaled_v2_continuous.json")["results"]
    trained = read_result("lm_successor/end_to_end_continuous.json")["results"]
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.65), sharey=True)
    fig.subplots_adjust(left=0.088, right=0.966, bottom=0.25, top=0.755, wspace=0.2)
    heading(fig, "Next-byte loss on continuous streams", "Bits per byte, lower is better. Each panel uses the same held-out bytes across models.")
    groups = (
        ("enwik8 test  /  4,999,999 target bytes", frozen[:1], trained[:1]),
        ("PG-19  /  1,048,568 target bytes", frozen[1:], trained[1:]),
    )
    labels = ("Local\nTransformer", "Frozen backbone\n+ successor memory", "Trained from\nscratch + memory")
    for ax, (title, frozen_entries, trained_entries) in zip(axes, groups):
        style(ax)
        values = [
            average_bpb(frozen_entries, "base_bpb"),
            average_bpb(frozen_entries, "spike_bpb"),
            average_bpb(trained_entries, "spike_bpb"),
        ]
        bars = ax.bar(np.arange(3), values, color=[GRAY, VIOLET, BLUE], width=0.61)
        ax.bar_label(bars, fmt="%.3f", padding=5, color=TEXT, weight="bold", fontsize=11)
        ax.set_xticks(np.arange(3), labels, fontsize=9)
        ax.set_title(title, loc="left", size=11, weight="bold", color=TEXT, pad=10)
        ax.set_ylim(0, 4)
        ax.yaxis.set_major_locator(MultipleLocator(1))
    axes[0].set_ylabel("Bits per byte", color=MUTED)
    fig.text(0.036, 0.094, "The local control and the model trained from scratch used the same 4.096M training bytes and seed.", size=9, color=MUTED)
    fig.text(0.036, 0.052, "The readout on the frozen backbone was fitted separately on enwik8. PG-19 uses eight excerpts without fitting on the books.", size=9, color=MUTED)
    save(fig, output, "dendrite_successor_loss.png")


def successor_calibration(output):
    rows = (
        ("Local Transformer", "scaled_v2", "base_bpb", GRAY),
        ("Tuned fixed mixture", "scaled_v2", "fixed_bpb", GRAY),
        ("Uniform spike increments", "verified_v1", "spike_bpb", VIOLET),
        ("Suffix-scaled spike increments", "scaled_v2", "spike_bpb", BLUE),
        ("Continuous gate (scaled)", "scaled_v2", "analog_bpb", ORANGE),
    )
    fig, ax = plt.subplots(figsize=(9.7, 4.14))
    fig.subplots_adjust(left=0.26, right=0.925, bottom=0.218, top=0.755)
    heading(fig, "Calibrating successor votes", "Frozen local backbone · 524,288 enwik8 target bytes · context 4,096")
    style(ax, grid_axis="x")
    for y, (_, tag, key, color) in enumerate(rows):
        scores = [read_result(f"lm_successor/successor_s{seed}_{tag}.json")["test"][key] for seed in range(3)]
        ax.plot([min(scores), max(scores)], [y, y], color=color, lw=3.5, alpha=0.5, solid_capstyle="round")
        ax.scatter(mean(scores), y, color=color, s=48, edgecolors=PAPER, linewidths=0.7, zorder=3)
        ax.annotate(f"{mean(scores):.3f}", (max(scores), y), xytext=(11, 0),
                    textcoords="offset points", va="center", fontsize=10, color=TEXT)
    ax.set_yticks(np.arange(len(rows)), [row[0] for row in rows])
    ax.set_ylim(len(rows) - 0.5, -0.5)
    ax.set_xlim(2.063, 2.489)
    ax.xaxis.set_major_locator(MultipleLocator(0.1))
    ax.set_xlabel("Test bits per byte (lower is better)", color=MUTED, labelpad=9)
    fig.text(0.036, 0.055, "Dots show means. Lines show the range across three readout fits with the same backbone.", size=9, color=MUTED)
    save(fig, output, "dendrite_successor_calibration.png")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=RESULTS / "figures")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "text.color": TEXT})
    recall_distance(args.output_dir)
    successor_loss(args.output_dir)
    successor_calibration(args.output_dir)


if __name__ == "__main__":
    main()
