"""Plot the paired-pulse recall comparison and the matched enwik8 runs."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from plot_dendrite import BLUE, GRAY, MUTED, ORANGE, PAPER, RESULTS, TEXT, VIOLET, heading, plt, save, style
from plot_temporal import SHARED_ARGS

import numpy as np
from matplotlib.ticker import MaxNLocator, PercentFormatter


RECALL_ARMS = (
    ("preserve", "Preserve phase", BLUE, "o", "-"),
    ("erase", "Erase after each probe", ORANGE, "s", "--"),
    ("clear_before_final", "Clear before final probe", VIOLET, "^", ":"),
)
LM_ARMS = {
    "temporal_pulse": ("Paired pulse, preserve", BLUE),
    "temporal_pulse_erase": ("Paired pulse, erase", ORANGE),
    "temporal_spike_reset": ("Spiking reset, 32 spikes", VIOLET),
    "transformer_local": ("Local Transformer", GRAY),
}
LM_VARIANTS_BY_TAG = {
    "pulse_3m": {"temporal_pulse", "temporal_pulse_erase"},
    "temporal_3m": {"temporal_spike_reset", "transformer_local"},
}


def plot_recall(path, output):
    report = json.loads(path.read_text(encoding="utf-8"))
    args = report["args"]
    runs = report["runs"]
    gaps = sorted(set(args["eval_gaps"]))
    fig, ax = plt.subplots(figsize=(9.4, 4.8), facecolor=PAPER)
    fig.subplots_adjust(left=0.10, right=0.968, bottom=0.285, top=0.734)
    heading(fig, "Paired-pulse memory across repeated reads",
            f"Final probe after overwrite · trained on gaps of 0–{args['train_max_gap']} tokens · "
            f"{len(runs)} seeds")
    style(ax)
    positions = np.arange(len(gaps))
    chance = report["chance_accuracy"]
    ax.axhline(chance, color=GRAY, lw=0.8, linestyle="-.", label=f"Chance ({chance:.1%})")

    for arm, label, color, marker, linestyle in RECALL_ARMS:
        evaluations = [run["arms"][arm]["evaluation"] for run in runs]
        if any([point["gap"] for point in points] != gaps for points in evaluations):
            raise ValueError(f"{path.name} has mismatched recall gaps for {arm}")
        accuracy = []
        for points in zip(*evaluations):
            scores = [point["scores"]["final_read"] for point in points]
            accuracy.append(sum(row["correct"] for row in scores) / sum(row["samples"] for row in scores))
        ax.plot(positions, accuracy, label=label, color=color, marker=marker,
                linestyle=linestyle, lw=1.9, markersize=5)

    ax.set_xticks(positions, [f"{gap:,}" for gap in gaps])
    ax.set_ylim(0, 1.05)
    ax.set_yticks(np.linspace(0, 1, 5))
    ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    ax.set_ylabel("Final probe accuracy", color=MUTED)
    ax.set_xlabel("Intervening tokens per gap", color=MUTED)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles[1:] + handles[:1], labels[1:] + labels[:1], loc="lower center",
               bbox_to_anchor=(0.5, 0.10), frameon=False, ncol=4, fontsize=9, labelcolor=TEXT)
    fig.text(0.036, 0.043, "Accuracies pool all evaluation episodes across seeds. "
             "Clear-before-final uses the preserve model with its memory cleared for the last probe.",
             size=8.5, color=MUTED)
    save(fig, output, "pulse_memory_recall.png")


def load_lm_runs(directory):
    groups = {variant: {} for variant in LM_ARMS}
    protocol = None
    for tag, variants in LM_VARIANTS_BY_TAG.items():
        for path in sorted(directory.glob(f"*_{tag}.json")):
            run = json.loads(path.read_text(encoding="utf-8"))
            cfg, args = run["cfg"], run["args"]
            variant = args["variant"]
            if variant not in variants or (variant == "temporal_spike_reset" and cfg["levels"] != 32):
                continue
            seed = args["seed"]
            if seed in groups[variant]:
                raise ValueError(f"{path.name} repeats seed {seed} for {variant}")
            settings = (*(args[name] for name in SHARED_ARGS), cfg["local_window"], run["data_sha1"])
            if protocol is not None and settings != protocol:
                raise ValueError(f"{path.name} has different data, training or evaluation settings")
            protocol = settings
            if not math.isfinite(run["test_bpb"]) or run["test_bpb"] < 0:
                raise ValueError(f"{path.name} has invalid test loss")
            groups[variant][seed] = run

    missing = [variant for variant, runs in groups.items() if not runs]
    if missing:
        raise ValueError(f"no runs in {directory} for {', '.join(missing)}")
    seeds = set(groups["transformer_local"])
    if any(set(runs) != seeds for runs in groups.values()):
        raise ValueError(f"{directory} does not contain matching seeds across the four language models")
    return groups


def plot_lm(directory, output):
    groups = load_lm_runs(directory)
    local = groups["transformer_local"]
    first = next(iter(local.values()))
    args = first["args"]
    train_bytes = args["bs"] * args["ctx"] * args["steps"]
    test_bytes = args["eval_bs"] * args["ctx"] * args["final_eval_batches"]
    fig, ax = plt.subplots(figsize=(9.4, 4.2), facecolor=PAPER)
    fig.subplots_adjust(left=0.224, right=0.884, bottom=0.24, top=0.723)
    heading(fig, "Paired-pulse models on enwik8",
            f"Context {args['ctx']} bytes · {train_bytes / 1e6:g}M training targets per model · "
            f"{test_bytes:,} sampled test targets")
    style(ax, grid_axis="x")
    scores_max = max(run["test_bpb"] for group in groups.values() for run in group.values())
    ax.set_xlim(0, math.ceil(scores_max + 0.15))
    ax.set_ylim(len(LM_ARMS) - 0.5, -0.5)
    ax.set_yticks(range(len(LM_ARMS)), [label for label, _ in LM_ARMS.values()])
    ax.xaxis.set_major_locator(MaxNLocator(5, integer=True))
    ax.set_xlabel("Final test bits per byte", color=MUTED)
    for row, (variant, (_, color)) in enumerate(LM_ARMS.items()):
        scores = np.array([groups[variant][seed]["test_bpb"] for seed in sorted(local)])
        offsets = np.linspace(-0.28, -0.03, len(scores))
        ax.scatter(scores, row + offsets, s=32, color=color, alpha=0.68,
                   edgecolors=PAPER, linewidths=0.6, zorder=3)
        ax.scatter(scores.mean(), row + 0.14, marker="D", s=62, color=color,
                   edgecolors=PAPER, linewidths=0.8, zorder=4)
        ax.annotate(f"{scores.mean():.3f}", (1.025, row), xycoords=("axes fraction", "data"),
                    ha="left", va="center", size=10, color=TEXT, annotation_clip=False)
    ax.annotate("Mean", (1.025, 1.075), xycoords="axes fraction", ha="left", size=9,
                color=MUTED, annotation_clip=False)
    fig.text(0.036, 0.052, f"Lower is better. Dots show {len(local)} matched seeds per model. "
             "Diamonds show seed means.", size=9, color=MUTED)
    save(fig, output, "pulse_memory_enwik8.png")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recall", type=Path, default=RESULTS / "pulse_recall" / "repeated_reads_v1.json")
    parser.add_argument("--lm-results", type=Path, default=RESULTS / "lm")
    parser.add_argument("--output-dir", type=Path, default=RESULTS / "figures")
    parser.add_argument("--figure", choices=("both", "recall", "lm"), default="both")
    args = parser.parse_args()
    try:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "text.color": TEXT})
        if args.figure in ("both", "recall"):
            plot_recall(args.recall, args.output_dir)
        if args.figure in ("both", "lm"):
            plot_lm(args.lm_results, args.output_dir)
    except (OSError, KeyError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
