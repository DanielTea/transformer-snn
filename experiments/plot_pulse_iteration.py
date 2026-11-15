"""Plot matched language-model losses and write-interference recall accuracy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from matplotlib.ticker import FuncFormatter, MaxNLocator, PercentFormatter

from plot_dendrite import BLUE, GRAY, MUTED, ORANGE, PAPER, RESULTS, TEXT, VIOLET, heading, plt, save, style
from plot_temporal import SHARED_ARGS


LM_VARIANTS = {
    "transformer_local": ("Local Transformer", GRAY),
    "temporal_spike_reset": ("Spiking decoder, no bank", VIOLET),
    "temporal_pulse": ("Two pulses, unrestricted", BLUE),
    "temporal_pulse_sparse": ("Two pulses, sparser writes", "#07867d"),
    "temporal_pulse_hold": ("Two pulses, protected writes", ORANGE),
    "temporal_pulse_timing": ("Four pulses, learned cuts", "#a33364"),
    "temporal_pulse_timing_fixed": ("Four pulses, fixed cuts", "#85601b"),
}
RECALL_ARMS = (
    ("unrestricted", "Two pulses, unrestricted", BLUE, "o", "-"),
    ("protected", "Two pulses, protected", ORANGE, "s", "--"),
    ("rate_matched", "Two pulses, rate matched", "#07867d", "^", ":"),
)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_language_results(training, evaluation, tag):
    groups = {variant: {} for variant in LM_VARIANTS}
    protocol = None
    for path in sorted(training.glob(f"*_{tag}.json")):
        run = read_json(path)
        args, cfg = run["args"], run["cfg"]
        variant, seed = args["variant"], args["seed"]
        if variant not in groups:
            continue
        current = (*(args[key] for key in SHARED_ARGS), cfg["local_window"], run["data_sha1"])
        if protocol is not None and current != protocol:
            raise ValueError(f"{path.name} has different data, training or evaluation settings")
        protocol = current
        if seed in groups[variant]:
            raise ValueError(f"{path.name} repeats seed {seed} for {variant}")
        groups[variant][seed] = (path.stem, run)

    seeds = sorted(set().union(*(group.keys() for group in groups.values())))
    if not seeds or any(sorted(group) != seeds for group in groups.values()):
        missing = [f"{variant} ({', '.join(map(str, group)) or 'none'})" for variant, group in groups.items()]
        raise ValueError(f"missing matching seeds in {training}: {', '.join(missing)}")

    eval_protocol = None
    for seed in seeds:
        path = evaluation / f"seed{seed}.json"
        data = read_json(path)
        metadata = {key: data[key] for key in (
            "data_sha1", "split", "history_bytes", "scored_bytes_per_window", "window_starts",
            "target_bytes", "prefix_bytes", "remote_min_distance_bytes", "remote_target_bytes",
            "phase_ablation_age_bytes",
        )}
        if metadata["data_sha1"] != protocol[-1] or metadata["split"] != "test":
            raise ValueError(f"{path.name} is not a held-out evaluation of the training corpus")
        if eval_protocol is not None and eval_protocol != metadata:
            raise ValueError(f"{path.name} scores different held-out bytes or uses a different intervention")
        eval_protocol = metadata
        models = {row["run_name"]: row for row in data["models"]}
        if len(models) != len(data["models"]):
            raise ValueError(f"{path.name} repeats a model")
        for variant, group in groups.items():
            name, run = group[seed]
            if name not in models:
                raise ValueError(f"{path.name} is missing {name}")
            row = models[name]
            if row["parameters"] != run["params"]:
                raise ValueError(f"{path.name} has a different parameter count for {name}")
            group[seed] = (run, row)
    return seeds, groups, eval_protocol


def plot_language(training, evaluation, tag, output):
    seeds, groups, protocol = load_language_results(training, evaluation, tag)
    local = groups["transformer_local"]
    args = local[seeds[0]][0]["args"]
    train_bytes = args["bs"] * args["ctx"] * args["steps"]
    window_targets = args["eval_bs"] * args["ctx"] * args["final_eval_batches"]
    fig, axes = plt.subplots(1, 4, figsize=(15.5, 5.45), sharey=True, facecolor=PAPER)
    fig.subplots_adjust(left=0.173, right=0.947, bottom=0.193, top=0.717, wspace=0.62)
    heading(fig, "Write protection and spike timing on enwik8",
            f"{train_bytes:,} training targets per model · {len(seeds)} matched seeds · "
            f"{args['ctx']:,}-byte training windows")

    # Keep each seed paired with its own reference in all panels.
    window = {name: np.array([group[seed][0]["test_bpb"] for seed in seeds]) for name, group in groups.items()}
    all_targets = {name: np.array([group[seed][1]["full"]["bpb"] for seed in seeds], dtype=float)
                   for name, group in groups.items()}
    remote = {name: np.array([group[seed][1]["full"]["remote_bpb"] for seed in seeds], dtype=float)
              for name, group in groups.items()}
    hidden = {}
    for variant, group in groups.items():
        interventions = [group[seed][1].get("old_phases_hidden_from_reads") for seed in seeds]
        if any(intervention is not None for intervention in interventions):
            if any(intervention is None for intervention in interventions):
                raise ValueError(f"{variant} is missing an old-phase intervention for at least one seed")
            hidden[variant] = (np.array([row["remote_bpb"] for row in interventions], dtype=float)
                               - remote[variant])
    if not hidden or protocol["phase_ablation_age_bytes"] is None:
        raise ValueError(f"{evaluation} has no old-phase intervention results")

    def paired(values):
        return {key: score - values["transformer_local"] for key, score in values.items()}

    window_deltas, all_deltas, remote_deltas = map(paired, (window, all_targets, remote))
    loss_points = np.concatenate([points for scores in (window_deltas, all_deltas, remote_deltas)
                                  for points in scores.values()])
    labels = [label for label, _ in LM_VARIANTS.values()]
    for ax, values, absolute, title, subtitle, xlabel in (
        (axes[0], window_deltas, window,
         "Windowed test", f"{window_targets:,} target bytes", "BPB vs. local Transformer"),
        (axes[1], all_deltas, all_targets,
         "Long-history test", f"{protocol['target_bytes']:,} target bytes", "BPB vs. local Transformer"),
        (axes[2], remote_deltas, remote,
         "Remote-prefix targets", f"{protocol['remote_target_bytes']:,} target bytes", "BPB vs. local Transformer"),
        (axes[3], hidden, None, "Hide old phases", f"Age ≥ {protocol['phase_ablation_age_bytes']:,} bytes",
         "Remote BPB change (×10⁻³)"),
    ):
        style(ax, grid_axis="x")
        ax.set_title(f"{title}\n{subtitle}", loc="left", color=TEXT, size=11, linespacing=1.6, pad=9)
        ax.axvline(0, color=GRAY, linestyle="--", linewidth=0.9, zorder=1)
        offsets = np.linspace(-0.20, 0.0, len(seeds)) if len(seeds) > 1 else np.zeros(1)
        axis_points = loss_points if absolute is not None else np.concatenate(list(values.values()))
        if not np.isfinite(axis_points).all():
            raise ValueError(f"nonfinite scores for {title}")
        left, right = min(axis_points.min(), 0), max(axis_points.max(), 0)
        pad = max(0.08 * (right - left), 0.0005)
        ax.set_xlim(left - pad, right + pad)
        ax.set_ylim(len(labels) - 0.5, -0.5)
        ax.set_yticks(range(len(labels)), labels)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
        if absolute is None:
            ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value * 1000:g}"))
        ax.set_xlabel(xlabel, color=MUTED, labelpad=7, fontsize=9.5)
        number_header = "Mean BPB" if absolute is not None else "Mean Δ"
        ax.annotate(number_header, (1.045, 1.035), xycoords="axes fraction", size=7.7,
                    color=MUTED, annotation_clip=False)
        for row, (variant, (_, color)) in enumerate(LM_VARIANTS.items()):
            if variant not in values:
                label = "—"
            else:
                points = values[variant]
                ax.scatter(points, row + offsets, s=26, color=color, alpha=0.64,
                           edgecolors=PAPER, linewidths=0.45, zorder=3)
                ax.scatter(points.mean(), row + 0.14, marker="D", s=43, color=color,
                           edgecolors=PAPER, linewidths=0.55, zorder=4)
                if absolute is not None:
                    label = f"{absolute[variant].mean():.4f}"
                else:
                    delta = round(1000 * float(points.mean()), 3)
                    label = f"{delta:+.3f}" if delta else "0.000"
            ax.annotate(label, (1.045, row), xycoords=("axes fraction", "data"),
                        va="center", color=TEXT, size=8.6, annotation_clip=False)

    fig.text(0.036, 0.074, "Dots show individual seeds. Diamonds show means. Lower loss is better. "
             f"The remote subset uses {protocol['prefix_bytes']}-byte prefixes last seen at least "
             f"{protocol['remote_min_distance_bytes']:,} bytes earlier.", color=MUTED, size=9)
    fig.text(0.036, 0.034, "The last panel subtracts intact-model loss from loss with old phases hidden. "
             "Its axis and mean values use 10⁻³ BPB.", color=MUTED, size=9)
    save(fig, output, "pulse_iteration_enwik8.png")


def plot_interference(path, timing, fixed, output):
    report = read_json(path)
    args, pooled = report["args"], report["aggregate"]
    gaps = sorted(set(args["eval_gaps"]))
    series = [(pooled[arm], label, color, marker, line) for arm, label, color, marker, line in RECALL_ARMS]
    for source, label, color, marker, line in (
        (timing, "Four pulses, learned cuts", "#a33364", "D", "-"),
        (fixed, "Four pulses, fixed cuts", "#85601b", "v", "--"),
    ):
        candidate = read_json(source)
        matched = ("eval_gaps", "eval_samples", "eval_seed", "train_max_gap", "steps", "bs", "seeds",
                   "keys", "classes", "dim", "cells", "spike_cap", "holds")
        if any(candidate["args"][key] != args[key] for key in matched) or candidate["chance_accuracy"] != report["chance_accuracy"]:
            raise ValueError(f"{source.name} uses different synthetic recall settings")
        series.append((candidate["aggregate"]["protected"], label, color, marker, line))

    fig, axes = plt.subplots(1, 2, figsize=(10.7, 5.4), sharey=True, facecolor=PAPER)
    fig.subplots_adjust(left=0.096, right=0.968, bottom=0.293, top=0.747, wspace=0.16)
    heading(fig, "Recall under competing writes",
            f"Final focal query · {len(report['runs'])} seeds · trained on gaps of 0–{args['train_max_gap']} tokens · "
            "four-pulse arms use write protection")
    rapid_scores = []
    for ax, competing, title in zip(axes, (True, False), ("With rival writes", "With neutral distractors")):
        style(ax)
        ax.set_title(title, loc="left", color=TEXT, weight="bold", size=11, pad=10)
        chance = report["chance_accuracy"]
        ax.axhline(chance, color=GRAY, linewidth=0.85, linestyle="-.", label=f"Chance ({chance:.0%})")
        for scores, label, color, marker, linestyle in series:
            points = sorted((row for row in scores if row["competing"] == competing), key=lambda row: row["gap"])
            if [row["gap"] for row in points] != gaps:
                raise ValueError(f"incomplete or repeated gap results for {label}")
            values = [row["scores"]["final"]["accuracy"] for row in points]
            ax.plot(range(len(gaps)), values, color=color, marker=marker, linestyle=linestyle,
                    linewidth=1.8, markersize=5.3, label=label)
            if competing:
                rapid = [row["scores"]["rapid_overwrite"] for row in points]
                rapid_scores.append(sum(row["correct"] for row in rapid) / sum(row["samples"] for row in rapid))

        ax.set_xticks(range(len(gaps)), [f"{gap:,}" for gap in gaps])
        ax.set_ylim(0, 1.05)
        ax.set_yticks(np.linspace(0, 1, 5))
        ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
        ax.set_xlabel("Intervening tokens (tested gaps)", color=MUTED, labelpad=5)
    axes[0].set_ylabel("Final query accuracy", color=MUTED)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles[1:] + handles[:1], labels[1:] + labels[:1], loc="lower center",
               bbox_to_anchor=(0.5, 0.069), frameon=False, ncol=3, fontsize=9, labelcolor=TEXT)
    fig.text(0.036, 0.03,
             f"Immediate overwrite accuracy with protection · {rapid_scores[1]:.1%} with two pulses · "
             f"{rapid_scores[3]:.1%} with learned cuts · {rapid_scores[4]:.1%} with fixed cuts.",
             color=MUTED, size=9)
    save(fig, output, "pulse_iteration_interference.png")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lm-results", type=Path, default=RESULTS / "lm")
    parser.add_argument("--lm-evaluation", type=Path, default=RESULTS / "pulse_iteration")
    parser.add_argument("--training-tag", default="pulse_iter_4m")
    parser.add_argument("--interference", type=Path, default=RESULTS / "pulse_recall" / "write_interference_v1.json")
    parser.add_argument("--timing-interference", type=Path,
                        default=RESULTS / "pulse_recall" / "write_interference_timing_v1.json")
    parser.add_argument("--fixed-interference", type=Path,
                        default=RESULTS / "pulse_recall" / "write_interference_timing_fixed_v1.json")
    parser.add_argument("--output-dir", type=Path, default=RESULTS / "figures")
    parser.add_argument("--figure", choices=("both", "lm", "interference"), default="both")
    args = parser.parse_args()
    try:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "text.color": TEXT})
        if args.figure in ("both", "lm"):
            plot_language(args.lm_results, args.lm_evaluation, args.training_tag, args.output_dir)
        if args.figure in ("both", "interference"):
            plot_interference(args.interference, args.timing_interference, args.fixed_interference, args.output_dir)
    except (OSError, KeyError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
