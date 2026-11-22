"""Figure of the learning-rate study (results/event_lm/knowledge_gap/learning_rate_with_data.json): the test bed's
spiking stack and its Transformer, each trained for 24,000 updates at five peak rates, scored on the first 2,000,000
validation bytes as 16 unbroken streams. Left, each model's loss against its peak rate (each seed and the mean);
right, the stack's gap to the Transformer at the same rate.

usage: python experiments/plot_learning_rate.py --scores FOLDER [--figure-dir DIR]
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

from plot_dendrite import GRAY, MUTED, ORANGE, RESULTS, TEXT, plt, save, style
from plot_state_reads import VIOLET

RATES = {"15": 1.5e-4, "": 3e-4, "6": 6e-4, "12": 1.2e-3, "24": 2.4e-3}
MODELS = {"bed24": ("Spiking stack (5.25M)", VIOLET), "tf24": ("Transformer (4.79M)", ORANGE)}


def collect(folder):
    """{model: {rate: {seed: bits per byte}}} from score records named like bed24-lr6-s1.json or tf24-s2.json."""
    points = {}
    for path in Path(folder).glob("*.json"):
        match = re.fullmatch(r"(bed24|tf24)(?:-lr(\d+))?-s(\d)", path.stem)
        if not match or (match.group(2) or "") not in RATES:
            continue
        record = json.loads(path.read_text())
        bpb = record["checkpoints"][0]["bpb"] if record.get("checkpoints") else record["fast_output_synapses"][0]["bpb"]
        points.setdefault(match.group(1), {}).setdefault(RATES[match.group(2) or ""], {})[int(match.group(3))] = bpb
    return points


def figure(points, output):
    fig, (losses, gaps) = plt.subplots(1, 2, figsize=(12.4, 4.8), width_ratios=(1.2, 1))
    fig.subplots_adjust(left=0.07, right=0.975, bottom=0.17, top=0.76, wspace=0.26)
    fig.text(0.035, 0.965, "Both models wanted a higher learning rate", size=20, weight="bold", color=TEXT, va="top")
    fig.text(0.036, 0.875, "Test bed, 24,000 updates (98.3M bytes), cosine schedule, scored on 2M validation bytes as 16 "
             "unbroken streams.\nOne line a seed (seed 2 dashed). The dotted line marks 3e-4, the rate of every "
             "earlier run.", size=10, color=MUTED, va="top")
    rates = sorted(RATES.values())
    for ax in (losses, gaps):
        style(ax, grid_axis="both")
        ax.set_xscale("log", base=2)
        ax.set_xticks(rates, [f"{rate:g}" for rate in rates])
        ax.set_xlabel("Peak learning rate", color=MUTED, labelpad=8)
        ax.axvline(3e-4, color=GRAY, linewidth=0.8, linestyle=":")
    styles = {1: "-", 2: "--", 3: "-.", 4: ":"}
    seeds = sorted({seed for curve in points.values() for rate in curve.values() for seed in rate})
    curves = {}
    for model, (label, color) in MODELS.items():
        for seed in seeds:
            xs = sorted(rate for rate, by_seed in points.get(model, {}).items() if seed in by_seed)
            if not xs:
                continue
            ys = [points[model][rate][seed] for rate in xs]
            curves[(model, seed)] = dict(zip(xs, ys))
            # A line breaks where a rate has not been run, so that it shows no trend nobody measured.
            gappy = [points[model].get(rate, {}).get(seed, np.nan) for rate in rates]
            losses.plot(rates, gappy, "o" + styles.get(seed, "-"), color=color, linewidth=2 if seed == 1 else 1.4,
                        markersize=5, alpha=1 if seed == 1 else 0.7, label=f"{label}, seed {seed}")
            best = min(curves[(model, seed)], key=curves[(model, seed)].get)
            if len(xs) == len(RATES):
                losses.annotate(f"best {best:g}", (best, curves[(model, seed)][best]), textcoords="offset points",
                                xytext=(0, -14), ha="center", color=color, size=9)
    losses.set_ylabel("Bits per byte", color=MUTED, labelpad=8)
    losses.legend(frameon=False, fontsize=8.5, labelcolor=TEXT, loc="upper right")
    means = {}
    for seed in seeds:
        stack, reference = curves.get(("bed24", seed), {}), curves.get(("tf24", seed), {})
        shared = sorted(set(stack) & set(reference))
        if not shared:
            continue
        gap = [stack[rate] - reference[rate] for rate in shared]
        means[f"gap seed {seed}"] = dict(zip(shared, gap))
        gappy = [stack[rate] - reference[rate] if rate in shared else np.nan for rate in rates]
        gaps.plot(rates, gappy, "o" + styles.get(seed, "-"), color=VIOLET, linewidth=2 if seed == 1 else 1.4,
                  markersize=6, alpha=1 if seed == 1 else 0.7, label=f"seed {seed}")
        if seed == 1:
            for rate, value in zip(shared, gap):
                gaps.annotate(f"{value:+.3f}", (rate, value), textcoords="offset points", xytext=(0, 9), ha="center",
                              color=TEXT, size=9)
    if len(means) > 1:
        gaps.legend(frameon=False, fontsize=8.5, labelcolor=TEXT, loc="upper left")
    means.update({f"{model} seed {seed}": curve for (model, seed), curve in curves.items()})
    gaps.set_ylabel("Stack minus Transformer, bits per byte", color=MUTED, labelpad=8)
    save(fig, output, "event_lm_learning_rate.png")
    return means


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--figure-dir", type=Path, default=RESULTS / "figures")
    args = parser.parse_args()
    means = figure(collect(args.scores), args.figure_dir)
    print(json.dumps({model: {f"{rate:g}": round(v, 5) for rate, v in curve.items()} for model, curve in means.items()},
                     indent=1))


if __name__ == "__main__":
    main()
