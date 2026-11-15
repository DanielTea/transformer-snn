"""Plot the matched enwik8 and PG-19 evaluation from its per-seed JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from plot_dendrite import BLUE, GRAY, MUTED, PAPER, RESULTS, TEXT, VIOLET, plt, save, style

import numpy as np
from matplotlib.ticker import MaxNLocator


MODELS = (
    ("local", "Local Transformer", GRAY),
    ("continuous", "Continuous gate", VIOLET),
    ("spike", "Spike gate", BLUE),
)
DATASETS = (("enwik8", "enwik8"), ("raw", "PG-19 excerpts"))
DISTANCES = (
    ("miss", "No hit"),
    ("local_only", "Local hits only"),
    ("remote", "Remote hit"),
)


def load_runs(directory):
    paths = [path for path in directory.glob("seed*.json") if path.stem[4:].isdigit()]
    runs = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    if not runs:
        raise ValueError(f"no seed<N>.json results in {directory}")
    runs.sort(key=lambda run: run["training"]["seed"])
    first = runs[0]
    if len({run["training"]["seed"] for run in runs}) != len(runs):
        raise ValueError("results contain a repeated seed")
    for run in runs:
        if any(run[key] != first[key] for key in ("remote_min_age", "cache_slots", "cache_orders")):
            raise ValueError("results use different cache settings or remote age thresholds")
        if {key: val for key, val in run["training"].items() if key != "seed"} != {
            key: val for key, val in first["training"].items() if key != "seed"
        }:
            raise ValueError("results use different training settings")
        for dataset, _ in DATASETS:
            if run["datasets"][dataset]["target_bytes_by_distance"] != (
                first["datasets"][dataset]["target_bytes_by_distance"]
            ):
                raise ValueError(f"{dataset} results use different evaluation strata")
    return runs


def values(runs, dataset, arm, distance=None):
    models = [run["datasets"][dataset]["models"][arm] for run in runs]
    if distance is None:
        return np.array([model["bpb"] for model in models])
    return np.array([
        np.nan if model["bpb_by_distance"][distance] is None else model["bpb_by_distance"][distance]
        for model in models
    ])


def draw_points(ax, score, row, color, spread):
    jitter = np.linspace(-spread, spread, len(score)) if len(score) > 1 else np.zeros(1)
    ax.scatter(score, row + jitter, s=32, color=color, edgecolors=PAPER, linewidths=0.65, alpha=0.66, zorder=3)
    ax.scatter(score.mean(), row, s=65, marker="D", color=color, edgecolors=PAPER, linewidths=0.8, zorder=4)


def format_axis(ax, labels, xlabel):
    style(ax, grid_axis="x")
    ax.set_yticks(np.arange(len(labels)), labels)
    ax.set_ylim(len(labels) - 0.5, -0.5)
    ax.xaxis.set_major_locator(MaxNLocator(5))
    ax.set_xlabel(xlabel, color=MUTED, labelpad=6)


def plot(runs, output):
    first = runs[0]
    fig, axes = plt.subplots(2, 2, figsize=(11.4, 6.35), facecolor=PAPER)
    fig.subplots_adjust(left=0.169, right=0.925, bottom=0.19, top=0.794, wspace=0.64, hspace=0.65)
    fig.text(0.036, 0.956, "Successor memory on held-out byte streams",
             size=20, weight="bold", color=TEXT, va="top")
    parameters = first["models"]["spike"]["parameters"]
    training = first["training"]
    trained_bytes = training["steps"] * training["bs"] * training["ctx"]
    seeds = "seed" if len(runs) == 1 else "seeds"
    fig.text(0.037, 0.894, f"{parameters:,} parameters in the successor models · "
             f"{trained_bytes:,} training byte targets per model · {len(runs)} paired {seeds}",
             size=10, color=MUTED, va="top")
    deltas = [
        values(runs, dataset, arm, distance) - values(runs, dataset, "local", distance)
        for dataset, _ in DATASETS for arm, _, _ in MODELS[1:] for distance, _ in DISTANCES
    ]
    data = np.concatenate([np.zeros(1), *deltas])
    width = max(np.nanmax(data) - np.nanmin(data), 0.03)

    for column, (dataset, label) in enumerate(DATASETS):
        overall, strata = axes[:, column]
        result = first["datasets"][dataset]
        format_axis(overall, [name for _, name, _ in MODELS], "Bits per byte")
        overall.set_title(f"{label} · {result['target_bytes']:,} target bytes",
                          loc="left", weight="bold", size=11, pad=9)
        for row, (arm, _, color) in enumerate(MODELS):
            score = values(runs, dataset, arm)
            draw_points(overall, score, row, color, spread=0.13)
            overall.annotate(f"{score.mean():.3f}", (1.025, row), xycoords=("axes fraction", "data"),
                             ha="left", va="center", color=TEXT, size=9, annotation_clip=False)
        overall.margins(x=0.12)

        counts = result["target_bytes_by_distance"]
        names = [f"{name} ({counts[distance] / result['target_bytes']:.0%})" for distance, name in DISTANCES]
        format_axis(strata, names, "Change from local Transformer (bits per byte)")
        strata.set_title("Loss change by cache evidence", loc="left", weight="bold", size=11, pad=9)
        strata.axvline(0, color=GRAY, lw=0.8, zorder=2)
        strata.set_xlim(np.nanmin(data) - 0.08 * width, np.nanmax(data) + 0.08 * width)
        for row, (distance, _) in enumerate(DISTANCES):
            if not counts[distance]:
                continue
            control = values(runs, dataset, "local", distance)
            for offset, (arm, _, color) in zip((-0.16, 0.16), MODELS[1:]):
                draw_points(strata, values(runs, dataset, arm, distance) - control,
                            row + offset, color, spread=0.07)
    for axis in axes.flatten():
        axis.title.set_color(TEXT)
    fig.text(0.036, 0.066, "Small dots show individual seeds. Diamonds show means. Negative loss changes are improvements.",
             size=9, color=MUTED)
    fig.text(0.036, 0.036, f"A remote hit means a candidate successor was observed at least "
             f"{first['remote_min_age']} bytes earlier. Percentages show each group's share of target bytes.",
             size=9, color=MUTED)
    save(fig, output, "dendrite_scaling.png")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=RESULTS / "lm_scaling")
    parser.add_argument("--output-dir", type=Path, default=RESULTS / "figures")
    args = parser.parse_args()
    try:
        runs = load_runs(args.results_dir)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "text.color": TEXT})
        plot(runs, args.output_dir)
    except (OSError, ValueError, KeyError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
