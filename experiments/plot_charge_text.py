"""Plot matched next-byte losses on enwik8 and verified PG-19 excerpts."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
from matplotlib.ticker import MaxNLocator

from plot_dendrite import BLUE, GRAY, MUTED, ORANGE, PAPER, RESULTS, TEXT, VIOLET, heading, plt, save, style
from plot_pulse_iteration import read_json
from plot_temporal import SHARED_ARGS


MODELS = {
    "transformer": ("Full-attention Transformer", TEXT),
    "transformer_local": ("Local Transformer", GRAY),
    "temporal_spike_reset": ("Spiking decoder, no bank", VIOLET),
    "temporal_charge": ("Spiking + conserved charge", BLUE),
    "temporal_charge_independent": ("Spiking + independent updates", ORANGE),
}
SEEDS = (0, 1, 2)
PG19_FIELDS = ("dataset", "split", "protocol", "manifest_sha256", "context_bytes", "window_stride_bytes",
               "unscored_leading_bytes_per_book", "max_bytes_per_book", "target_bytes")
BOOK_FIELDS = ("book_id", "excerpt_file", "excerpt_sha256", "source_start_raw_byte",
               "excerpt_bytes_used", "target_bytes", "windows")


def load_training(directory):
    groups = {name: {} for name in MODELS}
    protocol = None
    for path in sorted(directory.glob("*_charge_text_v1.json")):
        run = read_json(path)
        args = run["args"]
        variant, seed = args["variant"], args["seed"]
        if variant not in groups:
            continue
        current = (*(args[key] for key in SHARED_ARGS), run["cfg"]["local_window"], run["data_sha1"])
        if protocol is not None and protocol != current:
            raise ValueError(f"{path.name} has different training data, settings or evaluation windows")
        protocol = current
        if seed in groups[variant]:
            raise ValueError(f"{path.name} repeats seed {seed} for {variant}")
        groups[variant][seed] = (path.stem, run)
    for variant, runs in groups.items():
        if sorted(runs) != list(SEEDS):
            raise ValueError(f"{variant} needs seeds {SEEDS}. Found {sorted(runs)} in {directory}.")
    return groups


def load_pg19(directory, training):
    groups = {name: {} for name in MODELS}
    protocol = excerpts = None
    for seed in SEEDS:
        path = directory / f"pg19_seed{seed}.json"
        report = read_json(path)
        current = {key: report[key] for key in PG19_FIELDS}
        if (current["dataset"], current["split"], current["protocol"]) != ("PG-19", "test", "causal_reset_windows"):
            raise ValueError(f"{path.name} must use the held-out PG-19 reset-window protocol")
        if protocol is not None and current != protocol:
            raise ValueError(f"{path.name} uses different excerpts or reset windows")
        protocol = current
        names = {row["run_name"]: row for row in report["models"]}
        if len(names) != len(report["models"]):
            raise ValueError(f"{path.name} repeats a model")
        for variant, runs in training.items():
            name, train = runs[seed]
            row = names[name]
            if row["parameters"] != train["params"]:
                raise ValueError(f"{path.name} has a different parameter count for {name}")
            books = row["books"]
            signature = [tuple(book[key] for key in BOOK_FIELDS) for book in books]
            if excerpts is not None and signature != excerpts:
                raise ValueError(f"{path.name} scores different books or targets for {name}")
            excerpts = signature
            targets = sum(book["target_bytes"] for book in books)
            if not targets or targets != row["target_bytes"] or targets != protocol["target_bytes"]:
                raise ValueError(f"{path.name} has inconsistent target totals for {name}")
            bpb = math.fsum(book["nll_nats"] for book in books) / (targets * math.log(2))
            if not math.isclose(bpb, row["bpb"], rel_tol=1e-9):
                raise ValueError(f"{path.name} has inconsistent byte-weighted loss for {name}")
            groups[variant][seed] = bpb
    return groups, protocol, len(excerpts)


def draw_scores(ax, scores, title, subtitle, right):
    style(ax, grid_axis="x")
    ax.set_title(f"{title}\n{subtitle}", loc="left", color=TEXT, size=11, linespacing=1.65, pad=12)
    ax.set_xlim(0, right)
    ax.set_ylim(len(MODELS) - 0.5, -0.5)
    ax.set_yticks(range(len(MODELS)), [label for label, _ in MODELS.values()])
    ax.xaxis.set_major_locator(MaxNLocator(5, integer=True))
    ax.set_xlabel("Test bits per byte", color=MUTED)
    ax.annotate("Mean", (1.035, 1.03), xycoords="axes fraction", size=8.8, color=MUTED, annotation_clip=False)
    for row, (name, (_, color)) in enumerate(MODELS.items()):
        values = scores[name]
        offsets = np.linspace(-0.30, 0.06, len(values))
        ax.scatter(values, row + offsets, s=32, color=color, alpha=0.7,
                   edgecolors=PAPER, linewidths=0.65, zorder=3)
        ax.scatter(values.mean(), row + 0.27, marker="D", s=60, color=color,
                   edgecolors=PAPER, linewidths=0.8, zorder=4)
        ax.annotate(f"{values.mean():.3f}", (1.035, row), xycoords=("axes fraction", "data"),
                    ha="left", va="center", size=10, color=TEXT, annotation_clip=False)


def plot(training, pg19, output):
    groups = load_training(training)
    pg, protocol, num_books = load_pg19(pg19, groups)
    args = groups["transformer"][SEEDS[0]][1]["args"]
    training_targets = args["bs"] * args["ctx"] * args["steps"]
    enwik8_targets = args["eval_bs"] * args["ctx"] * args["final_eval_batches"]
    en = {name: np.array([group[seed][1]["test_bpb"] for seed in SEEDS]) for name, group in groups.items()}
    pg = {name: np.array([group[seed] for seed in SEEDS]) for name, group in pg.items()}
    all_values = np.concatenate([*en.values(), *pg.values()])
    if not np.isfinite(all_values).all() or all_values.min() < 0:
        raise ValueError("expected finite nonnegative test loss in both datasets")
    x_max = math.ceil(all_values.max() + 0.1)
    fig, (left, right) = plt.subplots(1, 2, figsize=(11.6, 4.7), sharey=True, facecolor=PAPER)
    fig.subplots_adjust(left=0.218, right=0.929, bottom=0.226, top=0.694, wspace=0.31)
    heading(fig, "Spiking charge memory on held-out text",
            f"{training_targets:,} enwik8 training targets per model · {args['ctx']:,}-byte training context · "
            f"{len(SEEDS)} matched seeds")
    draw_scores(left, en, "enwik8 test", f"{enwik8_targets:,} scored bytes · {args['ctx']:,}-byte windows", x_max)
    draw_scores(right, pg, "PG-19 transfer",
                f"{protocol['target_bytes']:,} scored bytes · {num_books} books", x_max)
    fig.text(0.036, 0.085, f"PG-19 resets state every {protocol['context_bytes']:,} bytes. "
             "All checkpoints score the same target bytes within each corpus.", size=9, color=MUTED)
    fig.text(0.036, 0.043, "Lower is better. Dots show seeds. Diamonds show means.", size=9, color=MUTED)
    save(fig, output, "charge_text_bpb.png")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-dir", type=Path, default=RESULTS / "lm")
    parser.add_argument("--pg19-dir", type=Path, default=RESULTS / "charge_text")
    parser.add_argument("--output-dir", type=Path, default=RESULTS / "figures")
    args = parser.parse_args()
    try:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "text.color": TEXT})
        plot(args.training_dir, args.pg19_dir, args.output_dir)
    except (OSError, KeyError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
