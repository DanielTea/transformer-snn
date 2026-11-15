"""Plot the cut controls and matched write-credit training results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from matplotlib.ticker import FuncFormatter, MaxNLocator, PercentFormatter

from plot_dendrite import BLUE, GRAY, MUTED, ORANGE, PAPER, RESULTS, TEXT, VIOLET, heading, plt, save, style


CUTS = (
    ("_fixed", "Fixed cuts", ORANGE, "v", "--"),
    ("_global", "Shared learned cuts", VIOLET, "s", ":"),
    ("", "Query-dependent cuts", BLUE, "D", "-"),
)
ARMS = {"ce": ("Cross-entropy only", GRAY), "paired": ("Paired credit", BLUE), "shuffled": ("Shuffled credit", ORANGE)}
SEEDS = (0, 1, 2)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_cut_scores(directory):
    reference = None
    output = []
    for suffix, label, color, marker, line in CUTS:
        path = directory / f"write_interference_timing{suffix}_v1.json"
        report = read_json(path)
        args = report["args"]
        settings = {key: value for key, value in args.items() if key not in ("fixed_cuts", "global_cuts")}
        protocol = (report["chance_accuracy"], report["write_holds"], settings)
        if reference is not None and protocol != reference:
            raise ValueError(f"{path.name} uses different synthetic data or training settings")
        reference = protocol
        args_modes = (bool(args.get("fixed_cuts")), bool(args.get("global_cuts")))
        if args_modes != (suffix == "_fixed", suffix == "_global") or args["read_bins"] != 4:
            raise ValueError(f"{path.name} has a different timing configuration")
        rows = sorted((row for row in report["aggregate"]["protected"] if row["competing"]), key=lambda row: row["gap"])
        if [row["gap"] for row in rows] != sorted(set(args["eval_gaps"])):
            raise ValueError(f"{path.name} has incomplete gap results")
        values = [row["scores"]["final"]["accuracy"] for row in rows]
        if not np.isfinite(values).all() or any(not 0 <= value <= 1 for value in values):
            raise ValueError(f"{path.name} has invalid accuracies")
        output.append((label, color, marker, line, values))
    return args, report["chance_accuracy"], output


def load_lm_scores(directory, tag, include_remote):
    window = {arm: [] for arm in ARMS}
    remote = {arm: [] for arm in ARMS}
    eval_protocol = None
    for seed in SEEDS:
        base, training = None, None
        if include_remote:
            path = directory / f"heldout_seed{seed}.json"
            evaluation = read_json(path)
            protocol = {key: value for key, value in evaluation.items() if key not in ("models", "device")}
            if eval_protocol is not None and protocol != eval_protocol:
                raise ValueError(f"{path.name} scores different held-out targets")
            eval_protocol = protocol
            rows = {row["run_name"]: row for row in evaluation["models"]}

        for arm in ARMS:
            path = directory / f"pulse_credit_{arm}_{tag}_s{seed}.json"
            run = read_json(path)
            args = run["args"]
            if run["arm"] != arm or args["seed"] != seed:
                raise ValueError(f"{path.name} has a different arm or seed")
            protocol = (run["data_sha1"], args)
            if training is not None and (training != protocol or base != run["base"]):
                raise ValueError(f"{path.name} uses a different checkpoint or training protocol for seed {seed}")
            base, training = run["base"], protocol
            window[arm].append(run["test_bpb"])
            if include_remote:
                if evaluation["data_sha1"] != run["data_sha1"] or evaluation["split"] != "test":
                    raise ValueError(f"heldout_seed{seed}.json is not a test result for the training corpus")
                remote[arm].append(rows[path.stem]["full"]["remote_bpb"])
    return window, remote, eval_protocol


def plot_accuracy(ax, args, chance, cases):
    style(ax)
    ax.set_title("Synthetic recall with competing writes", loc="left", size=11, weight="bold", pad=10)
    gaps = sorted(set(args["eval_gaps"]))
    ax.axhline(chance, color=GRAY, linewidth=0.8, linestyle="-.")
    ax.annotate(f"Chance {chance:.0%}", (0.01, chance), xycoords=("axes fraction", "data"),
                xytext=(0, -11), textcoords="offset points", color=MUTED, fontsize=8)
    for label, color, marker, line, accuracy in cases:
        ax.plot(range(len(gaps)), accuracy, color=color, marker=marker, linestyle=line,
                linewidth=1.8, markersize=5, label=label)
    ax.set_xticks(range(len(gaps)), [f"{gap:,}" for gap in gaps])
    ax.set_ylim(0, 1.04)
    ax.set_yticks(np.linspace(0, 1, 5))
    ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    ax.set_ylabel("Final query accuracy", color=MUTED)
    ax.set_xlabel("Intervening tokens (tested gaps)", color=MUTED)
    ax.legend(loc="upper right", frameon=False, labelcolor=TEXT, fontsize=9)


def plot_paired_loss(ax, title, values):
    style(ax, grid_axis="x")
    ax.set_title(title, loc="left", size=11, weight="bold", pad=10)
    raw = {arm: np.asarray(scores, dtype=float) for arm, scores in values.items()}
    delta = {arm: raw[arm] - raw["ce"] for arm in ARMS}
    points = np.concatenate(list(delta.values()))
    if not np.isfinite(points).all():
        raise ValueError(f"nonfinite loss in {title}")
    span = max(float(np.abs(points).max()) * 1.15, 0.00015)
    ax.set_xlim(-span, span)
    ax.axvline(0, color=GRAY, linestyle="--", linewidth=0.85, zorder=1)
    ax.set_ylim(len(ARMS) - 0.45, -0.45)
    ax.set_yticks(range(len(ARMS)), [label for label, _ in ARMS.values()])
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value * 1000:.4g}"))
    ax.set_xlabel("BPB vs. cross-entropy only (×10⁻³)", color=MUTED, fontsize=9)
    ax.annotate("Mean BPB", (1.025, 1.01), xycoords="axes fraction", size=8, color=MUTED, annotation_clip=False)
    for row, (arm, (_, color)) in enumerate(ARMS.items()):
        scores = delta[arm]
        ax.scatter(scores, row + np.linspace(-0.18, 0.01, len(scores)), s=29, color=color, alpha=0.63,
                   edgecolors=PAPER, linewidths=0.55, zorder=3)
        ax.scatter(scores.mean(), row + 0.13, s=53, color=color, marker="D",
                   edgecolors=PAPER, linewidths=0.6, zorder=4)
        ax.annotate(f"{raw[arm].mean():.5f}", (1.025, row), xycoords=("axes fraction", "data"),
                    va="center", size=9, color=TEXT, annotation_clip=False)


def plot(recall_dir, credit_dir, tag, include_remote, output):
    args, chance, cut_scores = load_cut_scores(recall_dir)
    window, remote, protocol = load_lm_scores(credit_dir, tag, include_remote)
    fig, axes = plt.subplots(1, 3 if include_remote else 2, figsize=(14.2 if include_remote else 10.0, 4.75),
                             facecolor=PAPER)
    fig.subplots_adjust(left=0.09, right=0.945 if include_remote else 0.918,
                        bottom=0.23, top=0.735, wspace=0.82 if include_remote else 0.56)
    heading(fig, "Spike timing and delayed write credit",
            "Four-pulse recall controls and matched write-gate training across three seeds")
    plot_accuracy(axes[0], args, chance, cut_scores)
    plot_paired_loss(axes[1], "Windowed enwik8 test", window)
    if include_remote:
        plot_paired_loss(axes[2], "Remote-prefix enwik8 targets", remote)
        span = max(ax.get_xlim()[1] for ax in axes[1:])
        for ax in axes[1:]:
            ax.set_xlim(-span, span)
    tail = f" The remote subset contains {protocol['remote_target_bytes']:,} target bytes." if include_remote else ""
    fig.text(0.036, 0.055, "Recall pools the protected-write episodes. Loss dots show matched seeds and diamonds show means. "
             "Lower loss is better." + tail, color=MUTED, size=9)
    save(fig, output, "pulse_write_credit.png")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recall-dir", type=Path, default=RESULTS / "pulse_recall")
    parser.add_argument("--credit-dir", type=Path, default=RESULTS / "pulse_credit")
    parser.add_argument("--tag", default="v1")
    parser.add_argument("--include-remote", action="store_true", help="also plot the three matched held-out reports")
    parser.add_argument("--output-dir", type=Path, default=RESULTS / "figures")
    args = parser.parse_args()
    try:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "text.color": TEXT})
        plot(args.recall_dir, args.credit_dir, args.tag, args.include_remote, args.output_dir)
    except (OSError, KeyError, TypeError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
