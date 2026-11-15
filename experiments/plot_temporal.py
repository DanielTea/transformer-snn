"""Plot validation curves and held-out test loss from the temporal enwik8 runs."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from plot_dendrite import BLUE, GRAY, MUTED, PAPER, RESULTS, TEXT, VIOLET, heading, plt, style

import numpy as np
from matplotlib.ticker import MaxNLocator


MODELS = (
    (("temporal_spike", 8), "Carry\n8 spikes", "Carry · max 8 spikes/rail/byte", BLUE, "-", "o"),
    (("temporal_spike_reset", 8), "Reset\n8 spikes", "Reset · max 8 spikes/rail/byte", BLUE, "--", "s"),
    (("temporal_spike", 32), "Carry\n32 spikes", "Carry · max 32 spikes/rail/byte", VIOLET, "-", "o"),
    (("temporal_spike_reset", 32), "Reset\n32 spikes", "Reset · max 32 spikes/rail/byte", VIOLET, "--", "s"),
    (("transformer_local", None), "Local\nTransformer", "Local Transformer", GRAY, "-", "^"),
)
SHARED_ARGS = (
    "dim", "depth", "heads", "ctx", "bs", "steps", "warmup", "lr", "wd",
    "eval_every", "eval_bs", "eval_batches", "final_eval_batches",
)


def load_runs(directory):
    groups = {model[0]: {} for model in MODELS}
    protocol = None
    for path in sorted(directory.glob("*_temporal_3m.json")):
        run = json.loads(path.read_text(encoding="utf-8"))
        args = run["args"]
        cfg = run["cfg"]
        variant = args["variant"]
        key = variant, None if variant == "transformer_local" else cfg["levels"]
        if key not in groups:
            raise ValueError(f"{path.name} has unexpected variant and spike cap {key}")
        seed = args["seed"]
        if seed in groups[key]:
            raise ValueError(f"{path.name} repeats seed {seed} for {key}")
        curve_steps = tuple(point["tokens"] for point in run["curve"])
        if not curve_steps:
            raise ValueError(f"{path.name} has no validation curve")
        run_protocol = (
            *(args[name] for name in SHARED_ARGS), cfg["local_window"], run["data_sha1"], curve_steps,
        )
        if protocol is not None and run_protocol != protocol:
            raise ValueError(f"{path.name} uses different data, training or evaluation settings")
        protocol = run_protocol
        groups[key][seed] = run
    if protocol is None:
        raise ValueError(f"no *_temporal_3m.json results in {directory}")
    return groups


def plot(groups, output):
    fig, (validation, test) = plt.subplots(1, 2, figsize=(11.3, 5.05), sharey=True, facecolor=PAPER)
    fig.subplots_adjust(left=0.074, right=0.976, bottom=0.33, top=0.743, wspace=0.13)
    first = next(run for group in groups.values() for run in group.values())
    args = first["args"]
    training_bytes = args["bs"] * args["ctx"] * args["steps"]
    test_bytes = args["eval_bs"] * args["ctx"] * args["final_eval_batches"]
    trained = np.array([point["tokens"] for point in first["curve"]]) / 1e6
    heading(fig, "Temporal spiking language modeling on enwik8",
            f"Context {args['ctx']} bytes · {training_bytes / 1e6:g}M training targets per model · "
            f"{test_bytes:,} sampled test targets")
    for axis, title in ((validation, "Validation during training"), (test, "Final test by configuration")):
        style(axis)
        axis.set_title(title, loc="left", color=TEXT, size=11, weight="bold", pad=10)

    labels = []
    top = 0.0
    for key, name, legend, color, linestyle, marker in MODELS:
        group = groups[key]
        if not group:
            continue
        runs = [group[seed] for seed in sorted(group)]
        curves = np.array([[point["val_bpb"] for point in run["curve"]] for run in runs])
        scores = np.array([run["test_bpb"] for run in runs])
        if not np.isfinite(curves).all() or not np.isfinite(scores).all():
            raise ValueError(f"{legend} has nonfinite loss")
        top = max(top, curves.max(), scores.max())
        if len(runs) > 1:
            validation.fill_between(trained, curves.min(axis=0), curves.max(axis=0), color=color, alpha=0.10)
        validation.plot(trained, curves.mean(axis=0), color=color, marker=marker,
                        linestyle=linestyle, label=legend, lw=1.7, markersize=4)

        column = len(labels)
        labels.append(f"{name}\n(n={len(runs)})")
        offsets = np.linspace(-0.23, -0.09, len(scores))
        test.scatter(column + offsets, scores, s=32, color=color,
                     edgecolors=PAPER, linewidths=0.65, alpha=0.7, zorder=3)
        test.scatter(column, scores.mean(), s=58, color=color, marker="D",
                     edgecolors=PAPER, linewidths=0.8, zorder=4)
        test.annotate(f"{scores.mean():.3f}", (column, scores.mean()),
                      xytext=(0, 10), textcoords="offset points", ha="center", color=TEXT, size=9)

    validation.set_ylim(0, math.ceil(top + 0.2))
    validation.yaxis.set_major_locator(MaxNLocator(5, integer=True))
    validation.set_ylabel("Bits per byte", color=MUTED)
    validation.set_xlabel("Sampled training targets (millions)", color=MUTED)
    validation.set_xticks(trained, [f"{point:g}" for point in trained])
    test.set_xticks(range(len(labels)), labels, fontsize=9)
    test.set_xlim(-0.55, len(labels) - 0.45)
    handles, names = validation.get_legend_handles_labels()
    fig.legend(handles, names, loc="lower center", bbox_to_anchor=(0.5, 0.101),
               frameon=False, ncol=3, fontsize=8.5, labelcolor=TEXT)
    fig.text(0.036, 0.047, "Lower is better. Lines and diamonds are seed means. "
             "Bands span the seed range. Test dots show individual seeds.", size=9, color=MUTED)
    fig.savefig(output, dpi=170, facecolor=PAPER, metadata={"Software": "Temporal enwik8 benchmark plots"})
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=RESULTS / "lm")
    parser.add_argument("--output", type=Path, default=RESULTS / "figures" / "temporal_enwik8.png")
    args = parser.parse_args()
    try:
        groups = load_runs(args.results_dir)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "text.color": TEXT})
        plot(groups, args.output)
    except (OSError, KeyError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
