"""Figure of the full-size spiking stacks release by release against the Transformer they are read beside: each
stack's registered reading after 90,000 updates (a stream reaching 8,192 cells with taught output synapses) on enwik8
test and validation, the same stream without fast output synapses as a hollow mark, and the Transformer at its best
rate, frozen. A stack joins the line once its record has a result.

usage: python experiments/plot_full_size_line.py [--figure-dir DIR]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from plot_dendrite import MUTED, ORANGE, RESULTS, TEXT, plt, save, style

DARK_VIOLET = "#4a2a85"
RECORDS = RESULTS / "event_lm" / "knowledge_gap"
# (label, record, the run's key in the record's scores): the stacks in the order they were released.
STACKS = (("Plain stack\n(v19)", "full_size_loop_run.json", "fsp-st"),
          ("Looped stack\n(v20)", "full_size_loop_run.json", "fsp-loop"),
          ("Lifted loop\n(v21)", "full_size_lifted_loop_run.json", "fsp-lift"),
          ("Power-law cells\n(v22)", "full_size_lifted_loop_second.json", "fsp-lift2"))


def taught(scores, split):
    """A run's taught reading of a split, whatever its key names about the passes."""
    keys = [key for key in scores if key.startswith(f"taught {split}")]
    if len(keys) != 1:
        raise ValueError(f"expected one taught {split} reading, found {keys}")
    return scores[keys[0]]


def paired_loose(result, split):
    """The record's paired gap of the stream at 8,192 without fast output synapses to the frozen Transformer, if any."""
    for name, block in result.items():
        if name.startswith("(1)") and isinstance(block, dict):
            for key, value in block.items():
                if (key.startswith(f"reach {split}") or key.startswith(f"reach 8,192 {split}")) and "frozen" in key:
                    return value["gap"]
    return None


def line(records=RECORDS):
    """The stacks that have results, with their test and validation readings, and the Transformer's."""
    found, transformer = [], None
    for label, name, run in STACKS:
        path = Path(records) / name
        record = json.loads(path.read_text()) if path.exists() else {}
        scores = record.get("result", {}).get("scores", {})
        if run not in scores:
            continue
        found.append((label, {split: taught(scores[run], split) for split in ("test", "validation")},
                      {split: scores[run].get(f"reach 8,192 {split}") for split in ("test", "validation")},
                      {split: paired_loose(record["result"], split) for split in ("test", "validation")}))
        if "fsp-tf" in scores:
            tf = scores["fsp-tf"]
            transformer = {"test": tf["frozen test (windows of 2,048)"], "validation": tf["frozen validation"]}
    return found, transformer


def figure(found, transformer, output):
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 6.3), sharey=True)
    fig.subplots_adjust(left=0.075, right=0.975, bottom=0.13, top=0.70, wspace=0.07)
    best = min(scores["test"] for _, scores, _, _ in found)
    fig.text(0.035, 0.97, "Release by release, the spiking stack against the Transformer", size=19.5,
             weight="bold", color=TEXT, va="top")
    fig.text(0.036, 0.895, "Bits per byte on enwik8 after 90,000 updates from random weights at full size (the stacks "
             "36.3M parameters, the Transformer 34.8M), one seed a run.\nThe stacks read each split as 16 unbroken "
             "streams reaching 8,192 cells, with fast output synapses (filled: the registered reading)\nand without "
             "them (hollow: chosen after the scores were seen); the Transformer reads windows of 2,048 bytes. Lower is "
             f"better.\nOn test the best registered reading is {best - transformer['test']:+.4f} from the "
             "Transformer.", size=10, color=MUTED, va="top", linespacing=1.45)
    places = list(range(len(found)))
    for ax, split in zip(axes, ("test", "validation")):
        style(ax)
        ax.set_title(f"enwik8 {split}", loc="left", size=11.5, weight="bold", color=TEXT, pad=10)
        values = [scores[split] for _, scores, _, _ in found]
        loose_values = [value[split] for _, _, value, _ in found]
        ax.plot(places, values, color=DARK_VIOLET, linewidth=2.2, marker="o", markersize=8, zorder=3,
                markeredgecolor="white", markeredgewidth=1.5)
        ax.scatter([p for p, v in zip(places, loose_values) if v is not None], [v for v in loose_values if v is not None],
                   s=64, facecolors="white", edgecolors=DARK_VIOLET, linewidths=1.6, zorder=4)
        ax.axhline(transformer[split], color=ORANGE, linewidth=2.0, zorder=2)
        ax.annotate(f"Transformer  {transformer[split]:.4f}", (places[0] - 0.35, transformer[split]),
                    textcoords="offset points", xytext=(0, 6), ha="left", va="bottom", color=TEXT, size=10.5,
                    weight="bold")
        for place, (_, scores, loose, gap) in zip(places, found):
            if loose[split] is not None and loose[split] < scores[split] - 0.002:
                text = f"{loose[split]:.4f}" + ("" if gap[split] is None else f"\n{gap[split]:+.4f}")
                ax.annotate(text, (place, loose[split]), textcoords="offset points", xytext=(0, -12), ha="center",
                            va="top", color=TEXT, size=9.5)
        for place, value in zip(places, values):
            ax.annotate(f"{value:.4f}\n{value - transformer[split]:+.4f}", (place, value), textcoords="offset points",
                        xytext=(0, 12), ha="center", va="bottom", color=TEXT, size=9.5)
        ax.set_xlim(-0.5, len(found) - 0.2 + 0.6)
        ax.set_xticks(places, [label for label, _, _, _ in found])
        ax.tick_params(axis="x", labelsize=10, labelcolor=TEXT)
        ax.grid(axis="x", visible=False)
    low = min(transformer.values())
    low = min([low] + [v[split] for _, _, v, _ in found for split in ("test", "validation") if v[split] is not None])
    high = max(scores[split] for _, scores, _, _ in found for split in ("test", "validation"))
    axes[0].set_ylim(low - 0.016, high + 0.014)
    axes[0].set_ylabel("Bits per byte", color=MUTED, labelpad=8)
    save(fig, output, "event_lm_full_size_line.png")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--figure-dir", type=Path, default=RESULTS / "figures")
    args = parser.parse_args()
    found, transformer = line()
    figure(found, transformer, args.figure_dir)


if __name__ == "__main__":
    main()
