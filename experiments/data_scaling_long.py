"""The stack's data-scaling curves against the Transformer's (results/event_lm/knowledge_gap/data_scaling_long.json).

Reads the long-stream score of every run of the study from a folder of score records (one JSON per run, named as the
runs are), takes each model at its best peak rate at 24,000 updates, fits each model's loss = E + B * bytes^-beta,
and fits for each stack design the slope of its gap to the Transformer per doubling of the training bytes. A design
whose slope is below zero scales the other way around: its gap shrinks as both see more data. Writes the summary and
the figure.

usage: python experiments/data_scaling_long.py --scores FOLDER [--json OUT] [--figure-dir DIR]
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import numpy as np

from plot_dendrite import GRAY, MUTED, ORANGE, RESULTS, TEXT, plt, save, style
from plot_state_reads import GREEN, VIOLET

BYTES_PER_UPDATE = 16 * 256
# Runs of other studies that are this study's points: name pattern -> (model at a rate, thousands of updates).
REUSED = {r"tf24-s(\d)": ("tf3", 24), r"tf24-lr6-s(\d)": ("tf6", 24), r"bed24-s(\d)": ("st3", 24),
          r"bed24-lr6-s(\d)": ("st6", 24), r"pl24-s(\d)": ("pl3", 24), r"bed6-s(\d)": ("st3", 6)}
MODELS = {"tf": ("Transformer", ORANGE), "st": ("Spiking stack, delayed fibres, trained on streams", VIOLET),
          "pl": ("Spiking stack, plain, trained in windows", GREEN)}
RATES = {"3": 3e-4, "6": 6e-4}
# Two-sided 90% points of Student's t for 1 to 12 degrees of freedom.
T90 = (6.314, 2.920, 2.353, 2.132, 2.015, 1.943, 1.895, 1.860, 1.833, 1.812, 1.796, 1.782)


def bpb(record):
    if record.get("checkpoints"):
        return record["checkpoints"][0]["bpb"]
    return record["fast_output_synapses"][0]["bpb"]


def collect(folder):
    """{model at a rate: {thousands of updates: {seed: bits per byte}}} from the score records in ``folder``."""
    points = {}
    for path in sorted(Path(folder).glob("*.json")):
        name = path.stem
        match = re.fullmatch(r"ds-(tf[36]|st[36]|pl3)-(\d+)k-s(\d)", name)
        if match:
            tag, size, seed = match.group(1), int(match.group(2)), int(match.group(3))
        else:
            for pattern, (tag, size) in REUSED.items():
                reused = re.fullmatch(pattern, name)
                if reused:
                    seed = int(reused.group(1))
                    break
            else:
                continue
        points.setdefault(tag, {}).setdefault(size, {})[seed] = bpb(json.loads(path.read_text()))
    return points


def means(curve):
    sizes = sorted(curve)
    return np.array(sizes, dtype=float), np.array([np.mean(list(curve[size].values())) for size in sizes])


def best_rate(points, family):
    """The family's rate with the lower mean at 24,000 updates over the seeds both rates have, 3e-4 if none."""
    low, high = points.get(f"{family}3", {}).get(24, {}), points.get(f"{family}6", {}).get(24, {})
    seeds = sorted(set(low) & set(high))
    if not seeds:
        return "3"
    return "6" if np.mean([high[s] for s in seeds]) < np.mean([low[s] for s in seeds]) else "3"


def fit_power(updates, losses):
    """loss = E + B * bytes^-beta by least squares in log(loss - E) over a grid of floors E below the lowest loss."""
    if len(updates) < 3:
        return None
    logs = np.log(updates * 1000 * BYTES_PER_UPDATE)
    best = None
    for floor in np.linspace(0.0, losses.min() - 1e-3, 600):
        y = np.log(losses - floor)
        slope, intercept = np.polyfit(logs, y, 1)
        error = float(np.square(y - (slope * logs + intercept)).sum())
        if best is None or error < best[0]:
            best = (error, floor, math.exp(intercept), -slope)
    return {"E": round(best[1], 4), "B": best[2], "beta": round(best[3], 4)}


def gap_slope(updates, gaps):
    """The least-squares change of the gap per doubling of the data, with a 90% interval from the residuals."""
    x = np.log2(updates)
    if len(x) < 3:
        return None
    slope, intercept = np.polyfit(x, gaps, 1)
    residual = gaps - (slope * x + intercept)
    dof = len(x) - 2
    spread = math.sqrt(float(np.square(residual).sum()) / dof / float(np.square(x - x.mean()).sum()))
    half = T90[min(dof, len(T90)) - 1] * spread
    return {"per_doubling": round(float(slope), 4), "interval_90": [round(float(slope - half), 4),
                                                                    round(float(slope + half), 4)],
            "scales_the_other_way": bool(slope + half < 0)}


def summarize(points):
    rates = {family: best_rate(points, family) for family in ("tf", "st")} | {"pl": "3"}
    reference = points.get(f"tf{rates['tf']}", {})
    summary = {"best_rates": {family: RATES[rate] for family, rate in rates.items()}, "models": {}, "gaps": {}}
    for tag, curve in sorted(points.items()):
        updates, losses = means(curve)
        summary["models"][tag] = {
            "rate": RATES[tag[-1]],
            "points": {f"{int(u)}k": {"bpb": round(float(l), 5), "seeds": len(curve[int(u)])}
                       for u, l in zip(updates, losses)},
            "power_law": fit_power(updates, losses)}
    for family in ("st", "pl"):
        tag = f"{family}{rates[family]}"
        shared = sorted(set(points.get(tag, {})) & set(reference))
        if not shared:
            continue
        gaps = np.array([np.mean(list(points[tag][s].values())) - np.mean(list(reference[s].values()))
                         for s in shared])
        summary["gaps"][tag] = {"against": f"tf{rates['tf']}",
                                "by_size": {f"{s}k": round(float(g), 5) for s, g in zip(shared, gaps)},
                                "slope": gap_slope(np.array(shared, dtype=float), gaps)}
    return summary, rates


def figure(points, rates, summary, output):
    fig, (curves, gaps) = plt.subplots(1, 2, figsize=(12.4, 4.9), width_ratios=(1.15, 1))
    fig.subplots_adjust(left=0.07, right=0.975, bottom=0.17, top=0.76, wspace=0.26)
    fig.text(0.035, 0.965, "Does the stack's gap shrink with data?", size=20, weight="bold", color=TEXT, va="top")
    fig.text(0.036, 0.875, "Test bed, about 5M parameters; each point is its own run with a cosine schedule over its "
             "length, scored on 2M validation bytes as 16 unbroken streams.\nEach model at its best peak rate "
             "(lighter: the other rate). A gap that falls to the right is a scaling law the other way around.",
             size=10, color=MUTED, va="top")
    ticks = [6, 12, 24, 48, 96]
    for ax in (curves, gaps):
        style(ax, grid_axis="both")
        ax.set_xscale("log", base=2)
        ax.set_xticks(ticks, [f"{t * 1000 * BYTES_PER_UPDATE / 1e6:.0f}M" for t in ticks])
        ax.set_xlim(5, 115)
        ax.set_xlabel("Training bytes (log scale)", color=MUTED, labelpad=8)
    for tag, curve in sorted(points.items()):
        family = tag[:2]
        label, color = MODELS[family]
        best = tag[-1] == rates[family]
        updates, losses = means(curve)
        curves.plot(updates, losses, "o-", color=color, alpha=1.0 if best else 0.35, linewidth=2 if best else 1.2,
                    markersize=6 if best else 4, label=f"{label}, rate {RATES[tag[-1]]:g}")
        fit = summary["models"][tag]["power_law"]
        if fit and best:
            grid = np.geomspace(updates.min(), updates.max(), 50)
            curves.plot(grid, fit["E"] + fit["B"] * (grid * 1000 * BYTES_PER_UPDATE) ** -fit["beta"], "--",
                        color=color, linewidth=1, alpha=0.8)
    curves.set_ylabel("Bits per byte", color=MUTED, labelpad=8)
    curves.legend(frameon=False, fontsize=8.5, labelcolor=TEXT, loc="lower left")
    gaps.axhline(0, color=GRAY, linewidth=1)
    for tag, row in summary["gaps"].items():
        label, color = MODELS[tag[:2]]
        sizes = np.array([int(s[:-1]) for s in row["by_size"]], dtype=float)
        values = np.array(list(row["by_size"].values()))
        gaps.plot(sizes, values, "o-", color=color, linewidth=2, markersize=6, label=label)
        if row["slope"]:
            slope = row["slope"]["per_doubling"]
            gaps.annotate(f"{slope:+.3f} per doubling", (sizes[-1], values[-1]), textcoords="offset points",
                          xytext=(-6, 9), ha="right", color=color, size=9)
    gaps.set_ylabel("Stack minus Transformer, bits per byte", color=MUTED, labelpad=8)
    if summary["gaps"]:
        gaps.legend(frameon=False, fontsize=8.5, labelcolor=TEXT, loc="best")
    save(fig, output, "event_lm_data_scaling_long.png")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--scores", type=Path, required=True, help="folder of the runs' score records")
    parser.add_argument("--json", type=Path, help="write the summary here")
    parser.add_argument("--figure-dir", type=Path, default=RESULTS / "figures")
    args = parser.parse_args()
    points = collect(args.scores)
    summary, rates = summarize(points)
    print(json.dumps(summary, indent=1))
    if args.json:
        args.json.write_text(json.dumps({"study": "data_scaling_long", **summary}, indent=1) + "\n")
    figure(points, rates, summary, args.figure_dir)


if __name__ == "__main__":
    main()
