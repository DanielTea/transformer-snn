"""The figures of the telemetry pilot (docs/TELEMETRY_PILOT.md) from the records of telemetry_evaluate.py."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from plot_dendrite import GRID, MUTED, ORANGE, RESULTS, TEXT, plt, save, style
from plot_state_reads import VIOLET

SLATE = "#687a92"


def plot_nab(evaluation, scores, pilot, arms, output):
    """The temperature with NAB's windows, and the smoothed score of each named Volley arm and the best baseline."""
    record = json.loads(evaluation.read_text())
    kept = np.load(scores)
    arrays = np.load(pilot.with_suffix(".npz"))
    calibration = json.loads(pilot.with_suffix(".json").read_text())["calibration_samples"]
    values, labels = arrays["run0_values"][:, 0], arrays["run0_labels"].astype(bool)
    best = record["paired_with_best_baseline"][arms[0][0]]["against"]
    rows = list(arms) + [(best, f"Best baseline: {BASELINE_NAMES.get(best, best)}")]
    days = np.arange(len(values)) / 288
    fig, axes = plt.subplots(len(rows) + 1, 1, figsize=(11, 2.1 * (len(rows) + 1)), sharex=True)
    edges = np.flatnonzero(np.diff(np.concatenate([[0], labels.astype(int), [0]])))
    for ax in axes:
        style(ax, grid_axis="y")
        for start, end in zip(edges[::2], edges[1::2]):
            ax.axvspan(days[start], days[end - 1], color=GRID, alpha=0.8, linewidth=0)
        ax.axvline(days[calibration], color=MUTED, linewidth=0.8, linestyle=(0, (2, 3)))
    axes[0].plot(days, values, color=TEXT, linewidth=0.6)
    axes[0].set_ylabel("Temperature", color=MUTED)
    axes[0].set_title("NAB machine temperature: four labelled windows (shaded), calibration before the dotted line",
                      loc="left", size=11, color=TEXT, weight="bold", pad=10)
    for ax, (method, label) in zip(axes[1:], rows):
        color = ORANGE if method == best else VIOLET
        score = kept[f"machine_temperature|{method}"]
        row = record["per_run"][method][0]
        ax.plot(days, np.clip(score, None, np.quantile(score, 0.999)), color=color, linewidth=0.8)
        ax.axhline(row["threshold"], color=color, linewidth=0.8, linestyle=(0, (4, 3)))
        ax.set_ylabel("Score", color=MUTED)
        ax.text(0.005, 0.97, f"{label} · AUC {row['auc']:.3f} · caught {row['caught']} of {row['anomalies']}",
                transform=ax.transAxes, color=color, size=9.5, va="top",
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 2})
    axes[-1].set_xlabel("Days", color=MUTED)
    fig.tight_layout()
    save(fig, output, "telemetry_nab.png")


def plot_skab(evaluation, output, names):
    record = json.loads(evaluation.read_text())
    summary = record["summary"]
    methods = list(summary)
    fig, axes = plt.subplots(1, 2, figsize=(11, 0.5 * len(methods) + 1.6))
    for ax, key, title in ((axes[0], "mean_auc", "Mean ROC-AUC over 34 runs"),
                           (axes[1], "mean_f1", "Mean F1 at the calibrated threshold")):
        style(ax, grid_axis="x")
        heights = [summary[method][key] or 0 for method in methods]
        colors = [VIOLET if method.startswith("volley") else SLATE for method in methods]
        ax.barh(range(len(methods)), heights, color=colors, height=0.62)
        ax.set_yticks(range(len(methods)), [names.get(method, method) for method in methods], color=TEXT)
        ax.invert_yaxis()
        for row, height in enumerate(heights):
            ax.text(height, row, f" {height:.3f}", va="center", size=9, color=TEXT)
        ax.set_title(title, loc="left", size=11, color=TEXT, weight="bold", pad=10)
        if key == "mean_auc":
            ax.axvline(0.5, color=MUTED, linewidth=0.8, linestyle=(0, (2, 3)))
    fig.tight_layout()
    save(fig, output, "telemetry_skab.png")


NAB_ARMS = (("volley_learning", "Volley, fast synapses learning throughout"),
            ("volley_frozen", "Volley, fast synapses frozen after calibration"))
NAMES = {"volley_learning": "Volley, learning", "volley_frozen": "Volley, frozen", "volley_no_fast": "Volley, no fast synapses",
         "volley_printable": "Volley, one character", "volley_delta": "Volley, delta", "volley_delta_frozen":
         "Volley, delta, frozen", "volley_reference": "Volley, against the normal capsule",
         "zscore": "Rolling z-score", "ewma": "EWMA forecast", "ar": "AR(8) forecast"}
BASELINE_NAMES = {"zscore": "rolling z-score", "ewma": "EWMA forecast", "ar": "AR(8) forecast"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nab-evaluation", type=Path, default=RESULTS / "telemetry/evaluation_nab.json")
    parser.add_argument("--nab-scores", type=Path, default=RESULTS / "telemetry/evaluation_nab_scores.npz")
    parser.add_argument("--nab-pilot", type=Path, default=RESULTS / "telemetry/nab_digits_taught")
    parser.add_argument("--skab-evaluation", type=Path, default=RESULTS / "telemetry/evaluation_skab.json")
    parser.add_argument("--output", type=Path, default=RESULTS / "figures")
    args = parser.parse_args()
    plot_nab(args.nab_evaluation, args.nab_scores, args.nab_pilot, NAB_ARMS, args.output)
    plot_skab(args.skab_evaluation, args.output, NAMES)


if __name__ == "__main__":
    main()
