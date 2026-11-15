"""Plot validation loss and integer pulse traffic for the enwik8 count sweep."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from plot_dendrite import BLUE, MUTED, ORANGE, RESULTS, TEXT, VIOLET, heading, plt, read_result, save, style


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=RESULTS / "figures")
    args = parser.parse_args()

    resolutions = (1, 2, 4, 8, 16, 32)
    seeds = (41, 42, 43)
    losses = np.array([
        [read_result(f"event_lm/event_steered_s{seed}_count{count}_memory_pilot.json")["val_bpb"]
         for count in resolutions]
        for seed in seeds
    ])
    activity = read_result("event_lm/spike_resolution_activity.json")["models"]
    pulse_counts = {(row["seed"], row["resolution"]): row["pulses_per_byte"] for row in activity}
    pulses = np.array([[pulse_counts[seed, count] for count in resolutions] for seed in seeds])
    pulses /= pulses[:, :1].copy()

    first, second, third = losses[0, :3]
    ratio = (second - third) / (first - second)
    floor = first - (first - second) / (1 - ratio)
    x = np.arange(len(resolutions))
    forecast = floor + (first - floor) * ratio**x

    fig, (loss_ax, pulse_ax) = plt.subplots(1, 2, figsize=(11.4, 4.2))
    fig.subplots_adjust(left=0.105, right=0.97, top=0.728, bottom=0.23, wspace=0.34)
    heading(fig, "Finer spikes give diminishing returns",
            "enwik8 validation · 391,296 parameters · 1.64 million training targets per run")
    for ax in (loss_ax, pulse_ax):
        style(ax)
        ax.set_xticks(x, [f"{count}×" for count in resolutions])
        ax.set_xlabel("Spike count resolution", color=MUTED, labelpad=8)
    loss_ax.set_ylabel("Validation bits per byte", color=MUTED, labelpad=10)
    loss_ax.fill_between(x, losses.min(axis=0), losses.max(axis=0), color=BLUE, alpha=0.15)
    loss_ax.plot(x, losses.mean(axis=0), "o-", color=BLUE, linewidth=2, label="Three-seed mean")
    loss_ax.plot(x[2:], forecast[2:], "--", color=ORANGE, linewidth=1.5, label="Forecast from seed 41")
    loss_ax.legend(loc="upper right", frameon=False, labelcolor=TEXT, fontsize=9)

    pulse_ax.set_ylabel("Pulses relative to 1×", color=MUTED, labelpad=10)
    pulse_ax.plot(x, pulses.mean(axis=0), "o-", color=VIOLET, linewidth=2)
    pulse_ax.set_yticks([0, 8, 16, 24, 32])
    pulse_ax.set_ylim(0, 33)
    fig.text(0.036, 0.064,
             "The forecast uses 1×, 2× and 4× only. Shading spans the three seeds. Pulse ratios are paired by seed.",
             color=MUTED, size=9)
    args.output.mkdir(parents=True, exist_ok=True)
    save(fig, args.output, "spike_resolution.png")


if __name__ == "__main__":
    main()
