"""Figure of fast output synapses that consolidate themselves: how much of one lesson the output still hears as the
lesson ages, for one to six stages, traced with the model's own rule (EventLM._consolidate)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
from matplotlib.colors import to_hex
from matplotlib.patches import Rectangle
import torch

from plot_dendrite import GRID, MUTED, RESULTS, TEXT, plt, save, style
from plot_read_clocks import SLATE
from plot_state_reads import VIOLET, tall_heading

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from snnformer.event_lm import EventLM  # noqa: E402

TRACE = RESULTS / "event_lm/cascade_synapses/lesson_trace.json"
LONGEST = 1 << 19
SIZE = 1 << 30


def trace_lesson(stages, ages):
    """The share of one lesson of SIZE charges that the first of ``stages`` stages holds at each age, in bytes."""
    state = SimpleNamespace(fast=torch.tensor([float(SIZE)], dtype=torch.float64),
                            deep=torch.zeros(stages - 1, 1, dtype=torch.float64))
    shares, wanted = [], iter(ages)
    age = next(wanted)
    for position in range(1, ages[-1] + 1):
        if stages > 1:
            EventLM._consolidate(state, position)
        while position == age:
            shares.append(float(state.fast[0]) / SIZE)
            age = next(wanted, None)
    levels = [float(state.fast[0])]
    levels += [float(charge) / 2 ** (index + 1) for index, charge in enumerate(state.deep[:, 0])]
    return shares, [level / SIZE for level in levels]


def read_trace():
    """The traces, computed once and kept beside the study's other results."""
    if TRACE.exists():
        return json.loads(TRACE.read_text())
    ages = sorted({int(round(2 ** (step / 8))) for step in range(0, 8 * 19 + 1)})
    traces = {"lesson_charges": SIZE, "ages": ages, "shares": {}, "levels_of_six_stages": {}}
    for stages in range(1, 7):
        traces["shares"][str(stages)], _ = trace_lesson(stages, ages)
    for age in (8, 1024):
        _, traces["levels_of_six_stages"][str(age)] = trace_lesson(6, [age])
    TRACE.write_text(json.dumps(traces, indent=1) + "\n")
    return traces


def shade(stages):
    """One hue from light to dark for two to six stages."""
    start, end = np.array([0.80, 0.76, 0.93]), np.array(plt.matplotlib.colors.to_rgb(VIOLET)) * 0.78
    return to_hex(start + (end - start) * (stages - 2) / 4)


def plot_lesson(output):
    traces = read_trace()
    ages = np.array(traces["ages"])
    fig = plt.figure(figsize=(11.4, 5.3))
    tall_heading(fig, "Fast synapses that consolidate themselves",
                 "Every fast output synapse holds a chain of charges. Bytes teach the first, the output hears the "
                 "first, and charge seeps on into\nlarger and slower stages, so a lesson fades as a power of its age "
                 "and is never wholly lost. Traced with the model's own whole-number rule.")
    vessels = fig.add_axes((0.035, 0.1, 0.36, 0.66))
    curves = fig.add_axes((0.49, 0.14, 0.33, 0.62))

    # Left: the six vessels of one synapse, as wide as their capacities, 8 and 1,024 bytes after one lesson that
    # filled the first to the top.
    gap, rows = 1.6, (("8", 1.32), ("1024", 0.0))
    vessels.set_xlim(-1.2, 63 + 5 * gap + 1.2)
    vessels.set_ylim(-0.3, 2.62)
    vessels.axis("off")
    for age, bottom in rows:
        left = 0.0
        for stage, level in enumerate(traces["levels_of_six_stages"][age]):
            width = 2 ** stage
            vessels.add_patch(Rectangle((left, bottom), width, level, facecolor=VIOLET, alpha=0.5, linewidth=0))
            vessels.add_patch(Rectangle((left, bottom), width, 1.0, fill=False, edgecolor=TEXT, linewidth=1.0))
            if stage < 5:
                # A tube to the next stage, half as wide as the one before: a quarter of its cross-section, as the
                # conductance falls fourfold from each pair of stages to the next.
                vessels.add_patch(Rectangle((left + width, bottom + 0.03), gap, max(0.06 / 2 ** stage, 0.006),
                                            facecolor=TEXT, linewidth=0))
            if bottom == 0:
                vessels.text(left + width / 2, -0.06, f"{width}", ha="center", va="top", size=9, color=MUTED)
            left += width + gap
        bytes_after = f"{int(age):,}"
        vessels.text(63 + 5 * gap, bottom + 1.06, f"{bytes_after} bytes after a lesson", ha="right", va="bottom",
                     size=9.5, color=TEXT)
    vessels.annotate("taught and heard here", (0.5, 2.32), xytext=(4, 14), textcoords="offset points", size=9,
                     color=TEXT, arrowprops={"arrowstyle": "-", "color": TEXT, "linewidth": 0.8, "shrinkA": 0,
                                             "shrinkB": 0})
    vessels.text(0, -0.2, "Capacities 1 to 32. Each tube has a quarter of the cross-section of the one before.",
                 ha="left", va="top", size=9, color=MUTED)

    # Right: the share of the lesson in the first stage against its age.
    style(curves, grid_axis="both")
    curves.set_xscale("log", base=2)
    curves.set_yscale("log", base=2)
    labels = []
    for stages in range(1, 7):
        shares = np.array(traces["shares"][str(stages)])
        color = SLATE if stages == 1 else shade(stages)
        curves.plot(ages, shares, color=color, linewidth=2 if stages in (1, 6) else 1.6,
                    linestyle=(0, (4, 3)) if stages == 1 else "-", zorder=3)
        name = "1 stage, v15's rule" if stages == 1 else f"{stages} stages, 1/{2 ** stages - 1}"
        labels.append((ages[-1], shares[-1], name, color))
    guide = ages[(ages >= 64) & (ages <= 1 << 15)]
    line = 0.27 * 0.5 * (guide / 64.0) ** (-1 / 3)
    curves.plot(guide, line, color=MUTED, linewidth=1, linestyle=(0, (1, 2.5)))
    curves.annotate(r"$\propto\,\mathrm{age}^{-1/3}$", (guide[0], line[0]), xytext=(-4, -10),
                    textcoords="offset points", ha="right", va="top", size=10, color=MUTED)
    curves.set_xlim(1, LONGEST)
    curves.set_ylim(1 / 90, 1.5)
    curves.set_xticks([1, 8, 64, 512, 4096, 32768, 262144])
    curves.set_xticklabels(["1", "8", "64", "512", "4K", "32K", "256K"])
    curves.set_yticks([1, 1 / 2, 1 / 4, 1 / 8, 1 / 16, 1 / 32, 1 / 64])
    curves.set_yticklabels(["1", "1/2", "1/4", "1/8", "1/16", "1/32", "1/64"])
    curves.set_xlabel("Bytes since the lesson", color=MUTED, labelpad=8)
    curves.set_ylabel("Share of the lesson that the output hears", color=MUTED, labelpad=8)
    for x, y, text, color in labels:
        curves.annotate(text, (x, y), xytext=(9, 0), textcoords="offset points", ha="left", va="center", size=9,
                        color=TEXT)
        curves.plot([x], [y], marker="o", markersize=4.5, color=color, zorder=4, clip_on=False)
    curves.grid(color=GRID, linewidth=0.75)
    save(fig, output, "event_lm_cascade_synapses.png")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=RESULTS / "figures")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    plot_lesson(args.output)


if __name__ == "__main__":
    main()
