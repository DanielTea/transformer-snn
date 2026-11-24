"""Figure of the full-size pilots of single parts (results/event_lm/knowledge_gap/full_size_*.json): every pilot is
the full-size spiking stack with one change, trained for 24,000 updates and scored on the first 2,000,000 validation
bytes as 16 unbroken streams. Each bar is the pilot's bits per byte minus the plain stack's at its best rate, so a
bar to the left is a gain. The Transformer's two pilots stand beside them.

usage: python experiments/plot_full_size_parts.py [--figure-dir DIR]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from plot_dendrite import GRAY, MUTED, ORANGE, RESULTS, TEXT, plt, save, style

DARK_VIOLET, LIGHT_VIOLET = "#4a2a85", "#8f73c9"
RECORDS = RESULTS / "event_lm" / "knowledge_gap"
# The axis ends here; a bar that would run further is cut and marked.
LIMIT = 0.024


def result(name, key="result"):
    return json.loads((RECORDS / f"{name}.json").read_text())[key]


def rows():
    """(label, note, bits per byte, kind) of every pilot, and the plain stack's score that they are read against."""
    rate = result("full_size_rate_pilot")["bpb"]
    base = rate["stack"]["6e-4"]
    loop = result("full_size_loop_pilot")["bpb"]
    unit = result("full_size_unit_screen")["bpb"]
    third = result("full_size_unit_screen", "result_third_arm")["bpb"]
    lifted_gate = result("full_size_lifted_gate")["bpb"]
    lifted = result("full_size_lifted_ceiling_pilot")["bpb"]["lifted"]
    budget = loop["fs-b507 (budget 507,000, no loop)"]
    looped = loop["fs-loop6 (loop, budget 507,000)"]
    found = [
        ("Transformer at its best rate, 1.2e-3", "the reference", rate["transformer"]["1.2e-3"], "transformer"),
        ("Transformer at the stack's rate, 6e-4", "", rate["transformer"]["6e-4"], "transformer"),
        ("Levels 2 to 8 run twice, with the pulses for it",
         f"{budget - looped:.4f} better than the same pulses without the loop", looped, "stack"),
        ("A pool that lifts the ceiling", "no membrane at its bound", lifted["6e-4"], "stack"),
        ("Noisy thresholds", "a gain on near copies only", unit["fs-noise6"], "stack"),
        ("Cells that gate themselves", "", third["fs-self6"], "stack"),
        ("More pulses alone: a budget of 507,000", "", budget, "stack"),
        ("Gated cells under the lifting pool", "", lifted_gate["fs-gate6-lift"], "stack"),
        ("Gated cells", "gates at their bound a fifth of the time", unit["fs-gate6"], "stack"),
    ]
    return found, base


def figure(found, base, output):
    fig, ax = plt.subplots(figsize=(12.4, 5.9))
    fig.subplots_adjust(left=0.375, right=0.955, bottom=0.13, top=0.775)
    fig.text(0.035, 0.965, "Full size: what each part is worth to the spiking stack", size=20, weight="bold",
             color=TEXT, va="top")
    fig.text(0.036, 0.888, f"One change to the full-size stack (34.8M parameters) in each run of 24,000 updates, "
             f"one seed, against the plain stack at its best rate ({base:.4f} bits per byte\non 2M validation bytes "
             "as 16 unbroken streams). Left of the line is better. The Transformer is the distance to close.",
             size=10, color=MUTED, va="top")
    style(ax, grid_axis="x")
    places = list(range(len(found)))[::-1]
    for place, (label, note, score, kind) in zip(places, found):
        change = score - base
        shown = min(LIMIT, change)
        color = ORANGE if kind == "transformer" else (DARK_VIOLET if change < 0 else GRAY)
        ax.barh(place, shown, height=0.56, color=color, zorder=3)
        cut = change > LIMIT
        if cut:
            # A break across the bar's end says that it runs on.
            for shift in (0.0012, 0.0024):
                ax.plot([LIMIT - shift - 0.0004, LIMIT - shift + 0.0004], [place - 0.36, place + 0.36], color="white",
                        linewidth=2.2, zorder=4, solid_capstyle="butt")
        side = -1 if change < 0 else 1
        # Small changes need their fourth decimal; the large ones are quoted to three, as the text quotes them.
        ax.annotate(f"{change:+.4f}" if abs(change) < 0.02 else f"{change:+.3f}", (shown, place),
                    textcoords="offset points", xytext=(7 * side, 0), ha="right" if side < 0 else "left", va="center",
                    color=TEXT, size=10, weight="bold", annotation_clip=False)
        ax.annotate(label, (0, place), xycoords=("axes fraction", "data"), textcoords="offset points",
                    xytext=(-12, 5 if note else 0), ha="right", va="center", color=TEXT, size=10.5,
                    annotation_clip=False)
        if note:
            ax.annotate(note, (0, place), xycoords=("axes fraction", "data"), textcoords="offset points",
                        xytext=(-12, -8), ha="right", va="center", color=MUTED, size=8.5, annotation_clip=False)
    ax.axvline(0, color=TEXT, linewidth=1.0, zorder=5)
    ax.annotate("the plain stack", (0, len(found) - 0.42), textcoords="offset points", xytext=(5, 0), color=MUTED,
                size=8.5, va="center")
    # The two models' rows stand apart from the stack's.
    ax.axhline(len(found) - 2.5, color=GRAY, linewidth=0.8, linestyle="--", zorder=1)
    ax.set_xlim(-0.0535, LIMIT + 0.0085)
    ax.set_ylim(-0.6, len(found) - 0.2)
    ax.set_yticks([])
    ticks = [-0.04, -0.03, -0.02, -0.01, 0, 0.01, 0.02]
    ax.set_xticks(ticks, [f"{tick:+.2f}" if tick else "0" for tick in ticks])
    ax.set_xlabel("Bits per byte against the plain stack (lower is better)", color=MUTED, labelpad=8)
    save(fig, output, "event_lm_full_size_parts.png")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--figure-dir", type=Path, default=RESULTS / "figures")
    args = parser.parse_args()
    figure(*rows(), args.figure_dir)


if __name__ == "__main__":
    main()
