"""Figure of where a stack's pulses go (results/event_lm/knowledge_gap/output_step_on_trained_weights.json): the
pulses a byte of every population as the pulse budget counts them, with output spikes as trained and with output
spikes that stand for 8 fine steps of evidence, on the test bed and at full size.

usage: python experiments/plot_pulse_populations.py [--data JSON] [--figure-dir DIR]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from plot_dendrite import GRID, MUTED, PAPER, RESULTS, TEXT, VIOLET, plt, save, style

REST = "#cdd5e0"  # every population whose spikes stay as they are
FINE, COARSE = "1 fine step a spike (1/128 bit)", "8 fine steps a spike (1/16 bit)"
# The populations in the order of the bars, each with the tool's names that it sums.
PARTS = (
    ("Reads", ("reads", "racing cells", "read pulses", "within reads")),
    ("Cells", ("cells", "gate dendrites")),
    ("Stream", ("stream after reads", "stream after cells")),
    ("Input, pools", ("input", "pools")),
)


def parts(populations):
    """Pulses a byte of every part of PARTS; whatever the tool names beside them and the output joins the last."""
    named = {name for _, names in PARTS for name in names} | {"output"}
    sums = [sum(populations.get(name, 0.0) for name in names) for _, names in PARTS]
    sums[-1] += sum(value for name, value in populations.items() if name not in named)
    return sums


def stacks(record):
    """For the test bed and the full size: a title, the parts' pulses a byte, the output's and the stack's pulses with
    fine and with coarse output spikes, the two scores and what they were scored on."""
    result = record["result"]
    bed = result["test bed, 24,000 updates (rd-base-24k-s3)"]
    full = result["full size (l2048-word-st)"]
    scored = full.get("complete test split") or full.get("sample of the test split")
    where = ("the complete test split" if "complete test split" in full else
             f"the first {scored['first bytes of the test split']:,} bytes of the test split" if scored else None)
    return (
        ("Test bed, 5.2M parameters, 24,000 updates",
         parts(result["test bed, pulses a byte by population at fine spikes"]["24,000 updates"]),
         (bed[FINE]["of them the output's"], bed[COARSE]["of them the output's"]),
         (bed[FINE]["pulses a byte"], bed[COARSE]["pulses a byte"]),
         (f"{bed[FINE]['bits per byte']:.5f}", f"+{bed[COARSE]['above fine spikes']:.5f}"),
         "the first 2,000,000 validation bytes"),
        ("Full size, 36.3M parameters: v18's best stack",
         parts(full["pulses a byte by population at fine spikes"]),
         (full["of them the output's"][FINE], full["of them the output's"][COARSE]),
         (full["pulses a byte"][FINE], full["pulses a byte"][COARSE]),
         (f"{scored['1 fine step a spike']:.5f}", f"+{scored['above fine spikes']:.5f}") if scored else
         ("1.11615 on test", "not scored yet"), where),
    )


def figure(record, output):
    fig, ax = plt.subplots(figsize=(12.4, 6.4))
    fig.subplots_adjust(left=0.225, right=0.69, bottom=0.225, top=0.765)
    fig.text(0.035, 0.965, "Output spikes of 1/16 bit: the same score for a fifth to a third fewer pulses", size=20,
             weight="bold", color=TEXT, va="top")
    fig.text(0.036, 0.89, "Pulses a byte by population, as the pulse budget counts them, on 65,536 validation bytes "
             "as streams. As trained, one spike of an output neuron stands\nfor 1/128 bit of evidence for its byte. "
             "Here it stands for 1/16 bit, with no weight changed: the output neurons round the same charge to a "
             "coarser grid.", size=10, color=MUTED, va="top")
    style(ax, grid_axis="x")
    ax.grid(False)  # the grid is drawn behind each stack's own two bars, and not through the titles
    margin = ax.get_yaxis_transform()  # x in the axes' own fraction, y in rows
    row, notes = 0.0, []
    for title, rest, outputs, totals, scores, where in stacks(record):
        whole = sum(rest) + outputs[0]
        ax.vlines(range(0, 101, 25), row - 0.4, row + 0.78 + 0.4, color=GRID, linewidth=0.75, zorder=0)
        ax.text(-0.405, row - 0.74, title, transform=margin, ha="left", va="center", color=TEXT, size=10.5,
                weight="bold")
        for kind, (spikes, total, label, score) in enumerate(zip(outputs, totals, ("1/128 bit a spike, as trained",
                                                                                 "1/16 bit a spike"), scores)):
            left = 0.0
            for (name, _), value in zip(PARTS, rest):
                share = 100 * value / whole
                ax.barh(row, share, left=left, height=0.56, color=REST, edgecolor=PAPER, linewidth=2)
                if kind == 0 and share >= 5.5:
                    ax.text(left + share / 2, row, f"{name}\n{share:.0f}%", ha="center", va="center", color=TEXT,
                            size=9, linespacing=1.25)
                left += share
            share = 100 * spikes / whole
            ax.barh(row, share, left=left, height=0.56, color=VIOLET, edgecolor=PAPER, linewidth=2)
            if share >= 5.5:
                ax.text(left + share / 2, row, f"Output\n{share:.0f}%", ha="center", va="center", color=PAPER,
                        size=9, linespacing=1.25, weight="bold")
            else:
                ax.text(left + share + 1.0, row, f"Output {share:.0f}%", ha="left", va="center", color=TEXT, size=9,
                        weight="bold")
            ax.text(-1.2, row, label, ha="right", va="center", color=TEXT if kind else MUTED, size=10)
            ax.text(104, row, f"{total:,}", ha="left", va="center", color=TEXT, size=10.5,
                    weight="bold" if kind else "normal")
            if kind:
                ax.text(117.5, row, f"{1 - total / totals[0]:.0%} fewer", ha="left", va="center", color=TEXT,
                        size=10.5, weight="bold")
            ax.text(138, row, score, ha="left", va="center", color=TEXT, size=10.5)
            row += 0.78
        if where:
            notes.append(f"{title.split(',')[0].lower()} on {where}")
        row += 1.15
    ax.text(104, -0.74, "Pulses a byte", ha="left", va="center", color=MUTED, size=9.5)
    ax.text(138, -0.74, "Bits per byte", ha="left", va="center", color=MUTED, size=9.5)
    ax.set_ylim(row - 1.15 - 0.3, -1.08)
    ax.set_yticks([])
    ax.set_xlim(0, 100)
    ax.set_xticks(range(0, 101, 25), [f"{tick}%" for tick in range(0, 101, 25)])
    ax.set_xlabel("Share of the stack's pulses with output spikes as trained", color=MUTED, labelpad=8)
    fig.text(0.036, 0.03, "Reads: the reads' own cells, their racing cells and the read pulses these pass on. "
             "Stream: the neurons that put a level's reads and cells back on the stream.\nThe thin part before the "
             "output: the input and the pools.\nBits per byte, as streams, the coarse spikes paired piece by piece "
             "with the fine ones: " + "; ".join(notes) + ".", size=8.8, color=MUTED, va="bottom", linespacing=1.35)
    save(fig, output, "event_lm_pulse_populations.png")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--data", type=Path,
                        default=RESULTS / "event_lm" / "knowledge_gap" / "output_step_on_trained_weights.json")
    parser.add_argument("--figure-dir", type=Path, default=RESULTS / "figures")
    args = parser.parse_args()
    figure(json.loads(args.data.read_text()), args.figure_dir)


if __name__ == "__main__":
    main()
