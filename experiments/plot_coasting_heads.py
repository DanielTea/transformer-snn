"""Figures and tables for the coasting read heads."""

from __future__ import annotations

import argparse

from plot_dendrite import BLUE, GRAY, MUTED, ORANGE, RESULTS, TEXT, VIOLET, heading, plt, read_result, save, style

STUDY = "event_lm/coasting_heads"
GREEN = "#187b62"
SYNAPSES, HEADS, BOTH = "v11 fast synapses", "heads", "synapses and heads"
# Label, replay record, whether the decoder is a full-attention Transformer.
DECODERS = (
    ("v11 release · surprise clock", "fit_only/test_clock", False),
    ("v11 release · 25.79M control", "fit_only/test_control", False),
    ("16× staged run · surprise clock", "recall_long16-clock", False),
    ("Continuation · plain", "recall_long-plain", False),
    ("Continuation · synapses in training", "recall_long-syn", False),
    ("Continuation · heads in training", "recall_long-heads", False),
    ("From scratch · plain", "recall_scratch-clock", False),
    ("From scratch · synapses in training", "recall_scratch-clock-syn", False),
    ("From scratch · heads in training", "recall_scratch-heads", False),
    ("Second pair · plain", "recall_rep-plain", False),
    ("Second pair · heads in training", "recall_rep-heads", False),
    ("Round 2 · same strengths", "recall_round2-same", False),
    ("Round 2 · refitted strengths", "recall_round2-refit", False),
    ("Mixture v2 · synapses in training", "recall_v2-syn", False),
    ("Mixture v2 · heads in training", "recall_v2-heads", False),
    ("16× staged run · Transformer", "recall_long16-transformer", True),
    ("From scratch · Transformer", "recall_scratch-transformer", True),
)
PLOTTED = ("fit_only/test_clock", "recall_long16-clock", "recall_long-plain", "recall_long-heads", "recall_scratch-clock",
           "recall_scratch-clock-syn", "recall_scratch-heads", "recall_long16-transformer", "recall_scratch-transformer")
BINS = ("1-8", "9-32", "33-96", "97-256", "257-512")
DOCUMENTS = (("fineweb_edu", "Web pages"), ("finemath", "Mathematics"), ("stack_edu", "Source code"))


def replay(name):
    """Test loss of a decoder alone, with refitted v11 synapses, with refitted heads, and with both."""
    rows = {row["name"]: row for row in read_result(f"{STUDY}/{name}.json")["records"]}
    return (rows[HEADS]["test_base_bpb"], rows[SYNAPSES]["test_bpb"], rows[HEADS]["test_bpb"],
            rows[BOTH]["test_bpb"] if BOTH in rows else None)


def gaps(record):
    return {pair["name"]: {"-".join(map(str, row["preceding_bytes"])): row for row in pair["histories"]}
            for pair in read_result(f"{STUDY}/{record}.json")["pairs"]}


def plot_test(output):
    rows = [(label, replay(name), full) for label, name, full in DECODERS if name in PLOTTED][::-1]
    fig, ax = plt.subplots(figsize=(11.6, 6.3))
    fig.subplots_adjust(left=0.3, right=0.965, bottom=0.2, top=0.79)
    heading(fig, "Read heads on a tape of the stream",
            "enwik8 test · 4,999,999 targets · recall replayed over each decoder's recorded outputs · "
            "strengths fitted on validation")
    style(ax, grid_axis="x")
    marks = ((0, GRAY, "Decoder alone"), (1, VIOLET, "With v11 fast synapses"), (2, GREEN, "With coasting read heads"))
    for index, (label, scores, full) in enumerate(rows):
        ax.plot([scores[2], scores[0]], [index, index], color="#c9d2de", linewidth=2.2, zorder=1)
        for column, color, _ in marks:
            ax.scatter(scores[column], index, s=58, color=color, zorder=2)
        ax.annotate(f"{scores[2]:.3f}", (scores[2], index), xytext=(-9, 0), textcoords="offset points", ha="right",
                    va="center", size=8.5, color=TEXT)
        ax.annotate(f"{scores[0]:.3f}", (scores[0], index), xytext=(9, 0), textcoords="offset points", ha="left",
                    va="center", size=8.5, color=MUTED)
    ax.set_yticks(range(len(rows)), [label for label, _, _ in rows])
    for tick, (_, _, full) in zip(ax.get_yticklabels(), rows, strict=True):
        tick.set_color(ORANGE if full else TEXT)
    ax.set_xlabel("Bits per byte", color=MUTED, labelpad=8)
    ax.set_xlim(1.1, 1.86)
    ax.legend(handles=[plt.Line2D([], [], marker="o", linestyle="", color=color, label=label) for _, color, label in marks],
              frameon=False, labelcolor=TEXT, fontsize=9, loc="upper center", bbox_to_anchor=(0.36, -0.13), ncol=3)
    fig.text(0.035, 0.03, "Orange rows are full-attention Transformers. Staged and continued runs keep the decoders of "
             "earlier releases as ancestors. One seed per row.", color=MUTED, size=9)
    save(fig, output, "event_lm_coasting_heads.png")


def plot_history(output):
    panels = (("history_gap_scratch", "Against the from-scratch Transformer",
               (("clock", "Plain", GRAY), ("clock-syn", "Synapses in training", VIOLET),
                ("heads", "Heads in training", GREEN))),
              ("history_gap_long", "Against the 16× staged Transformer",
               (("plain", "Plain", GRAY), ("syn", "Synapses in training", VIOLET), ("heads", "Heads in training", GREEN))))
    record = read_result(f"{STUDY}/history_gap_scratch.json")
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 5.2))
    fig.subplots_adjust(left=0.07, right=0.975, bottom=0.27, top=0.72, wspace=0.2)
    heading(fig, "The gap to full attention by bytes seen since a reset",
            f'enwik8 test · {record["windows"]:,} reset windows of {record["window_bytes"]} bytes · loss minus the '
            "Transformer · paired 95% intervals")
    width = 0.25
    for ax, (name, title, arms) in zip(axes, panels, strict=True):
        pairs = gaps(name)
        style(ax)
        for offset, (arm, label, color) in enumerate(arms):
            rows = [pairs[arm][key] for key in BINS]
            values = [row["gap_bpb"] for row in rows]
            errors = [[value - row["paired_95_bpb"][0] for value, row in zip(values, rows, strict=True)],
                      [row["paired_95_bpb"][1] - value for value, row in zip(values, rows, strict=True)]]
            positions = [index + (offset - 1) * width for index in range(len(BINS))]
            ax.bar(positions, values, width, color=color, label=label, yerr=errors, ecolor=TEXT,
                   error_kw={"linewidth": 0.8, "capsize": 1.5})
        ax.axhline(0, color=TEXT, linewidth=0.8)
        ax.set_xticks(range(len(BINS)), [key.replace("-", "–") for key in BINS])
        ax.set_xlabel("Bytes before the target", color=MUTED, labelpad=6)
        ax.set_title(title, fontsize=10, loc="left", color=TEXT)
    axes[0].set_ylabel("Bits per byte above full attention", color=MUTED, labelpad=8)
    axes[0].legend(frameon=False, labelcolor=TEXT, fontsize=9, loc="upper center", bbox_to_anchor=(1.1, -0.2), ncol=3)
    fig.text(0.035, 0.035, "Each window starts from an empty stream, so heads and synapses run inside the model. Left: 368.64M "
             "training targets each. Right: the decoders saw 157.29M more training targets than the Transformer.",
             color=MUTED, size=9)
    save(fig, output, "event_lm_coasting_heads_history.png")


def documents(record):
    return {row["checkpoint"]: {name: row["benchmarks"][name]["bpb"] for name, _ in DOCUMENTS}
            for row in read_result(f"{STUDY}/{record}.json")["models"]}


def plot_documents(output):
    scores = documents("heldout_scratch")
    arms = (("checkpoints/scratch-study/clock_s20001.pt", "Plain", GRAY),
            ("checkpoints/scratch-study/clock-syn_s20001.pt", "Synapses in training", VIOLET),
            ("checkpoints/v12-scratch/heads_s20001.pt", "Heads in training", GREEN),
            ("checkpoints/scratch-study/transformer_s20001.pt", "Full-attention Transformer", ORANGE))
    fig, ax = plt.subplots(figsize=(10.8, 5.0))
    fig.subplots_adjust(left=0.09, right=0.97, bottom=0.27, top=0.74)
    heading(fig, "Documents from outside the training mixture",
            "Held-out FineWeb-Edu, FineMath and Stack-Edu · 512-byte windows from an empty stream · "
            "368.64M training targets each")
    style(ax)
    width = 0.2
    for offset, (checkpoint, label, color) in enumerate(arms):
        values = [scores[checkpoint][name] for name, _ in DOCUMENTS]
        positions = [index + (offset - 1.5) * width for index in range(len(DOCUMENTS))]
        ax.bar(positions, values, width, color=color, label=label)
        for position, value in zip(positions, values, strict=True):
            ax.annotate(f"{value:.2f}", (position, value), xytext=(0, 4), textcoords="offset points", ha="center",
                        size=8.5, color=TEXT)
    ax.set_xticks(range(len(DOCUMENTS)), [label for _, label in DOCUMENTS])
    ax.set_ylabel("Bits per byte", color=MUTED, labelpad=8)
    ax.legend(frameon=False, labelcolor=TEXT, fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=4)
    fig.text(0.035, 0.04, "None of the three sources is in the training mixture of these runs. "
             "A decoder trained with heads or synapses runs them inside the model.", color=MUTED, size=9)
    save(fig, output, "event_lm_coasting_heads_documents.png")


def cell(value):
    return "" if value is None else f"{value:.4f}"


def tables():
    print("| Decoder | Alone | + v11 synapses | + heads | + both |\n|---|---:|---:|---:|---:|")
    for label, name, _ in DECODERS:
        print(f"| {label} | " + " | ".join(cell(value) for value in replay(name)) + " |")
    for record in ("history_gap_scratch", "history_gap_long", "history_gap_v2pair"):
        pairs = gaps(record)
        print(f"\n{record}\n| Arm | " + " | ".join(key.replace("-", "–") for key in (*BINS, "1-512")) + " |\n|---|"
              + "---:|" * (len(BINS) + 1))
        for arm, rows in pairs.items():
            print(f"| {arm} | " + " | ".join(f'{rows[key]["gap_bpb"]:+.4f}' for key in (*BINS, "1-512")) + " |")
    for record in ("heldout_scratch", "heldout_long", "heldout_v2pair", "heldout_partials"):
        print(f"\n{record}\n| Checkpoint | " + " | ".join(label for _, label in DOCUMENTS) + " |\n|---|---:|---:|---:|")
        for checkpoint, scores in documents(record).items():
            print(f"| {checkpoint} | " + " | ".join(f"{scores[name]:.4f}" for name, _ in DOCUMENTS) + " |")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tables", action="store_true", help="print the tables of docs/COASTING_HEADS.md instead")
    args = parser.parse_args()
    if args.tables:
        tables()
        return
    output = RESULTS / "figures"
    plot_test(output)
    plot_history(output)
    plot_documents(output)


if __name__ == "__main__":
    main()
