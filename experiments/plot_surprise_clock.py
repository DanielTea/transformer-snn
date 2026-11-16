"""Figures for the surprise clock and the surprise-written fast synapses."""

from __future__ import annotations

from plot_dendrite import BLUE, GRAY, MUTED, ORANGE, TEXT, VIOLET, heading, plt, read_result, save, style

STUDY = "event_lm/surprise_clock"
GREEN = "#187b62"
WORD_LABELS = {"start": "1st", "second": "2nd", "third": "3rd", "fourth": "4th", "fifth": "5th",
               "later": "6th+", "outside": "Between\nwords"}
AGE_LABELS = {"unseen": "Not seen\nbefore", "1-31": "1–31", "32-127": "32–127", "128-511": "128–511", "512+": "512+"}


def plot_words(output):
    measured = read_result(f"{STUDY}/surprise_validation.json")
    probe = read_result(f"{STUDY}/clock_probe_test.json")
    decoder = measured["models"][0]["surprise_by_word_position"]["surprisal_bits"]
    clock = next(row["clock"] for row in probe["models"] if row["run_name"].endswith("_surprise_clock_clock"))
    slots = list(WORD_LABELS)
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.9))
    fig.subplots_adjust(left=0.07, right=0.975, bottom=0.25, top=0.74, wspace=0.2)
    heading(fig, "Surprise gathers at the first byte of a word",
            "Left: frozen 25.79M decoder on enwik8 validation · right: trained clock on the enwik8 test split")
    left, right = axes
    style(left)
    left.bar(range(len(slots)), [decoder[slot]["mean"] for slot in slots], color=VIOLET, width=0.62)
    for index, slot in enumerate(slots):
        left.annotate(f'{decoder[slot]["mean"]:.2f}', (index, decoder[slot]["mean"]), xytext=(0, 4),
                      textcoords="offset points", ha="center", size=8.5, color=TEXT)
    left.set_xticks(range(len(slots)), [WORD_LABELS[slot] for slot in slots])
    left.set_ylabel("Mean loss, bits", color=MUTED, labelpad=8)
    left.set_title("Loss by position of the byte inside its word", fontsize=11, loc="left", color=TEXT)
    style(right)
    width = 0.36
    ticks = [100 * clock["tick_share_by_word_position"][slot] for slot in slots]
    bytes_ = [100 * clock["byte_share_by_word_position"][slot] for slot in slots]
    right.bar([index - width / 2 for index in range(len(slots))], bytes_, width, color=GRAY, label="Share of bytes")
    right.bar([index + width / 2 for index in range(len(slots))], ticks, width, color=ORANGE,
              label="Share of clock ticks")
    right.set_xticks(range(len(slots)), [WORD_LABELS[slot] for slot in slots])
    right.set_ylabel("Percent", color=MUTED, labelpad=8)
    right.set_title(f'Where the clock ticks · one tick per {clock["bytes_per_tick"]:.2f} bytes', fontsize=11,
                    loc="left", color=TEXT)
    right.legend(frameon=False, labelcolor=TEXT, fontsize=9, loc="upper center")
    fig.text(0.035, 0.045,
             f'{100 * clock["words_without_tick"]:.0f}% of words pass without a tick, '
             f'{100 * clock["words_with_one_tick"]:.0f}% get one and '
             f'{100 * clock["words_with_more_ticks"]:.0f}% get more. Word bytes are letters, digits and UTF-8 bytes.',
             color=MUTED, size=9)
    save(fig, output, "event_lm_surprise_words.png")


def plot_sweep(output):
    sweep = read_result(f"{STUDY}/synapse_sweep.json")
    fig, ax = plt.subplots(figsize=(10.8, 4.9))
    fig.subplots_adjust(left=0.1, right=0.96, bottom=0.25, top=0.74)
    heading(fig, "What the fast synapses add, by state and write rule",
            "Readout fitted on the first half of enwik8 validation · gain on the second half · six heads of four groups")
    style(ax)
    rules = ((1, VIOLET, "Stored on error spikes"), (0, GRAY, "Stored on every byte"))
    gains = {gate: {row["synapse_state_bytes"]: row["held_out"]["gain_bpb"] for row in sweep["settings"]
                    if row["gate"] == gate and row["groups"] == 4} for gate, _, _ in rules}
    for gate, color, label in rules:
        sizes = sorted(gains[gate])
        ax.plot([size / 1024 for size in sizes], [gains[gate][size] for size in sizes], color=color, marker="o",
                markersize=6, linewidth=1.8, label=label)
        for size in sizes:
            gain = gains[gate][size]
            # The larger of the two gains at a size is labelled above its point.
            above = gain >= gains[1 - gate][size]
            ax.annotate(f"{gain:.4f}", (size / 1024, gain), xytext=(0, 8 if above else -15),
                        textcoords="offset points", ha="center", size=8.5, color=TEXT)
    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_formatter("{x:g}")
    ax.set_xlabel("Synapse state per stream, KiB", color=MUTED, labelpad=8)
    ax.set_ylabel("Loss removed, bits per byte", color=MUTED, labelpad=8)
    ax.margins(x=0.08, y=0.18)
    ax.legend(frameon=False, labelcolor=TEXT, fontsize=9, loc="lower right")
    stored = next(row["stored_byte_share"] for row in sweep["settings"] if row["gate"] == 1)
    fig.text(0.035, 0.045, f"Error spikes store {100 * stored:.0f}% of the bytes. One byte of state per key neuron.",
             color=MUTED, size=9)
    save(fig, output, "event_lm_fast_synapse_sweep.png")


def plot_prefix_age(group, output):
    by_label = {row["model"]: row for row in group["rows"]}
    arms = (("25.79M SNN", "25.79M full attention", BLUE),
            ("34.85M SNN, surprise clock", "34.85M full attention", VIOLET),
            ("25.79M SNN with fast synapses", "25.79M full attention", GREEN))

    def ages(label):
        return read_result(by_label[label]["score_files"]["enwik8"])["models"][0]["prefix_age"]

    targets = ages(arms[0][1])
    fig, ax = plt.subplots(figsize=(10.8, 4.9))
    fig.subplots_adjust(left=0.1, right=0.96, bottom=0.27, top=0.74)
    heading(fig, "Where the spiking decoders stand against full attention",
            "enwik8 test · loss minus the size-matched Transformer · "
            "targets grouped by how far back their 8-byte context last appeared")
    style(ax)
    width = 0.26
    for offset, (label, reference, color) in enumerate(arms):
        scores, matched = ages(label), ages(reference)
        gaps = [scores[age]["bpb"] - matched[age]["bpb"] for age in AGE_LABELS]
        positions = [index + (offset - 1) * width for index in range(len(AGE_LABELS))]
        ax.bar(positions, gaps, width, color=color, label=label)
        for x, gap in zip(positions, gaps, strict=True):
            ax.annotate(f"{gap:+.3f}", (x, gap), xytext=(0, 4 if gap >= 0 else -11), textcoords="offset points",
                        ha="center", size=7.5, color=TEXT)
    ax.axhline(0, color=TEXT, linewidth=0.8)
    total = sum(row["target_bytes"] for row in targets.values())
    ax.set_xticks(range(len(AGE_LABELS)),
                  [f"{AGE_LABELS[age]}\n{100 * targets[age]['target_bytes'] / total:.0f}% of targets"
                   for age in AGE_LABELS])
    ax.set_ylabel("Bits per byte above full attention", color=MUTED, labelpad=8)
    ax.margins(y=0.2)
    ax.legend(frameon=False, labelcolor=TEXT, fontsize=9, loc="lower left", ncol=1)
    fig.text(0.035, 0.04, "Bytes since the context last appeared. Full attention reads 512 bytes. "
             "The fast synapses carry the whole stream.", color=MUTED, size=9)
    save(fig, output, "event_lm_surprise_prefix_age.png")


def plot_history_gap(output):
    record = read_result(f"{STUDY}/history_gap_test.json")
    bins = ("1-8", "9-32", "33-96", "97-256", "257-512")
    arms = (("control", "25.79M SNN", BLUE), ("depth", "34.85M SNN, every byte", "#7a6a50"),
            ("stride", "34.85M SNN, fixed stride", "#9a6fb0"), ("clock", "34.85M SNN, surprise clock", VIOLET),
            ("control-synapses", "25.79M SNN with fast synapses", GREEN))
    pairs = {pair["name"]: {"-".join(map(str, row["preceding_bytes"])): row for row in pair["histories"]}
             for pair in record["pairs"]}
    fig, ax = plt.subplots(figsize=(11.6, 5.3))
    fig.subplots_adjust(left=0.09, right=0.97, bottom=0.32, top=0.76)
    heading(fig, "The gap to full attention by bytes seen since a reset",
            f'enwik8 test · {record["windows"]:,} reset windows of {record["window_bytes"]} bytes · '
            "loss minus the size-matched Transformer · paired 95% intervals")
    style(ax)
    width = 0.16
    for offset, (name, label, color) in enumerate(arms):
        rows = [pairs[name][key] for key in bins]
        gaps = [row["gap_bpb"] for row in rows]
        errors = [[gap - row["paired_95_bpb"][0] for gap, row in zip(gaps, rows, strict=True)],
                  [row["paired_95_bpb"][1] - gap for gap, row in zip(gaps, rows, strict=True)]]
        positions = [index + (offset - (len(arms) - 1) / 2) * width for index in range(len(bins))]
        ax.bar(positions, gaps, width, color=color, label=label, yerr=errors, ecolor=TEXT,
               error_kw={"linewidth": 0.8, "capsize": 1.5})
    ax.axhline(0, color=TEXT, linewidth=0.8)
    ax.set_xticks(range(len(bins)), [f"{key.replace('-', '–')} bytes" for key in bins])
    ax.set_ylabel("Bits per byte above full attention", color=MUTED, labelpad=8)
    ax.legend(frameon=False, labelcolor=TEXT, fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.13),
              ncol=3)
    fig.text(0.035, 0.04, "Bytes that precede the target. Each window starts from an empty stream. "
             "The intervals resample windows.", color=MUTED, size=9)
    save(fig, output, "event_lm_surprise_history_gap.png")


def plot_surprise_study(group, output):
    plot_words(output)
    plot_sweep(output)
    plot_prefix_age(group, output)
    plot_history_gap(output)
