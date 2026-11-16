"""Build release charts and tables for the fixed byte transfer benchmarks."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from plot_dendrite import BLUE, GRAY, MUTED, ORANGE, RESULTS, TEXT, VIOLET, heading, plt, save, style


DATASETS = (("wikitext103", "WikiText-103"), ("c4", "C4"), ("enwik9", "enwik9"), ("text8", "text8"))
SOURCE_DIR = RESULTS / "event_lm" / "byte_transfer"


def read_transfer(path):
    record = json.loads(path.read_text())
    expected = {"manifest_sha256": "sources_v1.json", "catalogue_sha256": "prepared_v1.json"}
    for key, name in expected.items():
        if record[key] != hashlib.sha256((SOURCE_DIR / name).read_bytes()).hexdigest():
            raise ValueError(f"{path} uses a different fixed byte transfer dataset")
    reference, *comparisons = record["models"]
    for row in comparisons:
        for key, _ in DATASETS:
            left, right = reference["benchmarks"][key], row["benchmarks"][key]
            if left["target_bytes"] != right["target_bytes"] or left.get("overlap_flagged_targets") != right.get(
                "overlap_flagged_targets"
            ):
                raise ValueError(f"{row['checkpoint']} did not score the same {key} targets")
        for condition in ("passage", "last_sentence"):
            left = reference["benchmarks"]["lambada"][condition]
            right = row["benchmarks"]["lambada"][condition]
            if left["target_bytes"] != right["target_bytes"] or left["examples"] != right["examples"]:
                raise ValueError(f"{row['checkpoint']} did not score the same LAMBADA words")
    return record


def transfer_tables(record, labels):
    models = record["models"]
    if len(labels) != len(models):
        raise ValueError("supply one label per evaluated checkpoint")

    def table(headers, rows):
        lines = ["| " + " | ".join(headers) + " |", "|---|" + "---:|" * (len(headers) - 1)]
        lines += ["| " + " | ".join(row) + " |" for row in rows]
        return "\n".join(lines)

    corpus, overlap, lambada = [], [], []
    for label, model in zip(labels, models, strict=True):
        scores = model["benchmarks"]
        corpus.append([label] + [f'{scores[key]["bpb"]:.5f}' for key, _ in DATASETS])
        screened = [scores[key]["without_overlap_hits"]["bpb"] for key in ("wikitext103", "enwik9", "text8")]
        overlap.append([label] + [f"{score:.5f}" if score is not None else "—" for score in screened])
        full, last = (scores["lambada"][name] for name in ("passage", "last_sentence"))
        lambada.append([label, f'{full["bpb"]:.5f}', f'{last["bpb"]:.5f}', f'{full["exact_match_accuracy"]:.2%}',
                        f'{last["exact_match_accuracy"]:.2%}'])
    return (
        table(["Model"] + [label for _, label in DATASETS], corpus),
        table(["Model", "WikiText-103", "enwik9", "text8"], overlap),
        table(["Model", "Passage bpb", "Last sentence bpb", "Passage exact", "Last sentence exact"], lambada),
    )


def plot_transfer(record, labels, output):
    models = record["models"]
    if len(labels) != len(models) or len(models) < 2:
        raise ValueError("a comparison needs a label for each of at least two checkpoints")
    reference = models[0]["benchmarks"]
    colors = (ORANGE, GRAY, BLUE, VIOLET, "#187b62", "#9c4772")
    fig, (corpus, words) = plt.subplots(1, 2, figsize=(12.8, 4.75), width_ratios=(1.1, 1))
    fig.subplots_adjust(left=0.145, right=0.967, bottom=0.26, top=0.735, wspace=0.43)
    heading(fig, "Byte-model transfer", "Fixed English corpora and paired LAMBADA prompts")
    for axes in (corpus, words):
        style(axes, grid_axis="x")
    corpus.set_title("Raw corpus loss", loc="left", color=TEXT, fontsize=12)
    words.set_title("LAMBADA final-word accuracy", loc="left", color=TEXT, fontsize=12)

    for index, (model, label) in enumerate(zip(models, labels, strict=True)):
        color = colors[index % len(colors)]
        metrics = model["benchmarks"]
        if index:
            deltas = [metrics[key]["bpb"] - reference[key]["bpb"] for key, _ in DATASETS]
            offset = (index - len(models) / 2) * 0.17
            corpus.scatter(deltas, [y + offset for y in range(len(DATASETS))],
                           color=color, s=37, zorder=3, label=label)
        passage, last = (100 * metrics["lambada"][mode]["exact_match_accuracy"]
                         for mode in ("passage", "last_sentence"))
        high, low = max(passage, last), min(passage, last)
        words.hlines(index, low, high, color=color, linewidth=2)
        words.scatter(passage, index, color=color, s=42, zorder=3)
        words.scatter(last, index, color="white", edgecolors=color, linewidths=1.5, s=42, zorder=3)
    corpus.axvline(0, color=MUTED, linewidth=0.85)
    corpus.set_yticks(range(len(DATASETS)), [label for _, label in DATASETS])
    corpus.invert_yaxis()
    corpus.set_xlabel("Bits per byte relative to full attention", color=MUTED)
    corpus.legend(loc="upper center", bbox_to_anchor=(0.5, -0.24), frameon=False,
                  ncols=min(3, len(models) - 1), fontsize=8, labelcolor=TEXT)

    words.set_yticks(range(len(labels)), labels)
    words.invert_yaxis()
    words.set_xlabel("Greedy exact match (%)", color=MUTED)
    fig.text(0.035, 0.037, "Lower loss is better. Filled and open markers show full-passage and last-sentence scores.",
             color=MUTED, size=9)
    output.mkdir(parents=True, exist_ok=True)
    save(fig, output, "event_lm_byte_transfer.png")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record", type=Path, required=True)
    parser.add_argument("--labels", nargs="+", required=True, help="model labels in record order, full attention first")
    parser.add_argument("--output-dir", type=Path, default=RESULTS / "figures")
    args = parser.parse_args()
    record = read_transfer(args.record)
    plot_transfer(record, args.labels, args.output_dir)
    print("\n\n".join(transfer_tables(record, args.labels)))


if __name__ == "__main__":
    main()
