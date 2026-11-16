"""Audit the exact sampled training windows against the pinned PG-19 books."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import torch

from byte_training_mixture import TrainingMixture
from common import _atomic_output
from prepare_byte_benchmarks import file_hash, overlap_words
from prepare_byte_training_mixture import PG19_SPLITS, read_heldout_books
from train_lm import load_splits


EXACT_BYTES = 64
NORMALIZED_WORDS = 12


def span_index(sequences, minimum, stride):
    # Any minimum-length match contains a probe on the reference stride grid.
    width = minimum - stride + 1
    positions = defaultdict(list)
    for book, sequence in enumerate(sequences):
        for start in range(0, len(sequence) - width + 1, stride):
            positions[sequence[start:start + width]].append((book, start))
    return width, positions


def matching_books(sequence, books, index, minimum):
    width, positions = index
    matches = set()
    for start in range(len(sequence) - width + 1):
        for book, offset in positions.get(sequence[start:start + width], ()):
            if book in matches:
                continue
            reference = books[book]
            left = 0
            left_limit = min(start, offset, minimum - width)
            while left < left_limit and sequence[start - left - 1] == reference[offset - left - 1]:
                left += 1
            right = width
            right_limit = min(len(sequence) - start, len(reference) - offset, minimum - left)
            while right < right_limit and sequence[start + right] == reference[offset + right]:
                right += 1
            if left + right >= minimum:
                matches.add(book)
    return matches


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("training_record", type=Path, help="completed mixture training record")
    parser.add_argument("--enwik8", type=Path, default=Path("data/enwik8"))
    parser.add_argument("--training-mixture", type=Path, help="override the prepared catalogue path in the record")
    parser.add_argument("--pg19-test", type=Path, default=Path("data/pg19-test"))
    parser.add_argument("--pg19-validation", type=Path, default=Path("data/pg19-validation"))
    parser.add_argument("--json", required=True, type=Path)
    args = parser.parse_args()
    record = json.loads(args.training_record.read_text())
    if not record.get("training_mixture") or args.json.exists():
        parser.error("use a completed mixture record and a fresh output file")

    torch.set_num_threads(1)
    train, _, _ = load_splits(args.enwik8)
    recipe = record["args"]
    catalogue = args.training_mixture or Path(recipe["training_mixture"])
    mixture = TrainingMixture(catalogue, train, recipe["ctx"])
    generator = torch.Generator().manual_seed(recipe["seed"])

    books, _ = read_heldout_books({"test": args.pg19_test, "validation": args.pg19_validation})
    labels = [(split, entry["book_id"]) for split, path in PG19_SPLITS.items()
              for entry in json.loads(path.read_text())["books"]]
    words = [tuple(overlap_words(book)) for book in books]
    exact = span_index(books, EXACT_BYTES, stride=16)
    normalized = span_index(words, NORMALIZED_WORDS, stride=3)

    sources = tuple(mixture.corpora)
    counts = {source: {"windows": 0, "exact": Counter(), "normalized": Counter()} for source in sources}
    matching_windows = {kind: defaultdict(Counter) for kind in ("exact", "normalized")}
    window_number = 0
    for _ in range(recipe["steps"]):
        inputs, targets = mixture.batch(recipe["bs"], generator)
        windows = torch.cat((inputs[:, :1], targets), dim=1).to(torch.uint8).numpy()
        for window in windows:
            source = sources[mixture.schedule[window_number % len(mixture.schedule)]]
            window_number += 1
            counts[source]["windows"] += 1
            text = window.tobytes()
            matched = {
                "exact": matching_books(text, books, exact, EXACT_BYTES),
                "normalized": matching_books(tuple(overlap_words(text)), words, normalized, NORMALIZED_WORDS),
            }
            for kind, indices in matched.items():
                counts[source][kind].update({labels[index][0] for index in indices})
                matching_windows[kind][source].update(f"{labels[index][0]}_{labels[index][1]}"
                                                       for index in sorted(indices))
    if mixture.receipt() != record["training_mixture"]:
        parser.error("the reconstructed windows differ from the recorded training windows")

    result = {
        "training_record": str(args.training_record),
        "training_record_sha256": file_hash(args.training_record),
        "training_receipt": mixture.receipt(),
        "books": {split: file_hash(path) for split, path in PG19_SPLITS.items()},
        "exact_match_minimum_bytes": EXACT_BYTES, "normalized_match_minimum_words": NORMALIZED_WORDS,
        "matching_windows_by_split": counts, "matching_windows_by_book": matching_windows,
    }
    with _atomic_output(args.json, "w", overwrite=False) as output:
        json.dump(result, output, indent=2)
        output.write("\n")


if __name__ == "__main__":
    main()
