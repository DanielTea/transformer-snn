"""Prepare additional training text while screening the released benchmark inputs."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from importlib.metadata import version
from pathlib import Path
from urllib.request import urlopen

from eval_byte_transfer import load_payload
from prepare_byte_benchmarks import file_hash, overlap_words, scoring_windows, verified_source, write_collection
from prepare_pg19 import write_excerpt
from train_lm import load_splits


RESULTS = Path("results/event_lm")
DEFAULT_MANIFEST = RESULTS / "training_mixture/sources_v1.json"
PG19_SPLITS = {
    "test": RESULTS / "correction_pg19_full_book_manifest.json",
    "validation": RESULTS / "correction_pg19_validation_manifest.json",
}


def protected_text(transfer, transfer_catalogue, enwik8_val, enwik8_test, heldout_books):
    yield enwik8_val.tobytes()
    yield enwik8_test.tobytes()
    yield from heldout_books
    for entry in transfer_catalogue["corpora"].values():
        text = load_payload(transfer, entry)
        for doc in entry["documents"]:
            start = doc["offset"]
            document = text[start:start + doc["bytes"]]
            windows = scoring_windows(document, transfer_catalogue["context_bytes"],
                                      transfer_catalogue["targets_per_window"])
            yield from (window for _, _, window in windows)
    for case in json.loads(load_payload(transfer, transfer_catalogue["lambada"])):
        yield (case["passage"] + case["word"]).encode()


def overlap_matchers(texts, spec):
    import ahocorasick

    exact = ahocorasick.Automaton()
    normalized = ahocorasick.Automaton()
    for text in texts:
        chars = text.decode("latin-1")
        size = spec["exact_bytes"]
        for offset in range(0, len(chars) - size + 1, spec["exact_stride_bytes"]):
            exact.add_word(chars[offset:offset + size], None)
        words = overlap_words(text)
        size = spec["normalized_words"]
        for offset in range(0, len(words) - size + 1, spec["normalized_stride_words"]):
            normalized.add_word(" " + " ".join(words[offset:offset + size]) + " ", None)
    exact.make_automaton()
    normalized.make_automaton()
    return exact, normalized


def read_heldout_books(folders):
    books, ids = [], set()
    for split, manifest in PG19_SPLITS.items():
        for entry in json.loads(manifest.read_text())["books"]:
            book_id = str(entry["book_id"])
            path = folders[split] / entry["excerpt_file"]
            body = path.read_bytes()
            if hashlib.sha256(body).hexdigest() != entry["excerpt_sha256"] or book_id in ids:
                raise ValueError(f"unexpected held-out PG-19 content: {path}")
            ids.add(book_id)
            books.append(body)
    return books, ids


def c4_documents(path, shard):
    with gzip.open(path, "rb") as inputs:
        for row, line in enumerate(inputs, start=1):
            yield f"{shard}-{row}", json.loads(line)["text"].encode()


def fetch_book(folder, book):
    name = book["file"]
    if Path(name).name != name or name in ("", ".", ".."):
        raise ValueError(f"invalid PG-19 source filename {name!r}")
    path = folder / name
    if path.is_file():
        data = path.read_bytes()
    else:
        with urlopen(book["url"], timeout=60) as response:
            data = response.read(book["bytes"] + 1)
    if len(data) != book["bytes"] or hashlib.md5(data, usedforsecurity=False).hexdigest() != book["md5"]:
        raise ValueError(f"PG-19 source does not match the published size and MD5: {path}")
    if not path.is_file():
        write_excerpt(path, data)
    return book["id"], data


def select_documents(documents, minimum_bytes, context, matchers, seen):
    exact, normalized = matchers
    selection, excluded = [], Counter()
    total = 0
    for doc_id, text in documents:
        if len(text) <= context:
            excluded["short"] += 1
            continue
        digest = hashlib.sha256(text).digest()
        if digest in seen:
            excluded["duplicate"] += 1
            continue
        if next(exact.iter(text.decode("latin-1")), None) is not None:
            excluded["exact_overlap"] += 1
            continue
        if next(normalized.iter(" " + " ".join(overlap_words(text)) + " "), None) is not None:
            excluded["normalized_overlap"] += 1
            continue
        seen.add(digest)
        selection.append((doc_id, text))
        total += len(text)
        if total >= minimum_bytes:
            return selection, {"bytes": total, "documents": len(selection), "excluded": dict(excluded)}
    raise ValueError(f"exhausted the pinned documents with {total:,} of {minimum_bytes:,} requested training bytes")


def write_source(folder, name, documents):
    payload = write_collection(folder, name, documents)
    spans = payload.pop("documents")
    index = "".join(f"{doc['id']}\t{doc['bytes']}\n" for doc in spans).encode()
    destination = folder / f"{name}.spans"
    write_excerpt(destination, index)
    return dict(**payload, spans_file=destination.name, spans_sha256=hashlib.sha256(index).hexdigest(),
                documents=len(spans), bytes=sum(doc["bytes"] for doc in spans))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--enwik8", type=Path, default=Path("data/enwik8"))
    parser.add_argument("--transfer", type=Path, default=Path("data/transfer-v1/prepared"))
    parser.add_argument("--pg19-test", type=Path, default=Path("data/pg19-test"))
    parser.add_argument("--pg19-validation", type=Path, default=Path("data/pg19-validation"))
    parser.add_argument("--sources", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    raw_manifest = args.source_manifest.read_bytes()
    manifest = json.loads(raw_manifest)
    args.sources.mkdir(parents=True, exist_ok=True)
    args.output.mkdir(parents=True, exist_ok=True)
    train, val, test = load_splits(args.enwik8)
    if len(train) != manifest["enwik8_training_bytes"]:
        raise ValueError("enwik8 training split differs from the mixture specification")
    transfer_path = args.transfer / "catalogue.json"
    if file_hash(transfer_path) != manifest["transfer_catalogue_sha256"]:
        raise ValueError("prepare the pinned byte transfer catalogue before preparing the training mixture")
    transfer = json.loads(transfer_path.read_text())
    heldout, protected_ids = read_heldout_books({"test": args.pg19_test, "validation": args.pg19_validation})
    patterns = overlap_matchers(protected_text(args.transfer, transfer, val, test, heldout), manifest["overlap"])
    print(f"Indexed {len(patterns[0]):,} exact and {len(patterns[1]):,} normalized protected strings", flush=True)

    corpora = {"enwik8": {"sha256": hashlib.sha256(train).hexdigest(), "bytes": len(train), "documents": 1}}
    seen = set()
    c4, audit = [], {"c4": {}}
    context = manifest["context_bytes"]
    for shard in manifest["c4"]["sources"]:
        source = verified_source(args.sources, shard)
        docs, stats = select_documents(c4_documents(source, shard["id"]), manifest["c4"]["bytes_per_shard"],
                                       context, patterns, seen)
        c4.extend(docs)
        audit["c4"][shard["id"]] = stats
        print(f"C4 shard {shard['id']} selected {stats['documents']:,} documents ({stats['bytes']:,} bytes)", flush=True)
    corpora["c4"] = write_source(args.output, "c4", c4)
    del c4, docs

    books = manifest["pg19"]["sources"]
    if protected_ids.intersection(book["id"] for book in books):
        raise ValueError("a PG-19 training candidate appears in the held-out book manifest")
    with ThreadPoolExecutor(max_workers=12) as pool:
        documents = pool.map(lambda book: fetch_book(args.sources, book), books)
        selected, audit["pg19"] = select_documents(documents, manifest["pg19"]["bytes"], context, patterns, seen)
    corpora["pg19"] = write_source(args.output, "pg19", selected)
    catalogue = {
        "format": 1, "name": manifest["name"], "context_bytes": context,
        "sampling_weights": manifest["sampling_weights"], "corpora": corpora,
        "source_manifest_sha256": hashlib.sha256(raw_manifest).hexdigest(),
        "protection": {
            "transfer_catalogue_sha256": manifest["transfer_catalogue_sha256"],
            "pg19_manifests_sha256": {split: file_hash(path) for split, path in PG19_SPLITS.items()},
            "overlap": manifest["overlap"], "exact_probes": len(patterns[0]), "normalized_probes": len(patterns[1]),
            "selection": audit, "pyahocorasick_version": version("pyahocorasick"),
        },
    }
    write_excerpt(args.output / "catalogue.json", (json.dumps(catalogue, indent=2) + "\n").encode())
    print(f"Prepared {sum(entry['bytes'] for entry in corpora.values()):,} training bytes across three corpora", flush=True)


if __name__ == "__main__":
    main()
