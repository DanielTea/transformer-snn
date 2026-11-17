"""Prepare additional training text while screening the released benchmark inputs."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from importlib.metadata import version
from pathlib import Path
from urllib.error import HTTPError, URLError
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


def owner_matchers(documents, spec):
    """The matchers of ``overlap_matchers`` with every probe naming the documents it was cut from."""
    import ahocorasick

    exact, normalized = {}, {}
    for owner, (_, text) in enumerate(documents):
        chars = text.decode("latin-1")
        size = spec["exact_bytes"]
        for offset in range(0, len(chars) - size + 1, spec["exact_stride_bytes"]):
            exact.setdefault(chars[offset:offset + size], set()).add(owner)
        words = overlap_words(text)
        size = spec["normalized_words"]
        for offset in range(0, len(words) - size + 1, spec["normalized_stride_words"]):
            normalized.setdefault(" " + " ".join(words[offset:offset + size]) + " ", set()).add(owner)
    automata = []
    for probes in (exact, normalized):
        automaton = ahocorasick.Automaton()
        for probe, owners in probes.items():
            automaton.add_word(probe, tuple(owners))
        automaton.make_automaton()
        automata.append(automaton)
    return tuple(automata)


def matched_owners(matchers, texts):
    """The probe owners that share an exact or normalized string with any of ``texts``."""
    exact, normalized = matchers
    found = set()
    for text in texts:
        for automaton, body in ((exact, text.decode("latin-1")),
                                (normalized, " " + " ".join(overlap_words(text)) + " ")):
            if len(automaton):
                for _, owners in automaton.iter(body):
                    found.update(owners)
    return found


def pinned_source(folder, source):
    """A pinned download that survives a dropped connection."""
    for attempt in range(5):
        try:
            return verified_source(folder, source)
        except (URLError, TimeoutError, ConnectionError):
            if attempt == 4:
                raise
            time.sleep(5 * (attempt + 1))


def text_documents(path, shard, column):
    import pyarrow.parquet as pq

    row = 0
    for batch in pq.ParquetFile(path).iter_batches(batch_size=2048, columns=[column]):
        for body in batch.column(0).to_pylist():
            row += 1
            if body:
                yield f"{shard}-{row}", body.encode()


def fetch_code(folder, template, blob_id):
    """One source file from Software Heritage, checked against the SHA-1 that names it."""
    if len(blob_id) != 40 or not set(blob_id) <= set("0123456789abcdef"):
        raise ValueError(f"invalid source file identifier {blob_id!r}")
    path = folder / blob_id[:2] / blob_id
    cached = path.is_file()
    if cached:
        data = path.read_bytes()
    else:
        for attempt in range(6):
            try:
                with urlopen(template.format(blob_id=blob_id), timeout=60) as response:
                    data = gzip.decompress(response.read())
                break
            except HTTPError as error:
                if error.code in (403, 404):
                    return None
                if attempt == 5:
                    raise
            except (URLError, TimeoutError, ConnectionError, EOFError):
                if attempt == 5:
                    raise
            time.sleep(2 ** attempt)
    if hashlib.sha1(data, usedforsecurity=False).hexdigest() != blob_id:
        raise ValueError(f"source file {blob_id} does not match its SHA-1")
    if not cached:
        path.parent.mkdir(parents=True, exist_ok=True)
        write_excerpt(path, data)
    return data


def code_documents(path, shard, spec, context, folder, pool, dropped):
    """Permissively licensed files of one language in listing order, fetched by content hash."""
    import pyarrow.parquet as pq

    licenses = set(spec["license_types"])
    columns = ["blob_id", "license_type", "src_encoding", "length_bytes"]
    for batch in pq.ParquetFile(path).iter_batches(batch_size=512, columns=columns):
        rows = [row["blob_id"] for row in batch.to_pylist()
                if row["license_type"] in licenses and row["src_encoding"] == "UTF-8"
                and context < row["length_bytes"] <= spec["max_file_bytes"]]
        bodies = pool.map(lambda blob_id: fetch_code(folder, spec["content_sha1_url"], blob_id), rows)
        for blob_id, body in zip(rows, bodies):
            if body is None:
                dropped["unavailable"] += 1
                continue
            try:
                body.decode("utf-8")
            except UnicodeDecodeError:
                dropped["not_utf8"] += 1
                continue
            if b"\0" in body:
                dropped["zero_byte"] += 1
                continue
            yield f"{shard}-{blob_id}", body


def _matches(automaton, text):
    # A set of short documents can leave one of the two matchers without any probe.
    return len(automaton) > 0 and next(automaton.iter(text), None) is not None


def select_documents(documents, minimum_bytes, context, matchers, seen):
    pairs = matchers if isinstance(matchers, list) else [matchers]
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
        chars = text.decode("latin-1")
        if any(_matches(exact, chars) for exact, _ in pairs):
            excluded["exact_overlap"] += 1
            continue
        words = " " + " ".join(overlap_words(text)) + " "
        if any(_matches(normalized, words) for _, normalized in pairs):
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


def add_domains(args, manifest, raw_manifest, patterns, seen, corpora, audit, *earlier):
    """Add the web, math and code corpora, each with held-out documents that no training text overlaps.

    The first screened documents of every source become held-out candidates.
    A candidate that shares a protected string with the earlier corpora is
    dropped. Training documents are then drawn from the rest of each source and
    screened against the released benchmarks and the held-out documents alike.
    """
    context, plan = manifest["context_bytes"], manifest["heldout"]
    pool = ThreadPoolExecutor(max_workers=48)
    streams, dropped = {}, {}
    for name, spec in manifest.get("text", {}).items():
        streams[name] = [(source["id"], text_documents(pinned_source(args.sources, source), source["id"],
                                                        spec["column"])) for source in spec["sources"]]
    for name, spec in manifest.get("code", {}).items():
        dropped[name] = Counter()
        streams[name] = [(source["id"], code_documents(pinned_source(args.sources, source), source["id"], spec,
                                                        context, args.sources / "software-heritage", pool,
                                                        dropped[name])) for source in spec["sources"]]
    specs = {**manifest.get("text", {}), **manifest.get("code", {})}

    candidates = []
    for name, sources in streams.items():
        share = specs[name]["heldout_bytes"] * plan["candidate_percent"] // (100 * len(sources))
        for shard, stream in sources:
            docs, _ = select_documents(stream, share, context, patterns, seen)
            candidates.extend((name, shard, doc_id, body) for doc_id, body in docs)
        print(f"{name}: {sum(1 for row in candidates if row[0] == name):,} held-out candidates", flush=True)
    owners = owner_matchers([(doc_id, body) for _, _, doc_id, body in candidates], manifest["overlap"])
    shared = set()
    for texts in earlier:
        shared |= matched_owners(owners, texts)
    del owners
    kept, sizes, summary = {name: [] for name in streams}, Counter(), {}
    for index, (name, shard, doc_id, body) in enumerate(candidates):
        limit = specs[name]["heldout_bytes"] // len(streams[name])
        if index not in shared and sizes[name, shard] < limit:
            kept[name].append((doc_id, body))
            sizes[name, shard] += len(body)
    folder = args.output / "heldout"
    folder.mkdir(exist_ok=True)
    entries = {name: write_collection(folder, name, docs) for name, docs in kept.items()}
    for name, docs in kept.items():
        summary[name] = {"bytes": sum(len(body) for _, body in docs), "documents": len(docs),
                         "candidates": sum(1 for row in candidates if row[0] == name),
                         "shared_with_earlier_training_text": sum(1 for index, row in enumerate(candidates)
                                                                  if row[0] == name and index in shared)}
        print(f"{name}: held out {summary[name]['documents']:,} documents ({summary[name]['bytes']:,} bytes)",
              flush=True)
    suite = {"suite": plan["suite"], "manifest_sha256": hashlib.sha256(raw_manifest).hexdigest(),
             "context_bytes": context, "targets_per_window": plan["targets_per_window"], "corpora": entries,
             "overlap": {}, "overlap_protocol": manifest["overlap"],
             "preparation_versions": {"pyarrow": version("pyarrow"), "pyahocorasick": version("pyahocorasick")}}
    write_excerpt(folder / "catalogue.json", (json.dumps(suite, indent=2) + "\n").encode())

    guards = [patterns, overlap_matchers((body for docs in kept.values() for _, body in docs), manifest["overlap"])]
    for name, sources in streams.items():
        selected, audit[name] = [], {}
        for shard, stream in sources:
            docs, stats = select_documents(stream, specs[name]["bytes"] // len(sources), context, guards, seen)
            selected.extend(docs)
            audit[name][shard] = stats
            print(f"{name} {shard} selected {stats['documents']:,} documents ({stats['bytes']:,} bytes)", flush=True)
        if name in dropped:
            audit[name]["not_used"] = dict(dropped[name])
        corpora[name] = write_source(args.output, name, selected)
        del selected, docs
    pool.shutdown(cancel_futures=True)
    return {"folder": "heldout", "catalogue_sha256": file_hash(folder / "catalogue.json"), "selection": summary}


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
    c4_selected = c4 if "heldout" in manifest else None
    del c4, docs

    books = manifest["pg19"]["sources"]
    if protected_ids.intersection(book["id"] for book in books):
        raise ValueError("a PG-19 training candidate appears in the held-out book manifest")
    with ThreadPoolExecutor(max_workers=12) as pool:
        documents = pool.map(lambda book: fetch_book(args.sources, book), books)
        selected, audit["pg19"] = select_documents(documents, manifest["pg19"]["bytes"], context, patterns, seen)
    corpora["pg19"] = write_source(args.output, "pg19", selected)
    heldout = None
    if "heldout" in manifest:
        heldout = add_domains(args, manifest, raw_manifest, patterns, seen, corpora, audit,
                              [train.tobytes()], [body for _, body in c4_selected], [body for _, body in selected])
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
    if heldout is not None:
        catalogue["heldout"] = heldout
    write_excerpt(args.output / "catalogue.json", (json.dumps(catalogue, indent=2) + "\n").encode())
    print(f"Prepared {sum(entry['bytes'] for entry in corpora.values()):,} training bytes "
          f"across {len(corpora)} corpora", flush=True)


if __name__ == "__main__":
    main()
