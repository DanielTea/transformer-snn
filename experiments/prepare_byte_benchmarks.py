"""Prepare the fixed byte-level transfer benchmarks from pinned public data."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import heapq
import html
import json
import re
import tempfile
import zipfile
from importlib.metadata import version
from pathlib import Path
from urllib.request import urlopen

from prepare_pg19 import write_excerpt


DEFAULT_MANIFEST = Path("results/event_lm/byte_transfer/sources_v1.json")
CATALOGUE = "catalogue.json"
BUFFER_BYTES = 4 * 1024 * 1024


def file_hash(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verified_source(folder, source):
    name = source["file"]
    if name in ("", ".", "..") or Path(name).name != name:
        raise ValueError(f"invalid source file name {name!r}")
    destination = folder / name
    if destination.is_file():
        if destination.stat().st_size == source["bytes"] and file_hash(destination) == source["sha256"]:
            return destination
        raise ValueError(f"cached source {destination} failed the manifest check")

    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=folder, delete=False) as output:
            temporary = Path(output.name)
            digest = hashlib.sha256()
            length = 0
            with urlopen(source["url"], timeout=60) as response:
                for block in iter(lambda: response.read(BUFFER_BYTES), b""):
                    length += len(block)
                    if length > source["bytes"]:
                        raise ValueError(f"{name} exceeds its pinned size")
                    digest.update(block)
                    output.write(block)
        if length != source["bytes"] or digest.hexdigest() != source["sha256"]:
            raise ValueError(f"downloaded source {name} failed the manifest check")
        temporary.replace(destination)
        print(f"Verified {destination}", flush=True)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return destination


def scoring_windows(document, context_bytes, targets_per_window):
    history = context_bytes - targets_per_window
    if history < 1 or targets_per_window < 1:
        raise ValueError("evaluation windows need both context and targets")
    for target in range(1, len(document), targets_per_window):
        begin = max(0, target - history)
        stop = min(len(document), target + targets_per_window)
        yield target, target - begin, document[begin:stop]


def canonical_slice(path, member, spec, *, hash_name, prefix_sha1=None):
    start = spec["slice_offset"]
    stop = start + spec["slice_bytes"]
    if not 0 <= start < stop <= spec["raw_bytes"]:
        raise ValueError(f"{member} slice outside the corpus")
    digest = hashlib.new(hash_name)
    prefix = hashlib.sha1() if prefix_sha1 else None
    excerpt = bytearray()
    position = 0
    with zipfile.ZipFile(path) as archive, archive.open(member) as stream:
        for block in iter(lambda: stream.read(BUFFER_BYTES), b""):
            end = position + len(block)
            digest.update(block)
            if prefix is not None and position < 100_000_000:
                prefix.update(block[:100_000_000 - position])
            if start < end and position < stop:
                excerpt.extend(block[max(0, start - position):stop - position])
            position = end
    if position != spec["raw_bytes"] or digest.hexdigest() != spec[hash_name]:
        raise ValueError(f"{member} does not match the published corpus")
    if prefix is not None and prefix.hexdigest() != prefix_sha1:
        raise ValueError("enwik9 does not have the canonical enwik8 prefix")
    return bytes(excerpt)


def read_wikitext(path):
    from pyarrow import parquet

    # The original loader uses an empty record for a blank input line.
    rows = parquet.read_table(path, columns=["text"])["text"].to_pylist()
    if len(rows) != 4358 or any(row and not row.endswith("\n") for row in rows[:-1]):
        raise ValueError("unexpected WikiText-103 raw test line layout")
    return "".join(row or "\n" for row in rows).encode("utf-8")


def read_c4(paths, spec):
    documents = []
    for shard, path in enumerate(paths):
        def eligible(stream):
            for line, row in enumerate(stream, start=1):
                body = json.loads(row)["text"].encode("utf-8")
                if len(body) >= 2:
                    yield hashlib.sha256(body).hexdigest(), line, body[:spec["maximum_document_bytes"]]

        with gzip.open(path, "rb") as stream:
            selected = heapq.nsmallest(spec["documents_per_shard"], eligible(stream))
        if len(selected) != spec["documents_per_shard"]:
            raise ValueError(f"C4 shard {shard} has too few documents")
        documents.extend((f"{shard:02d}-{line}", body) for _, line, body in selected)
    return documents


def read_lambada(path, context_bytes):
    from pyarrow import parquet
    from pysbd import Segmenter

    texts = parquet.read_table(path, columns=["text"])["text"]
    if len(texts) != 5153:
        raise ValueError("unexpected LAMBADA test size")
    segmenter = Segmenter(language="en", clean=False, char_span=True)
    cases = []
    exclusions = {"over_context": 0, "no_last_sentence_prefix": 0}
    for index, item in enumerate(texts):
        text = item.as_py().rstrip()
        if len(text.encode("utf-8")) > context_bytes:
            exclusions["over_context"] += 1
            continue
        word = text.split()[-1]
        stop = len(text) - len(word)
        sentence = segmenter.segment(text)[-1]
        if not 0 <= sentence.start < stop or sentence.end != len(text):
            exclusions["no_last_sentence_prefix"] += 1
            continue
        cases.append({"id": index, "passage": text[:stop], "last_sentence": text[sentence.start:stop],
                      "word": word})
    if not cases:
        raise ValueError("empty LAMBADA evaluation set")
    return cases, {"source_examples": len(texts), "scored_examples": len(cases), "excluded": exclusions}


def overlap_words(text):
    text = html.unescape(text.decode("utf-8", errors="ignore")).lower()
    text = re.sub(r"<ref\b[^>]*>.*?</ref\s*>", " ", text, flags=re.DOTALL)
    text = re.sub(r"<[^<>]*>", " ", text)
    text = re.sub(r"\[\[([^\[\]]+)\]\]", lambda match: match[1].rsplit("|", 1)[-1], text)
    return re.findall(r"[a-z]+", text)


def matched_probes(probes, training):
    import ahocorasick

    if not probes:
        return set()
    index = ahocorasick.Automaton()
    for probe in probes:
        index.add_word(probe, probe)
    index.make_automaton()
    return {value for _, value in index.iter(training)}


def check_overlap(collections, enwik8, protocol):
    spec = protocol["overlap"]
    groups = ("wikitext103", "enwik9", "text8")
    context = protocol["context_bytes"]
    targets = protocol["targets_per_window"]
    exact_probes, normalized_probes = set(), set()
    audited = {name: [] for name in groups}
    for name in groups:
        for doc, body in collections[name]:
            for start, prefix, window in scoring_windows(body, context, targets):
                size = spec["exact_bytes"]
                exact = tuple(window[offset:offset + size].decode("latin-1") for offset in
                              range(0, len(window) - size + 1, spec["exact_stride_bytes"]))
                words = overlap_words(window)
                size = spec["normalized_words"]
                normalized = tuple(" " + " ".join(words[offset:offset + size]) + " " for offset in
                                   range(0, len(words) - size + 1, spec["normalized_stride_words"]))
                audited[name].append((doc, start, len(window) - prefix, exact, normalized))
                exact_probes.update(exact)
                normalized_probes.update(normalized)

    training = enwik8[:protocol["enwik8"]["training_bytes"]]
    exact_hits = matched_probes(exact_probes, training.decode("latin-1"))
    normalized_hits = matched_probes(normalized_probes, " " + " ".join(overlap_words(training)) + " ")
    summary = {}
    for name, windows in audited.items():
        flagged = []
        exact_windows = normalized_windows = flagged_targets = 0
        for doc, start, length, exact, normalized in windows:
            exact_match = not exact_hits.isdisjoint(exact)
            normalized_match = not normalized_hits.isdisjoint(normalized)
            exact_windows += exact_match
            normalized_windows += normalized_match
            if exact_match or normalized_match:
                flagged.append([doc, start])
                flagged_targets += length
        summary[name] = dict(windows=len(windows), exact_match_windows=exact_windows,
                             normalized_match_windows=normalized_windows, flagged_targets=flagged_targets,
                             flagged_windows=flagged)
    return summary


def write_collection(folder, name, documents):
    spans = []
    offset = 0
    for label, body in documents:
        spans.append({"id": label, "offset": offset, "bytes": len(body)})
        offset += len(body)
    content = b"".join(body for _, body in documents)
    file = f"{name}.bin"
    write_excerpt(folder / file, content)
    return dict(file=file, sha256=hashlib.sha256(content).hexdigest(), documents=spans)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--enwik8", required=True, type=Path, help="original enwik8 file used for training")
    parser.add_argument("--sources", type=Path, required=True, help="cache for the pinned downloads")
    parser.add_argument("--output", type=Path, required=True, help="write the prepared data and its catalogue here")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    args = parser.parse_args()
    manifest_bytes = args.manifest.read_bytes()
    protocol = json.loads(manifest_bytes)
    sources = protocol["sources"]
    args.sources.mkdir(parents=True, exist_ok=True)
    args.output.mkdir(parents=True, exist_ok=True)
    enwik8 = args.enwik8.read_bytes()
    if hashlib.sha1(enwik8).hexdigest() != protocol["enwik8"]["sha1"]:
        raise ValueError("overlap audit needs the canonical enwik8 training source")
    wiki = read_wikitext(verified_source(args.sources, sources["wikitext103"]))
    c4 = read_c4([verified_source(args.sources, shard) for shard in sources["c4"]], protocol["c4"])
    enwik9 = canonical_slice(verified_source(args.sources, sources["enwik9"]), "enwik9", protocol["enwik9"],
                             hash_name="sha1", prefix_sha1=protocol["enwik8"]["sha1"])
    text8 = canonical_slice(verified_source(args.sources, sources["text8"]), "text8", protocol["text8"],
                             hash_name="md5")
    cases, eligibility = read_lambada(verified_source(args.sources, sources["lambada"]), protocol["context_bytes"])
    collections = {"wikitext103": [("raw-test", wiki)], "c4": c4, "enwik9": [("slice", enwik9)],
                   "text8": [("tail-slice", text8)]}
    overlaps = check_overlap(collections, enwik8, protocol)
    corpora = {name: write_collection(args.output, name, docs) for name, docs in collections.items()}
    lambada = json.dumps(cases, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    write_excerpt(args.output / "lambada.json", lambada)
    catalogue = dict(suite=protocol["suite"], manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
                     context_bytes=protocol["context_bytes"], targets_per_window=protocol["targets_per_window"],
                     corpora=corpora, overlap=overlaps, overlap_protocol=protocol["overlap"],
                     overlap_training_bytes=protocol["enwik8"]["training_bytes"],
                     lambada=dict(file="lambada.json", sha256=hashlib.sha256(lambada).hexdigest(), **eligibility),
                     preparation_versions={module: version(module) for module in ("pyarrow", "pyahocorasick", "pysbd")})
    write_excerpt(args.output / CATALOGUE, (json.dumps(catalogue, indent=2) + "\n").encode("utf-8"))
    for name, entry in overlaps.items():
        print(f"{name}: {len(entry['flagged_windows'])}/{entry['windows']} windows flagged by the overlap screen")
    print(f"LAMBADA: {eligibility['scored_examples']} paired full-passage and last-sentence examples", flush=True)


if __name__ == "__main__":
    main()
