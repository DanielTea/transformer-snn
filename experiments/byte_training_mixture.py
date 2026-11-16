"""Sample matched next-byte windows from a pinned, document-bounded training mixture."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from prepare_byte_benchmarks import file_hash


class _Corpus:
    def __init__(self, data, lengths, context):
        lengths = np.asarray(lengths, dtype=np.int64)
        if (lengths.ndim != 1 or not lengths.size or (lengths < 1).any()
                or int(lengths.sum()) != len(data)):
            raise ValueError("document lengths must partition the training bytes")
        self.data = data
        self.offsets = lengths.cumsum() - lengths
        counts = np.maximum(lengths - context, 0)
        self.ends = counts.cumsum()
        self.beginnings = self.ends - counts
        self.total = int(self.ends[-1])
        if self.total == 0:
            raise ValueError("each source needs a document longer than the training context")

    def draw(self, count, context, generator):
        starts = torch.randint(self.total, (count,), generator=generator).numpy()
        documents = np.searchsorted(self.ends, starts, side="right")
        offsets = self.offsets[documents] + starts - self.beginnings[documents]
        return np.stack([self.data[offset:offset + context + 1] for offset in offsets])


def _checked_file(folder, name, digest):
    if Path(name).name != name or name in ("", ".", ".."):
        raise ValueError(f"invalid training mixture file name {name!r}")
    path = folder / name
    if file_hash(path) != digest:
        raise ValueError(f"training mixture file checksum differs: {path}")
    return path


class TrainingMixture:
    def __init__(self, catalogue_path, enwik8_train, context):
        self.context = context
        raw_catalogue = catalogue_path.read_bytes()
        catalogue = json.loads(raw_catalogue)
        if catalogue["format"] != 1 or catalogue["context_bytes"] != context:
            raise ValueError("the training mixture needs its recorded context and catalogue format")
        self.name = catalogue["name"]
        self.catalogue_sha256 = hashlib.sha256(raw_catalogue).hexdigest()
        sources = catalogue["corpora"]
        weights = catalogue["sampling_weights"]
        if (set(sources) != set(weights) or "enwik8" not in sources
                or any(type(weight) is not int or weight < 1 for weight in weights.values())):
            raise ValueError("record one positive integer sampling weight for every source")
        self.corpora = {}
        for name, spec in sources.items():
            if name == "enwik8":
                data, lengths = enwik8_train, [len(enwik8_train)]
                if hashlib.sha256(data).hexdigest() != spec["sha256"]:
                    raise ValueError("enwik8 train split differs from the training mixture")
            else:
                folder = catalogue_path.parent
                body = _checked_file(folder, spec["file"], spec["sha256"])
                index = _checked_file(folder, spec["spans_file"], spec["spans_sha256"])
                data = np.memmap(body, dtype=np.uint8, mode="r")
                lengths = [int(row.split("\t", 1)[1]) for row in index.read_text(encoding="utf-8").splitlines()]
            if len(data) != spec["bytes"] or len(lengths) != spec["documents"]:
                raise ValueError(f"{name} does not match its recorded training inventory")
            self.corpora[name] = _Corpus(data, lengths, context)
        self.schedule = np.repeat(np.arange(len(self.corpora)), [weights[name] for name in self.corpora])
        self.targets = dict.fromkeys(self.corpora, 0)
        self.windows = 0
        self.digest = hashlib.sha256()

    def batch(self, batch_size, generator):
        indices = self.schedule[(self.windows + np.arange(batch_size)) % len(self.schedule)]
        raw = np.empty((batch_size, self.context + 1), dtype=np.uint8)
        for source, (name, corpus) in enumerate(self.corpora.items()):
            positions = np.flatnonzero(indices == source)
            if positions.size:
                raw[positions] = corpus.draw(positions.size, self.context, generator)
                self.targets[name] += len(positions) * self.context
        self.windows += batch_size
        self.digest.update(raw)
        windows = torch.from_numpy(raw.astype(np.int64))
        return windows[:, :-1], windows[:, 1:]

    def receipt(self):
        return {
            "name": self.name, "catalogue_sha256": self.catalogue_sha256,
            "source_targets": self.targets.copy(),
            "ordered_input_and_target_windows_sha256": self.digest.hexdigest(),
        }
