"""Score a successor model on continuous byte streams with bounded working state."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch

from common import resolve_device
from probe_lm_context import load_model
from snnformer.successor_memory import successor_slot
from train_lm import DATA, ENWIK8_BYTES, ENWIK8_SHA1
from train_successor import next_byte_nll, target_probability


class ChunkSuccessorMemory:
    """Batch writes and reads while carrying the native direct-mapped tables."""

    def __init__(self, orders, slots):
        self.orders = tuple(orders)
        self.slots = slots
        self.position = 0
        self.history = np.zeros(max(orders), dtype=np.int64)
        shape = (len(orders), slots)
        self.tags = np.zeros(shape, dtype=np.int64)
        self.successors = np.zeros(shape, dtype=np.int64)
        self.observed_at = np.full(shape, -1, dtype=np.int64)

    def advance(self, tokens):
        length = len(tokens)
        width = len(self.history)
        history = np.concatenate((self.history, tokens))
        packed = np.zeros(length + 1, dtype=np.int64)
        for offset in range(width):
            first = width - offset - 1
            packed |= history[first : first + length + 1] << (8 * offset)
        self.history = history[-width:].copy()
        indices = np.arange(length)
        positions = self.position + indices
        shape = (length, len(self.orders))
        predictions = np.full(shape, -1, dtype=np.int64)
        ages = np.full(shape, -1, dtype=np.int64)

        for column, order in enumerate(self.orders):
            mask = (1 << (8 * order)) - 1 if order < 8 else -1
            write_tag, read_tag = packed[:-1] & mask, packed[1:] & mask
            write_slot = successor_slot(write_tag, self.slots)
            read_slot = successor_slot(read_tag, self.slots)
            tags = self.tags[column, read_slot]
            values = self.successors[column, read_slot]
            observed = self.observed_at[column, read_slot]

            writes = indices[positions >= order]
            if len(writes):
                # Position is the low part of the sort key. Including equal
                # positions gives read-after-write semantics within each slot.
                keys = write_slot[writes] * length + writes
                by_slot = keys.argsort()
                query = read_slot * length + indices
                preceding = np.searchsorted(keys[by_slot], query, side="right") - 1
                latest = writes[by_slot[preceding.clip(min=0)]]
                in_chunk = (preceding >= 0) & (write_slot[latest] == read_slot)
                tags[in_chunk] = write_tag[latest[in_chunk]]
                values[in_chunk] = tokens[latest[in_chunk]]
                observed[in_chunk] = positions[latest[in_chunk]]

                ordered = writes[by_slot]
                slots = write_slot[ordered]
                terminal = np.r_[slots[:-1] != slots[1:], True]
                last = ordered[terminal]
                slot = slots[terminal]
                self.tags[column, slot] = write_tag[last]
                self.successors[column, slot] = tokens[last]
                self.observed_at[column, slot] = positions[last]

            hit = (positions >= order - 1) & (observed >= 0) & (tags == read_tag)
            predictions[hit, column] = values[hit]
            ages[hit, column] = positions[hit] - observed[hit]

        self.position += length
        return predictions, ages


def local_log_probs(model, data, start, end, overlap, device):
    first = max(0, start - overlap)
    tokens = torch.tensor(data[first:end], dtype=torch.long, device=device)[None, :]
    if model.cfg.attn == "transformer_local":
        log_probs = model(tokens, position_offset=first).log_softmax(dim=-1, dtype=torch.float32)
    else:
        log_probs = model.local_log_probs(tokens, position_offset=first)
    return log_probs[:, start - first :]


def stream_cache(data, chunk, orders, slots):
    """Yield verified successors and their causal statistics for one stream."""
    store = ChunkSuccessorMemory(orders, slots)
    last_prediction = np.full((1, len(orders)), -1, dtype=np.int64)
    observed = np.zeros(len(orders), dtype=np.int64)
    correct = np.zeros_like(observed)
    for start in range(0, len(data) - 1, chunk):
        end = min(start + chunk, len(data) - 1)
        tokens = data[start:end]
        successors, ages = store.advance(tokens)
        previous = np.concatenate((last_prediction, successors[:-1]))
        running_observed = np.cumsum(previous >= 0, axis=0) + observed
        running_correct = np.cumsum(previous == tokens[:, None], axis=0) + correct
        last_prediction = successors[-1:].copy()
        observed, correct = running_observed[-1].copy(), running_correct[-1].copy()
        yield start, end, (successors, ages, running_observed, running_correct)


def stream_evidence(model, data, chunk, device):
    """Yield the model evidence without resetting between scoring chunks."""
    overlap = len(model.blocks) * (model.cfg.local_window - 1)
    for start, end, evidence in stream_cache(data, chunk, model.successors.orders, model.successors.slots):
        log_probs = local_log_probs(model, data, start, end, overlap, device)
        yield (log_probs, *(torch.as_tensor(array, device=device)[None, :] for array in evidence),
               torch.tensor(data[start + 1 : end + 1], dtype=torch.long, device=device)[None, :])


@torch.inference_mode()
def score(model, data, chunk, device):
    if len(data) < 2:
        raise ValueError("each evaluation stream requires at least two bytes")
    receptive_field = 1 + len(model.blocks) * (model.cfg.local_window - 1)
    sums = np.zeros(4)
    hits = far_hits = far_wins = count = 0
    for log_probs, successors, ages, observed, correct, targets in stream_evidence(model, data, chunk, device):
        base_p = target_probability(log_probs, targets)
        wins = successors == targets[..., None]
        pulses = model.readout.strength(log_probs, successors, ages, observed, correct, quantized=True)
        soft = model.readout.strength(log_probs, successors, ages, observed, correct, quantized=False)
        # State and readout evidence remain intact. Only the mass assigned to
        # successors inside the transformer's receptive field is removed.
        far = ages >= receptive_field
        losses = (-base_p.log(), next_byte_nll(base_p, wins, pulses),
                  next_byte_nll(base_p, wins, soft), next_byte_nll(base_p, wins, pulses * far))
        sums += [term.cpu().sum(dtype=torch.float64).item() for term in losses]
        hits += (successors >= 0).any(-1).sum().item()
        far_hits += far.any(-1).sum().item()
        far_wins += (wins & far).any(-1).sum().item()
        count += targets.numel()
    metrics = dict(zip(("base_bpb", "spike_bpb", "analog_bpb", "remote_only_bpb"),
                       (sums / (count * math.log(2))).tolist()))
    metrics.update(target_bytes=count, hit_rate=hits / count, remote_hit_rate=far_hits / count,
                   remote_correct_rate=far_wins / max(1, far_hits), remote_min_age=receptive_field)
    return metrics


def enwik8_test(path):
    if path.stat().st_size != ENWIK8_BYTES:
        raise ValueError(f"{path} must be the canonical raw enwik8 file")
    digest = hashlib.sha1(usedforsecurity=False)
    with path.open("rb") as source:
        while block := source.read(1 << 20):
            digest.update(block)
    if digest.hexdigest() != ENWIK8_SHA1:
        raise ValueError(f"{path} checksum does not match canonical enwik8")
    return np.memmap(path, dtype=np.uint8, mode="r", offset=95_000_000)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=Path(os.environ.get("ENWIK8_PATH", DATA)))
    parser.add_argument("--skip-enwik8", action="store_true", help="score only the requested raw files")
    parser.add_argument("--raw", type=Path, nargs="+", default=[], help="score independent raw byte streams")
    parser.add_argument("--chunk", type=int, default=4096, help="scoring chunk size without resetting memory")
    parser.add_argument("--threads", type=int, default=3)
    parser.add_argument("--device", choices=("cpu", "cuda", "mps", "auto"), default="cpu")
    parser.add_argument("--json", type=Path, help="optional path for the score report")
    args = parser.parse_args()
    if min(args.chunk, args.threads) < 1:
        parser.error("chunk size and thread count must be positive")
    if args.skip_enwik8 and not args.raw:
        parser.error("provide --raw streams if --skip-enwik8 is set")
    torch.set_num_threads(args.threads)
    start = time.monotonic()
    try:
        device = resolve_device(args.device)
        model, name = load_model(args.checkpoint.expanduser(), device)
        if model.cfg.attn != "transformer_successor":
            raise ValueError("expected a successor-model checkpoint")
        streams = [] if args.skip_enwik8 else [("enwik8 test", enwik8_test(args.data.expanduser()))]
        streams.extend((str(path), np.memmap(path.expanduser(), dtype=np.uint8, mode="r")) for path in args.raw)
        results = []
        for source, data in streams:
            metrics = {"source": source, **score(model, data, args.chunk, device)}
            results.append(metrics)
            print(metrics, flush=True)
    except (OSError, KeyError, RuntimeError, ValueError) as exc:
        parser.error(str(exc))
    if args.json is not None:
        payload = {
            "checkpoint": str(args.checkpoint), "run_name": name, "device": str(device),
            "data_sha1": None if args.skip_enwik8 else ENWIK8_SHA1, "chunk": args.chunk,
            "orders": list(model.successors.orders), "slots": model.successors.slots,
            "results": results, "seconds": time.monotonic() - start,
        }
        path = args.json.expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
