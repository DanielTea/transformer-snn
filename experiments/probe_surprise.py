"""Measure where a frozen decoder is surprised and what a successor memory could add.

The probe scores one continuous enwik8 split and records three candidate
surprise signals for every arriving byte. It then asks two questions that
decide the surprise-clock design before any training:

1. Does surprise concentrate at word starts, so that a clock driven by it
   would tick about once per word?
2. How much loss would an exact "what followed this context before" memory
   remove at several reaches, with a logit-additive readout?
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.nn import functional as F

from common import _atomic_output, resolve_device
from eval_event_lm import _logits_and_targets
from eval_stream_lm import Book
from snnformer import count_params
from snnformer.event_lm import EventLM, _CorrectionAccess, _DelayState
from train_event_lm import load_checkpoint, scoring_logits
from train_lm import ENWIK8_SHA1, load_splits


ORDERS = (3, 4, 6, 8, 12, 16)
REACHES = (512, 4096, 65536, None)
SIGNALS = ("surprisal_bits", "vocabulary_deficit", "code_error")
_WORD_SLOTS = ("start", "second", "third", "fourth", "fifth", "later", "outside")


def word_positions(data: np.ndarray) -> np.ndarray:
    """Index of each byte inside its word, or -1 outside a word.

    Letters, digits and every byte of a multi-byte UTF-8 character count as
    word bytes. Everything else, including markup, separates words.
    """
    table = np.zeros(256, dtype=bool)
    for low, high in ((48, 58), (65, 91), (97, 123), (128, 256)):
        table[low:high] = True
    inside = table[data]
    index = np.arange(len(data))
    last_outside = np.maximum.accumulate(np.where(inside, -1, index))
    return np.where(inside, index - last_outside - 1, -1)


def previous_occurrence(data: np.ndarray, order: int, stored: np.ndarray | None = None) -> np.ndarray:
    """For each context ending at i, the end of its most recent earlier copy, or -1.

    With ``stored`` only the marked occurrences can be recalled.
    """
    count = len(data) - order + 1
    keys = np.zeros(count, dtype=np.uint64)
    multiplier = np.uint64(1099511628211)
    with np.errstate(over="ignore"):
        for offset in range(order):
            keys = keys * multiplier + data[offset:offset + count].astype(np.uint64) + np.uint64(1)
    ranked = np.argsort(keys, kind="stable")
    sorted_keys = keys[ranked]
    slots = np.arange(count)
    if stored is None:
        earlier = slots - 1
    else:
        marked = stored[ranked + order - 1]
        latest = np.maximum.accumulate(np.where(marked, slots, -1))
        earlier = np.concatenate(([-1], latest[:-1]))
    valid = earlier >= 0
    valid[valid] = sorted_keys[earlier[valid]] == sorted_keys[valid]
    previous = np.full(len(data), -1, dtype=np.int64)
    previous[ranked[valid] + order - 1] = ranked[earlier[valid]] + order - 1
    return previous


@torch.inference_mode()
def score_stream(model, data, device, *, context, history, batch_size, candidates, counts_file=None):
    """Per-target log probabilities, surprise signals, and candidate log probabilities."""
    event = isinstance(model, EventLM)
    targets = len(data) - 1
    bits = np.zeros(targets, dtype=np.float32)
    deficit = np.zeros(targets, dtype=np.float32)
    correct = np.zeros(targets, dtype=bool)
    code_error = np.zeros(targets, dtype=np.float32) if event and model.cfg.route == "steered_correction_code" else None
    candidate_logp = np.zeros((targets, candidates.shape[1]), dtype=np.float32)
    stored = None
    if counts_file is not None:
        stored = np.lib.format.open_memmap(counts_file, mode="w+", dtype=np.int16 if event else np.float32,
                                           shape=(targets, 256))
    tensor = torch.from_numpy(data.copy())
    candidate_ids = torch.from_numpy(candidates.clip(min=0))

    def record(start, logits, labels, raw):
        stop = start + labels.numel()
        logp = logits.log_softmax(-1)
        chosen = logp.gather(-1, labels[:, None]).squeeze(-1)
        bits[start:stop] = (-chosen / math.log(2)).cpu().numpy()
        lead = raw.amax(-1) - raw.gather(-1, labels[:, None]).squeeze(-1)
        deficit[start:stop] = lead.float().cpu().numpy()
        correct[start:stop] = (raw.argmax(-1) == labels).cpu().numpy()
        ids = candidate_ids[start:stop].to(logp.device)
        candidate_logp[start:stop] = logp.gather(-1, ids).cpu().numpy()
        if stored is not None:
            stored[start:stop] = raw.cpu().numpy()
        return stop

    position = 0
    if code_error is not None:
        resolution = model.cfg.resolution
        keys = model.cfg.forecast_key_channels
        every_byte = torch.arange(256, device=device)
        observed = model.input_emit(model.input(every_byte) * resolution)
        byte_codes = model.prediction_memory.observation_code(observed)[..., keys:]
        layers = [_DelayState() for _ in model.blocks]
        correction = None
        for start in range(0, targets, context):
            window = tensor[start:start + context + 1].unsqueeze(0).to(device=device, dtype=torch.long)
            access = _CorrectionAccess(correction, 0, None, None, False, local_layers=layers, local_position=start)
            counts, forecast, _ = model(window[:, :-1], correction_access=access, return_prediction=True)
            correction = access.state
            layers = [copy.copy(layer) for layer in layers]
            labels = window[0, 1:]
            error = (byte_codes[labels] - forecast[0, ..., keys:]).abs().sum((-1, -2))
            code_error[start:start + labels.numel()] = (error / resolution).cpu().numpy()
            position = record(position, scoring_logits(counts[0], True, resolution), labels, counts[0])
    else:
        options = SimpleNamespace(context=context, history=history, batch_size=batch_size)
        book = Book(0, "enwik8", 0, "", tensor)
        resolution = model.cfg.resolution if event else 1
        for logits, labels in _logits_and_targets(model, book, options, device, {}):
            logits, labels = logits.flatten(0, 1), labels.flatten()
            raw = logits * (8 * resolution / math.log(2)) if event else logits
            position = record(position, logits, labels, raw)
    if position != targets:
        raise ValueError("scored targets do not cover the split")
    if stored is not None:
        stored.flush()
    signals = {"surprisal_bits": bits, "vocabulary_deficit": deficit}
    if code_error is not None:
        signals["code_error"] = code_error
    return signals, correct, candidate_logp


def word_profile(signal, slots):
    total = float(signal.sum())
    rows = {}
    for index, name in enumerate(_WORD_SLOTS):
        chosen = slots == index
        count = int(chosen.sum())
        mass = float(signal[chosen].sum())
        rows[name] = {"targets": count, "share_of_targets": count / len(signal),
                      "mean": mass / max(count, 1), "share_of_surprise": mass / total}
    start = rows["start"]
    rows["start_enrichment"] = start["share_of_surprise"] / start["share_of_targets"]
    return rows


def clock_ticks(signal, threshold, leak=0.0):
    """One integrate-and-fire neuron with subtractive reset and at most one tick per byte.

    A positive leak is a constant inhibitory current with the membrane held at or above rest.
    """
    ticks = np.zeros(len(signal), dtype=bool)
    membrane = 0.0
    for index, value in enumerate(signal.tolist()):
        membrane = max(membrane + value - leak, 0.0)
        if membrane >= threshold:
            ticks[index] = True
            membrane -= threshold
    return ticks


def threshold_for_spacing(signal, spacing, leak, sample=400_000):
    """Bisect the firing threshold that yields the requested bytes per tick."""
    part = signal[:sample]
    low, high = 0.0, float(part.sum())
    for _ in range(30):
        middle = (low + high) / 2
        if clock_ticks(part, middle, leak).sum() * spacing > len(part):
            low = middle
        else:
            high = middle
    return (low + high) / 2


def clock_profile(ticks, slots):
    """How ticks line up with words. Ticks and slots describe the arriving byte."""
    starts = slots == 0
    inside = slots != len(_WORD_SLOTS) - 1
    word_id = np.cumsum(starts)
    per_word = np.bincount(word_id[inside & ticks], minlength=word_id[-1] + 1)[1:]
    spacing = np.diff(np.flatnonzero(ticks))
    return {
        "ticks": int(ticks.sum()),
        "bytes_per_tick": len(ticks) / max(int(ticks.sum()), 1),
        "spacing_standard_deviation": float(spacing.std()) if len(spacing) else None,
        "share_of_ticks_on_word_start": float(ticks[starts].sum() / max(ticks.sum(), 1)),
        "share_of_ticks_on_first_two_bytes": float(ticks[slots <= 1].sum() / max(ticks.sum(), 1)),
        "share_of_word_starts_with_a_tick": float(ticks[starts].mean()),
        "words": int(len(per_word)),
        "share_of_words_without_tick": float((per_word == 0).mean()),
        "share_of_words_with_one_tick": float((per_word == 1).mean()),
        "share_of_words_with_more_ticks": float((per_word > 1).mean()),
    }


def memory_ceiling(target_logp, labels, candidates, candidate_logp, distances, reach, fit):
    """Fit one logit boost per context length on the first part, score the rest."""
    valid = (candidates >= 0) & ((distances <= reach) if reach is not None else True)
    same = candidates[:, :, None] == candidates[:, None, :]
    same &= valid[:, :, None] & valid[:, None, :]
    share = 1.0 / np.maximum(same.sum(-1), 1)
    hits = valid & (candidates == labels[:, None])
    tensors = [torch.from_numpy(np.ascontiguousarray(value)) for value in (
        target_logp, candidate_logp, same.astype(np.float32), (share * valid).astype(np.float32),
        hits.astype(np.float32),
    )]

    def loss(boosts, rows):
        logp, options, agree, weight, hit = (value[rows] for value in tensors)
        gain = torch.einsum("tij,j->ti", agree, boosts)
        partition = 1 + (weight * options.exp() * torch.expm1(gain)).sum(-1)
        return -(logp + hit @ boosts - partition.log()).mean() / math.log(2)

    boosts = torch.zeros(candidates.shape[1], requires_grad=True)
    optimizer = torch.optim.LBFGS([boosts], lr=0.5, max_iter=80, line_search_fn="strong_wolfe")
    train, held_out = slice(0, fit), slice(fit, len(labels))

    def closure():
        optimizer.zero_grad()
        value = loss(boosts, train)
        value.backward()
        return value

    optimizer.step(closure)
    with torch.no_grad():
        base = float(-tensors[0][held_out].mean() / math.log(2))
        with_memory = float(loss(boosts, held_out))
        longest = valid[fit:].any(-1)
        return {
            "reach_bytes": reach, "base_bpb": base, "memory_bpb": with_memory, "gain_bpb": base - with_memory,
            "boost_nats_by_order": dict(zip(map(str, ORDERS), boosts.tolist(), strict=True)),
            "share_of_targets_with_candidate": float(longest.mean()),
            "candidate_accuracy_by_order": {
                str(order): float(hits[fit:, index].sum() / max(valid[fit:, index].sum(), 1))
                for index, order in enumerate(ORDERS)
            },
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--enwik8", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True, nargs="+")
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--bytes", type=int, help="score only this leading part of the split")
    parser.add_argument("--device", choices=("cpu", "mps", "cuda", "auto"), default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--spacing", type=float, nargs="+", default=[4.0, 5.0, 6.0],
                        help="target bytes per clock tick")
    parser.add_argument("--cache-dir", type=Path, help="also keep every output count for later readout fits")
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.json.expanduser().exists() and not args.force:
        parser.error(f"{args.json} already exists. Choose a new path or use --force to replace it.")
    torch.set_num_threads(args.threads)
    device = resolve_device(args.device)
    splits = dict(zip(("train", "val", "test"), load_splits(args.enwik8.expanduser()), strict=True))
    data = np.ascontiguousarray(splits[args.split][:args.bytes])
    wide = data.astype(np.int64)
    labels = wide[1:]
    targets = len(labels)
    in_word = word_positions(data)[1:]
    slots = np.where(in_word < 0, len(_WORD_SLOTS) - 1, np.minimum(in_word, len(_WORD_SLOTS) - 2))

    previous = np.stack([previous_occurrence(data, order)[:-1] for order in ORDERS], axis=1)
    index = np.arange(targets)[:, None]
    candidates = np.where(previous >= 0, wide[np.clip(previous + 1, 0, None)], -1)
    distances = np.where(previous >= 0, index - previous, np.iinfo(np.int64).max)

    models = []
    for path in args.checkpoint:
        model, checkpoint = load_checkpoint(path.expanduser(), device)
        event = isinstance(model, EventLM)
        counts_file = None
        if args.cache_dir is not None:
            args.cache_dir.mkdir(parents=True, exist_ok=True)
            counts_file = args.cache_dir / f"{checkpoint['run_name']}_{args.split}_outputs.npy"
        signals, correct, candidate_logp = score_stream(
            model, data, device, context=8192 if event else 256, history=256, batch_size=8,
            candidates=candidates, counts_file=counts_file,
        )
        if args.cache_dir is not None:
            np.savez(args.cache_dir / f"{checkpoint['run_name']}_{args.split}_signals.npz", correct=correct, **signals)
        target_logp = -signals["surprisal_bits"].astype(np.float64) * math.log(2)
        record = {
            "run_name": checkpoint["run_name"], "checkpoint": str(path), "parameters": count_params(model),
            "target_bytes": targets, "bpb": float(signals["surprisal_bits"].mean()),
            "argmax_accuracy": float(correct.mean()),
            "surprise_by_word_position": {name: word_profile(signal, slots) for name, signal in signals.items()},
        }
        clocks = {}
        for name, signal in signals.items():
            rows = {}
            for spacing in args.spacing:
                threshold = spacing * float(signal.mean())
                rows[str(spacing)] = {"threshold": threshold, **clock_profile(clock_ticks(signal, threshold), slots)}
            clocks[name] = rows
        mean_spacing = args.spacing[len(args.spacing) // 2]
        for name in ("surprisal_bits", "vocabulary_deficit"):
            signal = signals[name]
            mean = float(signal.mean())
            rows = {}
            for leak in (0.5, 1.0, 2.0):
                threshold = threshold_for_spacing(signal, mean_spacing, leak * mean)
                rows[str(leak)] = {"leak": leak * mean, "threshold": threshold,
                                   **clock_profile(clock_ticks(signal, threshold, leak * mean), slots)}
            cut = float(np.quantile(signal, 1 - 1 / mean_spacing))
            rows["largest_only"] = {"threshold": cut, **clock_profile(signal >= max(cut, 1e-9), slots)}
            clocks[f"{name}_leaky_at_{mean_spacing}"] = rows
        generator = np.random.default_rng(0)
        shuffled = generator.permutation(signals["surprisal_bits"])
        clocks["controls"] = {
            str(spacing): {
                "periodic": clock_profile(np.arange(targets) % round(spacing) == 0, slots),
                "shuffled_surprisal": clock_profile(
                    clock_ticks(shuffled, spacing * float(shuffled.mean())), slots),
            } for spacing in args.spacing
        }
        record["clock"] = clocks
        fit = targets // 2
        record["successor_memory"] = {
            "fit_targets": fit, "held_out_targets": targets - fit, "orders": list(ORDERS),
            "store_every_byte": [
                memory_ceiling(target_logp.astype(np.float32), labels, candidates, candidate_logp, distances, reach, fit)
                for reach in REACHES
            ],
        }
        if counts_file is not None:
            # A memory written only on mistakes can recall only those occurrences.
            mistakes = np.concatenate((~correct, [False]))
            gated = np.stack([previous_occurrence(data, order, mistakes)[:-1] for order in ORDERS], axis=1)
            gated_candidates = np.where(gated >= 0, wide[np.clip(gated + 1, 0, None)], -1)
            gated_distances = np.where(gated >= 0, index - gated, np.iinfo(np.int64).max)
            outputs = np.load(counts_file, mmap_mode="r")
            scale = math.log(2) / (8 * model.cfg.resolution) if event else 1.0
            gated_logp = np.zeros_like(candidate_logp)
            for start in range(0, targets, 1 << 18):
                rows = slice(start, start + (1 << 18))
                normalized = (torch.from_numpy(np.array(outputs[rows])).float() * scale).log_softmax(-1)
                ids = torch.from_numpy(gated_candidates[rows].clip(min=0))
                gated_logp[rows] = normalized.gather(-1, ids).numpy()
            record["successor_memory"]["store_only_mistakes"] = [
                memory_ceiling(target_logp.astype(np.float32), labels, gated_candidates, gated_logp,
                               gated_distances, reach, fit)
                for reach in REACHES
            ]
        models.append(record)
        print(f"{record['run_name']}: {record['bpb']:.5f} bpb", flush=True)
        del model

    result = {
        "dataset": "enwik8", "data_sha1": ENWIK8_SHA1, "split": args.split,
        "scored_sha256": hashlib.sha256(data.tobytes()).hexdigest(), "target_bytes": targets,
        "word_slots": list(_WORD_SLOTS), "device": str(device), "models": models,
    }
    with _atomic_output(args.json.expanduser(), "w", args.force) as out:
        json.dump(result, out, indent=2)
        out.write("\n")


if __name__ == "__main__":
    main()
