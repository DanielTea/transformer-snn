"""Score EventLM and matched Transformers on enwik8 or verified PG-19 text."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import torch
from torch.nn import functional as F

from benchmark_decode import state_bytes
from common import _atomic_output, make_lm, resolve_device
from eval_stream_lm import Book, PG19_MANIFEST, load_books, score_book
from snnformer import SNNConfig, count_params
from snnformer.event_lm import EventLM, EventLMConfig
from train_event_lm import scoring_logits
from train_lm import ENWIK8_SHA1, load_splits


_AGE_BINS = ("unseen", "1-31", "32-127", "128-511", "512+")


def load_checkpoint(path: Path, device: torch.device):
    """Load a training checkpoint with its original floating-point parameters."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("task") != "event_lm":
        raise ValueError(f"{path}: expected a train_event_lm.py checkpoint")
    if checkpoint.get("model_type") == "EventLM":
        model = EventLM(EventLMConfig(**checkpoint["cfg"]))
    else:
        cfg = SNNConfig(**checkpoint["cfg"])
        options = checkpoint["model_kwargs"]
        if cfg.attn not in ("transformer", "transformer_local") or options.get("vocab", 256) != 256:
            raise ValueError(f"{path}: expected a byte-level full or local Transformer")
        model = make_lm(cfg, **options)
    model.load_state_dict(checkpoint["state_dict"])
    return model.to(device).eval(), checkpoint


def _prefix_ages(data, order=8):
    """Distance from each target's prefix to its prior occurrence in the passage."""
    raw = data.numpy().tobytes()
    ages = torch.zeros(len(raw) - 1, dtype=torch.int32).numpy()
    latest = {}
    for end in range(order, len(raw)):
        prefix = raw[end - order:end]
        previous = latest.get(prefix)
        if previous is not None:
            ages[end - 1] = end - previous
        latest[prefix] = end
    return torch.bucketize(torch.from_numpy(ages), torch.tensor([1, 32, 128, 512]), right=True)


def _loss_summary(nll, targets):
    return {"target_bytes": targets, "nll_nats": nll, "bpb": nll / (targets * math.log(2)) if targets else None}


def _logits_and_targets(model, book, args, device, options, counters=None):
    data = book.data
    event = isinstance(model, EventLM)
    resolution = model.cfg.resolution if event else 1
    if event and model.has_correction:
        state = None
        for start in range(0, data.numel() - 1, args.context):
            window = data[start:start + args.context + 1].unsqueeze(0).to(device=device, dtype=torch.long)
            measured = {} if counters is not None else None
            counts, state = model.forward_chunk(window[:, :-1], state, stats=measured, **options)
            if measured is not None:
                for key, value in measured.items():
                    counters[key] = counters.get(key, 0) + int(value.item())
            yield scoring_logits(counts, event, resolution), window[:, 1:]
        return

    full_windows, tail = divmod(data.numel() - 1, args.context)
    history = args.history
    if history or not full_windows:
        window = data[:args.context + 1].unsqueeze(0).to(device=device, dtype=torch.long)
        yield scoring_logits(model(window[:, :-1], **options), event, resolution), window[:, 1:]
    first_full = int(bool(history))
    if full_windows > first_full:
        start = first_full * args.context - history
        windows = data[start:].unfold(0, history + args.context + 1, args.context)
        for window in windows.split(args.batch_size):
            window = window.to(device=device, dtype=torch.long)
            logits = scoring_logits(model(window[:, :-1], **options), event, resolution)
            yield logits[:, history:], window[:, history + 1:]
    if full_windows and tail:
        window = data[full_windows * args.context - history:].unsqueeze(0).to(device=device, dtype=torch.long)
        logits = scoring_logits(model(window[:, :-1], **options), event, resolution)
        yield logits[:, history:], window[:, history + 1:]


@torch.inference_mode()
def _fit_temperature(model, books, args, device):
    losses = torch.zeros(len(args.temperatures), device=device)
    for book in books:
        for logits, labels in _logits_and_targets(model, book, args, device, {}):
            logits, labels = logits.flatten(0, 1), labels.flatten()
            losses += torch.stack([
                F.cross_entropy(logits / temperature, labels, reduction="sum") for temperature in args.temperatures
            ])
    return args.temperatures[int(losses.argmin().item())]


@torch.inference_mode()
def _score_passage(model, book, ages, args, device, options, temperature):
    """Score disjoint targets using continuous state or sufficient finite history."""
    masks = options.get("hide_age")
    simultaneous = isinstance(masks, tuple)
    paths = len(masks) if simultaneous else 1
    losses = [[] for _ in range(paths)]
    correct = [0] * paths
    raw_nll = [0.0] * paths
    counters = {} if args.memory_stats and isinstance(model, EventLM) and model.has_correction else None
    for logits, targets in _logits_and_targets(model, book, args, device, options, counters):
        targets = targets.flatten()
        for path, scores in enumerate(logits.reshape(paths, -1, logits.shape[-1]).unbind(0)):
            losses[path].append(F.cross_entropy(scores / temperature, targets, reduction="none").cpu())
            correct[path] += int((scores.argmax(-1) == targets).sum().item())
            if temperature != 1:
                raw_nll[path] += F.cross_entropy(scores, targets, reduction="sum").item()
    counts = ages.bincount(minlength=len(_AGE_BINS))
    results = []
    for path, chunks in enumerate(losses):
        errors = torch.cat(chunks)
        if errors.numel() != ages.numel() or not errors.isfinite().all():
            raise ValueError(f"{book.name}: nonfinite loss or inconsistent target count")
        groups = torch.zeros(len(_AGE_BINS), dtype=torch.float64).scatter_add_(0, ages, errors.to(torch.float64))
        results.append({
            "book_id": book.book_id, "excerpt_file": book.name, "excerpt_sha256": book.excerpt_sha256,
            "source_start_raw_byte": book.source_start, "excerpt_bytes_used": book.data.numel(),
            **_loss_summary(groups.sum().item(), errors.numel()),
            "raw_nll_nats": raw_nll[path] if temperature != 1 else groups.sum().item(),
            "correct_bytes": correct[path],
            **({"memory_stats": counters} if counters is not None else {}),
            "prefix_age": {key: _loss_summary(nll, count) for key, nll, count in
                           zip(_AGE_BINS, groups.tolist(), counts.tolist(), strict=True)},
        })
    return results if simultaneous else results[0]


def _evaluate(model, checkpoint, path, books, ages, args, device, options, temperature, streaming_bytes,
              *, scored_books=None, full_attention_context=None):
    name = checkpoint["run_name"]
    event = isinstance(model, EventLM)
    correction = event and model.has_correction
    resolution = model.cfg.resolution if event else 1

    def score_logits(tokens):
        return scoring_logits(model(tokens, **options), event, resolution)

    detailed = args.history or correction or temperature != 1
    if scored_books is not None:
        per_book = list(scored_books)
    elif detailed:
        if ages is None:
            ages = [_prefix_ages(book.data) for book in books]
        per_book = [_score_passage(model, book, age, args, device, options, temperature)
                    for book, age in zip(books, ages, strict=True)]
    else:
        per_book = [score_book(score_logits, book, args.context, args.batch_size, device) for book in books]
    nll = math.fsum(book["nll_nats"] for book in per_book)
    targets = sum(book["target_bytes"] for book in per_book)
    stats = _loss_summary(nll, targets)
    if detailed:
        stats["raw_bpb"] = math.fsum(book["raw_nll_nats"] for book in per_book) / (targets * math.log(2))
        stats["argmax_accuracy"] = sum(book["correct_bytes"] for book in per_book) / targets
        stats["prefix_age"] = {
            group: _loss_summary(math.fsum(book["prefix_age"][group]["nll_nats"] for book in per_book),
                                 sum(book["prefix_age"][group]["target_bytes"] for book in per_book))
            for group in _AGE_BINS
        }
    print(f"{name} | {targets:,} targets | {stats['bpb']:.6f} bpb" + (f" | {options}" if options else ""),
          flush=True)
    return {
        "run_name": name,
        "checkpoint": str(path),
        "parameters": count_params(model),
        "streaming_tensor_bytes": streaming_bytes,
        **({"full_attention_context_bytes": full_attention_context} if full_attention_context is not None else {}),
        "scoring": "normalized exponential of scaled output" if event else "softmax over logits",
        "temperature": temperature,
        "continuous_correction_state": correction,
        "training_context_bytes": checkpoint["args"]["ctx"],
        **({"intervention": options} if options else {}),
        **stats,
        "books": per_book,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    dataset = parser.add_mutually_exclusive_group(required=True)
    dataset.add_argument("--books-dir", type=Path, help="directory of prepared PG-19 excerpts")
    dataset.add_argument("--enwik8", type=Path, help="canonical enwik8 file, score the complete test split")
    parser.add_argument("--manifest", type=Path, default=PG19_MANIFEST)
    parser.add_argument("--checkpoint", required=True, type=Path, nargs="+")
    parser.add_argument("--context", type=int, default=256)
    parser.add_argument("--history", type=int, default=0, help="preceding unscored bytes before each target chunk")
    parser.add_argument("--delay-limit", type=int, action="append", default=[],
                        help="also evaluate selectable-delay models with taps older than this removed")
    parser.add_argument("--shuffle-phase", action="store_true",
                        help="also evaluate timing models with start ticks rotated across channels")
    parser.add_argument("--hide-memory-age", type=int, action="append", default=[],
                        help="also score with memory cells hidden from reads after this many bytes")
    parser.add_argument("--memory-stats", action="store_true", help="record correction traffic and read age")
    parser.add_argument("--shuffle-memory-keys", action="store_true",
                        help="also score stored keys rotated across slots without altering writes")
    parser.add_argument("--calibration-books", type=Path, help="directory of verified PG-19 validation excerpts")
    parser.add_argument("--calibration-manifest", type=Path)
    parser.add_argument("--temperatures", type=float, nargs="+", default=[0.8, 1.0, 1.25, 1.5, 1.75, 2.0])
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--device", choices=("cpu", "mps", "cuda", "auto"), default="cpu")
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--force", action="store_true", help="replace the JSON record")
    args = parser.parse_args()
    if min(args.context, args.batch_size, args.threads) < 1:
        parser.error("context, batch size, and threads must be positive")
    if not 0 <= args.history <= args.context or any(limit < 1 for limit in args.delay_limit):
        parser.error("history must be within the context and delay limits must be positive")
    if any(age < 1 for age in args.hide_memory_age):
        parser.error("memory ages must be positive")
    if bool(args.calibration_books) != bool(args.calibration_manifest):
        parser.error("calibration requires both its manifest and its book directory")
    if any(not math.isfinite(t) or t <= 0 for t in args.temperatures):
        parser.error("temperatures must be finite and positive")
    if args.json.expanduser().exists() and not args.force:
        parser.error(f"{args.json} already exists. Choose a new path or use --force to replace it.")
    torch.set_num_threads(args.threads)
    try:
        device = resolve_device(args.device)
        if args.enwik8 is None:
            manifest = args.manifest.expanduser().read_bytes()
            books = load_books(manifest, args.books_dir.expanduser(), max_bytes=0)
            source = {"dataset": "PG-19", "manifest": str(args.manifest),
                      "manifest_sha256": hashlib.sha256(manifest).hexdigest()}
        else:
            _, _, raw = load_splits(args.enwik8.expanduser())
            digest = hashlib.sha256(raw).hexdigest()
            books = [Book(0, "enwik8 test", 95_000_000, digest, torch.from_numpy(raw.copy()))]
            source = {"dataset": "enwik8", "data_sha1": ENWIK8_SHA1}
        ages = [_prefix_ages(book.data) for book in books] if args.history else None
        validation = None
        if args.calibration_books:
            calibration = args.calibration_manifest.read_bytes()
            validation = load_books(calibration, args.calibration_books, max_bytes=0, split="validation")
            if args.books_dir and {book.book_id for book in validation} & {book.book_id for book in books}:
                raise ValueError("calibration and evaluation must use disjoint books")
            source["calibration"] = {"manifest": str(args.calibration_manifest),
                                     "manifest_sha256": hashlib.sha256(calibration).hexdigest(),
                                     "temperatures": args.temperatures}
        models = []
        for path in args.checkpoint:
            model, checkpoint = load_checkpoint(path.expanduser(), device)
            temperature = 1.0 if validation is None else _fit_temperature(model, validation, args, device)
            streaming_bytes = None
            full_attention_context = None
            byte = books[0].data[:1].long()
            if isinstance(model, EventLM) and not model.cfg.continuous:
                _, state = model.to_integer().step(byte)
                streaming_bytes = state_bytes(state)
            elif not isinstance(model, EventLM) and model.cfg.attn == "transformer_local":
                _, state = model.step(byte.to(device))
                streaming_bytes = state_bytes(state)
            elif not isinstance(model, EventLM):
                # Each scored window uses its full causal prefix. Report the
                # K/V tensor size for the longest such prefix.
                targets = max(book.data.numel() - 1 for book in books)
                full_attention_context = max(min(targets, args.context),
                                             args.history + min(args.context, targets - args.context))
                kv_per_byte = 2 * sum(block.attn.heads * block.attn.head_dim for block in model.blocks)
                streaming_bytes = full_attention_context * kv_per_byte * model.tok.weight.element_size()
            conditions = [{}]
            if isinstance(model, EventLM):
                if model.blocks[0].selector is not None:
                    conditions.extend({"delay_limit": limit} for limit in args.delay_limit)
                if args.shuffle_phase and model.blocks[0].phase_bits:
                    conditions.append({"shuffle_phase": True})
                if model.has_correction:
                    conditions.extend({"hide_age": age} for age in args.hide_memory_age)
            if isinstance(model, EventLM) and model.prediction_memory is not None and args.hide_memory_age:
                # Only the last read changes. Score all masks with the same
                # local spikes, interval scan, and per-book write statistics.
                grouped = {"hide_age": (None, *args.hide_memory_age)}
                book_ages = ages if ages is not None else [_prefix_ages(book.data) for book in books]
                per_book = [_score_passage(model, book, age, args, device, grouped, temperature)
                            for book, age in zip(books, book_ages, strict=True)]
                scores = zip(*per_book, strict=True)
            else:
                scores = (None for _ in conditions)
            models.extend(_evaluate(model, checkpoint, path, books, ages, args, device, options, temperature,
                                    streaming_bytes, scored_books=rows, full_attention_context=full_attention_context)
                          for options, rows in zip(conditions, scores, strict=True))
            if args.shuffle_memory_keys and isinstance(model, EventLM) and model.cfg.match_forecast_key:
                models.append(_evaluate(model, checkpoint, path, books, ages, args, device,
                                        {"shuffle_keys": True}, temperature, streaming_bytes))
        result = {
            **source, "split": "test",
            "protocol": "causal_warmed_windows" if args.history else "causal_reset_windows",
            "device": str(device), "context_bytes": args.context, "window_stride_bytes": args.context,
            "unscored_history_bytes": args.history,
            "batch_size": args.batch_size, "unscored_leading_bytes_per_book": 1,
            "target_bytes": sum(book.data.numel() - 1 for book in books),
            "models": models,
        }
        if any(model["target_bytes"] != result["target_bytes"] for model in result["models"]):
            raise ValueError("model target counts differ from the corpus")
        with _atomic_output(args.json.expanduser(), "w", args.force) as out:
            json.dump(result, out, indent=2)
            out.write("\n")
    except (OSError, ValueError, RuntimeError, KeyError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
