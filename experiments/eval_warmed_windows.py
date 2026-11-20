"""Score checkpoints on a complete enwik8 split or on PG-19 books in warmed windows.

Every window starts from an empty state and holds ``--history`` unscored bytes
before its ``--context`` scored ones, so each target after the first
``--history`` bytes has at least that many bytes before it. This is the
protocol of the Transformer's reference score. A history longer than the
scored bytes, as 448 before 64, gives a model with a window of 512 bytes
nearly all of its context at every target. Here it is applied to any model that
train_event_lm.py saves, so a decoder and the Transformer see the same bytes
before every target. A decoder scored this way gives up the state it would
carry through a stream.

With ``--books-dir`` every book of a verified PG-19 manifest is a passage of
its own. ``--calibration-books`` fits one temperature per checkpoint on
separate books, as eval_event_lm.py does.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from common import _atomic_output, resolve_device
from eval_stream_lm import load_books
from snnformer import count_params
from snnformer.event_lm import EventLM
from train_event_lm import load_checkpoint, scoring_logits, training_exposure
from train_lm import load_splits

PG19_BOOKS = Path(__file__).resolve().parents[1] / "results/event_lm/correction_pg19_full_book_manifest.json"


def window_batches(data, context, history, batch_size):
    """Windows and the number of unscored bytes at the start of each, as eval_event_lm.py forms them.

    A history longer than the scored bytes gives every target more bytes before it: 448 unscored bytes before 64
    scored ones fill a window of 512, and no target after the first 448 has fewer than 448 bytes behind it.
    """
    full_windows, tail = divmod(data.numel() - 1, context)
    # The first windows begin with the passage and have less than the history before their scored bytes.
    early = -(-history // context)
    if not full_windows:
        yield data[:context + 1].unsqueeze(0), 0
    for window in range(min(early, full_windows)):
        yield data[:(window + 1) * context + 1].unsqueeze(0), window * context
    if full_windows > early:
        start = early * context - history
        for window in data[start:].unfold(0, history + context + 1, context).split(batch_size):
            yield window, history
    if full_windows and tail:
        start = max(0, full_windows * context - history)
        yield data[start:].unsqueeze(0), full_windows * context - start


@torch.inference_mode()
def score(model, passages, context, history, batch_size, device, temperatures=(1.0,)):
    """Bits of every scored window at each temperature, which windows are whole and warmed, and the targets."""
    event = isinstance(model, EventLM)
    resolution = model.cfg.resolution if event else 1
    bits, whole, targets = [], [], 0
    for data in passages:
        for window, skipped in window_batches(data, context, history, batch_size):
            window = window.to(device=device, dtype=torch.long)
            logits = scoring_logits(model(window[:, :-1]), event, resolution)[:, skipped:]
            labels = window[:, skipped + 1:]
            losses = [F.cross_entropy((logits if temperature == 1 else logits / temperature).flatten(0, 1),
                                      labels.flatten(), reduction="none").view(labels.shape)
                      for temperature in temperatures]
            stacked = torch.stack(losses)
            if stacked.device.type == "mps":
                # MPS has no 64-bit floats, so the sum is taken on the CPU.
                stacked = stacked.cpu()
            bits.append(stacked.sum(2, dtype=torch.float64).cpu() / math.log(2))
            whole += [skipped == history and labels.shape[1] == context] * labels.shape[0]
            targets += labels.numel()
    return torch.cat(bits, dim=1).numpy(), np.array(whole), targets


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, required=True, nargs="+")
    dataset = parser.add_mutually_exclusive_group(required=True)
    dataset.add_argument("--enwik8", type=Path, help="canonical enwik8 file, score a complete split")
    dataset.add_argument("--books-dir", type=Path, help="directory of prepared PG-19 excerpts")
    parser.add_argument("--manifest", type=Path, default=PG19_BOOKS, help="manifest of the PG-19 excerpts")
    parser.add_argument("--calibration-books", type=Path, help="directory of verified PG-19 validation excerpts")
    parser.add_argument("--calibration-manifest", type=Path)
    parser.add_argument("--temperatures", type=float, nargs="+", default=[0.8, 1.0, 1.25, 1.5, 1.75, 2.0])
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument("--bytes", type=int, help="score only the first bytes of the enwik8 split, for a quick look")
    parser.add_argument("--context", type=int, default=256, help="scored bytes per window")
    parser.add_argument("--history", type=int, default=256, help="unscored bytes before them")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", choices=("cpu", "mps", "cuda", "auto"), default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--json", type=Path, required=True)
    args = parser.parse_args()
    if min(args.context, args.batch_size, args.threads) < 1 or args.history < 0:
        parser.error("context, batch size and threads must be positive and the history not negative")
    if args.enwik8 is not None and args.calibration_books is not None:
        parser.error("temperatures are fitted for PG-19 books only")
    if args.bytes is not None and (args.enwik8 is None or args.bytes <= args.context + args.history):
        parser.error("--bytes shortens an enwik8 split to more than one window")
    if (args.calibration_books is None) != (args.calibration_manifest is None):
        parser.error("calibration books need their manifest")
    if min(args.temperatures) <= 0:
        parser.error("temperatures must be positive")
    if args.json.exists():
        parser.error(f"{args.json} already exists")
    torch.set_num_threads(args.threads)
    device = resolve_device(args.device)
    record = {"protocol": "causal_warmed_windows", "split": args.split, "context_bytes": args.context,
              "unscored_history_bytes": args.history, "device": str(device)}
    calibration = None
    if args.enwik8 is not None:
        _, validation, test = load_splits(args.enwik8)
        passages = [torch.from_numpy(np.array(test if args.split == "test" else validation))]
        if args.bytes is not None:
            # A shortened split is named in the record, so its score is not taken for the whole split's.
            passages = [passages[0][:args.bytes]]
            record["first_bytes_of_split"] = args.bytes
    else:
        manifest = args.manifest.read_bytes()
        books = load_books(manifest, args.books_dir, 0, split=args.split)
        passages = [book.data for book in books]
        record = {"dataset": "PG-19", "manifest": args.manifest.name,
                  "manifest_sha256": hashlib.sha256(manifest).hexdigest(), "books": len(books), **record}
        if args.calibration_books is not None:
            manifest = args.calibration_manifest.read_bytes()
            calibration = [book.data for book in load_books(manifest, args.calibration_books, 0, split="validation")]
            record["calibration"] = {"manifest": args.calibration_manifest.name,
                                     "manifest_sha256": hashlib.sha256(manifest).hexdigest(),
                                     "books": len(calibration), "temperatures": args.temperatures}
    resamples, rows, first = None, [], None
    for path in args.checkpoint:
        model, checkpoint = load_checkpoint(path, device)
        temperature = 1.0
        if calibration is not None:
            fitted, _, _ = score(model, calibration, args.context, args.history, args.batch_size, device,
                                 args.temperatures)
            temperature = args.temperatures[int(fitted.sum(1).argmin())]
        temperatures = (1.0,) if temperature == 1 else (1.0, temperature)
        bits, whole, targets = score(model, passages, args.context, args.history, args.batch_size, device, temperatures)
        row = {"checkpoint": str(path), "run_name": checkpoint.get("run_name"), "parameters": count_params(model),
               "training_targets": training_exposure(checkpoint), "targets": targets,
               "bpb": float(bits[0].sum() / targets)}
        if calibration is not None:
            row.update(temperature=temperature, calibrated_bpb=float(bits[-1].sum() / targets))
        # Whole warmed windows are resampled together for the difference to the first checkpoint.
        paired = bits[0][whole] if args.history else bits[0]
        if first is None:
            first = paired
            resamples = np.random.default_rng(5261).integers(len(paired), size=(3000, len(paired)))
        else:
            difference = (paired - first) / args.context
            row["bpb_over_first_whole_windows"] = float(difference.mean())
            row["bpb_over_first_paired_95"] = np.percentile(difference[resamples].mean(axis=1), (2.5, 97.5)).tolist()
        rows.append(row)
        extra = ("" if "bpb_over_first_whole_windows" not in row else
                 f" | {row['bpb_over_first_whole_windows']:+.4f} to the first "
                 f"[{row['bpb_over_first_paired_95'][0]:+.4f}, {row['bpb_over_first_paired_95'][1]:+.4f}]")
        fitted = "" if calibration is None else f" | {row['calibrated_bpb']:.6f} at temperature {temperature}"
        print(f"{path} | {row['parameters']:,} parameters | {targets:,} targets | {row['bpb']:.6f} bpb{fitted}{extra}",
              flush=True)
        del model
    record["checkpoints"] = rows
    with _atomic_output(args.json, "w", False) as stream:
        json.dump(record, stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
