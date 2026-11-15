"""Score byte language models on verified PG-19 excerpts with reset windows."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from common import RESULTS, _atomic_output, resolve_device
from prepare_pg19 import matches
from probe_lm_context import load_model
from snnformer import count_params


PG19_MANIFEST = RESULTS / "lm_scaling" / "pg19_test16_manifest.json"


@dataclass(frozen=True)
class Book:
    book_id: int
    name: str
    source_start: int
    excerpt_sha256: str
    data: torch.Tensor


def load_books(manifest_bytes: bytes, directory: Path, max_bytes: int, *, split: str = "test") -> list[Book]:
    manifest = json.loads(manifest_bytes)
    if manifest["dataset"] != "PG-19" or manifest["split"] != split:
        raise ValueError(f"expected a PG-19 {split} excerpt manifest")
    books = []
    for entry in manifest["books"]:
        name = entry["excerpt_file"]
        if name in ("", ".", "..") or Path(name).name != name:
            raise ValueError(f"invalid excerpt filename {name!r}")
        path = directory / name
        raw = path.read_bytes()
        if not matches(raw, entry["excerpt_bytes"], entry["excerpt_sha256"]):
            raise ValueError(f"{path}: excerpt size or checksum differs from the manifest")
        if max_bytes:
            raw = raw[:max_bytes]
        if len(raw) < 2:
            raise ValueError(f"{path}: need at least two raw bytes to score a prediction")
        data = torch.from_numpy(np.frombuffer(raw, dtype=np.uint8).copy())
        books.append(Book(entry["book_id"], name, entry["excerpt_start_raw_byte"], entry["excerpt_sha256"], data))
    if not books or len({book.book_id for book in books}) != len(books) or len({book.name for book in books}) != len(books):
        raise ValueError("manifest must list at least one excerpt and each book and filename must be unique")
    return books


def batch_loss(model, windows: torch.Tensor, device: torch.device) -> tuple[float, int]:
    windows = windows.to(device=device, dtype=torch.long)
    logits = model(windows[:, :-1])
    targets = windows[:, 1:]
    if logits.shape != (*targets.shape, 256):
        raise ValueError("checkpoint must emit 256 next-byte logits at each input position")
    loss = F.cross_entropy(logits.flatten(0, 1), targets.flatten(), reduction="sum")
    return loss.item(), targets.numel()


@torch.inference_mode()
def score_book(model, book: Book, context: int, batch_size: int, device: torch.device) -> dict:
    # Reuse the boundary byte as the next window's first input. Every byte
    # after the first is a target exactly once, including the partial tail.
    target_bytes = book.data.numel() - 1
    full_windows, tail = divmod(target_bytes, context)
    nll, scored = 0.0, 0
    if full_windows:
        windows = book.data.unfold(0, context + 1, context)
        for window_batch in windows.split(batch_size):
            loss, count = batch_loss(model, window_batch, device)
            nll += loss
            scored += count
    if tail:
        loss, count = batch_loss(model, book.data[full_windows * context:].unsqueeze(0), device)
        nll += loss
        scored += count
    if scored != target_bytes or not math.isfinite(nll):
        raise ValueError(f"{book.name}: expected {target_bytes:,} scored bytes and finite loss, got {scored:,}")
    return {
        "book_id": book.book_id,
        "excerpt_file": book.name,
        "excerpt_sha256": book.excerpt_sha256,
        "source_start_raw_byte": book.source_start,
        "excerpt_bytes_used": book.data.numel(),
        "target_bytes": scored,
        "windows": full_windows + bool(tail),
        "nll_nats": nll,
        "bpb": nll / (scored * math.log(2)),
    }


def evaluate(checkpoint: Path, books: list[Book], args, device: torch.device) -> dict:
    model, name = load_model(checkpoint, device)
    maximum_input = min(args.context, max(book.data.numel() - 1 for book in books))
    positions = getattr(model, "pos", None)
    if isinstance(positions, torch.Tensor) and maximum_input > positions.shape[0]:
        raise ValueError(f"{name}: requested context exceeds its learned position table ({positions.shape[0]})")
    per_book = [score_book(model, book, args.context, args.batch_size, device) for book in books]
    nll = math.fsum(book["nll_nats"] for book in per_book)
    targets = sum(book["target_bytes"] for book in per_book)
    bpb = nll / (targets * math.log(2))
    print(f"{name} | {targets:,} target bytes | {bpb:.6f} bpb", flush=True)
    return {"run_name": name, "checkpoint": str(checkpoint), "parameters": count_params(model),
            "target_bytes": targets, "nll_nats": nll, "bpb": bpb, "books": per_book}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--books-dir", required=True, type=Path, help="directory of prepared PG-19 excerpts")
    parser.add_argument("--manifest", type=Path, default=PG19_MANIFEST)
    parser.add_argument("--checkpoint", required=True, nargs="+", type=Path)
    parser.add_argument("--context", type=int, default=1024, help="maximum preceding bytes in one reset window")
    parser.add_argument("--max-bytes-per-book", type=int, default=0, help="use a prefix of each excerpt (0 uses full excerpts)")
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--device", choices=("cpu", "cuda", "mps", "auto"), default="cpu")
    parser.add_argument("--json", required=True, type=Path)
    parser.add_argument("--force", action="store_true", help="replace the JSON record")
    args = parser.parse_args()

    if min(args.context, args.batch_size, args.threads) < 1 or args.max_bytes_per_book < 0 or args.max_bytes_per_book == 1:
        parser.error("context, batch size, and threads must be positive, and the byte limit must be zero or at least two")
    path = args.json.expanduser()
    if path.exists() and not args.force:
        parser.error(f"{path} already exists. Choose a new path or use --force to replace it.")
    torch.set_num_threads(args.threads)
    try:
        device = resolve_device(args.device)
        manifest = args.manifest.expanduser().read_bytes()
        books = load_books(manifest, args.books_dir.expanduser(), args.max_bytes_per_book)
        target_bytes = sum(len(book.data) - 1 for book in books)
        print(f"PG-19 | {len(books)} verified test excerpts | {target_bytes:,} targets | reset windows", flush=True)
        models = [evaluate(checkpoint.expanduser(), books, args, device) for checkpoint in args.checkpoint]
        if any(model["target_bytes"] != target_bytes for model in models):
            raise ValueError("model target counts differ from the corpus")
        payload = {
            "dataset": "PG-19", "split": "test", "protocol": "causal_reset_windows",
            "manifest": str(args.manifest), "manifest_sha256": hashlib.sha256(manifest).hexdigest(),
            "device": str(device), "context_bytes": args.context, "window_stride_bytes": args.context,
            "batch_size": args.batch_size,
            "unscored_leading_bytes_per_book": 1, "max_bytes_per_book": args.max_bytes_per_book,
            "target_bytes": target_bytes, "models": models,
        }
        with _atomic_output(path, "w", args.force) as destination:
            json.dump(payload, destination, indent=2)
            destination.write("\n")
    except (OSError, ValueError, RuntimeError, KeyError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
