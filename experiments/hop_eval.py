"""Accuracy of trained models on multi-hop lookups, one hop count at a time (experiments/hop_tasks.py).

Every test line is scored where its answer stands, the byte after the query's '='. A stack reads the lines of one hop
count as one unbroken stream and carries its state from line to line, as the stack is always run; a Transformer sees a
window of the bytes before each answer. The answer counts as right when the model's most likely next byte is the
answer digit. The first LEAD lines of every file are read and not scored. Reported for every hop count: accuracy with
its 95% (Wilson) interval, the mean bits of the answer byte, and the share of answers where the most likely byte is
some digit at all.

usage: python experiments/hop_eval.py --checkpoint run/model.pt.averaged [more ...] --tests data/onemodel/hops/tests \\
    --json hops.json [--device mps] [--window 256]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import _atomic_output, resolve_device  # noqa: E402
from snnformer.event_lm import EventLM  # noqa: E402
from train_event_lm import load_checkpoint, scoring_logits  # noqa: E402

DIGITS = torch.tensor([ord(c) for c in "0123456789"])


def wilson(right: int, total: int, z: float = 1.96) -> list[float]:
    if not total:
        return [0.0, 1.0]
    p = right / total
    centre = (p + z * z / (2 * total)) / (1 + z * z / total)
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / (1 + z * z / total)
    return [round(centre - half, 4), round(centre + half, 4)]


def answer_places(text: bytes, lead: int) -> tuple[torch.Tensor, list[int]]:
    """The bytes of the lines as one stream, and the places of the answers after the first ``lead`` lines."""
    places, start = [], 0
    for number, line in enumerate(text.split(b"\n")[:-1]):
        if number >= lead:
            places.append(start + len(line) - 1)
        start += len(line) + 1
    return torch.frombuffer(bytearray(text), dtype=torch.uint8).long(), places


@torch.no_grad()
def answer_logits(model, stream: torch.Tensor, places: list[int], device, window: int, piece: int = 512):
    """The model's scores over the 256 bytes at every answer, as [answers, 256], from the bytes before it."""
    if isinstance(model, EventLM):
        rows, state, wanted = [], None, torch.tensor(places)
        for start in range(0, len(stream), piece):
            counts, state = model.stream(stream[None, start:start + piece].to(device), state)
            logits = scoring_logits(counts, True, model.cfg.resolution)[0]
            # The scores at place p - 1 predict the byte at p.
            inside = wanted[(wanted - 1 >= start) & (wanted - 1 < start + piece)]
            if len(inside):
                rows.append(logits[(inside - 1 - start).to(logits.device)].float().cpu())
        return torch.cat(rows)
    rows = []
    for chunk in range(0, len(places), 64):
        windows = []
        for place in places[chunk:chunk + 64]:
            before = stream[max(0, place - window):place]
            if len(before) < window:
                before = torch.cat((torch.full((window - len(before),), ord("\n")), before))
            windows.append(before)
        logits = model(torch.stack(windows).to(device))
        rows.append(logits[:, -1].float().cpu())
    return torch.cat(rows)


def score(model, path: Path, device, window: int, lead: int) -> dict:
    stream, places = answer_places(path.read_bytes(), lead)
    logits = answer_logits(model, stream, places, device, window)
    answers = stream[torch.tensor(places)]
    best = logits.argmax(-1)
    right = int((best == answers).sum())
    bits = F.cross_entropy(logits, answers, reduction="none") / math.log(2)
    return {"answers": len(places), "right": right, "accuracy": round(right / len(places), 4),
            "interval_95": wilson(right, len(places)), "answer_bits": round(float(bits.mean()), 4),
            "some_digit": round(float(torch.isin(best, DIGITS).float().mean()), 4)}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, required=True, nargs="+")
    parser.add_argument("--tests", type=Path, required=True, help="the folder of hops-H.txt files")
    parser.add_argument("--lead", type=int, default=16, help="lines of every file read and not scored")
    parser.add_argument("--window", type=int, default=256, help="the bytes a Transformer sees before an answer")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--json", type=Path, required=True)
    args = parser.parse_args()
    device = resolve_device(args.device)
    files = sorted(args.tests.glob("hops-*.txt"), key=lambda p: int(p.stem.split("-")[1]))
    if not files:
        parser.error(f"{args.tests}: no hops-H.txt files")
    record = {"tests": str(args.tests), "lead": args.lead, "window": args.window, "runs": {}}
    for path in args.checkpoint:
        model, checkpoint = load_checkpoint(path, device)
        found = {p.stem.split("-")[1]: score(model, p, device, args.window, args.lead) for p in files}
        record["runs"][path.parent.name] = {"run_name": checkpoint.get("run_name"), "weights": path.name,
                                            "stream": isinstance(model, EventLM), "by_hops": found}
        print(path.parent.name, " ".join(f"{h}:{row['accuracy']:.3f}" for h, row in found.items()), flush=True)
    with _atomic_output(args.json, "w", True) as handle:
        json.dump(record, handle, indent=1)
        handle.write("\n")


if __name__ == "__main__":
    main()
