"""How well a model's surprise finds the faults of simulated pumps (experiments/pump_sim.py tests), beside a classical
detector.

Every test pump is one stream of eight bytes a second. A stack reads all pumps side by side as unbroken streams and
scores every byte; a second's surprise is the bits of its eight bytes, and the anomaly score is the mean surprise of
the last WINDOW seconds. The classical detector reads the same bytes: for every channel, how many standard deviations
its value stands from the mean of the hour before (both running), the largest of the eight, averaged over the same
window. Seconds where a fault is active (from its start) are the positives, every second of the pump before is a
negative; the first hour of every pump is left out, where neither detector has a past. Reported: the area under the
ROC curve (AUC) of both detectors over all pumps, and for each kind of fault over its pumps and the healthy ones.

usage: python experiments/telemetry_eval.py --checkpoint run/model.pt.averaged --tests data/onemodel/telemetry/tests-v1.bin \\
    --json telemetry.json [--device mps]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import _atomic_output, resolve_device  # noqa: E402
from train_event_lm import load_checkpoint, scoring_logits  # noqa: E402

HOUR = 3600


def auc(scores: np.ndarray, labels: np.ndarray) -> float | None:
    """The probability that a positive scores above a negative (ties count half), or None without both."""
    positives, negatives = int(labels.sum()), int((~labels).sum())
    if not positives or not negatives:
        return None
    order = scores.argsort(kind="mergesort")
    ranks = np.empty(len(scores))
    sorted_scores = scores[order]
    # Average ranks over ties.
    starts = np.r_[0, np.flatnonzero(np.diff(sorted_scores)) + 1]
    ends = np.r_[starts[1:], len(scores)]
    for a, b in zip(starts, ends):
        ranks[order[a:b]] = (a + b + 1) / 2
    return float((ranks[labels].sum() - positives * (positives + 1) / 2) / (positives * negatives))


def running_mean(x: np.ndarray, window: int) -> np.ndarray:
    """The mean of the last ``window`` values at every place (fewer at the start), along the last axis."""
    total = np.cumsum(x, axis=-1)
    shifted = np.concatenate((np.zeros_like(total[..., :window]), total[..., :-window]), axis=-1)[..., :x.shape[-1]]
    count = np.minimum(np.arange(1, x.shape[-1] + 1), window)
    return (total - shifted) / count


def zscore_detector(codes: np.ndarray, window: int) -> np.ndarray:
    """[pumps, seconds]: the largest over channels of |value - mean of the hour before| / its spread, smoothed."""
    x = codes.astype(np.float64).transpose(0, 2, 1)          # [pumps, channels, seconds]
    before = np.concatenate((x[..., :1], x[..., :-1]), axis=-1)
    mean = running_mean(before, HOUR)
    spread = np.sqrt(np.maximum(running_mean(before ** 2, HOUR) - mean ** 2, 0.0)) + 0.5
    z = (np.abs(x - mean) / spread).max(1)
    return running_mean(z, window)


@torch.no_grad()
def surprise(model, codes: np.ndarray, device, piece: int = 512) -> np.ndarray:
    """[pumps, seconds]: the bits of every second's bytes, every pump read as one unbroken stream."""
    pumps, seconds, channels = codes.shape
    stream = torch.from_numpy(codes.reshape(pumps, -1).astype(np.int64))
    bits, state = [], None
    for start in range(0, stream.shape[1] - 1, piece):
        inputs = stream[:, start:start + piece]
        targets = stream[:, start + 1:start + piece + 1]
        counts, state = model.stream(inputs.to(device), state)
        logits = scoring_logits(counts, True, model.cfg.resolution)[:, :targets.shape[1]].float()
        loss = F.cross_entropy(logits.flatten(0, 1), targets.to(device).flatten(), reduction="none")
        bits.append((loss.view(pumps, -1) / math.log(2)).cpu())
    per_byte = torch.cat(bits, 1).numpy()
    # The first byte has no prediction; it counts as the mean of its second.
    per_byte = np.concatenate((per_byte[:, :1], per_byte), axis=1)[:, :seconds * channels]
    return per_byte.reshape(pumps, seconds, channels).sum(-1)


def labels_of(truth: dict) -> np.ndarray:
    seconds = truth["seconds_per_pump"]
    out = np.zeros((len(truth["pumps"]), seconds), dtype=bool)
    for index, pump in enumerate(truth["pumps"]):
        for fault in pump["faults"]:
            out[index, fault["start"]:] = True
    return out


def report(score: np.ndarray, labels: np.ndarray, kinds: list[str | None]) -> dict:
    keep = np.zeros_like(labels)
    keep[:, HOUR:] = True
    found = {"all": auc(score[keep], labels[keep])}
    for kind in sorted({k for k in kinds if k}):
        rows = [i for i, k in enumerate(kinds) if k in (kind, None)]
        found[kind] = auc(score[rows][keep[rows]], labels[rows][keep[rows]])
    return {k: None if v is None else round(v, 4) for k, v in found.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, nargs="*", default=[])
    parser.add_argument("--tests", type=Path, required=True)
    parser.add_argument("--window", type=int, default=60, help="seconds that both scores are averaged over")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--json", type=Path, required=True)
    args = parser.parse_args()
    truth = json.loads(Path(f"{args.tests}.truth.json").read_text())
    channels, seconds = truth["bytes_per_second"], truth["seconds_per_pump"]
    codes = np.frombuffer(args.tests.read_bytes(), dtype=np.uint8).reshape(len(truth["pumps"]), seconds, channels)
    labels = labels_of(truth)
    kinds = [pump["faults"][0]["kind"] if pump["faults"] else None for pump in truth["pumps"]]
    record = {"tests": str(args.tests), "window": args.window, "pumps": len(kinds),
              "zscore": report(zscore_detector(codes, args.window), labels, kinds), "runs": {}}
    print("zscore", record["zscore"], flush=True)
    device = resolve_device(args.device)
    for path in args.checkpoint:
        model, checkpoint = load_checkpoint(path, device)
        bits = surprise(model, codes, device)
        record["runs"][path.parent.name] = {"run_name": checkpoint.get("run_name"), "weights": path.name,
                                            "bits_per_second": round(float(bits[:, HOUR:].mean()), 4),
                                            "surprise": report(running_mean(bits, args.window), labels, kinds)}
        print(path.parent.name, record["runs"][path.parent.name], flush=True)
    with _atomic_output(args.json, "w", True) as handle:
        json.dump(record, handle, indent=1)
        handle.write("\n")


if __name__ == "__main__":
    main()
