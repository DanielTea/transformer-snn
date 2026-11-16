"""Score paired enwik8 predictions at each position after a stream reset."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from contextlib import contextmanager
from itertools import combinations
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from common import _atomic_output, resolve_device
from probe_event_state import added_residual_weights, muted_synapses
from snnformer.event_lm import EventLM, _route_packets, _steer_packet
from train_event_lm import load_checkpoint, scoring_logits, training_exposure
from train_lm import ENWIK8_SHA1, load_splits


CONTEXT_BYTES = 32
CONTEXT_BINS = ((1, 4), (5, 8), (9, 16), (17, 24), (25, 32), (17, 32), (1, 32))
DEFAULT_POSITIONS = Path("results/event_lm/trace_read_drive/control_validation_history.json")


@torch.inference_mode()
def score(model, windows, batch_size, device):
    losses = []
    for window in windows.split(batch_size):
        window = window.to(device)
        output = model(window[:, :-1])
        if isinstance(model, EventLM):
            output = scoring_logits(output, True, model.cfg.resolution)
        nll = F.cross_entropy(output.transpose(1, 2), window[:, 1:], reduction="none")
        losses.append(nll.cpu().double())
    return torch.cat(losses).numpy() / math.log(2)


@contextmanager
def count_arrival_inputs(model, enabled):
    if not enabled or not isinstance(model, EventLM) or not model.cfg.local_trace or model.cfg.trace_tap_read:
        yield None
        return
    width = sum(model.cfg.group_widths[:model.cfg.trace_arrival_groups])
    totals = torch.zeros((2, CONTEXT_BYTES), dtype=torch.int64)

    def record(_module, args, kwargs, output):
        selected, _ = _steer_packet(kwargs["delayed_content"][..., :width], args[1][..., :width])
        _, unmatched = _route_packets(output[0][..., :width], selected, "coincidence")
        for row, counts in zip(totals, (selected, unmatched), strict=True):
            row += counts.cpu().to(torch.int64).abs().sum((0, 2))

    handles = [block.trace.register_forward_hook(record, with_kwargs=True) for block in model.blocks]
    try:
        yield totals
    finally:
        for handle in handles:
            handle.remove()


def paired_interval(differences, resamples):
    window_means = differences.mean(axis=1)
    return np.percentile(window_means[resamples].mean(axis=1), (2.5, 97.5)).tolist()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True, nargs="+",
                        help="matched checkpoints, with the comparison reference first")
    parser.add_argument("--enwik8", type=Path, required=True)
    parser.add_argument("--positions-from", type=Path, default=DEFAULT_POSITIONS,
                        help="use relative target starts from this paired enwik8 history record")
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--profile-arrivals", action="store_true",
                        help="count selected and unmatched fixed-arrival pulses by age")
    parser.add_argument("--width-gap", nargs=4, type=Path,
                        metavar=("WIDE_SNN", "WIDE_TRANSFORMER", "NARROW_SNN", "NARROW_TRANSFORMER"),
                        help="compare the paired SNN-to-Transformer gap at two widths")
    parser.add_argument("--json", type=Path, required=True)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("batch size must be positive")
    if args.width_gap and (len(set(args.width_gap)) != 4 or not set(args.width_gap) <= set(args.checkpoint)):
        parser.error("--width-gap must list four distinct paths also supplied to --checkpoint")
    if args.json.exists():
        parser.error(f"{args.json} already exists")

    torch.set_num_threads(4)
    device = resolve_device(args.device)
    _, val, test = load_splits(args.enwik8)
    data = val if args.split == "val" else test
    source = json.loads(args.positions_from.read_text())
    if source["data_sha1"] != ENWIK8_SHA1:
        parser.error("the positions file must refer to the same enwik8 data")
    starts = np.asarray(source["tail_starts"], dtype=np.int64)
    starts = starts[(starts > 0) & (starts + CONTEXT_BYTES <= len(data))]
    if starts.size < 2 or np.any(np.diff(starts) < CONTEXT_BYTES):
        parser.error("the selected offsets must define disjoint 32-byte targets")
    positions = starts[:, None] + np.arange(-1, CONTEXT_BYTES)
    windows = torch.from_numpy(data[positions].astype(np.int64))
    rng = np.random.default_rng(5261)
    resamples = rng.integers(len(starts), size=(3000, len(starts)))

    reference = None
    models, losses = [], []
    for path in args.checkpoint:
        model, checkpoint = load_checkpoint(path, device)
        with count_arrival_inputs(model, args.profile_arrivals) as arrival_counts:
            actual = score(model, windows, args.batch_size, device)
        if reference is None:
            reference = actual
        else:
            losses.append((str(path), actual))
        arrival = None
        muted = None
        silenced_residual = None
        if isinstance(model, EventLM):
            prior_width = (checkpoint.get("initialization") or {}).get("residual_width_before")
            if prior_width is not None:
                with muted_synapses(added_residual_weights(model, prior_width)):
                    silenced_residual = score(model, windows, args.batch_size, device)
            arrival = model.cfg.trace_arrival
            if arrival is not None:
                with muted_synapses([block.arrival_project.weight for block in model.blocks]):
                    muted = score(model, windows, args.batch_size, device)
        ages = []
        for low, high in CONTEXT_BINS:
            span = slice(low - 1, high)
            bpb = actual[:, span]
            delta = bpb - reference[:, span]
            row = {"context_bytes": [low, high], "targets": bpb.size, "bpb": float(bpb.mean()),
                   "delta_vs_reference_bpb": float(delta.mean()), "paired_95_bpb": paired_interval(delta, resamples)}
            if arrival_counts is not None:
                selected, unmatched = arrival_counts[:, span].sum(1).tolist()
                row.update(selected_arrival_pulses=selected, unmatched_arrival_pulses=unmatched)
            if muted is not None:
                change = muted[:, span] - bpb
                row.update(muted_arrival_bpb=float(muted[:, span].mean()), muted_minus_intact_bpb=float(change.mean()),
                           muted_minus_intact_paired_95_bpb=paired_interval(change, resamples))
            if silenced_residual is not None:
                change = silenced_residual[:, span] - bpb
                row.update(silenced_residual_bpb=float(silenced_residual[:, span].mean()),
                           silenced_residual_minus_intact_bpb=float(change.mean()),
                           silenced_residual_paired_95_bpb=paired_interval(change, resamples))
            ages.append(row)
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        models.append({"checkpoint": str(path), "sha256": digest, "run_name": checkpoint["run_name"],
                       "parameters": sum(parameter.numel() for parameter in model.parameters()),
                       "training_targets": training_exposure(checkpoint), "trace_arrival": arrival, "ages": ages})
        late, all_ages = ages[-2:]
        print(f"{path.name} | 1–32 {all_ages['bpb']:.6f} bpb | 17–32 {late['bpb']:.6f} bpb", flush=True)
        if muted is not None:
            print(f"  muted minus intact at 17–32 {late['muted_minus_intact_bpb']:+.6f} bpb", flush=True)
        del model

    comparisons = []
    for (reference_path, reference_loss), (arm_path, arm_loss) in combinations(losses, 2):
        ages = []
        for low, high in CONTEXT_BINS:
            difference = (arm_loss - reference_loss)[:, low - 1:high]
            ages.append({"context_bytes": [low, high], "delta_bpb": float(difference.mean()),
                         "paired_95_bpb": paired_interval(difference, resamples)})
        comparisons.append({"reference": reference_path, "checkpoint": arm_path, "ages": ages})

    width_gap = None
    if args.width_gap:
        by_path = dict([(str(args.checkpoint[0]), reference), *losses])
        wide_snn, wide_full, narrow_snn, narrow_full = (by_path[str(path)] for path in args.width_gap)
        wide, narrow = wide_snn - wide_full, narrow_snn - narrow_full
        ages = []
        for low, high in CONTEXT_BINS:
            span = slice(low - 1, high)
            change = wide[:, span] - narrow[:, span]
            ages.append({"context_bytes": [low, high], "wide_gap_bpb": float(wide[:, span].mean()),
                         "narrow_gap_bpb": float(narrow[:, span].mean()), "gap_change_bpb": float(change.mean()),
                         "paired_95_bpb": paired_interval(change, resamples)})
        keys = ("wide_snn", "wide_transformer", "narrow_snn", "narrow_transformer")
        width_gap = {**dict(zip(keys, (str(path) for path in args.width_gap), strict=True)), "ages": ages}

    result = {
        "data_sha1": ENWIK8_SHA1, "split": args.split, "device": str(device), "positions_from": str(args.positions_from),
        "batch_size": args.batch_size, "target_start_offsets": starts.tolist(),
        "target_bytes": len(starts) * CONTEXT_BYTES,
        "reference": str(args.checkpoint[0]),
        "bootstrap": "3000 paired resamples of complete 32-target windows, seed 5261",
        "models": models, "nonreference_comparisons": comparisons,
        **({"width_gap": width_gap} if width_gap is not None else {}),
    }
    with _atomic_output(args.json, "w", overwrite=False) as output:
        json.dump(result, output, indent=2)
        output.write("\n")


if __name__ == "__main__":
    main()
