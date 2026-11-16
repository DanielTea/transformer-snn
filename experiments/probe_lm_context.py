"""Score the same held-out enwik8 target bytes with several amounts of history."""

from __future__ import annotations

import argparse
import json
import math
import os
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from common import make_lm, resolve_device
from snnformer import SNNConfig, count_params
from snnformer.event_lm import EventLM
from train_event_lm import load_checkpoint as load_event_checkpoint, scoring_logits
from train_lm import DATA, ENWIK8_SHA1, load_splits


def load_model(path: Path, device: torch.device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("task") != "lm" or checkpoint["model_kwargs"].get("vocab", 256) != 256:
        raise ValueError(f"{path}: expected a byte-level language-model checkpoint from train_lm.py")
    cfg = SNNConfig(**checkpoint["cfg"])
    model = make_lm(cfg, **checkpoint["model_kwargs"])
    model.load_state_dict(checkpoint["state_dict"])
    return model.to(device).eval(), checkpoint["run_name"]


@contextmanager
def memory_disabled(model):
    """Remove the hybrid's memory residual during inference with its weights intact."""
    def zero_output(_module, _inputs, output):
        return torch.zeros_like(output)

    hooks = [block.memory.register_forward_hook(zero_output) for block in model.blocks]
    try:
        yield
    finally:
        for hook in hooks:
            hook.remove()


@torch.inference_mode()
def score(model, data, starts, history, tail_bytes, batch_size, device):
    # Only score the tail. Its tokens are teacher forced the same way in
    # each condition, while the unscored context begins at different offsets.
    offsets = np.arange(-history, tail_bytes)
    nll = 0.0
    resolution = model.cfg.resolution if isinstance(model, EventLM) else None
    for index in range(0, len(starts), batch_size):
        positions = starts[index : index + batch_size, None] + offsets
        window = torch.as_tensor(data[positions], device=device, dtype=torch.long)
        logits = model(window[:, :-1])[:, -tail_bytes:]
        if resolution is not None:
            logits = scoring_logits(logits, True, resolution)
        targets = window[:, history:]
        nll += F.cross_entropy(logits.flatten(0, 1), targets.flatten(), reduction="sum").item()
    return nll / (len(starts) * tail_bytes * math.log(2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path, nargs="+")
    parser.add_argument("--event-lm", action="store_true",
                        help="load checkpoints from train_event_lm.py instead of train_lm.py")
    parser.add_argument("--data", type=Path, default=Path(os.environ.get("ENWIK8_PATH", str(DATA))))
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--histories", type=int, nargs="+", default=[32, 128, 256, 1024],
                        help="preceding byte counts before the first scored target")
    parser.add_argument("--tail-bytes", type=int, default=32, help="consecutive target bytes per tail")
    parser.add_argument("--tails", type=int, default=256, help="number of evenly spaced, disjoint score tails")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--device", choices=("cpu", "cuda", "mps", "auto"), default="cpu")
    parser.add_argument("--memory-ablation", action="store_true", help="also score hybrid checkpoints with memory silenced")
    parser.add_argument("--trace-read-drive-ablation", action="store_true",
                        help="also score an SNN's read synapses zeroed and with their input swapped")
    parser.add_argument("--json", type=Path, help="optional path for machine-readable results")
    args = parser.parse_args()

    histories = sorted(set(args.histories))
    if min(*histories, args.tail_bytes, args.tails, args.batch_size, args.threads) < 1:
        parser.error("histories, tail size, tail count, batch size, and thread count must all be positive")
    torch.set_num_threads(args.threads)
    try:
        device = resolve_device(args.device)
        _, val, test = load_splits(args.data.expanduser())
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    data = val if args.split == "val" else test
    if max(histories) + args.tails * args.tail_bytes > len(data):
        parser.error("the longest history and all disjoint target tails must fit the held-out split")
    starts = np.linspace(max(histories), len(data) - args.tail_bytes, args.tails, dtype=np.int64)
    targets = args.tails * args.tail_bytes
    print(f"enwik8 {args.split}: {targets} paired target bytes on {device}", flush=True)

    results = []
    for requested_path in args.checkpoint:
        path = requested_path.expanduser()
        try:
            if args.event_lm:
                model, metadata = load_event_checkpoint(path, device)
                name = metadata["run_name"]
            else:
                model, name = load_model(path, device)
        except (OSError, ValueError, KeyError, RuntimeError) as exc:
            parser.error(str(exc))
        positions = getattr(model, "pos", None)
        if positions is not None and max(histories) + args.tail_bytes - 1 > len(positions):
            parser.error(f"{name}: requested history and tail exceed its learned position table ({len(positions)})")
        if args.trace_read_drive_ablation:
            if not isinstance(model, EventLM) or model.cfg.trace_read_drive is None:
                parser.error(f"{name}: --trace-read-drive-ablation requires a driven held-charge read")
            source = model.cfg.trace_read_drive
            alternative = "present" if source.startswith("write") else "write"
            if source.endswith("-rails"):
                alternative += "-rails"
            zero = EventLM(model.cfg).to(device).eval()
            swapped = EventLM(replace(model.cfg, trace_read_drive=alternative)).to(device).eval()
            zero.load_state_dict(model.state_dict())
            swapped.load_state_dict(model.state_dict())
            with torch.no_grad():
                for block in zero.blocks:
                    block.trace.read_drive.weight.zero_()
        result = {"checkpoint": str(path), "run_name": name, "parameters": count_params(model), "histories": []}
        if args.trace_read_drive_ablation:
            result["trace_read_drive"] = model.cfg.trace_read_drive
        for history in histories:
            bpb = score(model, data, starts, history, args.tail_bytes, args.batch_size, device)
            row = {"history_bytes": history, "bpb": bpb}
            description = f"{name} | history {history:4d} | {bpb:.4f} bpb"
            if args.trace_read_drive_ablation:
                zero_bpb = score(zero, data, starts, history, args.tail_bytes, args.batch_size, device)
                swapped_bpb = score(swapped, data, starts, history, args.tail_bytes, args.batch_size, device)
                row.update(zero_trace_read_drive_bpb=zero_bpb, swapped_trace_read_drive_bpb=swapped_bpb)
                description += f" | zero synapses {zero_bpb:.5f} | swapped input {swapped_bpb:.5f}"
            if args.memory_ablation and getattr(model.cfg, "attn", None) == "transformer_event":
                with memory_disabled(model):
                    ablated = score(model, data, starts, history, args.tail_bytes, args.batch_size, device)
                row["memory_disabled_bpb"] = ablated
                row["memory_gain_bpb"] = ablated - bpb
                description += f" | memory disabled {ablated:.4f} bpb | gain {ablated - bpb:+.4f} bpb"
            result["histories"].append(row)
            print(description, flush=True)
        results.append(result)
        del model
        if args.trace_read_drive_ablation:
            del zero, swapped

    if args.json is not None:
        payload = {
            "data_sha1": ENWIK8_SHA1, "split": args.split, "device": str(device),
            "target_bytes": targets, "tail_bytes": args.tail_bytes,
            "tail_starts": starts.tolist(), "models": results,
        }
        destination = args.json.expanduser()
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
