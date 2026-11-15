"""Compare matched local, continuous-cache and spike-cache models on byte streams."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
import torch

from common import make_lm, resolve_device
from eval_successor import enwik8_test, local_log_probs, stream_cache
from snnformer import SNNConfig, count_params
from train_lm import DATA, ENWIK8_SHA1
from train_successor import next_byte_nll


ARMS = ("local", "continuous", "spike")
DISTANCES = ("miss", "local_only", "remote")
TRAINING_FIELDS = ("seed", "steps", "bs", "ctx", "lr", "wd", "warmup")


def load_arms(paths, device):
    models, records = {}, {}
    for arm, path in zip(ARMS, paths):
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        if checkpoint.get("task") != "lm" or checkpoint["model_kwargs"].get("vocab", 256) != 256:
            raise ValueError(f"{arm}: expected a byte-level checkpoint from train_lm.py")
        cfg = SNNConfig(**checkpoint["cfg"])
        expected = "transformer_local" if arm == "local" else "transformer_successor"
        if cfg.attn != expected or (arm != "local" and cfg.successor_spikes != (arm == "spike")):
            raise ValueError(f"{arm}: checkpoint has the wrong model or gate type")
        model = make_lm(cfg, **checkpoint["model_kwargs"])
        model.load_state_dict(checkpoint["state_dict"])
        models[arm] = model.to(device).eval()
        records[arm] = {
            "run_name": checkpoint["run_name"],
            "checkpoint": path.name,
            "parameters": count_params(model),
            "model_kwargs": checkpoint["model_kwargs"],
            "training": {name: checkpoint["args"][name] for name in TRAINING_FIELDS},
        }

    local, continuous, spike = (models[arm] for arm in ARMS)
    for arm in ("continuous", "spike"):
        for key in ("model_kwargs", "training"):
            if records[arm][key] != records["local"][key]:
                raise ValueError(f"{arm}: {key} does not match the local Transformer")
        cfg = models[arm].cfg
        settings = ("local_window", "position_embeddings", "mlp_ratio")
        if any(getattr(cfg, key) != getattr(local.cfg, key) for key in settings):
            raise ValueError(f"{arm}: backbone configuration does not match the local Transformer")
    cache_settings = ("successor_orders", "successor_slots", "successor_scaled_votes")
    if any(getattr(continuous.cfg, key) != getattr(spike.cfg, key) for key in cache_settings):
        raise ValueError("continuous and spike cache configurations differ")
    return models, records


@torch.inference_mode()
def score_stream(models, data, chunk, device):
    """Score one stream with independent model weights and shared cache evidence."""
    if len(data) < 2:
        raise ValueError("each stream must contain at least two bytes")
    reference = models["spike"]
    overlap = len(reference.blocks) * (reference.cfg.local_window - 1)
    counts = np.zeros(len(DISTANCES), dtype=np.int64)
    sums = np.zeros((len(ARMS), len(DISTANCES)), dtype=np.float64)
    for start, end, evidence in stream_cache(data, chunk, reference.successors.orders, reference.successors.slots):
        ages_np = evidence[1]
        # A remote hit means at least one candidate successor was observed
        # beyond the local backbone's full receptive field.
        distances = (ages_np >= 0).any(axis=1).astype(np.int64)
        distances[(ages_np >= overlap + 1).any(axis=1)] = 2
        counts += np.bincount(distances, minlength=len(DISTANCES))
        successors, ages, observed, correct = [torch.as_tensor(x, device=device)[None, :] for x in evidence]
        targets = torch.tensor(data[start + 1 : end + 1], dtype=torch.long, device=device)[None, :]
        wins = successors == targets[..., None]
        for index, arm in enumerate(ARMS):
            model = models[arm]
            log_probs = local_log_probs(model, data, start, end, overlap, device)
            target_log_probs = log_probs.gather(-1, targets[..., None]).squeeze(-1)
            if arm == "local":
                nll = -target_log_probs
            else:
                strength = model.readout.strength(
                    log_probs, successors, ages, observed, correct, quantized=model.cfg.successor_spikes,
                )
                nll = next_byte_nll(target_log_probs.exp(), wins, strength)
            sums[index] += np.bincount(distances, weights=nll.flatten().cpu().numpy(), minlength=len(DISTANCES))
    return counts, sums


def summarize(counts, sums):
    """Convert raw loss sums to exact byte-weighted results."""
    total = int(counts.sum())
    return {
        "target_bytes": total,
        "target_bytes_by_distance": dict(zip(DISTANCES, counts.tolist())),
        "models": {
            arm: {
                "bpb": float(losses.sum() / (total * math.log(2))),
                "bpb_by_distance": {
                    distance: float(loss / (count * math.log(2))) if count else None
                    for distance, loss, count in zip(DISTANCES, losses, counts)
                },
            }
            for arm, losses in zip(ARMS, sums)
        },
    }


def evaluate_streams(models, streams, limit, chunk, device):
    """Reset history between streams and aggregate by target bytes, not book count."""
    total_counts = np.zeros(len(DISTANCES), dtype=np.int64)
    total_sums = np.zeros((len(ARMS), len(DISTANCES)), dtype=np.float64)
    rows = []
    for name, data in streams:
        data = data[:limit + 1] if limit else data
        counts, sums = score_stream(models, data, chunk, device)
        total_counts += counts
        total_sums += sums
        row = {"source": name, **summarize(counts, sums)}
        rows.append(row)
        print(f"{name} ({row['target_bytes']:,} bytes): " + ", ".join(
            f"{arm} {row['models'][arm]['bpb']:.4f} bpb" for arm in ARMS
        ), flush=True)
    return {**summarize(total_counts, total_sums), "streams": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for arm in ARMS:
        parser.add_argument(f"--{arm}", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=Path(os.environ.get("ENWIK8_PATH", DATA)))
    parser.add_argument("--skip-enwik8", action="store_true", help="score only independent raw streams")
    parser.add_argument("--raw", type=Path, nargs="+", default=[], help="independent raw streams, such as PG-19 books")
    parser.add_argument("--enwik8-bytes", type=int, default=0, help="number of leading enwik8 targets (0 means all)")
    parser.add_argument("--raw-bytes", type=int, default=0, help="number of leading targets per raw stream (0 means all)")
    parser.add_argument("--chunk", type=int, default=4096, help="scoring chunk size without resetting cache")
    parser.add_argument("--threads", type=int, default=3)
    parser.add_argument("--device", choices=("cpu", "cuda", "mps", "auto"), default="cpu")
    parser.add_argument("--json", type=Path, required=True, help="paired benchmark output")
    args = parser.parse_args()
    if min(args.chunk, args.threads) < 1 or min(args.enwik8_bytes, args.raw_bytes) < 0:
        parser.error("chunk and thread counts must be positive, and byte limits must be non-negative")
    if args.skip_enwik8 and not args.raw:
        parser.error("provide --raw streams if --skip-enwik8 is set")
    torch.set_num_threads(args.threads)
    try:
        device = resolve_device(args.device)
        paths = [getattr(args, arm).expanduser() for arm in ARMS]
        models, records = load_arms(paths, device)
        datasets = {}
        if not args.skip_enwik8:
            data = enwik8_test(args.data.expanduser())
            datasets["enwik8"] = evaluate_streams(models, [("enwik8 test", data)], args.enwik8_bytes, args.chunk, device)
        if args.raw:
            streams = [(path.name, np.memmap(path.expanduser(), dtype=np.uint8, mode="r")) for path in args.raw]
            datasets["raw"] = evaluate_streams(models, streams, args.raw_bytes, args.chunk, device)
    except (OSError, KeyError, RuntimeError, ValueError) as exc:
        parser.error(str(exc))
    reference = models["spike"]
    payload = {
        "training": records["local"]["training"],
        "device": str(device),
        "chunk": args.chunk,
        "data_sha1": None if args.skip_enwik8 else ENWIK8_SHA1,
        "cache_orders": list(reference.successors.orders),
        "cache_slots": reference.successors.slots,
        "remote_min_age": 1 + len(reference.blocks) * (reference.cfg.local_window - 1),
        "models": records,
        "datasets": datasets,
    }
    destination = args.json.expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
