"""Measure fixed-width spiking residual ablations on one pinned byte corpus."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from common import _atomic_output, resolve_device
from eval_byte_transfer import corpus_windows, score_windows
from prepare_byte_benchmarks import CATALOGUE, file_hash
from probe_event_state import added_residual_weights, muted_synapses
from snnformer.event_lm import EventLM
from train_event_lm import load_checkpoint, training_exposure


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", required=True, type=Path)
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--checkpoint", required=True, type=Path, nargs="+")
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--json", required=True, type=Path)
    args = parser.parse_args()
    if args.batch_size < 1 or args.json.exists():
        parser.error("use a positive batch size and a fresh output file")

    torch.set_num_threads(4)
    device = resolve_device(args.device)
    source = args.prepared / CATALOGUE
    catalogue = json.loads(source.read_text())
    suites = corpus_windows(args.prepared, catalogue)
    if args.corpus not in suites:
        parser.error(f"unknown corpus {args.corpus!r}")
    windows = suites[args.corpus]
    rows = []
    for path in args.checkpoint:
        model, meta = load_checkpoint(path, device)
        if not isinstance(model, EventLM):
            parser.error("use spiking checkpoints from a residual-width comparison")
        intact = score_windows(model, windows, args.batch_size, device)
        print(f"{path.name} | intact {intact['bpb']:.6f} bpb", flush=True)
        prior_width = (meta.get("initialization") or {}).get("residual_width_before")
        ablations = {}
        if prior_width is not None:
            with muted_synapses(added_residual_weights(model, prior_width)):
                result = score_windows(model, windows, args.batch_size, device)
            delta = result["bpb"] - intact["bpb"]
            ablations["all_added"] = {**result, "delta_bpb": delta}
            print(f"{path.name} | all added muted {result['bpb']:.6f} bpb ({delta:+.6f})", flush=True)
        rows.append({"checkpoint": str(path), "checkpoint_sha256": file_hash(path), "run_name": meta["run_name"],
                     "training_targets": training_exposure(meta), "intact": intact, "ablations": ablations})
        del model

    with _atomic_output(args.json, "w", overwrite=False) as stream:
        json.dump({"corpus": args.corpus, "catalogue_sha256": file_hash(source), "device": str(device),
                   "batch_size": args.batch_size, "models": rows}, stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
