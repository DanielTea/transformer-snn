"""Compare the added residual projections at matched enwik8 and C4 context lengths."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from ablate_wide_snn_context import score, windows
from common import _atomic_output
from eval_byte_transfer import load_payload
from prepare_byte_benchmarks import CATALOGUE, file_hash
from probe_early_context import paired_interval
from probe_event_state import added_residual_weights, muted_synapses
from snnformer.event_lm import EventLM
from train_event_lm import load_checkpoint
from train_lm import ENWIK8_SHA1, load_splits


TARGET_BYTES = 16
HISTORIES = (17, 256)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--enwik8", type=Path, required=True)
    parser.add_argument("--early-record", type=Path, required=True)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--json", type=Path, required=True)
    args = parser.parse_args()
    if args.json.exists():
        parser.error("use a fresh output file")

    torch.set_num_threads(4)
    record = json.loads(args.early_record.read_text())
    if record["split"] != "val" or record["data_sha1"] != ENWIK8_SHA1:
        parser.error("the early-context record must use the canonical enwik8 validation split")
    enwik8_starts = np.asarray(record["target_start_offsets"], dtype=np.int64) + TARGET_BYTES
    _, val, _ = load_splits(args.enwik8)
    catalogue_path = args.prepared / CATALOGUE
    catalogue = json.loads(catalogue_path.read_text())
    c4_entry = catalogue["corpora"]["c4"]
    c4 = np.frombuffer(load_payload(args.prepared, c4_entry), dtype=np.uint8)
    c4_starts = np.asarray([doc["offset"] + HISTORIES[-1] for doc in c4_entry["documents"]
                            if doc["bytes"] >= HISTORIES[-1] + TARGET_BYTES], dtype=np.int64)
    model, meta = load_checkpoint(args.checkpoint, torch.device("cpu"))
    prior_width = (meta.get("initialization") or {}).get("residual_width_before")
    if not isinstance(model, EventLM) or prior_width is None:
        parser.error("use a residual-widened spiking checkpoint")
    weights = added_residual_weights(model, prior_width)

    rows = []
    cases = (("enwik8_validation", val, enwik8_starts), ("c4_validation", c4, c4_starts))
    for corpus, data, starts in cases:
        if starts.size == 0 or starts.min() < HISTORIES[-1] or starts.max() + TARGET_BYTES > len(data):
            parser.error(f"{corpus} has invalid target positions for the paired contexts")
        rng = np.random.default_rng(5261)
        resamples = rng.integers(len(starts), size=(3000, len(starts)))
        for history in HISTORIES:
            batches = windows(data, starts, history, TARGET_BYTES).split(24)
            with torch.inference_mode():
                intact = score(model, batches, TARGET_BYTES)
                with muted_synapses(weights):
                    muted = score(model, batches, TARGET_BYTES)
            change = muted - intact
            row = {"corpus": corpus, "history_bytes": history, "windows": len(starts),
                   "target_bytes": TARGET_BYTES * len(starts), "intact_bpb": float(intact.mean()),
                   "muted_bpb": float(muted.mean()), "muted_minus_intact_bpb": float(change.mean()),
                   "paired_95_bpb": paired_interval(change[:, None], resamples)}
            print(f"{corpus}, {history} history bytes | mute change {row['muted_minus_intact_bpb']:+.6f} bpb",
                  flush=True)
            rows.append(row)
    with _atomic_output(args.json, "w", overwrite=False) as stream:
        json.dump({"checkpoint": str(args.checkpoint), "checkpoint_sha256": file_hash(args.checkpoint),
                   "enwik8_windows": str(args.early_record), "c4_catalogue_sha256": file_hash(catalogue_path),
                   "target_bytes_per_window": TARGET_BYTES, "device": "cpu", "rows": rows}, stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
