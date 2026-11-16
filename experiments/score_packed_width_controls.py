"""Score existing continuation controls on the same enwik8 history tails."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from ablate_wide_snn_context import cache_integer_ticks, score, windows
from benchmark_decode import state_bytes
from train_event_lm import load_checkpoint, training_exposure
from train_lm import load_splits


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("checkpoints", type=Path, nargs="+", help="spiking continuation checkpoints")
    parser.add_argument("--data", type=Path, default=Path("data/enwik8"))
    parser.add_argument("--history-record", type=Path,
                        default=Path("results/event_lm/packed_width/wide_validation_history.json"))
    args = parser.parse_args()
    record_path = args.history_record
    record = json.loads(record_path.read_text())
    starts = np.asarray(record["tail_starts"])
    tail = record["tail_bytes"]
    _, val, test = load_splits(args.data)
    torch.set_num_threads(3)
    rows = []
    for path in args.checkpoints:
        model, meta = load_checkpoint(path, torch.device("cpu"))
        state = model.to_integer().step(torch.zeros(1, dtype=torch.long))[1]
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        row = {
            "checkpoint": str(path),
            "checkpoint_sha256": digest,
            "parameters": sum(p.numel() for p in model.parameters()),
            "training_targets": training_exposure(meta),
            "state_bytes": state_bytes(state),
        }
        cache_integer_ticks(model)
        batches = {}
        for split, data in [("val", val), ("test", test)]:
            batches[split] = {history: windows(data, starts, history, tail).split(48) for history in (1, 32)}
            row[split] = {history: float(score(model, sample, tail).mean())
                          for history, sample in batches[split].items()}
            print(path.name, split, row[split], flush=True)
        if model.cfg.steering_transfer is not None:
            for block in model.blocks:
                block.steering_transfer = None
            row["transfer_disabled_bpb_at_32"] = {
                split: float(score(model, conditions[32], tail).mean()) for split, conditions in batches.items()
            }
            print(path.name, "frozen transfer disabled", row["transfer_disabled_bpb_at_32"], flush=True)
        rows.append(row)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        "history_record": str(record_path),
        "target_bytes_per_condition": record["target_bytes"],
        "models": rows,
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
