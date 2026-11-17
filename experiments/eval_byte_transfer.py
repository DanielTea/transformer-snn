"""Evaluate fixed text windows and LAMBADA words with byte checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from common import _atomic_output, resolve_device
from prepare_byte_benchmarks import CATALOGUE, file_hash, scoring_windows
from snnformer.event_lm import EventLM
from train_event_lm import load_checkpoint, scoring_logits, training_exposure


WORD_BOUNDARIES = b" \t\r\n.,!?;:\"()[]{}"


@dataclass(frozen=True)
class Window:
    sequence: bytes
    prefix_bytes: int
    overlap_hit: bool = False


def load_payload(folder, entry):
    name = entry["file"]
    if name in ("", ".", "..") or Path(name).name != name:
        raise ValueError(f"invalid prepared file name {name!r}")
    data = (folder / name).read_bytes()
    if hashlib.sha256(data).hexdigest() != entry["sha256"]:
        raise ValueError(f"{name} failed the prepared data checksum")
    return data


def corpus_windows(folder, catalogue):
    context = catalogue["context_bytes"]
    targets = catalogue["targets_per_window"]
    suites = {}
    for name, entry in catalogue["corpora"].items():
        content = load_payload(folder, entry)
        overlap = catalogue["overlap"].get(name, {}).get("flagged_windows", [])
        flagged = {tuple(window) for window in overlap}
        windows = []
        for doc in entry["documents"]:
            offset = doc["offset"]
            stop = offset + doc["bytes"]
            if not 0 <= offset < stop <= len(content):
                raise ValueError(f"{name} document falls outside its prepared bytes")
            for target, prefix, sequence in scoring_windows(content[offset:stop], context, targets):
                windows.append(Window(sequence, prefix, (doc["id"], target) in flagged))
        if not windows:
            raise ValueError(f"no scoring windows in {name}")
        suites[name] = windows
    return suites


def lambada_windows(folder, catalogue):
    cases = json.loads(load_payload(folder, catalogue["lambada"]))
    if len(cases) != catalogue["lambada"]["scored_examples"]:
        raise ValueError("LAMBADA case count differs from the catalogue")
    suites = {}
    for condition in ("passage", "last_sentence"):
        windows = []
        for case in cases:
            prefix = case[condition].encode("utf-8")
            word = case["word"].encode("utf-8")
            if not prefix or not word or len(prefix) + len(word) > catalogue["lambada"]["max_passage_bytes"]:
                raise ValueError(f"invalid LAMBADA case {case['id']}")
            windows.append(Window(prefix + word, len(prefix)))
        suites[condition] = windows
    return suites


@torch.inference_mode()
def score_windows(model, windows, batch_size, device, *, word_accuracy=False, overlap=False):
    # Independent causal windows can be right-padded and sorted for efficient batches.
    ordered = sorted(windows, key=lambda window: len(window.sequence))
    event = isinstance(model, EventLM)
    boundary_table = torch.zeros(256, dtype=torch.bool, device=device)
    boundary_table[list(WORD_BOUNDARIES)] = True
    losses, unflagged_losses = [], []
    targets = unflagged_targets = correct = terminated = 0
    for start in range(0, len(ordered), batch_size):
        batch = ordered[start:start + batch_size]
        lengths = [len(window.sequence) for window in batch]
        prefixes = [window.prefix_bytes for window in batch]
        padded = np.zeros((len(batch), max(lengths)), dtype=np.int64)
        for row, window in enumerate(batch):
            padded[row, :lengths[row]] = np.frombuffer(window.sequence, dtype=np.uint8)
        tokens = torch.from_numpy(padded).to(device)
        output = model(tokens)
        if event:
            output = scoring_logits(output, True, model.cfg.resolution)
        end = torch.tensor(lengths, device=device) - 1
        begin = torch.tensor(prefixes, device=device) - 1
        positions = torch.arange(tokens.shape[1] - 1, device=device)
        scored = (positions >= begin[:, None]) & (positions < end[:, None])
        labels = tokens[:, 1:]
        logits = output[:, :-1].reshape(-1, output.shape[-1])
        nll = F.cross_entropy(logits, labels.flatten(), reduction="none").reshape_as(labels)
        sums = nll.masked_fill(~scored, 0).sum(1).cpu().tolist()
        losses.extend(sums)
        targets += sum(length - prefix for length, prefix in zip(lengths, prefixes, strict=True))
        if overlap:
            for index, window in enumerate(batch):
                if not window.overlap_hit:
                    unflagged_losses.append(sums[index])
                    unflagged_targets += lengths[index] - prefixes[index]
        if word_accuracy:
            predicted = output.argmax(-1)
            exact = ((predicted[:, :-1] == labels) | ~scored).all(1)
            boundary = boundary_table[predicted.gather(1, end[:, None]).squeeze(1)]
            correct += exact.sum().item()
            terminated += (exact & boundary).sum().item()

    if targets == 0:
        raise ValueError("the evaluation must include target bytes")
    nll = math.fsum(losses)
    result = {"target_bytes": targets, "nll_nats": nll, "bpb": nll / (targets * math.log(2))}
    if overlap:
        result["overlap_flagged_targets"] = targets - unflagged_targets
        result["without_overlap_hits"] = {
            "target_bytes": unflagged_targets,
            "bpb": (math.fsum(unflagged_losses) / (unflagged_targets * math.log(2))
                    if unflagged_targets else None),
        }
    if word_accuracy:
        result.update(examples=len(windows), exact_matches=correct, exact_match_accuracy=correct / len(windows),
                      boundary_exact_matches=terminated, boundary_exact_match_accuracy=terminated / len(windows))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True, help="output of prepare_byte_benchmarks.py")
    parser.add_argument("--checkpoint", type=Path, nargs="+", required=True)
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--json", type=Path, required=True)
    args = parser.parse_args()
    if args.batch_size < 1 or args.json.exists():
        parser.error("use a positive batch size and a fresh output file")

    torch.set_num_threads(4)
    device = resolve_device(args.device)
    source = args.prepared / CATALOGUE
    catalogue = json.loads(source.read_text(encoding="utf-8"))
    corpora = corpus_windows(args.prepared, catalogue)
    # A suite of held-out documents alone carries no LAMBADA questions.
    questions = lambada_windows(args.prepared, catalogue) if "lambada" in catalogue else {}
    header = {"suite": catalogue["suite"], "catalogue_sha256": file_hash(source),
              "manifest_sha256": catalogue["manifest_sha256"], "device": str(device),
              "batch_size": args.batch_size, "context_bytes": catalogue["context_bytes"],
              "targets_per_window": catalogue["targets_per_window"]}
    if questions:
        header.update({"lambada_examples": catalogue["lambada"]["scored_examples"],
                       "lambada_max_passage_bytes": catalogue["lambada"]["max_passage_bytes"],
                       "lambada_word_boundary_bytes": list(WORD_BOUNDARIES)})
    checkpoints = [(path, file_hash(path)) for path in args.checkpoint]
    partial = args.json.with_suffix(args.json.suffix + ".partial")
    rows = []
    if partial.exists():
        previous = json.loads(partial.read_text())
        rows = previous.pop("models")
        completed = [(Path(row["checkpoint"]), row["checkpoint_sha256"]) for row in rows]
        if previous != header or completed != checkpoints[:len(rows)]:
            parser.error(f"{partial} belongs to a different benchmark invocation")
        print(f"Resuming after {len(rows)} completed checkpoints", flush=True)
    for path, digest in checkpoints[len(rows):]:
        model, metadata = load_checkpoint(path, device)
        metrics = {}
        for name, windows in corpora.items():
            metrics[name] = score_windows(model, windows, args.batch_size, device, overlap=name in catalogue["overlap"])
            print(f"{path.name} | {name} | {metrics[name]['bpb']:.6f} bpb", flush=True)
        lambada = {}
        for name, windows in questions.items():
            score = score_windows(model, windows, args.batch_size, device, word_accuracy=True)
            lambada[name] = score
            print(f"{path.name} | LAMBADA {name} | {score['bpb']:.6f} bpb | "
                  f"{score['exact_match_accuracy']:.2%} exact", flush=True)
        if questions:
            metrics["lambada"] = lambada
        rows.append({"checkpoint": str(path), "checkpoint_sha256": digest, "run_name": metadata["run_name"],
                     "model_type": type(model).__name__, "attention": metadata["cfg"].get("attn"),
                     "parameters": sum(parameter.numel() for parameter in model.parameters()),
                     "training_targets": training_exposure(metadata), "benchmarks": metrics})
        with _atomic_output(partial, "w", overwrite=True) as output:
            json.dump({**header, "models": rows}, output, indent=2)
            output.write("\n")
        del model

    with _atomic_output(args.json, "w", overwrite=False) as output:
        json.dump({**header, "models": rows}, output, indent=2)
        output.write("\n")
    partial.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
