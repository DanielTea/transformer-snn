"""Score a decoder with its surprise clock and fast synapses intact, muted and displaced.

Every condition replays one continuous enwik8 split with the trained weights.
Muting removes the held correction of the clocked blocks or the vocabulary
spikes of the fast synapses. Displacing moves each chunk's ticks by half a
chunk, which keeps their number and removes their timing.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from common import _atomic_output, resolve_device
from probe_surprise import _WORD_SLOTS, word_positions
from snnformer.event_lm import EventLM
from train_event_lm import load_checkpoint, scoring_logits
from train_lm import ENWIK8_SHA1, load_splits

_CHUNK = 8192


@torch.inference_mode()
def replay(model, data, device):
    """Loss in bits per byte, tick positions and the share of bytes the synapses stored."""
    tensor = torch.from_numpy(data.copy())
    targets = len(data) - 1
    nll, state = 0.0, None
    model.clock_trace = [] if model.cfg.clock_start is not None else None
    if model.fast_synapses is not None:
        model.fast_synapses.capture = []
    stored = 0
    for start in range(0, targets, _CHUNK):
        window = tensor[start:start + _CHUNK + 1].unsqueeze(0).to(device=device, dtype=torch.long)
        counts, state = model.forward_chunk(window[:, :-1], state)
        logits = scoring_logits(counts, True, model.cfg.resolution)
        nll += F.cross_entropy(logits[0], window[0, 1:], reduction="sum").item()
        if model.fast_synapses is not None:
            stored += int((model.fast_synapses.capture.pop()[2] >= model.fast_synapses.gate).sum())
    ticks = None
    if model.clock_trace is not None:
        ticks = torch.cat(model.clock_trace, dim=1)[0].numpy()
        model.clock_trace = None
    if model.fast_synapses is not None:
        model.fast_synapses.capture = None
    written = None if state.synapses is None else float(state.synapses.written.float().mean())
    return nll / (targets * math.log(2)), ticks, stored / targets, written


def tick_profile(ticks, data):
    """Where the clock fired relative to words. Tick t fires as byte t arrives."""
    position = word_positions(data)[:len(ticks)]
    slots = np.where(position < 0, len(_WORD_SLOTS) - 1, np.minimum(position, len(_WORD_SLOTS) - 2))
    starts = position == 0
    # Ticks inside each word, attributed to the word whose bytes they arrive with.
    word = np.cumsum(starts)
    inside = position >= 0
    per_word = np.bincount(word[inside], weights=ticks[inside], minlength=int(word[-1]) + 1)[1:]
    gaps = np.diff(np.flatnonzero(ticks))
    return {
        "bytes_per_tick": len(ticks) / max(int(ticks.sum()), 1),
        "tick_share_by_word_position": {name: float(ticks[slots == index].sum() / max(ticks.sum(), 1))
                                        for index, name in enumerate(_WORD_SLOTS)},
        "byte_share_by_word_position": {name: float((slots == index).mean()) for index, name in enumerate(_WORD_SLOTS)},
        "words": int(len(per_word)),
        "words_without_tick": float((per_word == 0).mean()),
        "words_with_one_tick": float((per_word == 1).mean()),
        "words_with_more_ticks": float((per_word > 1).mean()),
        "longest_gap_bytes": int(gaps.max()) if len(gaps) else None,
        "median_gap_bytes": float(np.median(gaps)) if len(gaps) else None,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--enwik8", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True, nargs="+")
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--bytes", type=int, help="replay only this leading part of the split")
    parser.add_argument("--device", choices=("cpu", "mps", "cuda", "auto"), default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.json.expanduser().exists() and not args.force:
        parser.error(f"{args.json} already exists. Choose a new path or use --force to replace it.")
    torch.set_num_threads(args.threads)
    device = resolve_device(args.device)
    splits = dict(zip(("train", "val", "test"), load_splits(args.enwik8.expanduser()), strict=True))
    data = np.ascontiguousarray(splits[args.split][:args.bytes])
    models = []
    for path in args.checkpoint:
        model, checkpoint = load_checkpoint(path.expanduser(), device)
        if not isinstance(model, EventLM) or (model.cfg.clock_start is None and model.fast_synapses is None):
            parser.error(f"{path}: expected a decoder with a clock or fast synapses")
        intact, ticks, stored, written = replay(model, data, device)
        row = {"checkpoint": str(path), "run_name": checkpoint["run_name"], "intact_bpb": intact}
        print(f"{checkpoint['run_name']}: intact {intact:.5f}", flush=True)
        if ticks is not None:
            row["clock"] = {"source": model.cfg.clock_source, "threshold": model.cfg.clock_threshold,
                            "first_clocked_block": model.cfg.clock_start, **tick_profile(ticks, data)}
            for name in ("mute", "shuffle"):
                model.clock_intervention = name
                row["clock"][f"{name}_bpb"] = replay(model, data, device)[0]
                print(f"  clock {name}: {row['clock'][f'{name}_bpb']:.5f}", flush=True)
            model.clock_intervention = None
        if model.fast_synapses is not None:
            model.fast_synapses.muted = True
            row["fast_synapses"] = {"muted_bpb": replay(model, data, device)[0], "stored_byte_share": stored,
                                    "key_neurons_written_share": written}
            model.fast_synapses.muted = False
            print(f"  synapses muted: {row['fast_synapses']['muted_bpb']:.5f}", flush=True)
            if ticks is not None:
                model.fast_synapses.muted, model.clock_intervention = True, "mute"
                row["both_muted_bpb"] = replay(model, data, device)[0]
                model.fast_synapses.muted, model.clock_intervention = False, None
        models.append(row)
    result = {"data_sha1": ENWIK8_SHA1, "split": args.split, "target_bytes": len(data) - 1, "chunk_bytes": _CHUNK,
              "device": str(device), "models": models}
    with _atomic_output(args.json.expanduser(), "w", args.force) as out:
        json.dump(result, out, indent=2)
        out.write("\n")


if __name__ == "__main__":
    main()
