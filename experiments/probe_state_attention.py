"""Train one attention head over a frozen decoder's own top states.

The decoder is not changed. Its top state and output spikes are recorded once
for windows of its training mixture and for held-out enwik8 windows. Every
byte leaves a cell that holds the top state after the byte and the byte
itself, labelled by the top state before the byte. A head reads the cells of
its last W bytes, and a small reader adds what it read to the decoder's
logits. The arm with W = 1 reads only its own cell, so it measures what the
reader gains without an earlier state. A deep arm replaces the head by layers
of full attention over the cells. All arms train on identical windows and
score the enwik8 windows of the history-gap table.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from byte_training_mixture import TrainingMixture
from common import _atomic_output, resolve_device
from probe_early_context import paired_interval
from probe_early_context import score as score_model
from probe_history_gap import BINS
from probe_match_gap import earlier_matches, share
from snnformer.event_lm import EventLM
from snnformer.neurons import integer_spike
from train_event_lm import load_checkpoint, scoring_logits, training_exposure
from train_lm import ENWIK8_SHA1, load_splits

# Every arm is one head, or a reader of attention layers. The settings it leaves out are those of its class.
ARMS = {
    "own": {"window": 1},
    "soft8": {"window": 8},
    "soft32": {"window": 32},
    "soft128": {"window": 128},
    "soft512": {"window": 512},
    "bytes128": {"window": 128, "cell": "byte"},
    "novotes128": {"window": 128, "votes": False},
    "position128": {"window": 128, "address": "position"},
    "same128": {"window": 128, "label": "same"},
    "race128": {"window": 128, "address": "race"},
    "four128": {"window": 128, "heads": 4},
    "bytekeys128": {"window": 128, "cell": "byte", "keys": "bytes"},
    "deepown": {"layers": 2, "window": 1},
    "deepbytes128": {"layers": 2, "window": 128, "cell": "byte"},
    "deepstate128": {"layers": 2, "window": 128},
}
# Set before the first run: the head over 128 cells against the reader of its own cell.
CRITERION = {"arm": "soft128", "control": "own", "preceding_bytes": (33, 96), "threshold_bpb": 0.02}
# Set before the first run of the deep arms: two attention layers over 128 cells of states, on the targets that no
# earlier match in the window predicts, against the same layers over their own cell and over cells of bytes.
READER_CRITERION = {"arm": "deepstate128", "control": "deepown", "bytes": "deepbytes128", "preceding_bytes": (33, 512),
                    "targets": "not copyable", "threshold_bpb": 0.05, "states_over_bytes_bpb": 0.02}
SPLITS = ("validation", "test", "train")


class CellHead(nn.Module):
    """Heads over the cells of their last ``window`` bytes, and a reader that adds to the logits.

    ``soft`` weighs the cells by a softmax over key matches, ``position`` by
    their distance alone, and ``race`` lets each cell fire up to ``levels``
    spikes above a lateral inhibition and divides the read by the spikes.
    A cell is labelled by the state ``before`` its byte or by the ``same``
    state it holds, and it holds that ``state`` with its byte or the ``byte``
    alone. With ``votes`` the read cells also vote for their own bytes.
    The key neurons read the decoder's ``state`` or only the last ``bytes``.
    """

    def __init__(self, width, context, *, window, address="soft", label="before", cell="state", votes=True, heads=1,
                 keys="state", key_width=64, cell_width=256, hidden=512, levels=4, key_bytes=16):
        super().__init__()
        if (address not in ("soft", "position", "race") or label not in ("before", "same")
                or cell not in ("state", "byte") or keys not in ("state", "bytes") or not 1 <= window <= context
                or heads < 1 or (window == 1 and heads > 1)):
            raise ValueError("unknown head")
        self.window, self.address, self.label, self.votes, self.levels = window, address, label, votes, levels
        self.heads, self.keys = heads, keys

        def projection(outputs):
            return nn.Sequential(nn.Linear(width, outputs), nn.LayerNorm(outputs))

        # Every arm draws these shared parts first, so equal seeds give them equal starting weights.
        self.own, self.byte = projection(cell_width), nn.Embedding(256, cell_width)
        self.prior = nn.Sequential(nn.Linear(256, cell_width // 2), nn.LayerNorm(cell_width // 2))
        # The reader changes the 256 logits and sets the strength of each head's votes.
        self.reader = nn.Sequential(nn.Linear((1 + heads) * cell_width + cell_width // 2, hidden), nn.GELU(),
                                    nn.Linear(hidden, 256 + heads))
        # The head starts as the unchanged decoder, with votes of almost no strength.
        nn.init.zeros_(self.reader[-1].weight)
        nn.init.zeros_(self.reader[-1].bias)
        nn.init.constant_(self.reader[-1].bias[256:], -2.0)
        self.value = projection(cell_width) if cell == "state" else None
        self.matches = window > 1 and address != "position"
        if window > 1:
            distance = torch.arange(context)[:, None] - torch.arange(context)
            self.register_buffer("visible", (distance >= 0) & (distance < window), persistent=False)
            self.register_buffer("distance", distance.clamp(0, window - 1), persistent=False)
            self.distance_bias = nn.Parameter(torch.zeros(heads, window))
        if self.matches:
            source = width
            if keys == "bytes":
                # These key neurons see the last bytes through one hidden layer and never the decoder's state.
                source = cell_width
                self.key_byte, self.key_context = nn.Embedding(256, 64), nn.Conv1d(64, source, key_bytes)
            # Every head has its own key neurons.
            self.query, self.key = nn.Linear(source, heads * key_width), nn.Linear(source, heads * key_width)
            self.query_norm, self.key_norm = nn.LayerNorm(key_width), nn.LayerNorm(key_width)
            self.gain = nn.Parameter(torch.zeros(heads, 1, 1))
        if address == "race":
            self.inhibition, self.threshold = (nn.Parameter(torch.ones(heads, 1, 1)),
                                               nn.Parameter(torch.zeros(heads, 1, 1)))
            self.spacing = nn.Parameter(torch.zeros(heads, 1, 1))

    def weights(self, state, tokens):
        """The share of every cell in each head's read of every byte, as [batch or 1, heads, bytes, cells]."""
        steps = state.shape[1]
        visible = self.visible[:steps, :steps]
        scores = self.distance_bias[:, self.distance[:steps, :steps]][None]
        if self.matches:
            if self.keys == "bytes":
                # The convolution over the last bytes, as one matrix product: a 1-D convolution is slow on a CPU.
                reach = self.key_context.kernel_size[0]
                recent = F.pad(self.key_byte(tokens), (0, 0, reach - 1, 0)).unfold(1, reach, 1).flatten(2)
                state = F.gelu(F.linear(recent, self.key_context.weight.flatten(1), self.key_context.bias))
            # A cell is labelled by the state before its byte, which is empty at a reset.
            label = F.pad(state[:, :-1], (0, 0, 1, 0)) if self.label == "before" else state
            query = self.query_norm(self.query(state).unflatten(-1, (self.heads, -1))).transpose(1, 2)
            key = self.key_norm(self.key(label).unflatten(-1, (self.heads, -1))).transpose(1, 2)
            scores = scores + query @ key.transpose(2, 3) * (self.gain.exp() / math.sqrt(key.shape[-1]))
        if self.address != "race":
            return scores.masked_fill(~visible, -torch.inf).softmax(-1)
        # The inhibition follows the mean excitation, and a cell fires once more per spacing above it.
        mean = scores.masked_fill(~visible, 0).sum(-1, keepdim=True) / visible.sum(-1, keepdim=True)
        excess = (scores - self.inhibition * mean - self.threshold) / self.spacing.exp()
        spikes = (integer_spike((excess + 0.5) / self.levels, self.levels) * self.levels).masked_fill(~visible, 0)
        return spikes / spikes.sum(-1, keepdim=True).clamp_min(1)

    def forward(self, state, tokens, logits):
        cells = self.byte(tokens)
        if self.value is not None:
            cells = cells + self.value(state)
        votes = F.one_hot(tokens, 256).to(cells.dtype)[:, None] if self.votes else None
        if self.window > 1:
            weights = self.weights(state, tokens)
            cells = (weights @ cells[:, None]).transpose(1, 2).flatten(2)
            if self.votes:
                votes = weights @ votes
        change = self.reader(torch.cat((self.own(state), cells, self.prior(logits.log_softmax(-1))), -1))
        logits = logits + change[..., :256]
        if not self.votes:
            return logits
        # Each head's read cells vote for their own bytes, with a strength the reader sets.
        return logits + (F.softplus(change[..., 256:]).transpose(1, 2)[..., None] * votes).sum(1)


class AttentionBlock(nn.Module):
    """Heads that weigh the visible cells by key matches and a learned distance, then one hidden layer."""

    def __init__(self, model, heads, window):
        super().__init__()
        self.heads = heads
        self.norm, self.project, self.mix = nn.LayerNorm(model), nn.Linear(model, 3 * model), nn.Linear(model, model)
        self.norm_hidden = nn.LayerNorm(model)
        self.hidden = nn.Sequential(nn.Linear(model, 2 * model), nn.GELU(), nn.Linear(2 * model, model))
        self.distance_bias = nn.Parameter(torch.zeros(heads, window))

    def forward(self, cells, visible, distance):
        query, key, value = self.project(self.norm(cells)).unflatten(-1, (3, self.heads, -1)).permute(2, 0, 3, 1, 4)
        scores = query @ key.transpose(2, 3) / math.sqrt(key.shape[-1]) + self.distance_bias[:, distance]
        weights = scores.masked_fill(~visible, -torch.inf).softmax(-1)
        cells = cells + self.mix((weights @ value).transpose(1, 2).flatten(2))
        return cells + self.hidden(self.norm_hidden(cells)), weights


class DeepReader(nn.Module):
    """Layers of full attention over the cells of the last ``window`` bytes, and a reader that adds to the logits.

    A cell holds its byte, and with ``state`` also the top state after the
    byte. Every layer lets each cell read the cells of its last ``window``
    bytes and compute with what it read. The heads of the last layer also
    vote for the bytes of their cells, as the single head does.
    """

    def __init__(self, width, context, *, window, cell="state", layers=2, heads=4, model=256, cell_width=256,
                 hidden=512):
        super().__init__()
        if cell not in ("state", "byte") or not 1 <= window <= context or min(layers, heads) < 1 or model % heads:
            raise ValueError("unknown reader")
        self.window, self.layers, self.heads, self.votes = window, layers, heads, True

        def projection(outputs):
            return nn.Sequential(nn.Linear(width, outputs), nn.LayerNorm(outputs))

        # Every arm draws these shared parts first, so equal seeds give them equal starting weights.
        self.own, self.byte = projection(cell_width), nn.Embedding(256, model)
        self.prior = nn.Sequential(nn.Linear(256, cell_width // 2), nn.LayerNorm(cell_width // 2))
        self.reader = nn.Sequential(nn.Linear(cell_width + model + cell_width // 2, hidden), nn.GELU(),
                                    nn.Linear(hidden, 256 + heads))
        # The reader starts as the unchanged decoder, with votes of almost no strength.
        nn.init.zeros_(self.reader[-1].weight)
        nn.init.zeros_(self.reader[-1].bias)
        nn.init.constant_(self.reader[-1].bias[256:], -2.0)
        self.blocks = nn.ModuleList(AttentionBlock(model, heads, window) for _ in range(layers))
        self.norm = nn.LayerNorm(model)
        distance = torch.arange(context)[:, None] - torch.arange(context)
        self.register_buffer("visible", (distance >= 0) & (distance < window), persistent=False)
        self.register_buffer("distance", distance.clamp(0, window - 1), persistent=False)
        self.value = projection(model) if cell == "state" else None

    def read(self, state, tokens):
        """The cells after the last layer, and the share of every cell in the read of each of its heads."""
        steps = tokens.shape[1]
        cells = self.byte(tokens)
        if self.value is not None:
            cells = cells + self.value(state)
        for block in self.blocks:
            cells, weights = block(cells, self.visible[:steps, :steps], self.distance[:steps, :steps])
        return self.norm(cells), weights

    def weights(self, state, tokens):
        return self.read(state, tokens)[1]

    def forward(self, state, tokens, logits):
        cells, weights = self.read(state, tokens)
        change = self.reader(torch.cat((self.own(state), cells, self.prior(logits.log_softmax(-1))), -1))
        # Each cell's share goes to the byte it holds, which is the vote of that head.
        held = tokens[:, None, None].expand(-1, self.heads, tokens.shape[1], -1)
        votes = torch.zeros(*weights.shape[:3], 256, dtype=weights.dtype, device=weights.device)
        votes = votes.scatter_add(-1, held, weights)
        return (logits + change[..., :256]
                + (F.softplus(change[..., 256:]).transpose(1, 2)[..., None] * votes).sum(1))


@torch.no_grad()
def record(model, tokens):
    """The frozen decoder's top state and its output, both in spikes, for a batch of windows."""
    states = []
    handle = model.output.register_forward_pre_hook(lambda _module, inputs: states.append(inputs[0]))
    try:
        counts = model(tokens)
    finally:
        handle.remove()
    # The forecast neurons read the stream under the clock first. The last read is the top state.
    return states[-1], counts


def record_split(model, windows, stem, batch_size, device):
    """Write the top state and the output spikes of [windows, bytes + 1] byte windows as 16-bit counts."""
    arrays = {name: np.lib.format.open_memmap(f"{stem}_{name}.npy", mode="w+", dtype=np.int16,
                                              shape=(windows.shape[0], windows.shape[1] - 1, size))
              for name, size in (("state", model.output.weight.shape[1]), ("counts", 256))}
    started = time.perf_counter()
    for start in range(0, len(windows), batch_size):
        rows = slice(start, start + batch_size)
        for name, spikes in zip(arrays, record(model, windows[rows, :-1].to(device)), strict=True):
            spikes = spikes.cpu()
            if not torch.equal(spikes, spikes.round()) or spikes.abs().max() > 32767:
                raise ValueError("the decoder's spikes are not 16-bit counts")
            arrays[name][rows] = spikes.numpy().astype(np.int16)
        if rows.stop % (64 * batch_size) == 0:
            print(f"{Path(stem).name} | recorded {rows.stop} of {len(windows)} windows | "
                  f"{time.perf_counter() - started:.0f}s", flush=True)
    for array in arrays.values():
        array.flush()
    np.save(f"{stem}_tokens.npy", windows.numpy().astype(np.uint8))


class Recorded:
    """The recorded windows of one split."""

    def __init__(self, stem, resolution):
        self.tokens, self.state, self.counts = (np.load(f"{stem}_{name}.npy", mmap_mode="r")
                                                for name in ("tokens", "state", "counts"))
        self.resolution = resolution

    def __len__(self):
        return len(self.tokens)

    def batch(self, rows, device):
        """The state in spikes, the input bytes, the decoder's logits and the targets of some windows."""
        rows = np.sort(np.asarray(rows))
        tokens = torch.from_numpy(self.tokens[rows].astype(np.int64)).to(device)
        state = torch.from_numpy(self.state[rows].astype(np.float32)).to(device) / self.resolution
        counts = torch.from_numpy(self.counts[rows].astype(np.float32)).to(device)
        return state, tokens[:, :-1], scoring_logits(counts, True, self.resolution), tokens[:, 1:]

    def batches(self, count, batch_size, device):
        for start in range(0, count, batch_size):
            yield self.batch(np.arange(start, min(start + batch_size, count)), device)


@torch.no_grad()
def check_causal(head, width, context, device, *, trained):
    """The output at a byte must not change with anything that follows the byte."""
    generator = torch.Generator().manual_seed(0)
    if not trained:
        # A fresh reader adds almost nothing, so the check uses a copy with a random last layer.
        head = copy.deepcopy(head)
        nn.init.normal_(head.reader[-1].weight, std=0.05)
    cut = context // 2
    first = (torch.randn(2, context, width, generator=generator), torch.randint(256, (2, context), generator=generator),
             torch.randn(2, context, 256, generator=generator))
    second = tuple(torch.cat((part[:, :cut], part[:, cut:].flip(0)), dim=1) for part in first)
    before, after = (head(*(part.to(device) for part in parts)) for parts in (first, second))
    if not torch.allclose(before[:, :cut], after[:, :cut], atol=1e-3, rtol=1e-4):
        raise AssertionError("a head reads bytes that follow its own")
    added = [output[:, cut:] - parts[2][:, cut:].to(device) for output, parts in ((before, first), (after, second))]
    if torch.allclose(*added, atol=1e-3):
        raise AssertionError("the later bytes did not change what the head adds")


@torch.no_grad()
def score(heads, recorded, count, batch_size, device):
    """Bits per target of the decoder and of every head, as [windows, bytes] arrays."""
    losses = {name: [] for name in ("decoder", *heads)}
    for state, tokens, logits, targets in recorded.batches(count, batch_size, device):
        outputs = {"decoder": logits, **{name: head(state, tokens, logits) for name, head in heads.items()}}
        for name, output in outputs.items():
            losses[name].append(F.cross_entropy(output.transpose(1, 2), targets, reduction="none").cpu().double())
    return {name: torch.cat(parts).numpy() / math.log(2) for name, parts in losses.items()}


@torch.no_grad()
def share_by_distance(head, recorded, count, batch_size, device):
    """Mean share of the read at each distance, over the bytes that have a cell that far back."""
    total, cells = np.zeros(head.window), np.zeros(head.window)
    for state, tokens, *_ in recorded.batches(count, batch_size, device):
        weights = head.weights(state, tokens).cpu()
        for distance in range(head.window):
            shares = weights.diagonal(-distance, dim1=-2, dim2=-1)
            total[distance] += shares.sum(dtype=torch.float64).item()
            cells[distance] += shares.numel()
    return (total / cells).tolist()


def disjoint_windows(data, count, span):
    starts = np.linspace(0, len(data) - span, count, dtype=np.int64)
    if count > 1 and np.diff(starts).min() < span:
        raise ValueError("the windows must be disjoint")
    return starts, torch.from_numpy(data[starts[:, None] + np.arange(span)].astype(np.int64))


def file_digest(path):
    with path.expanduser().open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def record_decoder(args, device):
    """Record the frozen decoder once: validation and test windows of enwik8, then windows of its mixture."""
    model, checkpoint = load_checkpoint(args.checkpoint.expanduser(), device)
    if not isinstance(model, EventLM):
        raise ValueError("the checkpoint must be a spiking decoder")
    span = args.window_bytes + 1
    train, val, test = load_splits(args.enwik8.expanduser())
    mixture = TrainingMixture(args.training_mixture.expanduser(), train, args.window_bytes)
    generator = torch.Generator().manual_seed(args.seed)
    drawn = [mixture.batch(args.batch_size, generator) for _ in range(args.train_windows // args.batch_size)]
    windows = {"train": torch.cat([torch.cat((inputs, targets[:, -1:]), dim=1) for inputs, targets in drawn])}
    starts = {}
    for split, data, count in (("validation", val, args.curve_windows), ("test", test, args.windows)):
        starts[split], windows[split] = disjoint_windows(data, count, span)
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        record_split(model, windows[split], args.cache_dir / split, args.batch_size, device)
    manifest = {
        "format": 1, "checkpoint": str(args.checkpoint), "sha256": file_digest(args.checkpoint),
        "run_name": checkpoint["run_name"], "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "training_targets": training_exposure(checkpoint), "state_width": model.output.weight.shape[1],
        "resolution": model.cfg.resolution, "window_bytes": args.window_bytes, "seed": args.seed, "device": str(device),
        "train": {"windows": len(windows["train"]), "training_mixture": mixture.receipt()},
        **{split: {"windows": len(windows[split]), "window_starts": starts[split].tolist()} for split in starts},
    }
    with _atomic_output(args.cache_dir / "manifest.json", "w", False) as stream:
        json.dump(manifest, stream, indent=2)
        stream.write("\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True, help="the spiking decoder, which stays frozen")
    parser.add_argument("--enwik8", type=Path, required=True)
    parser.add_argument("--training-mixture", type=Path, required=True, help="catalogue of the decoder's mixture")
    parser.add_argument("--cache-dir", type=Path, required=True, help="the decoder's recorded states, tens of GB")
    parser.add_argument("--record-only", action="store_true", help="record the decoder and stop")
    parser.add_argument("--reference", type=Path, help="a Transformer scored on the same windows")
    parser.add_argument("--arms", default=",".join(ARMS), help="comma-separated arms to train")
    parser.add_argument("--train-windows", type=int, default=32768, help="recorded windows of the mixture")
    parser.add_argument("--epochs", type=int, default=4, help="passes over the recorded training windows")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--warmup", type=int, default=200)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=5309)
    parser.add_argument("--windows", type=int, default=2048, help="evenly spaced disjoint test windows")
    parser.add_argument("--window-bytes", type=int, default=512, help="bytes after each reset")
    parser.add_argument("--curve-every", type=int, default=512, help="updates between validation scores")
    parser.add_argument("--curve-windows", type=int, default=256, help="evenly spaced disjoint validation windows")
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--save", type=Path, help="write the trained heads here")
    parser.add_argument("--losses", type=Path, help="write every arm's bits per test target here, as an .npz file")
    parser.add_argument("--json", type=Path, help="the record of the trained arms")
    args = parser.parse_args()
    names = args.arms.split(",")
    if set(names) - set(ARMS) or len(set(names)) != len(names):
        parser.error(f"choose distinct arms from {', '.join(ARMS)}")
    if min(args.train_windows // args.batch_size, args.epochs, args.windows, args.window_bytes, args.curve_every,
           args.curve_windows) < 1:
        parser.error("sizes, passes and intervals must be positive")
    if not args.record_only and (args.json is None or any(
            path is not None and path.exists() for path in (args.json, args.save, args.losses))):
        parser.error("name a fresh record, and fresh files for the heads and the losses")

    torch.set_num_threads(args.threads)
    device = resolve_device(args.device)
    if not (args.cache_dir / "manifest.json").exists():
        record_decoder(args, device)
    manifest = json.loads((args.cache_dir / "manifest.json").read_text())
    expected = {"sha256": file_digest(args.checkpoint), "window_bytes": args.window_bytes, "seed": args.seed}
    if (any(manifest[name] != value for name, value in expected.items())
            or manifest["train"]["windows"] < args.train_windows or manifest["test"]["windows"] != args.windows
            or manifest["validation"]["windows"] != args.curve_windows):
        parser.error("the recorded states belong to another decoder or to other windows")
    if args.record_only:
        return
    recorded = {split: Recorded(args.cache_dir / split, manifest["resolution"]) for split in SPLITS}
    width, context = manifest["state_width"], args.window_bytes
    bins = [(low, high) for low, high in BINS if high <= context]
    low, high = CRITERION["preceding_bytes"]

    heads = {}
    for name in names:
        torch.manual_seed(args.seed)
        heads[name] = (DeepReader if "layers" in ARMS[name] else CellHead)(width, context, **ARMS[name]).to(device)
        check_causal(heads[name], width, context, device, trained=False)
    # The reader's criterion counts the targets that the longest earlier match in the window does not predict.
    first, last = READER_CRITERION["preceding_bytes"]
    judged = all(READER_CRITERION[role] in names for role in ("arm", "control", "bytes"))
    unmatched = {}
    for split in ("validation", "test") if judged else ():
        unmatched[split] = ~earlier_matches(np.asarray(recorded[split].tokens), context)[1]
        unmatched[split][:, :first - 1] = False
        unmatched[split][:, last:] = False
    optimizers = {name: torch.optim.AdamW(head.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
                  for name, head in heads.items()}
    per_pass = args.train_windows // args.batch_size
    updates = args.epochs * per_pass
    generator = torch.Generator().manual_seed(args.seed)
    # Every pass visits the first recorded windows once, in its own order.
    order = torch.stack([torch.randperm(args.train_windows, generator=generator)[:per_pass * args.batch_size]
                         for _ in range(args.epochs)]).view(updates, args.batch_size).numpy()

    def rate(step):
        if step < args.warmup:
            return (step + 1) / args.warmup
        return max(0.01, 0.5 * (1 + math.cos(math.pi * (step - args.warmup) / max(1, updates - args.warmup))))

    running, curve, started = dict.fromkeys(("decoder", *names), 0.0), [], time.perf_counter()
    for step in range(updates):
        state, tokens, logits, targets = recorded["train"].batch(order[step], device)
        running["decoder"] += F.cross_entropy(logits.transpose(1, 2), targets).item() / math.log(2)
        for name, head in heads.items():
            optimizer = optimizers[name]
            for group in optimizer.param_groups:
                group["lr"] = args.learning_rate * rate(step)
            loss = F.cross_entropy(head(state, tokens, logits).transpose(1, 2), targets)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            optimizer.step()
            running[name] += loss.item() / math.log(2)
        done = step + 1
        if done % 100 == 0:
            print(f"update {done} | {time.perf_counter() - started:.0f}s | training "
                  + " | ".join(f"{name} {total / 100:.4f}" for name, total in running.items()), flush=True)
            running = dict.fromkeys(running, 0.0)
        if done % args.curve_every == 0 or done == updates:
            losses = score(heads, recorded["validation"], args.curve_windows, args.batch_size, device)
            curve.append({"updates": done, "seconds": time.perf_counter() - started,
                          "validation_bpb": {name: float(loss.mean()) for name, loss in losses.items()},
                          f"validation_bpb_{low}-{high}": {name: float(loss[:, low - 1:high].mean())
                                                         for name, loss in losses.items()}})
            print(f"update {done} | validation {low}-{high} "
                  + " | ".join(f"{name} {bpb:.4f}" for name, bpb in curve[-1][f"validation_bpb_{low}-{high}"].items()),
                  flush=True)
            if judged:
                curve[-1][f"validation_bpb_not_copyable_{first}-{last}"] = {
                    name: float(loss[unmatched["validation"]].mean()) for name, loss in losses.items()}
                print(f"update {done} | validation {first}-{last} not copyable " + " | ".join(
                    f"{name} {bpb:.4f}" for name, bpb in curve[-1][f"validation_bpb_not_copyable_{first}-{last}"].items()),
                    flush=True)
    seconds = time.perf_counter() - started
    for head in heads.values():
        check_causal(head, width, context, device, trained=True)
    if args.save is not None:
        with _atomic_output(args.save, "wb", False) as stream:
            torch.save({"arms": {name: ARMS[name] for name in names}, "seed": args.seed, "updates": updates,
                        "state_dicts": {name: head.state_dict() for name, head in heads.items()}}, stream)

    losses = score(heads, recorded["test"], args.windows, args.batch_size, device)
    resamples = np.random.default_rng(5261).integers(args.windows, size=(3000, args.windows))
    reference = None
    if args.reference is not None:
        other, other_checkpoint = load_checkpoint(args.reference.expanduser(), device)
        test_windows = torch.from_numpy(np.asarray(recorded["test"].tokens).astype(np.int64))
        reference_loss = score_model(other, test_windows, args.batch_size, device)
        reference = {"checkpoint": str(args.reference), "run_name": other_checkpoint["run_name"],
                     "parameters": sum(parameter.numel() for parameter in other.parameters()),
                     "bpb_by_preceding_bytes": {f"{low}-{high}": float(reference_loss[:, low - 1:high].mean())
                                                for low, high in bins}}
        del other
    if args.losses is not None:
        kept = losses if reference is None else {**losses, "reference": reference_loss}
        with _atomic_output(args.losses, "wb", False) as stream:
            np.savez_compressed(stream, **{name: loss.astype(np.float32) for name, loss in kept.items()})
    baselines = {"decoder": losses["decoder"]}
    if CRITERION["control"] in losses:
        baselines["control"] = losses[CRITERION["control"]]

    def histories(loss, *, own=False):
        rows = []
        for low, high in bins:
            part = slice(low - 1, high)
            row = {"preceding_bytes": [low, high], "targets": loss[:, part].size, "bpb": float(loss[:, part].mean())}
            if own:
                for label, baseline in baselines.items():
                    gain = baseline[:, part] - loss[:, part]
                    row.update({f"gain_over_{label}_bpb": float(gain.mean()),
                                f"gain_over_{label}_paired_95_bpb": paired_interval(gain, resamples)})
            if reference is not None:
                gap = loss[:, part] - reference_loss[:, part]
                row.update(gap_bpb=float(gap.mean()), gap_paired_95_bpb=paired_interval(gap, resamples))
            rows.append(row)
        return rows

    arms = []
    for name, head in heads.items():
        arm = {"name": name, **{key: value for key, value in vars(head).items()
                                if key in ("window", "address", "label", "votes", "heads", "keys", "layers")},
               "cell": "byte" if head.value is None else "state",
               "parameters": sum(parameter.numel() for parameter in head.parameters()),
               "histories": histories(losses[name], own=True)}
        if head.window > 1:
            arm["share_by_distance"] = share_by_distance(head, recorded["test"], args.curve_windows, args.batch_size,
                                                         device)
        arms.append(arm)
        print(f"{name} | " + " | ".join(
            f"{row['preceding_bytes'][0]}-{row['preceding_bytes'][1]} {row['bpb']:.4f} ({row['gain_over_decoder_bpb']:+.4f})"
            for row in arm["histories"]), flush=True)

    criterion = dict(CRITERION, preceding_bytes=list(CRITERION["preceding_bytes"]))
    if CRITERION["arm"] in losses and CRITERION["control"] in losses:
        gain = (losses[CRITERION["control"]] - losses[CRITERION["arm"]])[:, low - 1:high]
        criterion.update(gain_bpb=float(gain.mean()), gain_paired_95_bpb=paired_interval(gain, resamples),
                         passed=bool(gain.mean() >= CRITERION["threshold_bpb"]))
        print(f"criterion | {CRITERION['arm']} over {CRITERION['control']} at {low}-{high} bytes "
              f"{criterion['gain_bpb']:+.4f} bpb | threshold {CRITERION['threshold_bpb']}", flush=True)
    reader_criterion = dict(READER_CRITERION, preceding_bytes=list(READER_CRITERION["preceding_bytes"]))
    if judged:
        chosen = unmatched["test"]
        for label, role in (("gain", "control"), ("states_over_bytes", "bytes")):
            difference = (losses[READER_CRITERION[role]] - losses[READER_CRITERION["arm"]]) * chosen
            mean = share(difference.sum(axis=1), chosen.sum(axis=1), resamples)
            reader_criterion.update({f"{label}_bpb": mean["share"], f"{label}_paired_95_bpb": mean["paired_95"]})
        reader_criterion.update(
            targets_counted=int(chosen.sum()),
            passed=bool(reader_criterion["gain_bpb"] >= READER_CRITERION["threshold_bpb"]
                        and reader_criterion["states_over_bytes_bpb"] >= READER_CRITERION["states_over_bytes_bpb"]))
        print(f"reader criterion | {READER_CRITERION['arm']} over {READER_CRITERION['control']} on the targets no "
              f"match predicts at {first}-{last} bytes {reader_criterion['gain_bpb']:+.4f} bpb | threshold "
              f"{READER_CRITERION['threshold_bpb']} | over {READER_CRITERION['bytes']} "
              f"{reader_criterion['states_over_bytes_bpb']:+.4f} bpb | threshold "
              f"{READER_CRITERION['states_over_bytes_bpb']}", flush=True)
    result = {
        "data_sha1": ENWIK8_SHA1, "split": "test", "device": str(device), "windows": args.windows,
        "window_bytes": context, "window_starts": manifest["test"]["window_starts"], "resamples": len(resamples),
        "decoder": {name: manifest[name] for name in ("checkpoint", "sha256", "run_name", "parameters",
                                                      "training_targets", "state_width")}
        | {"recorded_on": manifest["device"], "histories": histories(losses["decoder"])},
        "reference": reference,
        "training": {"seed": args.seed, "windows": args.train_windows, "passes": args.epochs, "updates": updates,
                     "batch_size": args.batch_size, "targets": updates * args.batch_size * context,
                     "learning_rate": args.learning_rate, "warmup": args.warmup, "weight_decay": args.weight_decay,
                     "seconds": seconds, "recorded_windows": manifest["train"]["windows"],
                     "training_mixture": manifest["train"]["training_mixture"]},
        "criterion": criterion, "reader_criterion": reader_criterion, "curve": curve, "arms": arms,
    }
    with _atomic_output(args.json, "w", False) as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
