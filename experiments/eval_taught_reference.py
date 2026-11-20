"""Score a reference Transformer along unbroken streams, with fast output synapses that every byte teaches.

This is the control to the fast output synapses of a stack (eval_stream.py
with ``--teach-levels``). The split is cut into the same parts as there, and
each part is one unbroken stream. A Transformer carries no state, so it reads
a stream in warmed windows: the next ``--context`` bytes are predicted from a
window that holds the ``--history`` bytes before them.

The activity of the Transformer's last layer at a byte, less its running mean
and scaled to unit length, drives one fast synapse to every output unit. The
fast synapses start at zero in every stream. When the next byte arrives, the
synapses to its unit grow by the share of the readout that the unit missed,
times that activity and ``--rates``, and the synapses to the other units
shrink by the shares those took. No byte is learned from before it is
predicted, and the unscored lead of a part teaches like any other byte.

The stack's fast synapses follow the same rule in whole numbers. Here it runs
in floats, and several rates are scored side by side on the same pass. A rate
of 0 leaves the Transformer as it is.
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
from eval_stream import stream_rows
from snnformer import count_params
from snnformer.event_lm import EventLM
from train_event_lm import load_checkpoint, training_exposure
from train_lm import load_splits

# Bytes over which a unit's running mean is taken, as for the fibres that a stack's fast synapses hear.
SLOW = 1024


@torch.inference_mode()
def score(model, data, streams, lead, context, history, rates, device, batch=8):
    """The bits of every ``context`` bytes of every stream at each rate as [rates, streams, windows], and the
    number of scored targets."""
    inputs, targets, scored = stream_rows(data, streams, lead)
    length = inputs.shape[1]
    seen = {}
    hook = model.norm.register_forward_hook(lambda module, args, output: seen.__setitem__("activity", output))
    rate = torch.tensor(rates, dtype=torch.float32, device=device)
    fast = mean = None
    bits, heard, start = [], 0, 0
    try:
        while start < length:
            # Windows of one width share a pass: all but the first of a stream, which hold less than the history,
            # and the last, which may predict fewer bytes.
            spans = [(start, min(start + context, length))]
            while start >= history and len(spans) < batch and spans[-1][1] + context <= length:
                spans.append((spans[-1][1], spans[-1][1] + context))
            skipped = min(start, history)
            window = torch.stack([inputs[:, begin - skipped:end] for begin, end in spans], 1).to(device)
            logits = model(window.flatten(0, 1)).unflatten(0, (streams, len(spans)))
            activity = seen["activity"].unflatten(0, (streams, len(spans)))
            # The bytes after the history are new. In the order of the stream they are the bytes of this pass.
            base = logits[:, :, skipped:].flatten(1, 2).float()
            heard_now = activity[:, :, skipped:].flatten(1, 2).float()
            stop = spans[-1][1]
            byte = targets[:, start:stop].to(device)
            if fast is None:
                fast = base.new_zeros(len(rates), streams, base.shape[-1], heard_now.shape[-1])
                mean = base.new_zeros(streams, heard_now.shape[-1])
            losses = base.new_zeros(len(rates), streams, byte.shape[1])
            for place in range(byte.shape[1]):
                centred = heard_now[:, place] - mean
                heard += 1
                mean += centred / min(heard, SLOW)
                active = centred / centred.norm(dim=-1, keepdim=True).clamp_min(1e-6)
                shares = F.log_softmax(base[:, place] + torch.einsum("rsud,sd->rsu", fast, active), -1)
                came = byte[:, place].view(1, -1, 1).expand(len(rates), -1, 1)
                losses[:, :, place] = -shares.gather(-1, came).squeeze(-1)
                # The byte that came teaches its unit the share it missed, and the others lose what they took.
                missed = -shares.exp()
                missed.scatter_add_(-1, came, torch.ones_like(came, dtype=missed.dtype))
                fast.addcmul_((rate[:, None, None] * missed)[..., None], active[None, :, None, :])
            # MPS has no 64-bit floats, so the sum is taken on the CPU.
            counted = (losses.cpu() * scored[:, start:stop]).double() / math.log(2)
            bits += [part.sum(-1) for part in counted.split(context, -1)]
            start = stop
    finally:
        hook.remove()
    return torch.stack(bits, -1).numpy(), int(scored.sum())


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--enwik8", type=Path, required=True, help="canonical enwik8 file")
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument("--bytes", type=int, help="score only the first bytes of the split, for a quick look")
    parser.add_argument("--streams", type=int, default=16, help="parts of the split that run side by side")
    parser.add_argument("--lead", type=int, default=8192, help="unscored bytes that a part after the first hears")
    parser.add_argument("--context", type=int, default=64, help="bytes that a window predicts")
    parser.add_argument("--history", type=int, default=448, help="bytes of the stream that a window holds before them")
    parser.add_argument("--rates", type=float, nargs="+", default=[0.0],
                        help="rates of the fast output synapses, scored side by side; 0 is the Transformer as it is")
    parser.add_argument("--batch", type=int, default=8, help="windows of a stream that share a pass")
    parser.add_argument("--device", choices=("cpu", "mps", "cuda", "auto"), default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--bits", type=Path,
                        help="keep the bits of every window's new bytes as [rates, streams, windows] in this .npy "
                             "file, so that a stack's scoring of the same streams can be compared piece by piece")
    args = parser.parse_args()
    if min(args.streams, args.context, args.batch, args.threads) < 1 or min(args.lead, args.history) < 0:
        parser.error("streams, context, batch and threads must be positive, the lead and the history not negative")
    if min(args.rates) < 0 or len(set(args.rates)) != len(args.rates):
        parser.error("the rates are distinct and not negative")
    if args.json.exists():
        parser.error(f"{args.json} already exists")
    if args.bits is not None and (args.bits.suffix != ".npy" or args.bits.exists()):
        parser.error("--bits names a new .npy file")
    torch.set_num_threads(args.threads)
    device = resolve_device(args.device)
    _, validation, test = load_splits(args.enwik8)
    data = torch.from_numpy(np.array(test if args.split == "test" else validation))
    record = {"protocol": "unbroken_streams_in_warmed_windows", "split": args.split, "streams": args.streams,
              "unscored_lead_bytes": args.lead, "context_bytes": args.context,
              "history_bytes": args.history, "device": str(device)}
    if args.bytes is not None:
        # A shortened split is named in the record, so its score is not taken for the whole split's.
        data = data[:args.bytes]
        record["first_bytes_of_split"] = args.bytes
    model, checkpoint = load_checkpoint(args.checkpoint, device)
    if isinstance(model, EventLM):
        parser.error(f"{args.checkpoint}: a stack is scored and taught as a stream by eval_stream.py")
    bits, targets = score(model, data, args.streams, args.lead, args.context, args.history, args.rates, device,
                          args.batch)
    record.update(checkpoint=str(args.checkpoint), run_name=checkpoint.get("run_name"),
                  parameters=count_params(model), training_targets=training_exposure(checkpoint), targets=targets)
    rows = [{"rate": rate, "bpb": float(bits[index].sum() / targets)} for index, rate in enumerate(args.rates)]
    if 0.0 in args.rates:
        # The windows are resampled together for the difference to the Transformer as it is.
        frozen = bits[args.rates.index(0.0)].reshape(-1)
        resamples = np.random.default_rng(5261).integers(len(frozen), size=(3000, len(frozen)))
        for index, row in enumerate(rows):
            if row["rate"]:
                paired = bits[index].reshape(-1) - frozen
                row["bpb_over_frozen"] = float(paired.sum() / targets)
                row["bpb_over_frozen_paired_95"] = np.percentile(paired[resamples].sum(axis=1) / targets,
                                                                 (2.5, 97.5)).tolist()
    record["fast_output_synapses"] = rows
    if args.bits is not None:
        np.save(args.bits, bits)
    for row in rows:
        extra = "" if "bpb_over_frozen" not in row else f" | {row['bpb_over_frozen']:+.4f} to the frozen Transformer"
        print(f"rate {row['rate']:g} | {targets:,} targets | {row['bpb']:.6f} bpb{extra}", flush=True)
    with _atomic_output(args.json, "w", False) as stream:
        json.dump(record, stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
