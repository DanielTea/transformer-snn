"""Attach fitted read-head strengths to a decoder checkpoint.

``probe_coasting_heads.py --readout-dir`` writes the strengths that a replay
fitted on enwik8 validation. This writes a checkpoint that carries them, so
the streaming evaluator, the native integer decoder and the pulse audit run
the heads inside the decoder. No decoder weight changes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from common import _atomic_output
from snnformer.event_lm import EventLM, EventLMConfig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True, help="a spiking decoder, with or without heads")
    parser.add_argument("--readout", type=Path, required=True, help="strengths in the form of --head-readout")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = args.checkpoint.expanduser()
    checkpoint = torch.load(source, map_location="cpu", weights_only=True)
    if checkpoint.get("model_type") != "EventLM":
        parser.error(f"{args.checkpoint}: expected a spiking decoder")
    text = args.readout.expanduser().read_bytes()
    fitted = json.loads(text)
    if fitted.get("synapse_strength_spikes") is not None:
        parser.error(f"{args.readout}: a joint fit also sets the fast synapses and is not attached here")
    setting = fitted["setting"]
    heads = {"head_orders": tuple(setting["orders"]),
             **{f"head_{name}": setting[name] for name in ("cells", "gate", "tape", "lookup", "miss", "relock")}}
    model = EventLM(EventLMConfig(**{**checkpoint["cfg"], **heads})).eval()
    weights = {name: value for name, value in checkpoint["state_dict"].items() if not name.startswith("read_heads.")}
    missing, unexpected = model.load_state_dict(weights, strict=False)
    if unexpected or set(missing) != {"read_heads.readout"}:
        parser.error(f"{args.checkpoint}: the weights do not fit the decoder with these heads")
    with torch.no_grad():
        model.read_heads.readout[:, 1:] = torch.tensor(fitted["head_strength_spikes"], dtype=torch.int32)
    saved = dict(checkpoint)
    saved["cfg"] = {**checkpoint["cfg"], **heads}
    saved["state_dict"] = model.state_dict()
    saved["run_name"] = f"{checkpoint['run_name']}_fitted_heads"
    with source.open("rb") as stream:
        source_hash = hashlib.file_digest(stream, "sha256").hexdigest()
    saved["read_heads"] = {"source_checkpoint": str(args.checkpoint), "source_sha256": source_hash,
                           "readout": str(args.readout), "readout_sha256": hashlib.sha256(text).hexdigest(),
                           "fit": fitted["fit"], "fit_run": fitted["source_run"]}
    args.output.expanduser().parent.mkdir(parents=True, exist_ok=True)
    with _atomic_output(args.output.expanduser(), "wb", True) as out:
        torch.save(saved, out)
    print(f"{saved['run_name']}: {len(heads['head_orders'])} heads from {args.readout} -> {args.output}")


if __name__ == "__main__":
    main()
