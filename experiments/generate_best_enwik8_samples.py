"""Generate paired continuations from the enwik8 leaders in the width study."""

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import torch

from generate_lm import display_text, generate
from snnformer.event_lm import EventLM
from train_event_lm import load_checkpoint, scoring_logits

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "checkpoints/packed-width-study"


class Decoder:
    def __init__(self, path):
        model, metadata = load_checkpoint(path, torch.device("cpu"))
        self.run_name = metadata["run_name"]
        self.event = isinstance(model, EventLM)
        if self.event:
            model = model.to_integer()
        else:
            assert model.cfg.attn == "transformer" and model.ctx == 512
            model.cfg = replace(model.cfg, attn="transformer_local", local_window=model.ctx)
            for block in model.blocks:
                block.attn.window = model.ctx
        self.model = model

    def step(self, token, state):
        counts, state = self.model.step(token, state)
        resolution = self.model.cfg.resolution if self.event else 1
        return scoring_logits(counts, self.event, resolution), state


def main():
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    checkpoints = {"SNN": STUDY / "wide_s7907.pt", "Transformer": STUDY / "transformer_s7907.pt"}
    prompts = [
        "The first chapter describes ",
        "The city was founded in ",
        "In mathematics, a prime number is ",
        "The Pacific Ocean is the largest of Earth's ocean basins. It extends from ",
    ]
    temperature, seed, count = 0.8, 7907, 144
    decoders = {name: Decoder(path) for name, path in checkpoints.items()}
    result = {
        "selection_metric": "full enwik8 test bits per byte",
        "device": "cpu", "temperature": temperature,
        "new_bytes": count, "transformer_context_bytes": 512,
        "models": {},
        "samples": [],
    }
    for name, path in checkpoints.items():
        with path.open("rb") as source:
            digest = hashlib.file_digest(source, "sha256").hexdigest()
        result["models"][name] = {
            "checkpoint": str(path.relative_to(ROOT)), "sha256": digest, "run_name": decoders[name].run_name,
        }
    for index, prompt in enumerate(prompts):
        prefix = prompt.encode("utf-8")
        prompt_seed = seed + index
        row = {"prompt": prompt, "seed": prompt_seed, "continuations": {}}
        for name, decoder in decoders.items():
            raw = generate(decoder, prefix, count, temperature, prompt_seed, torch.device("cpu"))
            assert raw.startswith(prefix) and len(raw) == len(prefix) + count
            suffix = raw[len(prefix):]
            row["continuations"][name] = {"text": display_text(suffix), "raw_hex": suffix.hex()}
        result["samples"].append(row)
    destination = STUDY / "results/best_enwik8_generation_samples.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    )
    for sample in result["samples"]:
        print(repr(sample["prompt"]))
        for name, completion in sample["continuations"].items():
            print(name, repr(completion["text"]))


if __name__ == "__main__":
    main()
