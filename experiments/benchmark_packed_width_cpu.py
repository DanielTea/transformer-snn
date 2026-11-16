"""Benchmark the widened checkpoints with equal prompts and cached CPU decoding."""

import argparse
import hashlib
import json
import platform
import statistics
import subprocess
from dataclasses import replace
from pathlib import Path

import torch

from benchmark_decode import time_trial
from train_event_lm import load_checkpoint


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/enwik8"))
    parser.add_argument("--study-dir", type=Path, default=Path("checkpoints/packed-width-study"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    context, steps, repeats = 512, 128, 7
    offset = 95_000_000
    study = args.study_dir
    with args.data.open("rb") as corpus:
        corpus.seek(offset)
        sample = corpus.read(context + steps)
    data = torch.tensor(list(sample))[None, :]
    prompt = data[:, :context].unbind(dim=1)
    cpu = torch.device("cpu")
    snn, _ = load_checkpoint(study / "wide_s7907.pt", cpu)
    snn = snn.to_integer()
    transformer, _ = load_checkpoint(study / "transformer_s7907.pt", cpu)
    uncapped_prompt = transformer(data[:, :context])
    # Reuse the repository's KV-ring path at the checkpoint's training context.
    assert transformer.cfg.attn == "transformer"
    transformer.cfg = replace(transformer.cfg, attn="transformer_local", local_window=context)
    for block in transformer.blocks:
        assert block.attn.window is None
        block.attn.window = context
    models = {"snn_int32": snn, "transformer_fp32": transformer}

    parity = {}
    for name, model in models.items():
        state = None
        streamed = []
        for token in data.unbind(dim=1):
            logits, state = model.step(token, state)
            streamed.append(logits)
        streamed = torch.stack(streamed, dim=1)
        if model is snn:
            reference, _ = model.forward_chunk(data)
            torch.testing.assert_close(streamed, reference, atol=0, rtol=0)
        else:
            reference = model(data)
            torch.testing.assert_close(streamed, reference, atol=0.0001, rtol=0.0001)
            torch.testing.assert_close(streamed[:, :context], uncapped_prompt, atol=0.0001, rtol=0.0001)
        parity[name] = {"max_abs_stream_vs_batch": (streamed - reference).abs().max().item()}
    print("Verified streamed vs. batched outputs through position 640", flush=True)

    def sync():
        pass

    for model in models.values():
        time_trial(model, prompt, steps, sync, record=False)
    trials = {name: [] for name in models}
    for index in range(repeats):
        order = list(models) if index % 2 == 0 else list(reversed(models))
        for name in order:
            prefill, decode, state_prefill, state_decode = time_trial(
                models[name], prompt, steps, sync, record=True
            )
            trials[name].append({
                "sequential_prefill_ms": prefill, "decode_ms_per_byte": decode,
                "state_after_prefill_bytes": state_prefill, "state_after_decode_bytes": state_decode,
            })
        print(f"Finished trial {index + 1}/{repeats}", flush=True)

    result = {
        "hardware": (subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
                     if platform.system() == "Darwin" else platform.processor() or platform.machine()),
        "device": "cpu", "torch_version": str(torch.__version__),
        "threads": 2, "interop_threads": 1, "batch_size": 1, "prompt_bytes": context,
        "transformer_kv_window_bytes": context, "decode_steps": steps,
        "repeats": repeats, "warmups": 1, "order": "alternating",
        "prompt_file": str(args.data), "prompt_offset": offset,
        "prompt_sha256": hashlib.sha256(sample[:context]).hexdigest(),
        "parity": parity, "models": {},
    }
    for name, runs in trials.items():
        decode = statistics.median(run["decode_ms_per_byte"] for run in runs)
        result["models"][name] = {
            "median_decode_ms_per_byte": decode, "decode_bytes_per_second": 1000 / decode,
            "median_sequential_prefill_ms": statistics.median(run["sequential_prefill_ms"] for run in runs),
            "trials": runs,
        }
    destination = args.output or study / "results/decode_wide_matched_cpu.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({name: {k: v for k, v in row.items() if k != "trials"}
                      for name, row in result["models"].items()}, indent=2), flush=True)


if __name__ == "__main__":
    main()
