"""Measure single-stream greedy decoding for byte model checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import random
import statistics
import subprocess
import sys
import time
from dataclasses import fields, is_dataclass
from pathlib import Path


def state_bytes(state):
    import torch

    # Count backing storage once if two state fields share it.
    storages = {}

    def visit(value):
        if isinstance(value, torch.Tensor):
            storage = value.untyped_storage()
            storages[(value.device, storage.data_ptr())] = storage.nbytes()
        elif is_dataclass(value):
            for field in fields(value):
                visit(getattr(value, field.name))
        elif isinstance(value, (tuple, list)):
            for item in value:
                visit(item)

    visit(state)
    return sum(storages.values())


def peak_rss_bytes():
    if sys.platform not in ("darwin", "linux"):
        return None
    import resource

    maximum = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return maximum if sys.platform == "darwin" else 1024 * maximum


def time_trial(model, prompt, steps, sync, *, record):
    sync()
    start = time.perf_counter()
    state = None
    for token in prompt:
        logits, state = model.step(token, state)
    sync()
    prefill_seconds = time.perf_counter() - start
    prompt_state_bytes = state_bytes(state) if record else None

    start = time.perf_counter()
    for _ in range(steps):
        logits, state = model.step(logits.argmax(dim=-1), state)
    sync()
    decode_seconds = time.perf_counter() - start
    final_state_bytes = state_bytes(state) if record else None
    return prefill_seconds * 1000, decode_seconds * 1000 / steps, prompt_state_bytes, final_state_bytes


def worker(options):
    import torch

    from common import resolve_device
    from snnformer import count_params

    torch.set_num_threads(options["threads"])
    torch.set_num_interop_threads(1)
    device = resolve_device(options["device"])
    checkpoint = Path(options["checkpoint"])
    if options.get("integer_event"):
        from snnformer.event_lm import EventLM
        from train_event_lm import load_checkpoint

        if device.type != "cpu":
            raise ValueError("the integer event decoder requires CPU")
        model, metadata = load_checkpoint(checkpoint, device)
        if not isinstance(model, EventLM):
            raise ValueError(f"{checkpoint}: expected a spiking byte model from train_event_lm.py")
        run_name = metadata["run_name"]
        model = model.to_integer()
        details = {"variant": "integer_event", "local_history_bytes": model.local_history,
                   "spike_resolution": model.cfg.resolution}
    else:
        from probe_lm_context import load_model

        model, run_name = load_model(checkpoint, device)
        if model.cfg.attn not in ("transformer_local", "transformer_successor"):
            raise ValueError(f"{checkpoint}: streaming requires a local Transformer or successor model")
        model.float()
        successor = model.cfg.attn == "transformer_successor"
        details = {
            "variant": ("spike" if model.cfg.successor_spikes else "continuous") if successor else "local",
            "local_window": model.cfg.local_window,
            "successor_slots_per_order": model.cfg.successor_slots if successor else None,
            "successor_orders": list(model.cfg.successor_orders) if successor else None,
        }

    def synchronize():
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elif device.type == "mps":
            torch.mps.synchronize()

    prompt = torch.tensor(list(bytes.fromhex(options["prompt"])), dtype=torch.long, device=device)
    rows = []
    with torch.inference_mode():
        for context in options["contexts"]:
            tokens = prompt[:context, None].unbind()
            steps = options["steps"]
            for _ in range(options["warmups"]):
                time_trial(model, tokens, steps, synchronize, record=False)
            trials = [
                time_trial(model, tokens, steps, synchronize, record=True)
                for _ in range(options["repeats"])
            ]
            prefill, decode, after_prompt, after_decode = zip(*trials)
            rows.append({
                "context_bytes": context,
                "sequential_prompt_ms": list(prefill),
                "median_sequential_prompt_ms": statistics.median(prefill),
                "decode_ms_per_byte": list(decode),
                "median_decode_ms_per_byte": statistics.median(decode),
                "state_bytes_after_prompt": max(after_prompt),
                "state_bytes_after_decode": max(after_decode),
            })
    return {
        "checkpoint": str(checkpoint),
        "run_name": run_name,
        "parameters": count_params(model),
        **details,
        "device": str(device),
        "torch_version": str(torch.__version__),
        "process_peak_rss_bytes": peak_rss_bytes(),
        "contexts": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, nargs="+", required=True)
    parser.add_argument("--integer-event", action="store_true",
                        help="benchmark the native integer export of train_event_lm.py checkpoints on CPU")
    parser.add_argument("--contexts", type=int, nargs="+", default=[32, 512, 1024])
    parser.add_argument("--steps", type=int, default=128, help="greedy decode steps per repeat")
    parser.add_argument("--warmups", type=int, default=2, help="unmeasured repeats at each context")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--device", choices=("cpu", "cuda", "mps", "auto"), default="cpu")
    parser.add_argument("--prompt-file", type=Path, help="raw bytes, otherwise use a seeded random prompt")
    parser.add_argument("--prompt-offset", type=int, default=0)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    if min(*args.contexts, args.steps, args.repeats, args.threads) < 1 or args.warmups < 0:
        parser.error("contexts, steps, repeats and threads must be positive and warmups must be nonnegative")
    if args.prompt_offset < 0 or (args.prompt_offset and args.prompt_file is None):
        parser.error("a nonnegative prompt offset applies only with --prompt-file")
    contexts = sorted(set(args.contexts))
    length = max(contexts)
    prompt_seed = None
    if args.prompt_file is None:
        prompt_seed = 2026
        rng = random.Random(prompt_seed)
        prompt = bytes(rng.randrange(256) for _ in range(length))
    else:
        try:
            with args.prompt_file.expanduser().open("rb") as source:
                source.seek(args.prompt_offset)
                prompt = source.read(length)
        except OSError as exc:
            parser.error(str(exc))
        if len(prompt) < length:
            parser.error(f"prompt file has fewer than {length} bytes after the requested offset")

    options = {
        "contexts": contexts,
        "steps": args.steps,
        "warmups": args.warmups,
        "repeats": args.repeats,
        "threads": args.threads,
        "device": args.device,
        "integer_event": args.integer_event,
        "prompt": prompt.hex(),
    }
    results = []
    for path in args.checkpoint:
        # Use one fresh process per model so RSS cannot include other checkpoints.
        child = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--worker"],
            input=json.dumps(options | {"checkpoint": str(path.expanduser())}),
            text=True,
            stdout=subprocess.PIPE,
        )
        if child.returncode:
            parser.exit(child.returncode, f"benchmark failed for {path}\n")
        result = json.loads(child.stdout)
        results.append(result)
        for row in result["contexts"]:
            print(f'{result["run_name"]} | context {row["context_bytes"]:5d} B | '
                  f'{row["median_decode_ms_per_byte"]:.3f} ms/byte | '
                  f'{row["state_bytes_after_decode"]:,} tensor state bytes', flush=True)

    output = {
        "system": platform.platform(),
        "dtype": "int32" if args.integer_event else "float32",
        "threads": args.threads,
        "interop_threads": 1,
        "batch_size": 1,
        "decode_steps": args.steps,
        "warmups": args.warmups,
        "repeats": args.repeats,
        "prompt_file": str(args.prompt_file) if args.prompt_file is not None else None,
        "prompt_offset": args.prompt_offset,
        "prompt_seed": prompt_seed,
        "prompt_sha256": hashlib.sha256(prompt).hexdigest(),
        "timing": "wall clock with device synchronization at phase boundaries, greedy argmax included in decode",
        "rss_scope": "peak CPU resident bytes per checkpoint process, including imports, weights and all contexts",
        "state_scope": "allocated tensor storage reachable from decoder state, excluding weights and temporary tensors",
        "models": results,
    }
    if args.json is not None:
        destination = args.json.expanduser()
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(output, indent=2) + "\n")


if __name__ == "__main__":
    if sys.argv[1:] == ["--worker"]:
        print(json.dumps(worker(json.load(sys.stdin))))
    else:
        main()
