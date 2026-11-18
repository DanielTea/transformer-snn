"""Can the state reads learn a sharp lookup? A small decoder on sequences whose second part repeats the first.

usage: copy_task.py ARM STEPS [key=value ...]    ARM: plain | soft | race | rate
"""
import sys
import time

import torch
from torch.nn import functional as F

sys.path.insert(0, ".")
from snnformer.event_lm import EventLM, EventLMConfig  # noqa: E402

torch.set_num_threads(4)
arm, steps = sys.argv[1], int(sys.argv[2])
extra = dict(item.split("=") for item in sys.argv[3:])
lr = float(extra.pop("lr", 3e-4))
symbols, length, batch = int(extra.pop("symbols", 64)), 96, 16
base = dict(dim=192, hidden=64, depth=6, route="steered_correction_code", carry=False, memory_heads=4,
            memory_slots=14, resolution=16, memory_width=32, forecast_key_channels=2, match_forecast_key=True,
            learn_context_keys=True, compact_delays=True, compact_axon_state=True, activity_inhibition=True,
            inhibition_by_expansion=True, local_trace=True, trace_write_credit=True, router_charge_bits=8,
            residual_correction_start=3, compact_trace_state=True, delay_group_widths=(20, 20, 12, 12),
            trace_arrival="selected", trace_arrival_groups=2, clock_start=4, clock_threshold=8)
reads = dict(read_after=(1, 3), read_width=128, read_heads=4, read_window=96)
changes = {"plain": {}, "soft": dict(reads), "race": dict(reads, read_mode="race", read_resolution=4),
           "rate": dict(reads, read_mode="race", read_budget=256, read_resolution=4)}[arm]
for name, value in extra.items():
    changes[name] = type(getattr(EventLMConfig, name))(value) if not isinstance(getattr(EventLMConfig, name), bool) \
        else value == "True"
torch.manual_seed(0)
model = EventLM(EventLMConfig(**dict(base, **changes))).train()
optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.99), weight_decay=0.05)
generator = torch.Generator().manual_seed(1)


def sample(count):
    """Random symbols, repeated after a prefix of random length, and the mask of the repeated targets."""
    tokens = torch.randint(0, symbols, (count, length + 1), generator=generator)
    period = torch.randint(24, 49, (count,), generator=generator)
    position = torch.arange(length + 1)
    source = torch.where(position[None] >= period[:, None], position[None] - period[:, None], position[None])
    # Copy from the start so that later repeats copy what was itself copied.
    for index in range(length + 1):
        tokens[:, index] = torch.where(index >= period, tokens.gather(1, source[:, index:index + 1])[:, 0],
                                       tokens[:, index])
    repeated = position[None, 1:] >= period[:, None] + 2
    return tokens, repeated


start = time.perf_counter()
for step in range(1, steps + 1):
    rate = lr * min(1.0, step / 200)
    for group in optimizer.param_groups:
        group["lr"] = rate
    tokens, repeated = sample(batch)
    logits = model(tokens[:, :-1]) / 16
    loss = F.cross_entropy(logits.flatten(0, 1), tokens[:, 1:].flatten(), reduction="none").view(batch, length)
    (loss.mean()).backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    if step % 250 == 0 or step == steps:
        with torch.no_grad():
            model.eval()
            tokens, repeated = sample(64)
            logits = model(tokens[:, :-1]) / 16
            loss = F.cross_entropy(logits.flatten(0, 1), tokens[:, 1:].flatten(), reduction="none").view(64, length)
            accuracy = (logits.argmax(-1) == tokens[:, 1:])[repeated].float().mean()
            line = (f"{arm} step {step}: repeated part {loss[repeated].mean() / 0.6931:.3f} bits accuracy {accuracy:.3f} | "
                    f"first part {loss[~repeated].mean() / 0.6931:.3f} bits | {time.perf_counter() - start:.0f}s")
            if model.state_reads is not None:
                captured = {}
                hooks = [read.register_forward_hook(lambda m, i, o, n=name: captured.__setitem__(n, i[0]))
                         for name, read in model.state_reads.items()]
                model(tokens[:, :-1])
                for hook in hooks:
                    hook.remove()
                for name, spikes in captured.items():
                    shares = model.state_reads[name].shares(spikes)[0][:, :, 56:]
                    line += f" | read {name} largest share {[round(v, 2) for v in shares.amax(-1).mean((0, 2)).tolist()]}"
            print(line, flush=True)
            model.train()
