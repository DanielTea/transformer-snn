"""Where a stack's pulses go and how far its cells' gates stand open, on the first bytes of a split as streams.

Two measures of a trained stack, taken in one pass:

* The pulses a byte of every population, as the trainer's pulse budget counts them: the spikes of every firing
  population and what the model counts itself (in a race read, the racing cells' spikes and the read pulses that
  they pass on). A dendrite's steps stay inside its cell and count none. A gated stack whose gate dendrites fire
  spends part of its budget on those steps, and the price then takes that part from the reads and the output.
* For the cells of every level: the share of cell-bytes at which the body fires, its mean count and its spikes a
  byte. For gated cells also the share at which the gate is open, the share at which it is wide open (it lets in all
  of the value dendrite's charge or more), the mean share of the charge that an open gate lets in and how the open
  gates spread over that share by quarters, the gate's steps a byte whether or not they count as pulses, the share
  at which the body's count stands at the most it can fire, and the pool's mean quanta of threshold. A gate that
  lets in nearly all whenever it is open is a switch; a gate whose open states spread over the quarters is graded.

Streams run as eval_stream.py runs them, and the lead bytes of every stream are left out.

usage: gate_state.py --checkpoint run/model.pt.averaged [more ...] --enwik8 data/enwik8 \\
    --split validation --bytes 65536 --json gate_state.json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import _atomic_output, resolve_device  # noqa: E402
from eval_stream import coarsen_output, stream_rows  # noqa: E402
from snnformer.event_lm import EventLM, _GatedFire, _IntegrateFire, _StateRead, emitted_pulses  # noqa: E402
from train_event_lm import load_checkpoint  # noqa: E402
from train_lm import load_splits  # noqa: E402

_NAMES = (
    (r"^state_reads\.\d+\.fire$", "reads"),
    (r"^state_reads\.\d+\.emit$", "stream after reads"),
    (r"^granule_layers\.\d+\.fire$", "cells"),
    (r"^granule_layers\.\d+\.open$", "gate dendrites"),
    (r"^granule_layers\.\d+\.emit$", "stream after cells"),
    (r"(\.|^output_)pool$", "pools"),
    (r"^output_emit$", "output"),
    (r"^input_emit$", "input"),
    (r"^forecast_emit$", "forecast"),
)


_QUARTERS = torch.tensor([0.25, 0.5, 0.75], dtype=torch.float64)


def population_of(name):
    """What a firing population is, from its module name. An unknown one keeps its name with its level folded."""
    for pattern, label in _NAMES:
        if re.search(pattern, name):
            return label
    return re.sub(r"\.\d+\.", ".N.", name)


def _level(name):
    return int(re.search(r"\.(\d+)\.", name).group(1))


def gate_state(model, inputs, piece, device, skip=0):
    """The pulses a byte by population and the state of every level's cells, over the bytes after ``skip``."""
    pulses = defaultdict(float)
    cells = defaultdict(lambda: defaultdict(float))
    # The spikes of a stream are the same however it is cut, so one cut falls at the end of the lead: every piece is
    # then left out or counted as a whole, also in what the model counts itself.
    counting, own, marks, handles = [False], [], {}, []

    def per_byte(tensor, like):
        return tensor.to(torch.int32).expand_as(like)

    for name, module in model.named_modules():
        if isinstance(module, _StateRead):
            # A read counts for the budget itself what no firing population sends: in a race the spikes of its racing
            # cells and the read pulses that they pass on.
            def before(read, _args):
                marks[read] = len(own)

            def after(read, _args, _output):
                mark = marks.pop(read)
                added = [float(value) for value in own[mark:]]
                del own[mark:]
                if read.mode == "race" and len(added) == 2:
                    pulses["racing cells"] += added[0]
                    pulses["read pulses"] += added[1]
                elif added:
                    pulses["within reads"] += sum(added)

            handles += [module.register_forward_pre_hook(before), module.register_forward_hook(after)]
        if not isinstance(module, _IntegrateFire):
            continue

        def count(population, _args, fired, name=name):
            if counting[0]:
                fired = fired[0] if isinstance(fired, tuple) else fired
                pulses[population_of(name)] += float(emitted_pulses(population, fired.detach()))

        handles.append(module.register_forward_hook(count))
        if not (name.startswith("granule_layers.") and name.endswith(".fire")):
            continue

        def body(fire, args, kwargs, fired, name=name):
            if not counting[0]:
                return
            fired = (fired[0] if isinstance(fired, tuple) else fired).detach()
            tally = cells[_level(name)]
            tally["entries"] += fired.numel()
            tally["bytes"] += fired.shape[0] * fired.shape[1]
            tally["firing"] += float((fired != 0).sum())
            tally["count"] += float(fired.abs().sum())
            if not isinstance(fire, _GatedFire):
                return
            units, opened = per_byte(args[1].detach(), fired), kwargs["opened"].detach().to(torch.int32)
            # A lifted body counts up to its cap at any threshold; a bounded one up to what its membrane holds.
            most = torch.full_like(units, int(fire.capped)) if fire.lifted else \
                ((fire.cap * fire.quanta_per_threshold) // units).clamp(max=fire.capped)
            share = opened.double() / torch.as_tensor(fire.parts(units, opened)).double()
            is_open = opened > 0
            tally["gate_open"] += float(is_open.sum())
            tally["gate_steps"] += float(opened.sum())
            tally["gate_wide_open"] += float((share >= 1).sum())
            tally["share"] += float(share[is_open].sum())
            # Up to a quarter of the charge, up to a half, up to three quarters, and more.
            for quarter, held in enumerate(torch.bincount(torch.bucketize(share[is_open], _QUARTERS), minlength=4)):
                tally[f"quarter {quarter}"] += float(held)
            tally["body_at_most"] += float(((fired.abs() >= most) & (fired != 0)).sum())
            tally["quanta"] += float(units.sum())

        handles.append(module.register_forward_hook(body, with_kwargs=True))
    length, counted = inputs.shape[1], 0
    cuts = sorted({*range(0, length, piece), min(skip, length), length})
    try:
        state = None
        with torch.no_grad():
            for start, end in zip(cuts, cuts[1:]):
                counting[0] = start >= skip
                _, state = model.stream(inputs[:, start:end].to(device), state,
                                        memory_pulses=own if counting[0] else None)
                if counting[0]:
                    counted += inputs.shape[0] * (end - start)
                    # What the model counts outside its reads: the pacemakers that all reads hear, and the like.
                    if own:
                        pulses["other counts of the model"] += sum(float(value) for value in own)
                        own.clear()
    finally:
        for handle in handles:
            handle.remove()
    targets = max(counted, 1)
    by_population = {label: total / targets for label, total in sorted(pulses.items(), key=lambda item: -item[1])}
    by_level = {}
    for level, tally in sorted(cells.items()):
        entries, heard = max(tally["entries"], 1), max(tally["bytes"], 1)
        row = {"firing": tally["firing"] / entries, "mean_count": tally["count"] / entries,
               "spikes_a_byte": tally["count"] / heard}
        if "gate_open" in tally:
            row |= {"gate_open": tally["gate_open"] / entries, "gate_wide_open": tally["gate_wide_open"] / entries,
                    "share_when_open": tally["share"] / tally["gate_open"] if tally["gate_open"] else 0.0,
                    "open_gates_by_quarter_share": [tally[f"quarter {quarter}"] / max(tally["gate_open"], 1)
                                                    for quarter in range(4)],
                    "gate_steps_a_byte": tally["gate_steps"] / heard,
                    "body_at_most": tally["body_at_most"] / entries, "mean_quanta": tally["quanta"] / entries}
        by_level[str(level)] = row
    return {"pulses_a_byte": sum(by_population.values()), "pulses_by_population": by_population,
            "cells_by_level": by_level}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, required=True, nargs="+")
    parser.add_argument("--enwik8", type=Path, required=True)
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--bytes", type=int, default=65536, help="the first bytes of the split")
    parser.add_argument("--streams", type=int, default=16)
    parser.add_argument("--lead", type=int, default=1024, help="unscored bytes that a part after the first hears")
    parser.add_argument("--piece", type=int, default=512)
    parser.add_argument("--output-step", type=int, metavar="STEPS",
                        help="every spike of the output neurons stands for this many fine steps of evidence, a "
                             "power of two no finer than the stack's own (as eval_stream.py takes it)")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--json", type=Path, required=True)
    args = parser.parse_args()
    device = resolve_device(args.device)
    _, validation, test = load_splits(args.enwik8)
    data = torch.from_numpy(np.array(test if args.split == "test" else validation))[:args.bytes]
    inputs, _, _ = stream_rows(data, args.streams, args.lead)
    record = {"split": args.split, "first_bytes_of_split": args.bytes, "streams": args.streams, "lead": args.lead,
              "runs": {}}
    for path in args.checkpoint:
        model, checkpoint = load_checkpoint(path, device)
        if not isinstance(model, EventLM) or not model.cfg.stack:
            parser.error(f"{path}: the gates are measured on a stack of reads and cells")
        if args.output_step is not None:
            try:
                coarsen_output(model, args.output_step)
            except ValueError as error:
                parser.error(f"{path}: {error}")
        found = gate_state(model, inputs, args.piece, device, skip=args.lead)
        record["runs"][path.parent.name] = {"run_name": checkpoint.get("run_name"), "weights": path.name,
                                            "peak_rate": checkpoint["args"].get("lr"),
                                            "schedule_updates": checkpoint["args"].get("steps"),
                                            "lifted": bool(model.cfg.pool_lifts_ceiling),
                                            "output_step": model.output_step, **found}
        gated = [row for row in found["cells_by_level"].values() if "gate_open" in row]
        print(path.parent.name, f"{found['pulses_a_byte']:,.0f} pulses a byte;",
              " ".join(f"{label} {value:,.0f}" for label, value in list(found["pulses_by_population"].items())[:5]),
              f"; open gates let in {min(r['share_when_open'] for r in gated):.2f} to "
              f"{max(r['share_when_open'] for r in gated):.2f}" if gated else "", flush=True)
    with _atomic_output(args.json, "w", True) as handle:
        json.dump(record, handle, indent=1)


if __name__ == "__main__":
    main()
