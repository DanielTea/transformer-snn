"""Verify spike parity and gradient flow after changing a trained EventLM."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import torch
from torch.nn import functional as F

from audit_spike_code import _IntegerExecution
from benchmark_decode import state_bytes
from common import _atomic_output, resolve_device
from eval_event_lm import load_checkpoint
from probe_event_state import added_expansion_rows
from snnformer.event_lm import EventLM, _IntegrateFire, _route_packets, _steer_packet
from train_event_lm import initialize_from_checkpoint, scoring_logits
from train_lm import load_splits, random_batch


@torch.no_grad()
def measure(model, tokens, device, *, profile_coincidence, prior_depth=None):
    native = model.to_integer()
    emissions, memory_emissions, selected_fractions, added_inhibition = [], [], [], []

    def count_emissions(_module, _inputs, counts):
        emissions.append(counts.abs().sum(dtype=torch.int64).item())

    def count_coincidence(_module, inputs, outputs):
        content, steering = inputs[:2]
        read = outputs[0]
        selected = _steer_packet(content, steering)[0].abs().sum(dtype=torch.int64).item()
        confirmed = _route_packets(read, content, "coincidence")[0].abs().sum(dtype=torch.int64).item()
        selected_fractions.append(confirmed / selected if selected else None)

    handles = [module.register_forward_hook(count_emissions)
               for module in native.modules() if isinstance(module, _IntegrateFire)]
    if profile_coincidence:
        handles.extend(block.trace.register_forward_hook(count_coincidence) for block in native.blocks)
    if prior_depth is not None:
        def count_added_inhibition(_module, _inputs, counts):
            added_inhibition.append(counts.sum(dtype=torch.int64).item())

        handles.extend(block.activity_inhibition.register_forward_hook(count_added_inhibition)
                       for block in native.blocks[prior_depth:] if block.activity_inhibition is not None)
    with _IntegerExecution():
        counts = native(tokens, memory_pulses=memory_emissions)
        for handle in handles:
            handle.remove()

        chunks, chunk_state = [], None
        for chunk in tokens.split(173, dim=1):
            output, chunk_state = native.forward_chunk(chunk, chunk_state)
            chunks.append(output)
        torch.testing.assert_close(torch.cat(chunks, dim=1), counts, rtol=0, atol=0)

        steps, state = [], None
        for byte in tokens[:, :513].unbind(1):
            output, state = native.step(byte, state)
            steps.append(output)
        torch.testing.assert_close(torch.stack(steps, dim=1), counts[:, :513], rtol=0, atol=0)

    device_counts = model.to(device)(tokens.to(device)).cpu()
    torch.testing.assert_close(device_counts, counts, rtol=0, atol=0, check_dtype=False)
    metrics = {
        "parameters": sum(weight.numel() for weight in model.parameters()),
        "persistent_tensor_bytes": state_bytes(state),
        "total_pulses": sum(emissions) + sum(value.item() for value in memory_emissions),
    }
    if profile_coincidence:
        metrics["fraction_of_two_way_matches_also_confirmed"] = selected_fractions
    if prior_depth is not None:
        metrics["added_inhibition_pulses_by_block"] = added_inhibition
    return counts, metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", required=True, type=Path)
    parser.add_argument("--enwik8", required=True, type=Path)
    parser.add_argument("--device", default="cpu", choices=("cpu", "mps", "cuda"))
    growth = parser.add_mutually_exclusive_group()
    growth.add_argument("--dim", type=int, help="grow the residual stream (default width 768)")
    growth.add_argument("--hidden", type=int, help="grow content and steering within each delay group")
    growth.add_argument("--depth", type=int, help="append silent residual correction blocks to this depth")
    growth.add_argument("--trace-write-confirm", action="store_true", help="add silent write confirmation neurons")
    growth.add_argument("--redistribute-inhibition", action="store_true", help="add silent inhibitory transfers")
    growth.add_argument("--tag-read-competition", action="store_true", help="add silent tag read inhibition")
    growth.add_argument("--late-steering", action="store_true", help="add silent feedback steering")
    growth.add_argument("--refine-router-charge", type=int, choices=(8,), help="refine charge at a fixed width")
    parser.add_argument("--router-charge-bits", type=int, choices=(6, 8))
    parser.add_argument("--trace-coincidence", action="store_true")
    parser.add_argument("--json", required=True, type=Path)
    args = parser.parse_args()

    torch.set_num_threads(4)
    torch.manual_seed(711)
    device = resolve_device(args.device)
    train, validation, _ = load_splits(args.enwik8)
    sample = validation[:4096]
    tokens = torch.tensor(sample, dtype=torch.long)[None]
    parent, checkpoint = load_checkpoint(args.parent, torch.device("cpu"))
    if not isinstance(parent, EventLM) or not parent.cfg.local_trace:
        parser.error("the parent must be a held-charge EventLM")
    prior = parent.cfg
    circuit_options = ("trace_write_confirm", "redistribute_inhibition", "tag_read_competition", "late_steering")
    added_circuit = next((option for option in circuit_options if getattr(args, option)), None)
    if ((args.depth is not None or args.hidden is not None or added_circuit)
            and (args.router_charge_bits is not None or args.trace_coincidence)):
        parser.error("charge precision and trace coincidence options apply to residual growth")
    if added_circuit:
        if getattr(prior, added_circuit):
            parser.error(f"the parent already uses {added_circuit.replace('_', ' ')}")
        cfg = replace(prior, **{added_circuit: True})
        grown = EventLM(cfg)
        initialize_from_checkpoint(grown, {}, args.parent, freeze_router=False)
    elif args.refine_router_charge is not None:
        if args.router_charge_bits is not None or args.trace_coincidence:
            parser.error("use --refine-router-charge on its own for fixed-width refinement")
        cfg = replace(prior, router_charge_bits=args.refine_router_charge)
        grown = EventLM(cfg)
        initialize_from_checkpoint(grown, {}, args.parent, freeze_router=False, preserve_router_charge=True)
    elif args.depth is not None:
        start = prior.residual_correction_start if prior.residual_correction_start is not None else prior.depth
        cfg = replace(prior, depth=args.depth, residual_correction_start=start)
        grown = EventLM(cfg)
        initialize_from_checkpoint(grown, {}, args.parent, freeze_router=False, grow_depth=True)
    elif args.hidden is None:
        cfg = replace(prior, dim=768 if args.dim is None else args.dim,
                      router_charge_bits=prior.router_charge_bits if args.router_charge_bits is None
                      else args.router_charge_bits,
                      compact_axon_state=True, inhibition_by_expansion=True, trace_write_credit=True,
                      trace_coincidence_start=prior.dim if args.trace_coincidence else prior.trace_coincidence_start)
        grown = EventLM(cfg)
        grown.initialize_wider_residual(prior, checkpoint["state_dict"])
    else:
        cfg = replace(prior, hidden=args.hidden)
        grown = EventLM(cfg)
        grown.initialize_wider_expansion(prior, checkpoint["state_dict"])
    parent_counts, parent_metrics = measure(parent, tokens, device, profile_coincidence=False)
    grown_counts, grown_metrics = measure(grown.eval(), tokens, device, profile_coincidence=args.trace_coincidence,
                                          prior_depth=prior.depth if args.depth is not None else None)
    torch.testing.assert_close(parent_counts, grown_counts, rtol=0, atol=0)
    pulse_delta = grown_metrics["total_pulses"] - parent_metrics["total_pulses"]
    if pulse_delta != sum(grown_metrics.get("added_inhibition_pulses_by_block", [])):
        raise AssertionError("an added routing branch emitted at initialization")

    generator = torch.Generator().manual_seed(711)
    inputs, labels = random_batch(train, 1, 512, generator)
    counts = grown.train()(inputs.to(device))
    logits = scoring_logits(counts, True, cfg.resolution)
    F.cross_entropy(logits.flatten(0, 1), labels.to(device).flatten()).backward()
    if args.trace_write_confirm:
        gradient_field = "active_write_confirmation_gradients_on_512_train_targets"
        gradients = [block.trace.write_confirm.gain.grad.count_nonzero().item() for block in grown.blocks]
    elif args.redistribute_inhibition:
        gradient_field = "active_inhibition_gradients_on_512_train_targets"
        gradients = [block.inhibition_redistribution.synapse.weight.grad.count_nonzero().item()
                     for block in grown.blocks]
    elif args.tag_read_competition:
        gradient_field = "active_read_competition_heads_on_512_train_targets"
        gradients = grown.prediction_memory.read_inhibition.gain.grad.ne(0).int().tolist()
    elif args.late_steering:
        gradient_field = "active_late_steering_gradients_on_512_train_targets"
        gradients = [block.late_steering.gain.grad.count_nonzero().item() for block in grown.blocks]
    elif args.refine_router_charge is not None:
        gradient_field = "active_router_gradients_on_512_train_targets"
        gradients = [block.expand.weight.grad.count_nonzero().item() for block in grown.blocks]
    elif args.depth is not None:
        gradient_field = "active_added_content_gradients_on_512_train_targets"
        gradients = [block.expand.weight.grad[:cfg.hidden].count_nonzero().item() for block in grown.blocks[prior.depth:]]
        grown_metrics["active_added_steering_gradients_on_512_train_targets"] = [
            block.expand.weight.grad[cfg.hidden:].count_nonzero().item() for block in grown.blocks[prior.depth:]
        ]
    elif args.hidden is None:
        gradient_field = "active_new_projection_gradients_on_512_train_targets"
        gradients = [block.project.weight.grad[prior.dim:].count_nonzero().item() for block in grown.blocks]
    else:
        gradient_field = "active_new_expansion_gradients_on_512_train_targets"
        gradients = [
            sum(rows.count_nonzero().item() for rows in added_expansion_rows(block, prior.hidden, block.expand.weight.grad))
            for block in grown.blocks
        ]
    if not all(gradients):
        raise AssertionError("the added synapses received no gradient in a population")

    with args.parent.open("rb") as source:
        parent_sha256 = hashlib.file_digest(source, "sha256").hexdigest()
    record = {
        "parent": str(args.parent),
        "parent_sha256": parent_sha256,
        "validation_sample_sha256": hashlib.sha256(sample).hexdigest(),
        "validation_bytes": sample.size,
        "device": str(device),
        "native_step_chunk_and_device_counts_equal": True,
        "parent_count_and_emission_parity": pulse_delta == 0,
        **({"parent_counts_equal": True, "additional_pulses": pulse_delta} if args.depth is not None else {}),
        "models": {"parent": parent_metrics, "candidate" if added_circuit else "grown": grown_metrics},
        gradient_field: gradients,
    }
    with _atomic_output(args.json, "w", overwrite=False) as stream:
        json.dump(record, stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
