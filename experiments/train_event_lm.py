"""Train and score integer-event and Transformer decoders on canonical enwik8."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import asdict, replace
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from byte_training_mixture import TrainingMixture
from common import (
    VARIANTS,
    Timer,
    _atomic_output,
    checkpoint_path,
    cosine_lr,
    make_lm,
    make_optimizer,
    resolve_device,
    result_exists,
    save_result,
    setup,
)
from snnformer import SNNConfig, count_params
from snnformer.event_lm import (
    EventLM, EventLMConfig, _CLOCK_SOURCES, _TRACE_ARRIVAL_MODES, _TRACE_READ_DRIVE_SOURCES,
)
from train_lm import DATA, ENWIK8_SHA1, load_splits, random_batch


EVENT_ROUTES = {
    "event": "coincidence",
    "event_tonic": "tonic",
    "event_steered": "steered",
    "event_continuous": "steered",
    "event_correction": "steered_correction",
    "event_correction_residual": "steered_correction_residual",
    "event_correction_binding": "steered_correction_binding",
    "event_correction_observed": "steered_correction_observed",
    "event_correction_error": "steered_correction_error",
    "event_correction_code": "steered_correction_code",
    "event_steered_local": "steered_local",
    "event_convergent": "convergent",
    "event_static": "steered_static",
    "event_feedback": "steered_feedback",
    "event_margin": "steered_margin",
    "event_timed": "steered_timed",
    "event_phase_swap": "steered_phase_swap",
    "event_phase_delay": "steered_phase_delay",
    "event_balanced": "steered_balanced",
}


def scoring_logits(output: torch.Tensor, event: bool, resolution: int = 1) -> torch.Tensor:
    """Score the signed spike evidence outside the event decoder."""
    return output.float() * (math.log(2) / (8 * resolution)) if event else output


def load_checkpoint(path: Path, device: torch.device):
    """Load a training checkpoint with its original floating-point parameters."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("task") != "event_lm":
        raise ValueError(f"{path}: expected a train_event_lm.py checkpoint")
    if checkpoint.get("model_type") == "EventLM":
        model = EventLM(EventLMConfig(**checkpoint["cfg"]))
    else:
        cfg = SNNConfig(**checkpoint["cfg"])
        options = checkpoint["model_kwargs"]
        if cfg.attn not in ("transformer", "transformer_local") or options.get("vocab", 256) != 256:
            raise ValueError(f"{path}: expected a byte-level full or local Transformer")
        model = make_lm(cfg, **options)
    model.load_state_dict(checkpoint["state_dict"])
    return model.to(device).eval(), checkpoint


def training_exposure(checkpoint):
    options = checkpoint["args"]
    previous = checkpoint.get("initialization") or {}
    return options["steps"] * options["bs"] * options["ctx"] + previous.get("training_targets", 0)


def guided_language_loss(logits, targets, teacher_logits, weight):
    student = logits.log_softmax(dim=-1)
    target = targets.unsqueeze(-1)
    hard = -student.gather(-1, target).squeeze(-1)
    with torch.no_grad():
        teacher = teacher_logits.log_softmax(dim=-1)
        use_teacher = teacher.gather(-1, target).squeeze(-1) > -hard
    divergence = (teacher.exp() * (teacher - student)).sum(-1)
    loss = torch.where(use_teacher, (1 - weight) * hard + weight * divergence, hard)
    return loss.mean(), hard.mean(), use_teacher.float().mean()


# A continued run may change these without changing what it trains.
_RESUME_FREE = frozenset(("stop_at", "force", "threads", "device"))


def _on_cpu(value):
    """The tensors of a nested state on the CPU, so a saved run can continue on another device."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: _on_cpu(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_on_cpu(item) for item in value)
    return value


@contextmanager
def averaged_weights(model, average):
    """Run the model on the running average of its weights and give it its own weights back afterwards."""
    own = [parameter.detach().clone() for parameter in model.parameters()]
    try:
        with torch.no_grad():
            for parameter, mean in zip(model.parameters(), average, strict=True):
                parameter.copy_(mean)
        yield
    finally:
        with torch.no_grad():
            for parameter, saved in zip(model.parameters(), own, strict=True):
                parameter.copy_(saved)


@torch.inference_mode()
def evaluate(model, data, *, context, batches, batch_size, device, event, pulse_counts=None):
    was_training = model.training
    model.eval()
    window_count = batches * batch_size
    stride = (len(data) - context - 1) // window_count
    offsets = np.arange(context + 1)
    nll = 0.0
    correct = 0
    pulses = 0
    resolution = getattr(getattr(model, "cfg", None), "resolution", 1)
    try:
        for start in range(0, window_count, batch_size):
            positions = np.arange(start, start + batch_size)[:, None] * stride + offsets
            window = torch.from_numpy(data[positions].astype(np.int64)).to(device)
            targets = window[:, 1:]
            if pulse_counts is None:
                output = model(window[:, :-1])
            else:
                pulse_counts.clear()
                output = model(window[:, :-1], memory_pulses=pulse_counts)
                pulses += torch.stack(pulse_counts).sum().item()
            logits = scoring_logits(output, event, resolution)
            nll += F.cross_entropy(logits.flatten(0, 1), targets.flatten(), reduction="sum").item()
            correct += (output.argmax(dim=-1) == targets).sum().item()
    finally:
        model.train(was_training)
    count = window_count * context
    return {"bpb": nll / (count * math.log(2)), "argmax_accuracy": correct / count, "target_bytes": count,
            **({"pulses_per_byte": pulses / count} if pulse_counts is not None else {})}


def build_model(args):
    model_kwargs = dict(vocab=256, dim=args.dim, heads=args.heads, depth=args.depth, ctx=args.ctx)
    if args.variant in EVENT_ROUTES:
        groups = None if args.delay_group_widths is None else tuple(map(int, args.delay_group_widths.split(",")))
        synapses, readout = {}, None
        if getattr(args, "synapse_readout", None) is not None:
            # The synapses stay as fitted. Only the decoder that reads beside them is trained.
            donor = torch.load(args.synapse_readout.expanduser(), map_location="cpu", weights_only=True)
            synapses = {name: tuple(value) if isinstance(value, list) else value
                        for name, value in donor["cfg"].items() if name.startswith("synapse_")}
            if synapses.get("synapse_orders") is None:
                raise ValueError(f"{args.synapse_readout}: expected a decoder with fitted fast synapses")
            readout = donor["state_dict"]["fast_synapses.readout"]
        heads, head_readout, joint_readout = {}, None, None
        if getattr(args, "head_readout", None) is not None:
            # The heads stay as fitted, like the synapses.
            fitted = json.loads(args.head_readout.expanduser().read_text(encoding="utf-8"))
            setting = fitted["setting"]
            heads = dict(head_orders=tuple(setting["orders"]), head_cells=setting["cells"], head_gate=setting["gate"],
                         head_tape=setting["tape"], head_lookup=setting["lookup"], head_miss=setting["miss"],
                         head_relock=setting["relock"])
            head_readout = torch.tensor(fitted["head_strength_spikes"], dtype=torch.int32)
            if fitted.get("synapse_strength_spikes") is not None:
                if readout is None:
                    raise ValueError(f"{args.head_readout}: jointly fitted synapses need --synapse-readout")
                joint_readout = torch.tensor(fitted["synapse_strength_spikes"], dtype=torch.int32)
        cfg = EventLMConfig(dim=args.dim, hidden=args.hidden, depth=args.depth, route=EVENT_ROUTES[args.variant],
                            carry=not args.no_carry, continuous=args.variant == "event_continuous",
                            memory_heads=args.memory_heads, memory_slots=args.memory_slots,
                            resolution=args.spike_resolution, memory_width=getattr(args, "memory_width", None),
                            forecast_key_channels=args.forecast_key_channels, match_forecast_key=args.match_forecast_key,
                            tag_read_competition=args.tag_read_competition,
                            learn_context_keys=args.learn_context_keys, learn_packet_thresholds=args.learn_packet_thresholds,
                            output_reference=args.output_reference, compact_delays=args.compact_delays,
                            departure_coincidence=args.departure_coincidence, delay_window=args.delay_window,
                            delay_residual=args.delay_residual,
                            router_charge_bits=args.router_charge_bits, activity_inhibition=args.activity_inhibition,
                            inhibition_by_expansion=args.inhibition_by_expansion,
                            learn_inhibition_thresholds=args.learn_inhibition_thresholds,
                            inhibition_threshold_bits=args.inhibition_threshold_bits,
                            redistribute_inhibition=args.redistribute_inhibition,
                            local_trace=args.local_trace, trace_coincidence_start=args.trace_coincidence_start,
                            trace_write_credit=args.trace_write_credit, trace_write_confirm=args.trace_write_confirm,
                            trace_integration=args.trace_integration, trace_tap_read=args.trace_tap_read,
                            trace_read_drive=args.trace_read_drive,
                            trace_read_drive_bits=args.trace_read_drive_bits,
                            trace_read_drive_carry=args.trace_read_drive_carry,
                            trace_read_share_steering=args.trace_read_share_steering,
                            trace_arrival=args.trace_arrival, trace_arrival_groups=args.trace_arrival_groups,
                            steering_transfer=args.steering_transfer,
                            delay_group_widths=groups,
                            steering_gradient_at_zero=args.steering_gradient_at_zero,
                            late_steering=args.late_steering,
                            bank_write_credit=args.bank_write_credit,
                            trace_read_veto=args.trace_read_veto,
                            delay_filter=args.delay_filter,
                            opponent_steering=args.opponent_steering,
                            spread_coincidence=args.spread_coincidence,
                            residual_membrane=args.residual_membrane,
                            residual_correction_start=args.residual_correction_start,
                            compact_axon_state=args.compact_axon_state,
                            compact_trace_state=args.compact_trace_state,
                            tag_phase_address=args.tag_phase_address,
                            clock_start=args.clock_start, clock_threshold=args.clock_threshold,
                            clock_leak=args.clock_leak, clock_source=args.clock_source,
                            read_after=(None if getattr(args, "read_after", None) is None
                                        else tuple(map(int, args.read_after.split(",")))),
                            read_width=getattr(args, "read_width", EventLMConfig.read_width),
                            read_heads=getattr(args, "read_heads", EventLMConfig.read_heads),
                            read_window=getattr(args, "read_window", EventLMConfig.read_window),
                            read_mode=getattr(args, "read_mode", EventLMConfig.read_mode),
                            read_levels=getattr(args, "read_levels", EventLMConfig.read_levels),
                            read_resolution=getattr(args, "read_resolution", EventLMConfig.read_resolution),
                            read_signed=getattr(args, "read_signed", EventLMConfig.read_signed),
                            read_budget=getattr(args, "read_budget", EventLMConfig.read_budget),
                            read_address_rate=getattr(args, "read_address_rate", EventLMConfig.read_address_rate),
                            read_synapse_gain=getattr(args, "read_synapse_gain", EventLMConfig.read_synapse_gain),
                            read_beats=getattr(args, "read_beats", EventLMConfig.read_beats),
                            read_beat_start=getattr(args, "read_beat_start", EventLMConfig.read_beat_start),
                            read_clock=getattr(args, "read_clock", EventLMConfig.read_clock),
                            read_clock_rate=getattr(args, "read_clock_rate", EventLMConfig.read_clock_rate),
                            read_marks=getattr(args, "read_marks", EventLMConfig.read_marks),
                            granule_after=(None if getattr(args, "granule_after", None) is None
                                           else tuple(map(int, args.granule_after.split(",")))),
                            granule_clusters=getattr(args, "granule_clusters", EventLMConfig.granule_clusters),
                            granule_cells=getattr(args, "granule_cells", EventLMConfig.granule_cells),
                            granule_fan=getattr(args, "granule_fan", EventLMConfig.granule_fan),
                            granule_gain=getattr(args, "granule_gain", EventLMConfig.granule_gain),
                            granule_rest=getattr(args, "granule_rest", EventLMConfig.granule_rest),
                            granule_price=getattr(args, "granule_price", EventLMConfig.granule_price),
                            granule_bits=getattr(args, "granule_bits", EventLMConfig.granule_bits),
                            granule_budget=getattr(args, "granule_budget", EventLMConfig.granule_budget),
                            granule_delays=getattr(args, "granule_delays", EventLMConfig.granule_delays),
                            granule_relays=getattr(args, "granule_relays", EventLMConfig.granule_relays),
                            granule_traces=getattr(args, "granule_traces", EventLMConfig.granule_traces),
                            granule_beats=getattr(args, "granule_beats", EventLMConfig.granule_beats),
                            granule_clock=getattr(args, "granule_clock", EventLMConfig.granule_clock),
                            granule_hold=getattr(args, "granule_hold", EventLMConfig.granule_hold),
                            granule_clock_rate=getattr(args, "granule_clock_rate", EventLMConfig.granule_clock_rate),
                            **synapses, **heads)
        model = EventLM(cfg)
        with torch.no_grad():
            if readout is not None:
                model.fast_synapses.readout.copy_(readout)
            if joint_readout is not None:
                model.fast_synapses.readout[:, 1:] = joint_readout
            if head_readout is not None:
                model.read_heads.readout[:, 1:] = head_readout
        return model, cfg, {}
    cfg = replace(VARIANTS[args.variant], mlp_ratio=args.mlp_ratio, local_window=args.local_window)
    if args.local_lookback is not None:
        per_layer, extra = divmod(args.local_lookback - 1, args.depth)
        model_kwargs["local_windows"] = tuple(1 + per_layer + (layer < extra) for layer in range(args.depth))
    return make_lm(cfg, **model_kwargs), cfg, model_kwargs


_BANK_PARAMETERS = ("prediction_memory.", "recall_synapse.")
_OPTIONAL_ROUTER_PARAMETERS = (
    ".packet_thresholds.", ".delay_correction.", ".delay_filters.", ".opponent_steering.", ".trace.read_veto.",
    ".spread_coincidence.", ".trace.write_confirm.", ".trace.input.", ".trace.read_drive.", ".trace_taps.",
    ".inhibition_redistribution.", ".inhibition_thresholds.", ".late_steering.", ".steering_transfer.",
)


def initialize_router(model, source, source_cfg):
    """Load a trained local spike router, leaving any new bank at its initialization."""
    if source_cfg.route != "steered":
        raise ValueError("--initialize-from needs a matching spiking configuration or a plain router checkpoint")
    fields = ("dim", "hidden", "depth", "carry", "continuous", "resolution")
    if (any(getattr(source_cfg, name) != getattr(model.cfg, name) for name in fields)
            or source_cfg.group_widths != model.cfg.group_widths):
        raise ValueError("the checkpoint and model must use the same local router settings")
    if model.cfg.route != "steered" and model.prediction_memory is None:
        raise ValueError("router initialization supports the plain router and post-router banks")
    weights = source["state_dict"]
    local_names = {
        name for name in model.state_dict()
        if not name.startswith(_BANK_PARAMETERS)
        and (name in weights or all(part not in name for part in _OPTIONAL_ROUTER_PARAMETERS))
    }
    if set(weights) != local_names:
        raise ValueError("the checkpoint does not cover exactly the local router weights")
    model.load_state_dict(weights, strict=False)


def initialize_from_checkpoint(model, model_kwargs, path, *, freeze_router, grow_residual=False,
                               grow_feed_forward=False, grow_expansion=False, grow_depth=False, preserve_router_charge=False):
    source = torch.load(path, map_location="cpu", weights_only=True)
    if source.get("task") != "event_lm" or source.get("model_type") != type(model).__name__:
        raise ValueError("--initialize-from needs a checkpoint of the same decoder family")
    if preserve_router_charge and (not isinstance(model, EventLM) or grow_residual or grow_expansion or grow_depth):
        raise ValueError("preserving router charge applies to a spiking continuation at the same width")
    growth = {}
    if grow_residual or grow_expansion or (grow_depth and isinstance(model, EventLM)):
        if not isinstance(model, EventLM) or freeze_router:
            raise ValueError("growth needs a trainable spiking decoder")
        prior = EventLMConfig(**source["cfg"])
        if grow_residual:
            model.initialize_wider_residual(prior, source["state_dict"])
            growth["residual_width_before"] = prior.dim
        elif grow_expansion:
            model.initialize_wider_expansion(prior, source["state_dict"])
            growth["expansion_width_before"] = prior.hidden
            if prior.delay_group_widths is not None:
                growth["expansion_groups_before"] = prior.group_widths
        else:
            model.initialize_deeper(prior, source["state_dict"])
            growth["depth_before"] = prior.depth
    elif isinstance(model, EventLM):
        source_cfg = EventLMConfig(**source["cfg"])
        adding_arrival = source_cfg.trace_arrival is None and model.cfg.trace_arrival is not None
        # Training checkpoints store weights in physical units. These switches can
        # reuse the matrices even when the forward pass uses a finer charge grid.
        source_geometry = replace(source_cfg, resolution=model.cfg.resolution,
                                  output_reference=model.cfg.output_reference, compact_delays=model.cfg.compact_delays,
                                  compact_axon_state=model.cfg.compact_axon_state,
                                  compact_trace_state=model.cfg.compact_trace_state,
                                  departure_coincidence=model.cfg.departure_coincidence,
                                  delay_window=model.cfg.delay_window, router_charge_bits=model.cfg.router_charge_bits,
                                  delay_residual=source_cfg.delay_residual or model.cfg.delay_residual,
                                  activity_inhibition=model.cfg.activity_inhibition, local_trace=model.cfg.local_trace,
                                  trace_coincidence_start=model.cfg.trace_coincidence_start,
                                  inhibition_by_expansion=model.cfg.inhibition_by_expansion,
                                  learn_inhibition_thresholds=(source_cfg.learn_inhibition_thresholds
                                                               or model.cfg.learn_inhibition_thresholds),
                                  inhibition_threshold_bits=model.cfg.inhibition_threshold_bits,
                                  redistribute_inhibition=source_cfg.redistribute_inhibition or model.cfg.redistribute_inhibition,
                                  trace_write_credit=model.cfg.trace_write_credit,
                                  trace_write_confirm=source_cfg.trace_write_confirm or model.cfg.trace_write_confirm,
                                  trace_integration=source_cfg.trace_integration or model.cfg.trace_integration,
                                  trace_tap_read=model.cfg.trace_tap_read,
                                  trace_read_drive=source_cfg.trace_read_drive or model.cfg.trace_read_drive,
                                  trace_read_drive_bits=model.cfg.trace_read_drive_bits,
                                  trace_read_drive_carry=model.cfg.trace_read_drive_carry,
                                  trace_read_share_steering=model.cfg.trace_read_share_steering,
                                  trace_arrival=source_cfg.trace_arrival or model.cfg.trace_arrival,
                                  trace_arrival_groups=(model.cfg.trace_arrival_groups if adding_arrival
                                                        else source_cfg.trace_arrival_groups),
                                  steering_transfer=source_cfg.steering_transfer or model.cfg.steering_transfer,
                                  steering_gradient_at_zero=model.cfg.steering_gradient_at_zero,
                                  late_steering=source_cfg.late_steering or model.cfg.late_steering,
                                  bank_write_credit=model.cfg.bank_write_credit,
                                  trace_read_veto=source_cfg.trace_read_veto or model.cfg.trace_read_veto,
                                  residual_membrane=model.cfg.residual_membrane,
                                  residual_correction_start=model.cfg.residual_correction_start,
                                  tag_phase_address=model.cfg.tag_phase_address,
                                  tag_read_competition=source_cfg.tag_read_competition or model.cfg.tag_read_competition,
                                  delay_filter=source_cfg.delay_filter or model.cfg.delay_filter,
                                  opponent_steering=source_cfg.opponent_steering or model.cfg.opponent_steering,
                                  spread_coincidence=source_cfg.spread_coincidence or model.cfg.spread_coincidence)
        if source_geometry == model.cfg:
            expected = set()
            if model.cfg.delay_residual and not source_cfg.delay_residual:
                expected.update(f"blocks.{index}.delay_correction.gain" for index in range(model.cfg.depth))
            if model.cfg.delay_filter and not source_cfg.delay_filter:
                expected.update(name for name in model.state_dict() if ".delay_filters." in name)
            if model.cfg.opponent_steering and not source_cfg.opponent_steering:
                expected.update(name for name in model.state_dict() if ".opponent_steering." in name)
            if model.cfg.trace_read_veto and not source_cfg.trace_read_veto:
                expected.update(name for name in model.state_dict() if ".trace.read_veto." in name)
            if model.cfg.trace_write_confirm and not source_cfg.trace_write_confirm:
                expected.update(name for name in model.state_dict() if ".trace.write_confirm." in name)
            if model.cfg.trace_integration and not source_cfg.trace_integration:
                expected.update(name for name in model.state_dict() if ".trace.input." in name)
            if model.cfg.trace_tap_read is not None and source_cfg.trace_tap_read is None:
                expected.update(name for name in model.state_dict() if ".trace_taps." in name)
            if model.cfg.trace_read_drive is not None and source_cfg.trace_read_drive is None:
                expected.update(f"blocks.{index}.trace.read_drive.weight" for index in range(model.cfg.depth))
            if adding_arrival:
                expected.update(f"blocks.{index}.arrival_project.weight" for index in range(model.cfg.depth))
            if model.cfg.steering_transfer is not None and source_cfg.steering_transfer is None:
                expected.update(name for name in model.state_dict() if ".steering_transfer." in name)
            if model.cfg.spread_coincidence and not source_cfg.spread_coincidence:
                expected.update(name for name in model.state_dict() if ".spread_coincidence." in name)
            if model.cfg.redistribute_inhibition and not source_cfg.redistribute_inhibition:
                expected.update(f"blocks.{index}.inhibition_redistribution.synapse.weight"
                                for index in range(model.cfg.depth))
            if model.cfg.learn_inhibition_thresholds and not source_cfg.learn_inhibition_thresholds:
                expected.update(f"blocks.{index}.inhibition_thresholds.relative" for index in range(model.cfg.depth))
            if model.cfg.tag_read_competition and not source_cfg.tag_read_competition:
                expected.add("prediction_memory.read_inhibition.gain")
            if model.cfg.late_steering and not source_cfg.late_steering:
                expected.update(f"blocks.{index}.late_steering.gain" for index in range(model.cfg.depth))
            if expected:
                missing, unexpected = model.load_state_dict(source["state_dict"], strict=False)
                if set(missing) != expected or unexpected:
                    raise ValueError("the checkpoint must supply every weight except the new router synapses")
            else:
                model.load_state_dict(source["state_dict"])
        else:
            initialize_router(model, source, source_cfg)
        if preserve_router_charge:
            model.preserve_router_charge(source_cfg.router_charge_bits)
        if freeze_router:
            if model.prediction_memory is None:
                raise ValueError("--freeze-router needs a model with a trainable prediction bank")
            for name, parameter in model.named_parameters():
                if name in source["state_dict"] and not name.startswith(_BANK_PARAMETERS):
                    parameter.requires_grad_(False)
    else:
        changed_geometry = ("ctx", "depth") if grow_depth else ("ctx",)
        source_geometry = {name: value for name, value in source["model_kwargs"].items() if name not in changed_geometry}
        geometry = {name: value for name, value in model_kwargs.items() if name not in changed_geometry}
        prior = SNNConfig(**source["cfg"])
        if source_geometry != geometry:
            raise ValueError("the checkpoint and Transformer must use the same geometry")
        if grow_depth:
            prior_depth = source["model_kwargs"]["depth"]
            width = model.initialize_deeper(prior, source["state_dict"], prior_depth)
            growth["depth_before"] = prior_depth
            if width is not None:
                growth["feed_forward_width_before"] = width
        elif grow_feed_forward:
            growth["feed_forward_width_before"] = model.initialize_wider_feed_forward(prior, source["state_dict"])
        elif prior != model.cfg:
            raise ValueError("the checkpoint and Transformer must use the same geometry")
        else:
            model.load_state_dict(source["state_dict"])
    return {
        "checkpoint": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "training_targets": training_exposure(source),
        **growth,
    }


def _lr_scales(text):
    """Pairs of a parameter name part and a learning-rate factor, from ``PART=FACTOR,...``."""
    pairs = []
    for item in text.split(","):
        part, _, factor = item.partition("=")
        if not part or not math.isfinite(float(factor)) or float(factor) <= 0:
            raise ValueError(f"{item!r} is not a name part and a positive factor")
        pairs.append((part, float(factor)))
    return tuple(pairs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=(*EVENT_ROUTES, "transformer_local", "transformer"),
                        default="event")
    parser.add_argument("--data", type=Path, default=Path(os.environ.get("ENWIK8_PATH", str(DATA))))
    parser.add_argument("--training-mixture", type=Path,
                        help="sample training windows from a pinned byte mixture catalogue")
    parser.add_argument("--device", choices=("cpu", "mps", "cuda", "auto"), default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dim", type=int, default=96)
    parser.add_argument("--hidden", type=int, default=288, help="event feed-forward width")
    parser.add_argument("--delay-group-widths", help="four comma-separated axon group widths that sum to --hidden")
    parser.add_argument("--memory-heads", type=int, default=4)
    parser.add_argument("--memory-slots", type=int, default=16)
    parser.add_argument("--memory-width", type=int, help="total width of a prediction bank before splitting keys")
    parser.add_argument("--forecast-key-channels", type=int, default=0, help="forecast channels reserved per cell")
    parser.add_argument("--match-forecast-key", action="store_true", help="use key mismatches to inhibit reads")
    parser.add_argument("--tag-phase-address", action="store_true",
                        help="use signed forecast tags to time each address spike within a short clock")
    parser.add_argument("--tag-read-competition", action="store_true",
                        help="have stored tags inhibit candidate reads before their address competition")
    parser.add_argument("--tag-read-initial-gain", type=float,
                        help="set all stored-tag read gains at the start of training")
    parser.add_argument("--bank-read-rank-weight", type=float, default=0.0,
                        help="supervise read competition with the next observed forecast error")
    parser.add_argument("--learn-context-keys", action="store_true",
                        help="learn reserved answers to memory queries through next-byte loss")
    parser.add_argument("--learn-packet-thresholds", action="store_true",
                        help="learn shared integer thresholds for content and steering groups")
    parser.add_argument("--output-reference", action="store_true",
                        help="subtract a shared spike reference before vocabulary neurons fire")
    parser.add_argument("--compact-delays", action="store_true",
                        help="allocate a history ring of its own length to each fixed axon group")
    parser.add_argument("--compact-axon-state", action="store_true",
                        help="store axon counts in the narrowest signed integer that holds every pulse")
    parser.add_argument("--compact-trace-state", action="store_true",
                        help="store held charge in the narrowest signed integer that holds its bounds")
    parser.add_argument("--departure-coincidence", action="store_true",
                        help="route delayed pulses by coincidence with both departure and arrival steering")
    parser.add_argument("--delay-window", type=int, default=1,
                        help="number of adjacent delay taps per group available for spike matching")
    parser.add_argument("--delay-residual", action="store_true",
                        help="emit learned correction pulses from matching delays into the fixed split")
    parser.add_argument("--router-charge-bits", type=int, default=EventLMConfig.router_charge_bits,
                        help="fractional charge bits for local synapses and firing thresholds")
    parser.add_argument("--preserve-router-charge", action="store_true",
                        help="put loaded router weights on the old charge grid before refining it")
    parser.add_argument("--activity-inhibition", action="store_true",
                        help="raise content and steering thresholds with pooled incoming spike activity")
    parser.add_argument("--inhibition-by-expansion", action="store_true",
                        help="size the inhibitory threshold from the content and steering populations")
    parser.add_argument("--redistribute-inhibition", action="store_true",
                        help="learn to transfer inhibitory pulses between pairs of delay groups")
    parser.add_argument("--learn-inhibition-thresholds", action="store_true",
                        help="learn a shared content and steering threshold in each inhibited delay group")
    parser.add_argument("--inhibition-threshold-bits", type=int, default=EventLMConfig.inhibition_threshold_bits,
                        help="fractional bits for the learned base threshold as a share of the original threshold")
    parser.add_argument("--local-trace", action="store_true",
                        help="hold each channel's charge within its steering interval across bytes")
    parser.add_argument("--trace-coincidence-start", type=int,
                        help="first residual neuron whose present pulses must also match the signed held read")
    parser.add_argument("--residual-membrane", action="store_true",
                        help="retain the signed subthreshold charge at the output of each routing block")
    parser.add_argument("--residual-correction-start", type=int,
                        help="first block whose residual axon carries input spikes and emits only corrections")
    parser.add_argument("--trace-write-credit", action="store_true",
                        help="send held-charge gradients to the last bound that moved the charge")
    parser.add_argument("--trace-write-confirm", action="store_true",
                        help="use delayed content spikes to extend the held-charge write interval")
    parser.add_argument("--trace-integration", action="store_true",
                        help="integrate learned current and arriving spikes before correcting held charge")
    parser.add_argument("--trace-tap-read", choices=("fixed", "race"),
                        help="read the shortest candidate tap or let held spikes race among recent taps")
    parser.add_argument("--trace-read-drive", choices=_TRACE_READ_DRIVE_SOURCES,
                        help="choose the read input and whether to learn independent gains for its two rails")
    parser.add_argument("--trace-read-drive-bits", type=int, default=EventLMConfig.trace_read_drive_bits,
                        help="fixed-point charge precision of the driven read neurons")
    parser.add_argument("--trace-read-drive-carry", action="store_true",
                        help="retain the driven read neuron's fractional charge across bytes")
    parser.add_argument("--trace-read-share-steering", action="store_true",
                        help="match blocked read charge to spare steering pulses within each delay group")
    parser.add_argument("--trace-arrival", choices=_TRACE_ARRIVAL_MODES,
                        help="project the selected fixed arrival or its unmatched spikes against the held read")
    parser.add_argument("--trace-arrival-groups", type=int, default=EventLMConfig.trace_arrival_groups,
                        help="number of shortest fixed delay groups with arrival synapses")
    parser.add_argument("--clock-start", type=int,
                        help="first block that runs only when the surprise clock ticks")
    parser.add_argument("--clock-threshold", type=int, default=0,
                        help="unmatched winner output spikes that the clock neuron integrates per tick")
    parser.add_argument("--clock-leak", type=int, default=0,
                        help="inhibitory spikes per byte against the clock membrane")
    parser.add_argument("--clock-source", choices=_CLOCK_SOURCES, default="surprise",
                        help="integrate surprise, or one unit per byte as the fixed-stride control")
    parser.add_argument("--read-after", help="comma-separated byte-rate blocks that a state read follows")
    parser.add_argument("--read-width", type=int, default=EventLMConfig.read_width, help="channels of a state read")
    parser.add_argument("--read-heads", type=int, default=EventLMConfig.read_heads)
    parser.add_argument("--read-window", type=int, default=EventLMConfig.read_window, help="cells a state read can address")
    parser.add_argument("--read-mode", choices=("soft", "race", "age"), default=EventLMConfig.read_mode,
                        help="softmax shares as the reference, cell spikes above a lateral inhibition, "
                             "or shares by the cells' age alone as the control without an address")
    parser.add_argument("--read-levels", type=int, default=EventLMConfig.read_levels, help="most spikes of one cell per read")
    parser.add_argument("--read-resolution", type=int, default=EventLMConfig.read_resolution,
                        help="read pulses per payload pulse of a single addressed cell, the stream's resolution if unset")
    parser.add_argument("--read-synapse-gain", type=float, default=EventLMConfig.read_synapse_gain,
                        help="starting scale of a state read's probe, tag and payload synapses")
    parser.add_argument("--read-budget", type=int, default=EventLMConfig.read_budget,
                        help="cell spikes that end a race read of rates, the race within reach of the leader if unset")
    parser.add_argument("--read-address-rate", type=float, default=EventLMConfig.read_address_rate,
                        help="factor on the stored age offsets and gains of a state read")
    parser.add_argument("--read-beats", type=int, default=EventLMConfig.read_beats,
                        help="a state read also knows each cell's age in beats of a clock, counted up to this many")
    parser.add_argument("--read-beat-start", type=float, default=EventLMConfig.read_beat_start,
                        help="charge that each head starts with for one age in beats of its own")
    parser.add_argument("--read-clock", choices=("word", "end", "learned"), default=EventLMConfig.read_clock,
                        help="what beats for a state read's ages: the start of every word, the first byte after "
                             "every word, or a clock cell of each head that learns when to beat")
    parser.add_argument("--read-clock-rate", type=float, default=EventLMConfig.read_clock_rate,
                        help="the share of bytes at which a learned read clock beats before it has learned")
    parser.add_argument("--read-marks", action=argparse.BooleanOptionalAction, default=EventLMConfig.read_marks,
                        help="a beat marks the cell it falls on, and every head has a second offset for the "
                             "marked cell of each age in beats")
    parser.add_argument("--read-signed", action=argparse.BooleanOptionalAction, default=EventLMConfig.read_signed,
                        help="keep the negative pulses of a race read's probe and tag")
    parser.add_argument("--granule-after", help="comma-separated byte-rate blocks that a layer of granule cells follows")
    parser.add_argument("--granule-clusters", type=int, default=EventLMConfig.granule_clusters,
                        help="clusters of granule cells in a layer, each on its own stream channels")
    parser.add_argument("--granule-cells", type=int, default=EventLMConfig.granule_cells, help="cells of a cluster")
    parser.add_argument("--granule-fan", type=int, default=EventLMConfig.granule_fan,
                        help="stream channels a cluster listens to, and channels its axons reach")
    parser.add_argument("--granule-gain", type=float, default=EventLMConfig.granule_gain,
                        help="starting scale of the granule cells' input synapses")
    parser.add_argument("--granule-rest", type=float, default=EventLMConfig.granule_rest,
                        help="starting spikes of charge a granule cell needs before its threshold counts")
    parser.add_argument("--granule-price", type=float, default=EventLMConfig.granule_price,
                        help="share of the pulse budget's price that pushes on the granule cells' own spikes")
    parser.add_argument("--granule-bits", type=int, default=EventLMConfig.granule_bits,
                        help="charge bits of the granule cells' synapses; the router's charge bits when omitted")
    parser.add_argument("--granule-budget", type=int, default=EventLMConfig.granule_budget,
                        help="spikes per byte that the cells of a granule cluster race for; threshold cells when omitted")
    parser.add_argument("--granule-delays", type=int, default=EventLMConfig.granule_delays,
                        help="conduction delays among a granule cluster's fibres: each brings its channel as it was "
                             "zero to one less than this many bytes ago")
    parser.add_argument("--granule-traces", type=int, default=EventLMConfig.granule_traces,
                        help="kinds of slow fibres in a granule cluster: the first passes a quarter of the charge it "
                             "holds on per byte, the second a sixteenth, and so on")
    parser.add_argument("--granule-beats", type=int, default=EventLMConfig.granule_beats,
                        help="stretches of bytes that a granule cluster's beat fibres hold: a fibre brings its "
                             "channel over the present stretch or over one of this many stretches before it, and a "
                             "beat of the cluster's clock opens a new stretch; no beat fibres when zero")
    parser.add_argument("--granule-clock", choices=("word", "end", "every", "learned"),
                        default=EventLMConfig.granule_clock,
                        help="what beats for a granule cluster's beat fibres: the start of every word, the first "
                             "byte after every word, every byte (which makes them delayed fibres), or a clock cell "
                             "of the cluster that learns when")
    parser.add_argument("--granule-hold", choices=("sum", "mean", "last"), default=EventLMConfig.granule_hold,
                        help="what a beat fibre brings of a stretch: its summed count, its mean count per byte, or "
                             "the count at its last byte")
    parser.add_argument("--granule-clock-rate", type=float, default=EventLMConfig.granule_clock_rate,
                        help="beats per byte at which a learned clock cell starts")
    parser.add_argument("--granule-relays", type=int, default=EventLMConfig.granule_relays,
                        help="relay neurons of a granule layer that hear the whole stream and whose axons are the "
                             "cells' fibres; the fibres come from stream channels when zero")
    parser.add_argument("--synapse-readout", type=Path,
                        help="decoder with fitted fast synapses whose key neurons and fixed strengths run in training")
    parser.add_argument("--head-readout", type=Path,
                        help="fitted coasting heads from probe_coasting_heads.py whose fixed strengths run in training")
    parser.add_argument("--load-weights", type=Path,
                        help="start from the weights of a checkpoint of the same decoder, whatever its recall circuits")
    parser.add_argument("--save-every", type=int, default=0,
                        help="also write the weights to <checkpoint>.partial after every this many steps")
    parser.add_argument("--resume", action="store_true",
                        help="write the whole training state to <checkpoint>.resume with every partial "
                             "and continue from that file when it exists")
    parser.add_argument("--stop-at", type=int, metavar="STEP",
                        help="leave after this update with its training state saved; --resume continues the schedule")
    parser.add_argument("--early-readout", type=int,
                        help="number of leading blocks whose stream also drives the vocabulary neurons")
    parser.add_argument("--early-readout-weight", type=float, default=0.0,
                        help="weight of the next-byte loss on the early readout")
    parser.add_argument("--steering-transfer", choices=("present", "delayed"),
                        help="move steering pulses within each pair in response to present or arriving content")
    parser.add_argument("--steering-gradient-at-zero", action="store_true",
                        help="use the right training derivative for steering at zero synaptic charge")
    parser.add_argument("--late-steering", action="store_true",
                        help="let the first residual spikes steer the remaining immediate bypass spikes")
    parser.add_argument("--bank-write-credit", action="store_true",
                        help="send each correction-cell gradient to the last write that moved it")
    parser.add_argument("--trace-read-veto", action="store_true",
                        help="use charge correction spikes to inhibit mismatched reads")
    parser.add_argument("--opponent-steering", action="store_true",
                        help="use opposing steering spikes to protect held charge from writes")
    parser.add_argument("--spread-coincidence", action="store_true",
                        help="spread a learned share of immediate pulses across the steering clock")
    parser.add_argument("--delay-filter", action="store_true",
                        help="correct the coincidence path with learned spike differences across fixed taps")
    parser.add_argument("--pulse-budget", type=float, help="target average spike emissions per training byte")
    parser.add_argument("--spike-resolution", type=int, default=1, help="number of finer spikes per original unit")
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--no-carry", action="store_true", help="reset event membrane before each token")
    parser.add_argument("--heads", type=int, default=4, help="Transformer attention heads")
    parser.add_argument("--mlp-ratio", type=float, default=4.4, help="Transformer feed-forward expansion")
    parser.add_argument("--local-window", type=int, default=32, help="local Transformer attention window")
    parser.add_argument("--local-lookback", type=int, help="effective context including the current byte")
    parser.add_argument("--ctx", type=int, default=256)
    parser.add_argument("--bs", type=int, default=12)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.003)
    parser.add_argument("--wd", type=float, default=0.05)
    parser.add_argument("--adam-beta2", type=float, default=0.99,
                        help="memory of the optimizer's squared-gradient average")
    parser.add_argument("--rotated-adam", type=int, metavar="STEPS",
                        help="run Adam in the eigenbasis of each weight matrix's gradient statistics and "
                             "update the bases every this many steps")
    parser.add_argument("--rotate-stacks", action="store_true",
                        help="with --rotated-adam, also turn every matrix of a three-dimensional weight, such as "
                             "the synapses of granule cells")
    parser.add_argument("--lr-scale", type=_lr_scales, default=(), metavar="PART=FACTOR,...",
                        help="multiply the learning rate of every parameter whose name contains a part by that "
                             "part's factor; of several matching parts the longest counts")
    parser.add_argument("--weight-average", type=float,
                        help="also keep a running average of the weights with this decay, score it at every "
                             "evaluation and save it beside the checkpoint")
    parser.add_argument("--eval-every", type=int, default=250)
    parser.add_argument("--eval-bs", type=int, default=12)
    parser.add_argument("--eval-batches", type=int, default=8)
    parser.add_argument("--final-eval-batches", type=int, default=100)
    parser.add_argument("--validation-only", action="store_true", help="leave the test split untouched during tuning")
    parser.add_argument("--initialize-from", type=Path, help="matching model or compatible plain spike router checkpoint")
    parser.add_argument("--teacher-checkpoint", type=Path, help="frozen byte model used only during training")
    parser.add_argument("--teacher-weight", type=float, default=0.5,
                        help="share of the next-byte gradient from the teacher when it predicts the target better")
    growth = parser.add_mutually_exclusive_group()
    growth.add_argument("--grow-residual", action="store_true",
                        help="start extra residual neurons silent while loading the narrower checkpoint")
    growth.add_argument("--grow-expansion", action="store_true",
                        help="grow silent content and steering neurons inside each delay group")
    growth.add_argument("--grow-feed-forward", action="store_true",
                        help="grow a Transformer reference's SwiGLU width from a narrower checkpoint")
    growth.add_argument("--grow-depth", action="store_true",
                        help="append identity-initialized blocks to the loaded model")
    parser.add_argument("--freeze-router", action="store_true", help="update only the newly added bank")
    parser.add_argument("--code-loss-weight", type=float, default=0.0, help="weight for the next-byte spike code loss")
    parser.add_argument("--corrected-code-loss", action="store_true",
                        help="supervise the next-byte forecast after adding its retrieved correction spikes")
    parser.add_argument("--tag", default="")
    parser.add_argument("--results-dir", type=Path, help="write training JSONs to this directory")
    parser.add_argument("--checkpoint", help="save weights and the exact model configuration")
    parser.add_argument("--force", action="store_true", help="replace existing result or checkpoint")
    args = parser.parse_args()
    sizes = (args.threads, args.dim, args.hidden, args.depth, args.heads, args.local_window, args.ctx, args.bs,
             args.steps, args.eval_every, args.eval_bs, args.eval_batches, args.final_eval_batches)
    if min(sizes) < 1 or not 0 <= args.warmup <= args.steps:
        parser.error("sizes must be positive and warmup must be within the training step budget")
    if args.local_lookback is not None and (args.variant != "transformer_local" or args.local_lookback < 1):
        parser.error("--local-lookback needs a positive total width and a local Transformer")
    if not (math.isfinite(args.lr) and args.lr > 0 and math.isfinite(args.wd) and args.wd >= 0):
        parser.error("learning rate must be positive and weight decay must be nonnegative")
    if not 0 <= args.adam_beta2 < 1 or (args.weight_average is not None and not 0 <= args.weight_average < 1):
        parser.error("--adam-beta2 and --weight-average must be in [0, 1)")
    if args.rotated_adam is not None and args.rotated_adam < 1:
        parser.error("--rotated-adam needs a positive number of steps")
    if args.rotate_stacks and args.rotated_adam is None:
        parser.error("--rotate-stacks needs --rotated-adam")
    if not 0 < args.teacher_weight <= 1:
        parser.error("--teacher-weight must be in (0, 1]")
    if args.freeze_router and args.initialize_from is None:
        parser.error("--freeze-router requires --initialize-from")
    if (args.grow_residual or args.grow_expansion) and (
        args.initialize_from is None or args.variant not in EVENT_ROUTES or args.freeze_router
    ):
        parser.error("spiking growth needs an --initialize-from checkpoint and a trainable router")
    if args.grow_feed_forward and (args.initialize_from is None or args.variant not in ("transformer", "transformer_local")):
        parser.error("--grow-feed-forward needs a Transformer --initialize-from checkpoint")
    if args.grow_depth and (args.initialize_from is None or args.freeze_router):
        parser.error("--grow-depth needs a trainable --initialize-from checkpoint")
    if args.residual_correction_start is not None and args.variant not in EVENT_ROUTES:
        parser.error("--residual-correction-start needs a spiking variant")
    if args.preserve_router_charge and (args.initialize_from is None or args.variant not in EVENT_ROUTES):
        parser.error("--preserve-router-charge needs a spiking --initialize-from checkpoint")
    if args.bank_write_credit and args.variant not in EVENT_ROUTES:
        parser.error("--bank-write-credit needs a spiking correction bank")
    if args.freeze_router and args.variant not in EVENT_ROUTES:
        parser.error("--freeze-router needs a spiking variant")
    if not math.isfinite(args.code_loss_weight) or args.code_loss_weight < 0:
        parser.error("--code-loss-weight must be finite and nonnegative")
    if args.code_loss_weight and (args.variant != "event_correction_code" or args.ctx < 2):
        parser.error("spike code loss needs the forecast bank and at least two input bytes")
    if args.corrected_code_loss and not args.code_loss_weight:
        parser.error("--corrected-code-loss needs a positive --code-loss-weight")
    if (args.forecast_key_channels or args.match_forecast_key or args.learn_context_keys
            or args.tag_phase_address or args.tag_read_competition) and args.variant != "event_correction_code":
        parser.error("forecast keys need --variant event_correction_code")
    if args.tag_read_initial_gain is not None and (
        not args.tag_read_competition or not 0 <= args.tag_read_initial_gain <= 1
    ):
        parser.error("a read gain in [0, 1] needs --tag-read-competition")
    if not math.isfinite(args.bank_read_rank_weight) or args.bank_read_rank_weight < 0:
        parser.error("bank read ranking weight must be finite and nonnegative")
    if args.bank_read_rank_weight and (not args.tag_read_competition or args.ctx < 2):
        parser.error("bank read ranking needs --tag-read-competition and at least two input bytes")
    if args.delay_group_widths is not None and args.variant not in EVENT_ROUTES:
        parser.error("custom delay groups need a spiking decoder")
    if (args.clock_start is not None or args.early_readout is not None) and args.variant not in EVENT_ROUTES:
        parser.error("the surprise clock and the early readout need a spiking decoder")
    if not math.isfinite(args.early_readout_weight) or args.early_readout_weight < 0:
        parser.error("--early-readout-weight must be finite and nonnegative")
    if args.early_readout_weight and args.early_readout is None and args.clock_start is None:
        parser.error("an early readout loss needs --early-readout or a clocked model")
    if (args.learn_packet_thresholds or args.pulse_budget is not None or args.output_reference or args.compact_delays
            or args.compact_axon_state or args.compact_trace_state or args.residual_membrane
            or args.departure_coincidence or args.delay_window != 1 or args.delay_residual or args.activity_inhibition
            or args.inhibition_by_expansion or args.redistribute_inhibition or args.learn_inhibition_thresholds
            or args.inhibition_threshold_bits != EventLMConfig.inhibition_threshold_bits
            or args.trace_coincidence_start is not None or args.steering_gradient_at_zero or args.late_steering
            or args.local_trace or args.trace_write_credit or args.trace_write_confirm or args.trace_read_veto
            or args.trace_integration or args.trace_tap_read is not None or args.steering_transfer is not None
            or args.trace_read_drive is not None or args.trace_read_drive_carry or args.trace_read_share_steering
            or args.trace_read_drive_bits != EventLMConfig.trace_read_drive_bits
            or args.trace_arrival is not None or args.trace_arrival_groups != EventLMConfig.trace_arrival_groups
            or args.delay_filter or args.opponent_steering or args.spread_coincidence
            or args.router_charge_bits != EventLMConfig.router_charge_bits) and args.variant not in (
        "event_steered", "event_correction_code"
    ):
        parser.error("packet thresholds, pulse budgets and delay or output options need a steered spike router")
    if args.pulse_budget is not None and (not math.isfinite(args.pulse_budget) or args.pulse_budget <= 0):
        parser.error("pulse budget must be positive and finite")
    if args.resume and (args.checkpoint is None or args.save_every < 1):
        parser.error("--resume needs --checkpoint and --save-every")
    if args.stop_at is not None and (not args.resume or not 0 < args.stop_at < args.steps
                                     or args.stop_at % args.save_every):
        parser.error("--stop-at needs --resume and an update below --steps at which --save-every writes")
    event = args.variant in EVENT_ROUTES
    name = "_".join(part for part in (args.variant, f"s{args.seed}", args.tag) if part)
    results_dir = None if args.results_dir is None else args.results_dir.expanduser()
    if result_exists("event_lm", name, args.force, directory=results_dir):
        return
    setup(args)
    try:
        device = resolve_device(args.device)
        checkpoint = checkpoint_path(args)
        train, val, test = load_splits(args.data.expanduser())
        mixture = (None if args.training_mixture is None
                   else TrainingMixture(args.training_mixture.expanduser(), train, args.ctx))
        max_windows = args.eval_bs * max(args.eval_batches, args.final_eval_batches)
        if (len(val) - args.ctx - 1) // max_windows < args.ctx:
            raise ValueError("evaluation windows would overlap; reduce batch size or evaluation batches")
        model, cfg, model_kwargs = build_model(args)
        initialization = loaded_weights = None
        if args.initialize_from is not None:
            initialization = initialize_from_checkpoint(
                model, model_kwargs, args.initialize_from.expanduser(),
                freeze_router=args.freeze_router, grow_residual=args.grow_residual, grow_feed_forward=args.grow_feed_forward,
                grow_expansion=args.grow_expansion, grow_depth=args.grow_depth,
                preserve_router_charge=args.preserve_router_charge,
            )
        if args.load_weights is not None:
            # Synapses and heads carry no trained weights, so a decoder may gain or drop them here.
            recall = ("fast_synapses.", "read_heads.")
            parent = torch.load(args.load_weights.expanduser(), map_location="cpu", weights_only=True)
            own = asdict(cfg)
            # A read's offsets for the age in beats start at zero, and a new read or cell layer starts silent. Each
            # leaves the decoder as it was, so a trained decoder may gain them here. A checkpoint from before the
            # reads had such ages has none.
            ages = ("read_beats", "read_beat_start", "read_clock", "read_clock_rate", "read_marks",
                    "read_address_rate")
            earlier = {**{name: getattr(EventLMConfig, name) for name in ages}, **(parent.get("cfg") or {})}
            grown = {"read_beats", "read_marks", "read_address_rate"}
            silent = [".beat_age", ".beat_mark"]
            if not earlier["read_beats"]:
                # With the ages it may choose their clock: the clock cells of a read start deaf.
                grown |= {"read_clock", "read_clock_rate"}
                silent += [f"state_reads.{index}.clock" for index in own["read_after"] or ()]
            for place, part in (("read_after", "state_reads"), ("granule_after", "granule_layers")):
                before, now = set(earlier.get(place) or ()), set(own[place] or ())
                if before < now:
                    grown.add(place)
                    silent += [f"{part}.{index}." for index in sorted(now - before)]
            different = sorted(name for name, value in earlier.items()
                               if not name.startswith(("synapse_", "head_")) and name in own and name not in grown
                               and json.dumps(value) != json.dumps(own[name]))
            if different:
                raise ValueError(f"{args.load_weights}: the decoder differs in {', '.join(different)}")
            weights = {name: value for name, value in parent["state_dict"].items() if not name.startswith(recall)}
            # A read stores its offsets and gains as shares of its address rate. Under another rate the shares
            # change so that the offsets and gains, and with them the decoder, stay as they were.
            share = earlier["read_address_rate"] / own["read_address_rate"]
            if share != 1.0:
                shares = {f"state_reads.{index}.{name}" for index in earlier.get("read_after") or ()
                          for name in ("age", "gain", "spacing", "beat_age", "beat_mark")}
                weights = {name: value * share if name in shares else value for name, value in weights.items()}
            missing, unexpected = model.load_state_dict(weights, strict=False)
            if unexpected or any(not name.startswith(recall) and not any(part in name for part in silent)
                                 for name in missing):
                raise ValueError(f"{args.load_weights}: expected the weights of the same decoder")
            with args.load_weights.expanduser().open("rb") as stream:
                loaded_weights = {"checkpoint": str(args.load_weights), "run_name": parent.get("run_name"),
                                  "sha256": hashlib.file_digest(stream, "sha256").hexdigest()}
        if args.tag_read_initial_gain is not None:
            with torch.no_grad():
                model.prediction_memory.read_inhibition.gain.fill_(args.tag_read_initial_gain)
        model = model.to(device)
        teacher = None
        teacher_details = {}
        if args.teacher_checkpoint is not None:
            teacher_path = args.teacher_checkpoint.expanduser()
            teacher, source = load_checkpoint(teacher_path, device)
            teacher.requires_grad_(False)
            with teacher_path.open("rb") as stream:
                teacher_details["teacher"] = {
                    "checkpoint": str(teacher_path), "sha256": hashlib.file_digest(stream, "sha256").hexdigest(),
                    "model_type": source["model_type"], "parameters": count_params(teacher),
                    "training_targets": training_exposure(source), "run_name": source["run_name"],
                }
    except (OSError, RuntimeError, ValueError) as exc:
        parser.error(str(exc))

    pulse_counts = None
    if args.pulse_budget is not None:
        from snnformer.event_lm import _IntegrateFire, emitted_pulses

        pulse_counts = []

        def count_spikes(module, _inputs, counts):
            pulse_counts.append(emitted_pulses(module, counts))

        for module in model.modules():
            if isinstance(module, _IntegrateFire):
                module.register_forward_hook(count_spikes)
        spike_price = torch.zeros((), device=device)

    evaluate_options = dict(context=args.ctx, batch_size=args.eval_bs, device=device, event=event,
                            pulse_counts=pulse_counts)
    optimizer = make_optimizer(model, args.lr, args.wd, betas=(0.9, args.adam_beta2), rotated=args.rotated_adam,
                               stacks=args.rotate_stacks, scales=args.lr_scale)
    average = None if args.weight_average is None else [p.detach().clone() for p in model.parameters()]
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    timer, curve = Timer(), []
    parameters = count_params(model)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{name}: params={parameters} trainable={trainable_params}", flush=True)
    reached = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    start, earlier_seconds, resumed_at = 0, 0.0, []
    resume_path = Path(f"{checkpoint}.resume") if args.resume else None
    if resume_path is not None and resume_path.exists():
        state = torch.load(resume_path, map_location="cpu", weights_only=True)
        different = sorted(key for key in {*state["args"], *reached} - _RESUME_FREE
                           if json.dumps(state["args"].get(key)) != json.dumps(reached.get(key)))
        if different:
            parser.error(f"{resume_path}: the run differs in {', '.join(different)}")
        if args.stop_at is not None and args.stop_at <= state["step"]:
            parser.error(f"{resume_path}: the run is already at update {state['step']}")
        model.load_state_dict(state["state_dict"])
        optimizer.load_state_dict(state["optimizer"])
        if average is not None:
            with torch.no_grad():
                for mean, saved in zip(average, state["average"], strict=True):
                    mean.copy_(saved)
        if pulse_counts is not None:
            spike_price = state["spike_price"].to(device)
        curve, start, earlier_seconds = state["curve"], state["step"], state["seconds"]
        resumed_at = [*state["resumed_at"], start]
        # Drawing the earlier batches again puts the generator and the mixture's receipt where they were.
        for _ in range(start):
            if mixture is None:
                random_batch(train, args.bs, args.ctx, generator)
            else:
                mixture.batch(args.bs, generator)
        print(f"{name}: continuing after {start} of {args.steps} updates", flush=True)
    teacher_event = isinstance(teacher, EventLM)
    teacher_resolution = teacher.cfg.resolution if teacher_event else 1
    for step in range(start + 1, args.steps + 1):
        for group in optimizer.param_groups:
            group["lr"] = cosine_lr(step - 1, args.steps, args.warmup, args.lr) * group["lr_scale"]
        inputs, labels = (random_batch(train, args.bs, args.ctx, generator) if mixture is None
                          else mixture.batch(args.bs, generator))
        inputs = inputs.to(device)
        if pulse_counts is not None:
            pulse_counts.clear()
        forward_options = {} if pulse_counts is None else {"memory_pulses": pulse_counts}
        if args.bank_read_rank_weight:
            bank_read_losses = []
            forward_options["bank_read_losses"] = bank_read_losses
        if args.early_readout_weight:
            forward_options["early_readout"] = args.clock_start if args.early_readout is None else args.early_readout
        if args.code_loss_weight:
            counts, predicted, observed, *early = model(
                inputs, return_prediction=True, corrected_prediction=args.corrected_code_loss, **forward_options,
            )
            if args.learn_context_keys:
                predicted = predicted[..., args.forecast_key_channels:]
                observed = observed[..., args.forecast_key_channels:]
            resolution = args.spike_resolution
            code_loss = F.smooth_l1_loss(predicted[:, :-1] / resolution, observed[:, 1:].detach() / resolution)
        elif args.early_readout_weight:
            counts, *early = model(inputs, **forward_options)
        else:
            counts = model(inputs, **forward_options)
        logits = scoring_logits(counts, event, args.spike_resolution)
        targets = labels.to(device)
        if teacher is None:
            language_loss = F.cross_entropy(logits.flatten(0, 1), targets.flatten())
            loss = language_loss
        else:
            with torch.no_grad():
                guidance = scoring_logits(teacher(inputs), teacher_event, teacher_resolution)
            loss, language_loss, teacher_fraction = guided_language_loss(logits, targets, guidance, args.teacher_weight)
        if args.code_loss_weight:
            loss = loss + args.code_loss_weight * code_loss
        if args.early_readout_weight:
            early_logits = scoring_logits(early[0], event, model.forecast_resolution)
            early_loss = F.cross_entropy(early_logits.flatten(0, 1), targets.flatten())
            loss = loss + args.early_readout_weight * early_loss
        if args.bank_read_rank_weight:
            (bank_read_loss,) = bank_read_losses
            loss = loss + args.bank_read_rank_weight * bank_read_loss
        if pulse_counts is not None:
            pulses = torch.stack(pulse_counts).sum() / inputs.numel()
            gap = pulses / args.pulse_budget - 1.0
            loss = loss + spike_price * gap + 0.5 * gap.relu().square()
            spike_price = (spike_price + 0.04 * gap.detach()).clamp_min(0)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if average is not None:
            # The average follows the weights closely at first, so the starting weights do not linger in it.
            decay = min(args.weight_average, step / (step + 10))
            with torch.no_grad():
                for mean, parameter in zip(average, model.parameters(), strict=True):
                    mean.lerp_(parameter, 1 - decay)
        if step % args.eval_every == 0 or step == args.steps:
            metrics = evaluate(model, val, batches=args.eval_batches, **evaluate_options)
            averaged_metrics = None
            if average is not None:
                with averaged_weights(model, average):
                    averaged_metrics = evaluate(model, val, batches=args.eval_batches, **evaluate_options)
                print(f"{name} averaged weights at {step}: {averaged_metrics['bpb']:.4f} bpb", flush=True)
            curve.append(dict(step=step, tokens=step * args.bs * args.ctx,
                              train_bpb=language_loss.item() / math.log(2), val_bpb=metrics["bpb"],
                              val_argmax_accuracy=metrics["argmax_accuracy"],
                              **({} if averaged_metrics is None else {"averaged_val_bpb": averaged_metrics["bpb"]}),
                              **({"train_pulses_per_byte": pulses.item(), "val_pulses_per_byte": metrics["pulses_per_byte"],
                                  "spike_price": spike_price.item()} if pulse_counts is not None else {}),
                              **({"teacher_selected_fraction": teacher_fraction.item()} if teacher is not None else {}),
                              **({"bank_read_rank_loss": bank_read_loss.item()} if args.bank_read_rank_weight else {}),
                              **({"code_loss": code_loss.item()} if args.code_loss_weight else {}),
                              **({"early_train_bpb": early_loss.item() / math.log(2)}
                                 if args.early_readout_weight else {})))
            activity = f" pulses/byte={metrics['pulses_per_byte']:.0f}" if pulse_counts is not None else ""
            print(f"{name} step {step} validation={metrics['bpb']:.4f} bpb "
                  f"accuracy={metrics['argmax_accuracy']:.4f} ({earlier_seconds + timer():.1f}s){activity}", flush=True)

        if args.save_every and checkpoint is not None and step % args.save_every == 0 and step < args.steps:
            # A long run that is stopped early keeps its last weights under another name.
            partial = Path(f"{checkpoint}.partial")
            partial.parent.mkdir(parents=True, exist_ok=True)
            torch.save(dict(format_version=1, model_type=type(model).__name__, task="event_lm", run_name=name,
                            cfg=asdict(cfg), model_kwargs=model_kwargs, args=dict(reached, steps=step),
                            initialization=initialization,
                            state_dict={key: value.detach().cpu() for key, value in model.state_dict().items()},
                            **({} if average is None else {"averaged_parameters": {
                                key: mean.cpu() for (key, _), mean in zip(model.named_parameters(), average, strict=True)
                            }})),
                       f"{partial}.tmp")
            os.replace(f"{partial}.tmp", partial)
            if resume_path is not None:
                torch.save(dict(format_version=1, step=step, args=reached, resumed_at=resumed_at,
                                state_dict={key: value.detach().cpu() for key, value in model.state_dict().items()},
                                average=None if average is None else [mean.cpu() for mean in average],
                                optimizer=_on_cpu(optimizer.state_dict()),
                                spike_price=None if pulse_counts is None else spike_price.cpu(),
                                curve=curve, seconds=earlier_seconds + timer()),
                           f"{resume_path}.tmp")
                os.replace(f"{resume_path}.tmp", resume_path)
        if step == args.stop_at:
            print(f"{name} stopped after {step} of {args.steps} updates", flush=True)
            return

    val_score = evaluate(model, val, batches=args.final_eval_batches, **evaluate_options)
    test_score = None if args.validation_only else evaluate(model, test, batches=args.final_eval_batches,
                                                            **evaluate_options)
    averaged_score = None
    if average is not None:
        with averaged_weights(model, average):
            averaged_score = evaluate(model, val, batches=args.final_eval_batches, **evaluate_options)
            averaged_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        print(f"{name} averaged weights: validation={averaged_score['bpb']:.4f} bpb", flush=True)
    args_record = dict(vars(args), data=str(args.data), results_dir=None if results_dir is None else str(results_dir),
                       synapse_readout=None if args.synapse_readout is None else str(args.synapse_readout),
                       head_readout=None if args.head_readout is None else str(args.head_readout),
                       load_weights=loaded_weights,
                       training_mixture=None if args.training_mixture is None else str(args.training_mixture),
                       initialize_from=None if args.initialize_from is None else str(args.initialize_from),
                       teacher_checkpoint=None if args.teacher_checkpoint is None else str(args.teacher_checkpoint))
    config = asdict(cfg)
    mixture_details = {} if mixture is None else {"training_mixture": mixture.receipt()}
    if checkpoint is not None:
        payload = dict(format_version=1, model_type=type(model).__name__, task="event_lm", run_name=name,
                       cfg=config, model_kwargs=model_kwargs, args=args_record, initialization=initialization,
                       state_dict={key: value.detach().cpu() for key, value in model.state_dict().items()},
                       **teacher_details, **mixture_details)
        with _atomic_output(checkpoint, "wb", overwrite=args.force) as out:
            torch.save(payload, out)
        if average is not None:
            with _atomic_output(Path(f"{checkpoint}.averaged"), "wb", overwrite=args.force) as out:
                torch.save(dict(payload, state_dict=averaged_state), out)
    record = dict(args=args_record, cfg=config, device=str(device), params=parameters,
                  trainable_params=trainable_params, initialization=initialization,
                  data_sha1=ENWIK8_SHA1, train_target_bytes=args.steps * args.bs * args.ctx, curve=curve,
                  val_bpb=val_score["bpb"], test_bpb=test_score["bpb"] if test_score else None,
                  **({} if averaged_score is None else {"averaged_val_bpb": averaged_score["bpb"]}),
                  val=val_score, test=test_score,
                  scoring=("normalized exponential of scaled output" if event else "softmax over logits"),
                  seconds=earlier_seconds + timer(), **({"resumed_at_steps": resumed_at} if resumed_at else {}),
                  **teacher_details, **mixture_details)
    destination = save_result("event_lm", name, record, overwrite=args.force, directory=results_dir)
    if resume_path is not None:
        resume_path.unlink(missing_ok=True)
    split, metric = ("validation", val_score) if test_score is None else ("test", test_score)
    print(f"{name} {split}={metric['bpb']:.4f} bpb accuracy={metric['argmax_accuracy']:.4f} ({destination})")


if __name__ == "__main__":
    main()
