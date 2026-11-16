"""Train and score integer-event and Transformer decoders on canonical enwik8."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
import math
import os
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

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
from snnformer.event_lm import EventLM, EventLMConfig
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
                            tag_phase_address=args.tag_phase_address)
        return EventLM(cfg), cfg, {}
    cfg = replace(VARIANTS[args.variant], mlp_ratio=args.mlp_ratio, local_window=args.local_window)
    if args.local_lookback is not None:
        per_layer, extra = divmod(args.local_lookback - 1, args.depth)
        model_kwargs["local_windows"] = tuple(1 + per_layer + (layer < extra) for layer in range(args.depth))
    return make_lm(cfg, **model_kwargs), cfg, model_kwargs


_BANK_PARAMETERS = ("prediction_memory.", "recall_synapse.")
_OPTIONAL_ROUTER_PARAMETERS = (
    ".packet_thresholds.", ".delay_correction.", ".delay_filters.", ".opponent_steering.", ".trace.read_veto.",
    ".spread_coincidence.", ".trace.write_confirm.", ".trace.input.", ".trace_taps.",
    ".inhibition_redistribution.", ".inhibition_thresholds.", ".late_steering.",
)


def initialize_router(model, source, source_cfg):
    """Load a trained local spike router, leaving any new bank at its initialization."""
    if source_cfg.route != "steered":
        raise ValueError("--initialize-from needs a matching spiking configuration or a plain router checkpoint")
    fields = ("dim", "hidden", "depth", "carry", "continuous", "resolution")
    if any(getattr(source_cfg, name) != getattr(model.cfg, name) for name in fields):
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
        else:
            model.initialize_deeper(prior, source["state_dict"])
            growth["depth_before"] = prior.depth
    elif isinstance(model, EventLM):
        source_cfg = EventLMConfig(**source["cfg"])
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=(*EVENT_ROUTES, "transformer_local", "transformer"),
                        default="event")
    parser.add_argument("--data", type=Path, default=Path(os.environ.get("ENWIK8_PATH", str(DATA))))
    parser.add_argument("--device", choices=("cpu", "mps", "cuda", "auto"), default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dim", type=int, default=96)
    parser.add_argument("--hidden", type=int, default=288, help="event feed-forward width")
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
    if (args.learn_packet_thresholds or args.pulse_budget is not None or args.output_reference or args.compact_delays
            or args.compact_axon_state or args.compact_trace_state or args.residual_membrane
            or args.departure_coincidence or args.delay_window != 1 or args.delay_residual or args.activity_inhibition
            or args.inhibition_by_expansion or args.redistribute_inhibition or args.learn_inhibition_thresholds
            or args.inhibition_threshold_bits != EventLMConfig.inhibition_threshold_bits
            or args.trace_coincidence_start is not None or args.steering_gradient_at_zero or args.late_steering
            or args.local_trace or args.trace_write_credit or args.trace_write_confirm or args.trace_read_veto
            or args.trace_integration or args.trace_tap_read is not None
            or args.delay_filter or args.opponent_steering or args.spread_coincidence
            or args.router_charge_bits != EventLMConfig.router_charge_bits) and args.variant not in (
        "event_steered", "event_correction_code"
    ):
        parser.error("packet thresholds, pulse budgets and delay or output options need a steered spike router")
    if args.pulse_budget is not None and (not math.isfinite(args.pulse_budget) or args.pulse_budget <= 0):
        parser.error("pulse budget must be positive and finite")
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
        max_windows = args.eval_bs * max(args.eval_batches, args.final_eval_batches)
        if (len(val) - args.ctx - 1) // max_windows < args.ctx:
            raise ValueError("evaluation windows would overlap; reduce batch size or evaluation batches")
        model, cfg, model_kwargs = build_model(args)
        initialization = None
        if args.initialize_from is not None:
            initialization = initialize_from_checkpoint(
                model, model_kwargs, args.initialize_from.expanduser(),
                freeze_router=args.freeze_router, grow_residual=args.grow_residual, grow_feed_forward=args.grow_feed_forward,
                grow_expansion=args.grow_expansion, grow_depth=args.grow_depth,
                preserve_router_charge=args.preserve_router_charge,
            )
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
        from snnformer.event_lm import _IntegrateFire

        pulse_counts = []

        def count_spikes(_module, _inputs, counts):
            pulse_counts.append(counts.abs().sum())

        for module in model.modules():
            if isinstance(module, _IntegrateFire):
                module.register_forward_hook(count_spikes)
        spike_price = torch.zeros((), device=device)

    evaluate_options = dict(context=args.ctx, batch_size=args.eval_bs, device=device, event=event,
                            pulse_counts=pulse_counts)
    optimizer = make_optimizer(model, args.lr, args.wd)
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    timer, curve = Timer(), []
    parameters = count_params(model)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{name}: params={parameters} trainable={trainable_params}", flush=True)
    teacher_event = isinstance(teacher, EventLM)
    teacher_resolution = teacher.cfg.resolution if teacher_event else 1
    for step in range(1, args.steps + 1):
        for group in optimizer.param_groups:
            group["lr"] = cosine_lr(step - 1, args.steps, args.warmup, args.lr)
        inputs, labels = random_batch(train, args.bs, args.ctx, generator)
        inputs = inputs.to(device)
        if pulse_counts is not None:
            pulse_counts.clear()
        forward_options = {} if pulse_counts is None else {"memory_pulses": pulse_counts}
        if args.bank_read_rank_weight:
            bank_read_losses = []
            forward_options["bank_read_losses"] = bank_read_losses
        if args.code_loss_weight:
            counts, predicted, observed = model(
                inputs, return_prediction=True, corrected_prediction=args.corrected_code_loss, **forward_options,
            )
            if args.learn_context_keys:
                predicted = predicted[..., args.forecast_key_channels:]
                observed = observed[..., args.forecast_key_channels:]
            resolution = args.spike_resolution
            code_loss = F.smooth_l1_loss(predicted[:, :-1] / resolution, observed[:, 1:].detach() / resolution)
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
        if step % args.eval_every == 0 or step == args.steps:
            metrics = evaluate(model, val, batches=args.eval_batches, **evaluate_options)
            curve.append(dict(step=step, tokens=step * args.bs * args.ctx,
                              train_bpb=language_loss.item() / math.log(2), val_bpb=metrics["bpb"],
                              val_argmax_accuracy=metrics["argmax_accuracy"],
                              **({"train_pulses_per_byte": pulses.item(), "val_pulses_per_byte": metrics["pulses_per_byte"],
                                  "spike_price": spike_price.item()} if pulse_counts is not None else {}),
                              **({"teacher_selected_fraction": teacher_fraction.item()} if teacher is not None else {}),
                              **({"bank_read_rank_loss": bank_read_loss.item()} if args.bank_read_rank_weight else {}),
                              **({"code_loss": code_loss.item()} if args.code_loss_weight else {})))
            activity = f" pulses/byte={metrics['pulses_per_byte']:.0f}" if pulse_counts is not None else ""
            print(f"{name} step {step} validation={metrics['bpb']:.4f} bpb "
                  f"accuracy={metrics['argmax_accuracy']:.4f} ({timer():.1f}s){activity}", flush=True)

    val_score = evaluate(model, val, batches=args.final_eval_batches, **evaluate_options)
    test_score = None if args.validation_only else evaluate(model, test, batches=args.final_eval_batches,
                                                            **evaluate_options)
    args_record = dict(vars(args), data=str(args.data), results_dir=None if results_dir is None else str(results_dir),
                       initialize_from=None if args.initialize_from is None else str(args.initialize_from),
                       teacher_checkpoint=None if args.teacher_checkpoint is None else str(args.teacher_checkpoint))
    config = asdict(cfg)
    if checkpoint is not None:
        payload = dict(format_version=1, model_type=type(model).__name__, task="event_lm", run_name=name,
                       cfg=config, model_kwargs=model_kwargs, args=args_record, initialization=initialization,
                       state_dict={key: value.detach().cpu() for key, value in model.state_dict().items()},
                       **teacher_details)
        with _atomic_output(checkpoint, "wb", overwrite=args.force) as out:
            torch.save(payload, out)
    record = dict(args=args_record, cfg=config, device=str(device), params=parameters,
                  trainable_params=trainable_params, initialization=initialization,
                  data_sha1=ENWIK8_SHA1, train_target_bytes=args.steps * args.bs * args.ctx, curve=curve,
                  val_bpb=val_score["bpb"], test_bpb=test_score["bpb"] if test_score else None,
                  val=val_score, test=test_score,
                  scoring=("normalized exponential of scaled output" if event else "softmax over logits"),
                  seconds=timer(), **teacher_details)
    destination = save_result("event_lm", name, record, overwrite=args.force, directory=results_dir)
    split, metric = ("validation", val_score) if test_score is None else ("test", test_score)
    print(f"{name} {split}={metric['bpb']:.4f} bpb accuracy={metric['argmax_accuracy']:.4f} ({destination})")


if __name__ == "__main__":
    main()
