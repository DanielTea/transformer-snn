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
from snnformer import count_params
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
                            learn_context_keys=args.learn_context_keys, learn_packet_thresholds=args.learn_packet_thresholds,
                            output_reference=args.output_reference, compact_delays=args.compact_delays,
                            departure_coincidence=args.departure_coincidence, delay_window=args.delay_window,
                            delay_residual=args.delay_residual,
                            router_charge_bits=args.router_charge_bits, activity_inhibition=args.activity_inhibition,
                            local_trace=args.local_trace, delay_filter=args.delay_filter)
        return EventLM(cfg), cfg, {}
    cfg = replace(VARIANTS[args.variant], mlp_ratio=args.mlp_ratio, local_window=args.local_window)
    if args.local_lookback is not None:
        per_layer, extra = divmod(args.local_lookback - 1, args.depth)
        model_kwargs["local_windows"] = tuple(1 + per_layer + (layer < extra) for layer in range(args.depth))
    return make_lm(cfg, **model_kwargs), cfg, model_kwargs


_BANK_PARAMETERS = ("prediction_memory.", "recall_synapse.")
_OPTIONAL_ROUTER_PARAMETERS = (".packet_thresholds.", ".delay_correction.", ".delay_filters.")


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


def initialize_from_checkpoint(model, model_kwargs, path, *, freeze_router):
    from snnformer import SNNConfig

    source = torch.load(path, map_location="cpu", weights_only=True)
    if source.get("task") != "event_lm" or source.get("model_type") != type(model).__name__:
        raise ValueError("--initialize-from needs a checkpoint of the same decoder family")
    if isinstance(model, EventLM):
        source_cfg = EventLMConfig(**source["cfg"])
        # Training checkpoints store weights in physical units. These switches can
        # reuse the matrices even when the forward pass uses a finer charge grid.
        source_geometry = replace(source_cfg, resolution=model.cfg.resolution,
                                  output_reference=model.cfg.output_reference, compact_delays=model.cfg.compact_delays,
                                  departure_coincidence=model.cfg.departure_coincidence,
                                  delay_window=model.cfg.delay_window, router_charge_bits=model.cfg.router_charge_bits,
                                  delay_residual=source_cfg.delay_residual or model.cfg.delay_residual,
                                  activity_inhibition=model.cfg.activity_inhibition, local_trace=model.cfg.local_trace,
                                  delay_filter=source_cfg.delay_filter or model.cfg.delay_filter)
        if source_geometry == model.cfg:
            expected = set()
            if model.cfg.delay_residual and not source_cfg.delay_residual:
                expected.update(f"blocks.{index}.delay_correction.gain" for index in range(model.cfg.depth))
            if model.cfg.delay_filter and not source_cfg.delay_filter:
                expected.update(name for name in model.state_dict() if ".delay_filters." in name)
            if expected:
                missing, unexpected = model.load_state_dict(source["state_dict"], strict=False)
                if set(missing) != expected or unexpected:
                    raise ValueError("the checkpoint must supply every weight except the new delay synapses")
            else:
                model.load_state_dict(source["state_dict"])
        else:
            initialize_router(model, source, source_cfg)
        if freeze_router:
            if model.prediction_memory is None:
                raise ValueError("--freeze-router needs a model with a trainable prediction bank")
            for name, parameter in model.named_parameters():
                if name in source["state_dict"] and not name.startswith(_BANK_PARAMETERS):
                    parameter.requires_grad_(False)
    else:
        source_geometry = {name: value for name, value in source["model_kwargs"].items() if name != "ctx"}
        geometry = {name: value for name, value in model_kwargs.items() if name != "ctx"}
        if SNNConfig(**source["cfg"]) != model.cfg or source_geometry != geometry:
            raise ValueError("the checkpoint and Transformer must use the same geometry")
        model.load_state_dict(source["state_dict"])
    options = source["args"]
    previous = source.get("initialization") or {}
    return {
        "checkpoint": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "training_targets": options["steps"] * options["bs"] * options["ctx"] + previous.get("training_targets", 0),
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
    parser.add_argument("--learn-context-keys", action="store_true",
                        help="learn reserved answers to memory queries through next-byte loss")
    parser.add_argument("--learn-packet-thresholds", action="store_true",
                        help="learn shared integer thresholds for content and steering groups")
    parser.add_argument("--output-reference", action="store_true",
                        help="subtract a shared spike reference before vocabulary neurons fire")
    parser.add_argument("--compact-delays", action="store_true",
                        help="allocate a history ring of its own length to each fixed axon group")
    parser.add_argument("--departure-coincidence", action="store_true",
                        help="route delayed pulses by coincidence with both departure and arrival steering")
    parser.add_argument("--delay-window", type=int, default=1,
                        help="number of adjacent delay taps per group available for spike matching")
    parser.add_argument("--delay-residual", action="store_true",
                        help="emit learned correction pulses from matching delays into the fixed split")
    parser.add_argument("--router-charge-bits", type=int, default=EventLMConfig.router_charge_bits,
                        help="fractional charge bits for local synapses and firing thresholds")
    parser.add_argument("--activity-inhibition", action="store_true",
                        help="raise content and steering thresholds with pooled incoming spike activity")
    parser.add_argument("--local-trace", action="store_true",
                        help="hold each channel's charge within its steering interval across bytes")
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
    parser.add_argument("--freeze-router", action="store_true", help="update only the newly added bank")
    parser.add_argument("--code-loss-weight", type=float, default=0.0, help="weight for the next-byte spike code loss")
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
    if args.freeze_router and args.initialize_from is None:
        parser.error("--freeze-router requires --initialize-from")
    if args.freeze_router and args.variant not in EVENT_ROUTES:
        parser.error("--freeze-router needs a spiking variant")
    if not math.isfinite(args.code_loss_weight) or args.code_loss_weight < 0:
        parser.error("--code-loss-weight must be finite and nonnegative")
    if args.code_loss_weight and (args.variant != "event_correction_code" or args.ctx < 2):
        parser.error("spike code loss needs the forecast bank and at least two input bytes")
    if (args.forecast_key_channels or args.match_forecast_key or args.learn_context_keys) and args.variant != "event_correction_code":
        parser.error("forecast keys need --variant event_correction_code")
    if (args.learn_packet_thresholds or args.pulse_budget is not None or args.output_reference or args.compact_delays
            or args.departure_coincidence or args.delay_window != 1 or args.delay_residual or args.activity_inhibition
            or args.local_trace or args.delay_filter
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
                model, model_kwargs, args.initialize_from.expanduser(), freeze_router=args.freeze_router,
            )
        model = model.to(device)
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
    for step in range(1, args.steps + 1):
        for group in optimizer.param_groups:
            group["lr"] = cosine_lr(step - 1, args.steps, args.warmup, args.lr)
        inputs, labels = random_batch(train, args.bs, args.ctx, generator)
        inputs = inputs.to(device)
        if pulse_counts is not None:
            pulse_counts.clear()
        forward_options = {} if pulse_counts is None else {"memory_pulses": pulse_counts}
        if args.code_loss_weight:
            counts, predicted, observed = model(inputs, return_prediction=True, **forward_options)
            if args.learn_context_keys:
                predicted = predicted[..., args.forecast_key_channels:]
                observed = observed[..., args.forecast_key_channels:]
            resolution = args.spike_resolution
            code_loss = F.smooth_l1_loss(predicted[:, :-1] / resolution, observed[:, 1:].detach() / resolution)
        else:
            counts = model(inputs, **forward_options)
        logits = scoring_logits(counts, event, args.spike_resolution)
        language_loss = F.cross_entropy(logits.flatten(0, 1), labels.to(device).flatten())
        loss = language_loss + args.code_loss_weight * code_loss if args.code_loss_weight else language_loss
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
                              **({"code_loss": code_loss.item()} if args.code_loss_weight else {})))
            print(f"{name} step {step} validation={metrics['bpb']:.4f} bpb "
                  f"accuracy={metrics['argmax_accuracy']:.4f} ({timer():.1f}s)", flush=True)

    val_score = evaluate(model, val, batches=args.final_eval_batches, **evaluate_options)
    test_score = None if args.validation_only else evaluate(model, test, batches=args.final_eval_batches,
                                                            **evaluate_options)
    args_record = dict(vars(args), data=str(args.data), results_dir=None if results_dir is None else str(results_dir),
                       initialize_from=None if args.initialize_from is None else str(args.initialize_from))
    config = asdict(cfg)
    if checkpoint is not None:
        payload = dict(format_version=1, model_type=type(model).__name__, task="event_lm", run_name=name,
                       cfg=config, model_kwargs=model_kwargs, args=args_record, initialization=initialization,
                       state_dict={key: value.detach().cpu() for key, value in model.state_dict().items()})
        with _atomic_output(checkpoint, "wb", overwrite=args.force) as out:
            torch.save(payload, out)
    record = dict(args=args_record, cfg=config, device=str(device), params=parameters,
                  trainable_params=trainable_params, initialization=initialization,
                  data_sha1=ENWIK8_SHA1, train_target_bytes=args.steps * args.bs * args.ctx, curve=curve,
                  val_bpb=val_score["bpb"], test_bpb=test_score["bpb"] if test_score else None,
                  val=val_score, test=test_score,
                  scoring=("normalized exponential of scaled output" if event else "softmax over logits"),
                  seconds=timer())
    destination = save_result("event_lm", name, record, overwrite=args.force, directory=results_dir)
    split, metric = ("validation", val_score) if test_score is None else ("test", test_score)
    print(f"{name} {split}={metric['bpb']:.4f} bpb accuracy={metric['argmax_accuracy']:.4f} ({destination})")


if __name__ == "__main__":
    main()
