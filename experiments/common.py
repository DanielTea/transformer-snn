from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import tempfile
import time
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from snnformer import SNNConfig, SpikingLM  # noqa: E402
from snnformer.successor_lm import SuccessorDendriteLM  # noqa: E402
from snnformer.transformer_reference import TransformerLM  # noqa: E402

RESULTS = Path(__file__).resolve().parents[1] / "results"

# integer LIF, T=1 x D=4 (the "1x4" regime of E-SpikeFormer), saturating at 4 (spike unit 1)
_B = dict(neuron="ilif", levels=4, T=1, v_th=4.0)
_DENDRITE = dict(**_B, attn="dendritic", mlp="cdglu", causal_norm=True, position_embeddings=False)
VARIANTS: dict[str, SNNConfig] = {
    # references
    "ann": SNNConfig(neuron="ann", attn="softmax", mlp="mlp"),
    "transformer": SNNConfig(neuron="ann", attn="transformer"),
    "transformer_local": SNNConfig(neuron="ann", attn="transformer_local"),
    "transformer_event": SNNConfig(neuron="ann", attn="transformer_event"),
    "transformer_successor": SNNConfig(neuron="ann", attn="transformer_successor", successor_scaled_votes=True),
    "spk_softmax": SNNConfig(**_B, attn="softmax", mlp="mlp"),  # spiking, but softmax (not spike-driven)
    # spike-driven baselines (same backbone, same neurons)
    "spikformer": SNNConfig(**_B, attn="ssa", mlp="mlp"),  # SSA == SDSA-3 of Meta-SpikeFormer
    "ssa_norm": SNNConfig(**_B, attn="ssa_norm", mlp="mlp"),
    "sdt": SNNConfig(**_B, attn="sdsa", mlp="mlp"),  # Spike-driven Transformer V1
    "wta": SNNConfig(**_B, attn="wta", mlp="mlp", wta_k=1),  # WTA Spiking Transformer (2026), hard WTA
    "wta3": SNNConfig(**_B, attn="wta", mlp="mlp", wta_k=3),  # ... top-3
    # this work
    "csa": SNNConfig(**_B, attn="csa", mlp="mlp"),  # attention only
    "ours": SNNConfig(**_B, attn="csa", mlp="cdglu"),  # full model
    "ours_k4": SNNConfig(**_B, attn="csa", mlp="cdglu", csa_levels=4),  # graded attention spikes (CSA-4)
    "dendrite": SNNConfig(**_DENDRITE),
    "dendrite_relay": SNNConfig(**_DENDRITE, memory_relay=True),
    "dendrite_nodelay": SNNConfig(**_DENDRITE, memory_delay=0),
    "dendrite_event": replace(SNNConfig(**_DENDRITE), attn="dendritic_event", memory_delay=0),
    # ablations
    "ours_nocomp": SNNConfig(**_B, attn="csa", mlp="cdglu", csa_compete=False),
    "ours_nonorm": SNNConfig(**_B, attn="csa", mlp="cdglu", csa_normalize=False),
    "ours_binqk": SNNConfig(**_B, attn="csa", mlp="cdglu", qk_levels=1),
    "ours_nopbn": SNNConfig(**_B, attn="csa", mlp="cdglu", pre_bn=False),
    "spikformer_nopbn": SNNConfig(**_B, attn="ssa", mlp="mlp", pre_bn=False),  # original backbone
    "spikformer_cdglu": SNNConfig(**_B, attn="ssa", mlp="cdglu"),
}


TRANSFORMER_MIXERS = {"transformer", "transformer_local", "transformer_event", "transformer_successor"}


def add_common_args(p: argparse.ArgumentParser, *, vision: bool = False, bytes_only: bool = False) -> None:
    variants = [name for name, cfg in VARIANTS.items()
                if (not vision or cfg.attn not in TRANSFORMER_MIXERS | {"dendritic_event"})
                and (bytes_only or cfg.attn != "transformer_successor")]
    p.add_argument("--variant", default="ours", choices=sorted(variants))
    p.add_argument("--neuron", default=None, choices=["ilif", "lif", "ann", "relu", "relu1", "qrelu"])
    p.add_argument("--v-th", dest="v_th", type=float, default=None)
    p.add_argument("--sel-alpha", dest="sel_alpha", type=float, default=None)
    p.add_argument("--qk-levels", dest="qk_levels", type=int, default=None)
    p.add_argument("--softmax-temp", dest="softmax_temp_init", type=float, default=None)
    p.add_argument("--no-pre-bn", dest="pre_bn", action="store_const", const=False, default=None)
    p.add_argument("--levels", type=int, default=None)
    p.add_argument("--T", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--wd", type=float, default=0.05)
    p.add_argument("--tag", default="")


def add_sequence_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps", "auto"])
    p.add_argument("--strict-causal", dest="causal_norm", action="store_true", default=None,
                   help="use normalization that cannot see future positions during training")
    p.add_argument("--position", dest="position_embeddings", action=argparse.BooleanOptionalAction,
                   default=None, help="enable or disable learned absolute position embeddings")
    p.add_argument("--address-bits", type=int, help="Dendrite detectors per address (default: 4)")
    p.add_argument("--memory-block", type=int, help="Dendrite prefix block width (default: 32)")
    p.add_argument("--memory-delay", type=int, help="Dendrite lag on odd heads (unless set by the variant: 1)")
    p.add_argument("--local-window", type=int, help="local Transformer attention window")
    p.add_argument("--event-width", type=int, help="spike memory width inside the hybrid Transformer")
    p.add_argument("--mlp-ratio", type=float, help="feed-forward expansion for parameter-matched comparisons")
    p.add_argument("--checkpoint", help="path to save the trained model weights and configuration")
    p.add_argument("--force", action="store_true", help="overwrite existing result or checkpoint files")


def build_cfg(args) -> SNNConfig:
    cfg = VARIANTS[args.variant]
    overrides = ("neuron", "levels", "T", "v_th", "sel_alpha", "qk_levels", "softmax_temp_init",
                 "pre_bn", "causal_norm", "position_embeddings", "address_bits", "memory_block",
                 "memory_delay", "local_window", "event_width", "mlp_ratio")
    over = {k: getattr(args, k) for k in overrides if getattr(args, k, None) is not None}
    if cfg.neuron == "ann":
        over.pop("neuron", None)
    return replace(cfg, **over)


def make_lm(cfg: SNNConfig, **kwargs):
    if cfg.attn == "transformer_successor":
        return SuccessorDendriteLM(cfg, **kwargs)
    cls = TransformerLM if cfg.attn in TRANSFORMER_MIXERS else SpikingLM
    return cls(cfg, **kwargs)


def setup(args) -> None:
    torch.set_num_threads(args.threads)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)


def resolve_device(requested: str) -> torch.device:
    available = {"cpu": True, "cuda": torch.cuda.is_available(), "mps": torch.backends.mps.is_available()}
    if requested == "auto":
        requested = next(d for d in ("cuda", "mps", "cpu") if available[d])
    if not available[requested]:
        raise ValueError(f"PyTorch cannot use {requested} on this machine")
    return torch.device(requested)


def checkpoint_path(args) -> Path | None:
    if args.checkpoint is None:
        return None
    path = Path(args.checkpoint).expanduser()
    if path.exists() and not args.force:
        raise ValueError(f"{path} already exists; choose a new checkpoint path or use --force")
    return path


def make_optimizer(model, lr, wd):
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        (decay if p.ndim >= 2 and "pos" not in n else no_decay).append(p)
    return torch.optim.AdamW(
        [{"params": decay, "weight_decay": wd}, {"params": no_decay, "weight_decay": 0.0}],
        lr=lr,
        betas=(0.9, 0.99),
    )


def cosine_lr(step: int, total: int, warmup: int, base: float) -> float:
    if step < warmup:
        return base * (step + 1) / warmup
    t = (step - warmup) / max(1, total - warmup)
    return base * (0.02 + 0.98 * 0.5 * (1 + math.cos(math.pi * t)))


@contextmanager
def _atomic_output(path: Path, mode: str, overwrite: bool):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=path.parent) as temp_dir:
        staged = Path(temp_dir) / path.name
        with staged.open(mode) as f:
            yield f
        if overwrite:
            staged.replace(path)
        else:
            os.link(staged, path)


def save_result(task: str, name: str, payload: dict, *, overwrite: bool = True) -> Path:
    path = RESULTS / task / f"{name}.json"
    with _atomic_output(path, "w", overwrite) as f:
        json.dump(payload, f, indent=1)
    return path


def save_checkpoint(path: Path, model, model_kwargs: dict, *, args, task: str, name: str) -> None:
    checkpoint = {
        "format_version": 1,
        "model_type": type(model).__name__,
        "cfg": model.cfg.to_dict(),
        "model_kwargs": model_kwargs,
        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "task": task,
        "run_name": name,
        "args": vars(args),
    }
    with _atomic_output(path, "wb", args.force) as f:
        torch.save(checkpoint, f)


def result_exists(task: str, name: str, force: bool) -> bool:
    path = RESULTS / task / f"{name}.json"
    if path.exists() and not force:
        print(f"skip {path.name} (exists; use --tag for a new result or --force to overwrite)")
        return True
    return False


class Timer:
    def __init__(self):
        self.t0 = time.time()

    def __call__(self) -> float:
        return time.time() - self.t0


def run_name(args, cfg) -> str:
    parts = [args.variant, f"{cfg.neuron}{cfg.spike_levels}", f"T{cfg.T}", f"s{args.seed}"]
    if args.tag:
        parts.append(args.tag)
    return "_".join(parts)


def env_threads() -> None:
    os.environ.setdefault("OMP_NUM_THREADS", "4")
