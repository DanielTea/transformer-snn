from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from snnformer import SNNConfig  # noqa: E402

RESULTS = Path(__file__).resolve().parents[1] / "results"

# integer LIF, T=1 x D=4 (the "1x4" regime of E-SpikeFormer), saturating at 4 (spike unit 1)
_B = dict(neuron="ilif", levels=4, T=1, v_th=4.0)
VARIANTS: dict[str, SNNConfig] = {
    # references
    "ann": SNNConfig(neuron="ann", attn="softmax", mlp="mlp"),
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
    # ablations
    "ours_nocomp": SNNConfig(**_B, attn="csa", mlp="cdglu", csa_compete=False),
    "ours_nonorm": SNNConfig(**_B, attn="csa", mlp="cdglu", csa_normalize=False),
    "ours_binqk": SNNConfig(**_B, attn="csa", mlp="cdglu", qk_levels=1),
    "ours_nopbn": SNNConfig(**_B, attn="csa", mlp="cdglu", pre_bn=False),
    "spikformer_nopbn": SNNConfig(**_B, attn="ssa", mlp="mlp", pre_bn=False),  # original backbone
    "spikformer_cdglu": SNNConfig(**_B, attn="ssa", mlp="cdglu"),
}


def add_common_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--variant", default="ours", choices=sorted(VARIANTS))
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


def build_cfg(args) -> SNNConfig:
    cfg = VARIANTS[args.variant]
    over = {k: getattr(args, k) for k in ("neuron", "levels", "T", "v_th", "sel_alpha", "qk_levels", "softmax_temp_init", "pre_bn") if getattr(args, k) is not None}
    if cfg.neuron == "ann":
        over.pop("neuron", None)
    return replace(cfg, **over)


def setup(args) -> None:
    torch.set_num_threads(args.threads)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)


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


def save_result(task: str, name: str, payload: dict) -> Path:
    out = RESULTS / task
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{name}.json"
    path.write_text(json.dumps(payload, indent=1))
    return path


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
