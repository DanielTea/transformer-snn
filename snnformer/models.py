"""Vision and language models built from the same spiking transformer block."""

from __future__ import annotations

import torch
import torch.nn as nn

from .attention import make_attention
from .config import InputNeuron, SNNConfig, make_neuron
from .mlp import make_mlp


class Block(nn.Module):
    """Membrane-shortcut block: the residual stream stays real-valued and every
    sub-block starts with a spiking neuron, so all weight layers see spikes."""

    def __init__(self, dim: int, heads: int, cfg: SNNConfig, causal: bool) -> None:
        super().__init__()
        self.attn = make_attention(dim, heads, cfg, causal)
        self.mlp = make_mlp(dim, cfg)

    def forward(self, x):
        x = x + self.attn(x)
        return x + self.mlp(x)


class _Conv(nn.Module):
    """SN -> Conv2d -> BN on [T, B, C, H, W] tensors."""

    def __init__(self, cin, cout, cfg, stride=1, groups=1, spike_in=True):
        super().__init__()
        self.sn = make_neuron(cfg) if spike_in else nn.Identity()
        self.conv = nn.Conv2d(cin, cout, 3, stride, 1, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(cout)
        if not spike_in:
            self.conv.real_input = True  # first layer: encodes the analog image (MACs)

    def forward(self, x):
        T, B = x.shape[:2]
        y = self.bn(self.conv(self.sn(x).flatten(0, 1)))
        return y.view(T, B, *y.shape[1:])


class SpikingViT(nn.Module):
    """Small spiking vision transformer for 32x32 inputs.

    Conv tokenizer (32x32 -> 8x8 = 64 tokens, Spikformer-SPS-like but lighter),
    spiking relative-position conv, ``depth`` transformer blocks, and a
    spike-count-pooled linear head. Direct (analog) input encoding, as in the
    Spikformer family."""

    def __init__(self, cfg: SNNConfig, num_classes=10, dim=128, depth=4, heads=4, in_ch=3):
        super().__init__()
        self.cfg = cfg
        self.stem = nn.Sequential(
            _Conv(in_ch, dim // 4, cfg, spike_in=False),
            _Conv(dim // 4, dim // 2, cfg, stride=2),
            _Conv(dim // 2, dim, cfg, stride=2),
        )
        self.rpe = _Conv(dim, dim, cfg)
        self.blocks = nn.ModuleList(Block(dim, heads, cfg, causal=False) for _ in range(depth))
        self.head_sn = InputNeuron(dim, cfg)
        self.head = nn.Linear(dim, num_classes)

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        x = img.unsqueeze(0).expand(self.cfg.T, *img.shape)  # [T,B,C,H,W]
        x = self.stem(x)
        x = x + self.rpe(x)
        x = x.flatten(3).transpose(2, 3)  # [T,B,N,C]
        for blk in self.blocks:
            x = blk(x)
        # linear head on each token's spikes, then average (== head(mean(spikes)))
        return self.head(self.head_sn(x)).mean((0, 2))


class SpikingLM(nn.Module):
    """Causal spiking transformer language model (byte/char level)."""

    def __init__(self, cfg: SNNConfig, vocab=256, dim=128, depth=4, heads=4, ctx=128):
        super().__init__()
        self.cfg = cfg
        self.tok = nn.Embedding(vocab, dim)  # N(0, 1) init
        # Same scale as the token embedding: with the usual 0.02 init the position
        # signal is far below the first neuron's quantisation step and is erased.
        self.pos = nn.Parameter(torch.randn(ctx, dim))
        self.blocks = nn.ModuleList(Block(dim, heads, cfg, causal=True) for _ in range(depth))
        self.head_sn = InputNeuron(dim, cfg)
        self.head = nn.Linear(dim, vocab)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        B, N = idx.shape
        x = self.tok(idx) + self.pos[:N]
        x = x.unsqueeze(0).expand(self.cfg.T, B, N, x.shape[-1])
        for blk in self.blocks:
            x = blk(x)
        return self.head(self.head_sn(x)).mean(0)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
