"""A conventional causal Transformer for comparisons at the same training budget."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import SNNConfig
from .ops import OpCounter


@dataclass
class LocalAttentionState:
    """Rotary keys and values stored in a fixed-width ring."""

    keys: torch.Tensor
    values: torch.Tensor


@dataclass
class LocalTransformerState:
    """One ring per layer and the number of tokens already consumed."""

    layers: list[LocalAttentionState | None]
    length: int = 0


def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    half = cos.shape[-1]
    first, second = x[..., :half], x[..., half : 2 * half]
    rotated = (first * cos - second * sin, first * sin + second * cos)
    return torch.cat((*rotated, x[..., 2 * half :]), dim=-1)


def _sliding_window_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, window: int,
) -> torch.Tensor:
    """Apply exact causal SDPA on overlapping blocks of at most 2 * window keys."""
    batch, heads, length, width = q.shape
    if length <= window:
        return F.scaled_dot_product_attention(q, k, v, is_causal=True)

    first = F.scaled_dot_product_attention(
        q[:, :, :window], k[:, :, :window], v[:, :, :window], is_causal=True,
    )
    pad = (-length) % window
    blocks = (length + pad) // window - 1

    # Queries in each subsequent block see only keys from that block or its
    # predecessor. Pack the blocks into the batch for the SDPA kernels.
    q_blocks = F.pad(q[:, :, window:], (0, 0, 0, pad))
    q_blocks = q_blocks.unflatten(2, (blocks, window)).transpose(1, 2)
    q_blocks = q_blocks.reshape(batch * blocks, heads, window, width)

    def key_blocks(x):
        padded = F.pad(x, (0, 0, 0, pad))
        windows = padded.unfold(2, 2 * window, window).permute(0, 2, 1, 4, 3)
        return windows.reshape(batch * blocks, heads, 2 * window, width)

    queries = torch.arange(window, device=q.device)[:, None]
    keys = torch.arange(-window, window, device=q.device)
    distance = queries - keys
    visible = (distance >= 0) & (distance < window)
    rest = F.scaled_dot_product_attention(q_blocks, key_blocks(k), key_blocks(v), attn_mask=visible)
    rest = rest.reshape(batch, blocks, heads, window, width).transpose(1, 2).flatten(2, 3)
    return torch.cat((first, rest), dim=2)[:, :, :length]


class _CausalAttention(nn.Module):
    def __init__(self, dim: int, heads: int, window: int | None) -> None:
        super().__init__()
        self.heads = heads
        self.head_dim = dim // heads
        self.window = window
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)

    def _project(self, x, rope):
        batch, length, _ = x.shape
        qkv = self.qkv(x).reshape(batch, length, 3, self.heads, self.head_dim)
        q, k, v = (t.transpose(1, 2) for t in qkv.unbind(dim=2))
        if rope is not None:
            cos, sin = (t.to(q.dtype) for t in rope)
            q, k = _apply_rope(q, cos, sin), _apply_rope(k, cos, sin)
        return q, k, v

    def forward(self, x: torch.Tensor, rope: tuple[torch.Tensor, torch.Tensor] | None) -> torch.Tensor:
        batch, length, dim = x.shape
        q, k, v = self._project(x, rope)
        if self.window is None:
            mixed = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            mixed = _sliding_window_attention(q, k, v, self.window)
        if (counter := OpCounter.active) is not None:
            # Both dense contractions for each visible query/key pair.
            window = min(length, self.window) if self.window is not None else length
            pairs = batch * self.heads * (window * (window + 1) // 2 + (length - window) * window)
            counter.mac += 2 * pairs * self.head_dim
            counter.nonlinear += 2 * pairs  # exp and division in softmax
        return self.out(mixed.transpose(1, 2).reshape(batch, length, dim))

    def step(self, x, rope, state, position):
        assert self.window is not None
        q, k, v = self._project(x, rope)
        shape = (x.shape[0], self.heads, self.window, self.head_dim)
        if state is None:
            state = LocalAttentionState(k.new_empty(shape), v.new_empty(shape))
        elif state.keys.shape != shape or state.keys.device != k.device or state.keys.dtype != k.dtype:
            raise ValueError("streaming state must keep the same batch size, window, device and dtype")

        write = position % self.window
        state.keys[:, :, write : write + 1].copy_(k)
        state.values[:, :, write : write + 1].copy_(v)
        visible = min(position + 1, self.window)
        # Once the ring is full, its permutation of time positions leaves
        # attention unchanged. Each key already includes its rotary phase.
        mixed = F.scaled_dot_product_attention(q, state.keys[:, :, :visible], state.values[:, :, :visible])
        return self.out(mixed.transpose(1, 2).reshape_as(x)), state


class _TransformerBlock(nn.Module):
    def __init__(self, dim: int, heads: int, hidden: int, window: int | None) -> None:
        super().__init__()
        self.attn_norm = nn.LayerNorm(dim)
        self.attn = _CausalAttention(dim, heads, window)
        self.mlp_norm = nn.LayerNorm(dim)
        self.mlp_in = nn.Linear(dim, 2 * hidden, bias=False)
        self.mlp_out = nn.Linear(hidden, dim, bias=False)

    def _mix(self, x: torch.Tensor, rope: tuple[torch.Tensor, torch.Tensor] | None) -> torch.Tensor:
        return self.attn(x, rope)

    def _feed_forward(self, x):
        gate, value = self.mlp_in(self.mlp_norm(x)).chunk(2, dim=-1)
        return x + self.mlp_out(F.silu(gate) * value)

    def forward(self, x: torch.Tensor, rope: tuple[torch.Tensor, torch.Tensor] | None) -> torch.Tensor:
        return self._feed_forward(x + self._mix(self.attn_norm(x), rope))

    def step(self, x, rope, state, position):
        mixed, state = self.attn.step(self.attn_norm(x), rope, state, position)
        return self._feed_forward(x + mixed), state


class _EventTransformerBlock(_TransformerBlock):
    def __init__(self, dim: int, heads: int, hidden: int, window: int, cfg: SNNConfig) -> None:
        super().__init__(dim, heads, hidden, window)
        from .dendritic_event import DendriticEventAttention

        if cfg.event_width is None:
            self.memory_in = nn.Identity()
            self.memory_out = nn.Identity()
            self.memory = DendriticEventAttention(dim, heads, cfg, causal=True)
            self.memory_gain = nn.Parameter(torch.tensor(0.25))
        else:
            width = cfg.event_width
            if width < 1 or width % heads:
                raise ValueError("event_width must be positive and divisible by heads")
            self.memory_in = nn.Linear(dim, width, bias=False)
            self.memory = DendriticEventAttention(width, heads, cfg, causal=True)
            self.memory_out = nn.Linear(width, dim, bias=False)
            # The standardized memory branch emits a much larger initial
            # residual than the small-initialization attention projection.
            nn.init.normal_(self.memory_out.weight, std=0.003)
            self.memory_gain = nn.Parameter(torch.tensor(1.0))

    def _mix(self, x: torch.Tensor, rope: tuple[torch.Tensor, torch.Tensor] | None) -> torch.Tensor:
        memory = self.memory(self.memory_in(x).unsqueeze(0)).squeeze(0)
        return self.attn(x, rope) + self.memory_gain * self.memory_out(memory)


class TransformerLM(nn.Module):
    """Decoder-only pre-LayerNorm Transformer with SwiGLU and tied embeddings.

    Shares the SpikingLM constructor and [batch, tokens, vocab] output shape.
    Here ``cfg.position_embeddings`` enables RoPE instead of absolute learned
    embeddings, so ``ctx`` is the nominal training length and imposes no size
    limit. ``cfg.mlp_ratio`` controls the parameter budget; the SwiGLU hidden
    width is two-thirds the width of a two-matrix MLP at the same budget.

    The local, successor and event variants use windowed SDPA. The event
    variant adds a spiking memory to the same attention residual in each block.
    """

    def __init__(self, cfg: SNNConfig, vocab=256, dim=128, depth=4, heads=4, ctx=128, *, local_windows=None) -> None:
        super().__init__()
        if heads < 1 or dim < 1 or dim % heads:
            raise ValueError("dim must be positive and divisible by the number of heads")
        if depth < 1 or cfg.mlp_ratio <= 0:
            raise ValueError("depth and mlp_ratio must be positive")

        self.cfg = cfg
        self.ctx = ctx
        self.rotary_dim = 2 * ((dim // heads) // 2) if cfg.position_embeddings else 0
        hidden = max(1, round(2 * cfg.mlp_ratio * dim / 3))
        self.tok = nn.Embedding(vocab, dim)
        window = cfg.local_window if cfg.attn in ("transformer_local", "transformer_event", "transformer_successor") else None
        if window is not None and window < 1:
            raise ValueError("local_window must be positive")
        if local_windows is not None and (
            cfg.attn != "transformer_local" or len(local_windows) != depth or min(local_windows) < 1
        ):
            raise ValueError("local_windows needs one positive width per local Transformer layer")
        windows = (window,) * depth if local_windows is None else local_windows
        if cfg.attn == "transformer_event":
            memory_cfg = replace(cfg, neuron="ilif", levels=4, v_th=4.0, T=1, causal_norm=True)
            self.blocks = nn.ModuleList(
                _EventTransformerBlock(dim, heads, hidden, window, memory_cfg) for _ in range(depth)
            )
        else:
            self.blocks = nn.ModuleList(_TransformerBlock(dim, heads, hidden, width) for width in windows)
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab, bias=False)
        self._init_weights(self.tok)
        self.head.weight = self.tok.weight
        residual_std = 0.02 / math.sqrt(2 * depth)
        for block in self.blocks:
            # Keep the memory module's own spike-specific initialization.
            block.attn.apply(self._init_weights)
            self._init_weights(block.mlp_in)
            nn.init.normal_(block.attn.out.weight, std=residual_std)
            nn.init.normal_(block.mlp_out.weight, std=residual_std)

    @torch.no_grad()
    def initialize_wider_feed_forward(self, prior: SNNConfig, previous: dict[str, torch.Tensor]) -> int:
        """Load the trained feed-forward units and start new units with silent output."""
        weights, width = self._wider_feed_forward_weights(prior, previous, len(self.blocks))
        self.load_state_dict(weights)
        return width

    def _wider_feed_forward_weights(self, prior, previous, retained_depth):
        if replace(prior, mlp_ratio=self.cfg.mlp_ratio) != self.cfg:
            raise ValueError("feed-forward growth needs the same Transformer backbone")
        weights = previous.copy()
        widths = set()
        for index, block in enumerate(self.blocks[:retained_depth]):
            incoming = f"blocks.{index}.mlp_in.weight"
            outgoing = f"blocks.{index}.mlp_out.weight"
            old_input, old_output = previous[incoming], previous[outgoing]
            if old_output.ndim != 2:
                raise ValueError(f"invalid SwiGLU output matrix: {outgoing}")
            width = old_output.shape[1]
            widths.add(width)
            if (width >= block.mlp_out.in_features or old_output.shape[0] != block.mlp_out.out_features
                    or old_input.shape != (2 * width, block.mlp_in.in_features)):
                raise ValueError("the checkpoint must have a narrower SwiGLU layer at the same residual width")
            new_input = block.mlp_in.weight.detach().clone()
            for target, source in zip(new_input.chunk(2), old_input.chunk(2), strict=True):
                target[:width].copy_(source)
            new_output = torch.zeros_like(block.mlp_out.weight)
            new_output[:, :width].copy_(old_output)
            weights[incoming], weights[outgoing] = new_input, new_output
        if len(widths) != 1:
            raise ValueError("the checkpoint must use the same feed-forward width in every block")
        return weights, widths.pop()

    @torch.no_grad()
    def initialize_deeper(self, prior: SNNConfig, previous: dict[str, torch.Tensor], prior_depth: int) -> int | None:
        """Copy existing layers and start each new residual branch at zero."""
        if not 0 < prior_depth < len(self.blocks) or replace(prior, mlp_ratio=self.cfg.mlp_ratio) != self.cfg:
            raise ValueError("depth growth needs the same Transformer backbone with additional blocks")
        if prior.mlp_ratio == self.cfg.mlp_ratio:
            weights, width = previous.copy(), None
        else:
            weights, width = self._wider_feed_forward_weights(prior, previous, prior_depth)

        new_blocks = self.blocks[prior_depth:]
        added = {f"blocks.{index}.{name}" for index, block in enumerate(new_blocks, prior_depth)
                 for name in block.state_dict()}
        if self.state_dict().keys() - added != previous.keys():
            raise ValueError("the checkpoint must supply all original Transformer weights")
        for index, block in enumerate(new_blocks, prior_depth):
            block.attn.out.weight.zero_()
            block.mlp_out.weight.zero_()
            weights.update({f"blocks.{index}.{name}": value for name, value in block.state_dict().items()})
        self.load_state_dict(weights)
        return width

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=0.02)

    def _rope(self, positions):
        if not self.rotary_dim:
            return None
        angles = torch.arange(0, self.rotary_dim, 2, device=positions.device, dtype=torch.float32)
        inv_freq = torch.exp(angles * (-math.log(10_000) / self.rotary_dim))
        phase = positions[:, None] * inv_freq
        return phase.cos(), phase.sin()

    def forward(self, idx: torch.Tensor, *, position_offset: int = 0) -> torch.Tensor:
        positions = torch.arange(position_offset, position_offset + idx.shape[1],
                                 device=idx.device, dtype=torch.float32)
        rope = self._rope(positions)
        x = self.tok(idx)
        for block in self.blocks:
            x = block(x, rope)
        return self.head(self.norm(x))

    @torch.no_grad()
    def step(
        self, idx: torch.Tensor, state: LocalTransformerState | None = None,
    ) -> tuple[torch.Tensor, LocalTransformerState]:
        """Consume one token per stream and return [batch, vocab] next-token logits.

        Evaluation mode and a local attention backbone are required. The state
        is updated in place. Start an independent stream by passing `None`.
        """
        if self.training or self.cfg.attn not in ("transformer_local", "transformer_successor") or idx.ndim != 1:
            raise ValueError("step requires eval mode, a local attention backbone, and input of shape [batch]")
        if state is None:
            state = LocalTransformerState([None] * len(self.blocks))
        elif len(state.layers) != len(self.blocks):
            raise ValueError("streaming state must have one entry per model layer")

        position = state.length
        rope = self._rope(torch.tensor([position], device=idx.device, dtype=torch.float32))
        x = self.tok(idx).unsqueeze(1)
        for layer, block in enumerate(self.blocks):
            x, state.layers[layer] = block.step(x, rope, state.layers[layer], position)
        state.length += 1
        return self.head(self.norm(x)).squeeze(1), state
