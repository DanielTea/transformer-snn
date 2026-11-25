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
    def __init__(self, dim: int, heads: int, window: int | None, ages: int = 0, age_rate: float = 1.0) -> None:
        super().__init__()
        self.heads = heads
        self.head_dim = dim // heads
        self.window = window
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)
        # With ages every head adds a learned offset to a key by its distance from the query, as the heads of the
        # spike decoder's state reads do. The offsets start level and move ``age_rate`` times as fast as a weight.
        self.age = nn.Parameter(torch.zeros(heads, ages)) if ages else None
        self.age_rate = age_rate

    def _offsets(self, length):
        distance = torch.arange(length, device=self.age.device)
        distance = distance[:, None] - distance
        # A distance beyond the table takes its last offset. Selecting along one axis keeps the way back a scattered
        # sum: an indexed assignment sorts its indices on a GPU, and waits for it in every layer.
        chosen = distance.clamp(0, self.age.shape[1] - 1).reshape(-1)
        offsets = (self.age * self.age_rate).index_select(1, chosen).view(-1, length, length)
        return offsets.masked_fill(distance < 0, -torch.inf)

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
        if self.age is not None:
            mixed = F.scaled_dot_product_attention(q, k, v, attn_mask=self._offsets(length).to(q.dtype))
        elif self.window is None:
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


# The feed-forward units by name: whether a unit is gated, and whether it opens smoothly.
_UNITS = {"swiglu": (True, True), "reglu": (True, False), "gelu": (False, True), "relu": (False, False),
          "rampglu": (True, False), "shuntglu": (True, False), "sigglu": (True, True), "bilinear": (True, True)}
# Further controls of the gated unit, by what the gate does with its drive. Three gates are bounded and never pass
# more than their value: one opens in proportion and is wide open at a drive of one ("rampglu"), one is a
# conductance that lets in g / (1 + g) of the value ("shuntglu"), as the gates of a stack's gated cells do, and
# one is the logistic gate, half open at rest ("sigglu"). The last is no gate at all: the product of the two
# drives, of either sign ("bilinear").
_GATES = {"rampglu": lambda drive: drive.clamp(0, 1), "shuntglu": lambda drive: F.relu(drive) / (1 + F.relu(drive)),
          "sigglu": torch.sigmoid, "bilinear": lambda drive: drive}


class _PoolNorm(nn.Module):
    """A control of LayerNorm: the stream divided by its mean absolute activity and a learned rest, with a gain per
    channel and no centring, as a stack's inhibitory pool raises the threshold of every neuron with the summed
    activity of the stream it hears."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.rest = nn.Parameter(torch.full((), 1e-3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weight * x / (self.rest.abs() + x.abs().mean(-1, keepdim=True))


class _CentredPoolNorm(_PoolNorm):
    """The pool control with centring: the stream less its mean, divided by its mean absolute deviation and a
    learned rest, with a gain per channel, as a balanced inhibition that subtracts the stream's mean activity would
    let a stack's pool divide."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        centred = x - x.mean(-1, keepdim=True)
        return self.weight * centred / (self.rest.abs() + centred.abs().mean(-1, keepdim=True))


def _norm(dim: int, kind: str) -> nn.Module:
    return {"layer": nn.LayerNorm, "pool": _PoolNorm, "cpool": _CentredPoolNorm}[kind](dim)


class _TransformerBlock(nn.Module):
    def __init__(self, dim: int, heads: int, hidden: int, window: int | None, *, plain: int = 0, smooth: bool = True,
                 ages: int = 0, age_rate: float = 1.0, gate: str | None = None, norm: str = "layer") -> None:
        super().__init__()
        self.attn_norm = _norm(dim, norm)
        self.attn = _CausalAttention(dim, heads, window, ages, age_rate)
        self.mlp_norm = _norm(dim, norm)
        # ``plain`` units each have one row of input weights. A gated unit has two rows: one opens the unit and
        # the other is what it passes on. A unit opens smoothly, or not at all below a threshold at zero.
        self.plain, self.smooth = plain, smooth
        # The name of a control gate in ``_GATES``, in place of the gate that ``smooth`` picks.
        self.gate = gate
        self.mlp_in = nn.Linear(dim, plain or 2 * hidden, bias=False)
        self.mlp_out = nn.Linear(plain or hidden, dim, bias=False)

    def _mix(self, x: torch.Tensor, rope: tuple[torch.Tensor, torch.Tensor] | None) -> torch.Tensor:
        return self.attn(x, rope)

    def _feed_forward(self, x):
        if self.plain:
            drive = self.mlp_in(self.mlp_norm(x))
            return x + self.mlp_out(F.gelu(drive) if self.smooth else F.relu(drive))
        gate, value = self.mlp_in(self.mlp_norm(x)).chunk(2, dim=-1)
        if self.gate:
            return x + self.mlp_out(_GATES[self.gate](gate) * value)
        return x + self.mlp_out((F.silu(gate) if self.smooth else F.relu(gate)) * value)

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
        if cfg.feed_forward not in _UNITS or cfg.age_offsets < 0:
            raise ValueError("the feed-forward units are swiglu, reglu, gelu or relu, or a control gate (rampglu, "
                             "shuntglu, sigglu, bilinear), and the rate of the age offsets is not negative")
        if cfg.norm not in ("layer", "pool", "cpool"):
            raise ValueError("the reference's normalization is a LayerNorm or, as a control, a pool or a centred pool")
        if (cfg.feed_forward != "swiglu" or cfg.age_offsets or cfg.untied_output or cfg.norm != "layer") \
                and cfg.attn != "transformer":
            raise ValueError("other units, age offsets, an output layer of its own and a pool in place of the norm "
                             "are controls of the full Transformer")
        # Offsets by age replace the rotary phases: a head then addresses by content and by distance in two terms.
        self.rotary_dim = 2 * ((dim // heads) // 2) if cfg.position_embeddings and not cfg.age_offsets else 0
        hidden = max(1, round(2 * cfg.mlp_ratio * dim / 3))
        gated, smooth = _UNITS[cfg.feed_forward]
        options = dict(plain=0 if gated else max(1, round(cfg.mlp_ratio * dim)), smooth=smooth,
                       ages=ctx if cfg.age_offsets else 0, age_rate=cfg.age_offsets or 1.0,
                       gate=cfg.feed_forward if cfg.feed_forward in _GATES else None, norm=cfg.norm)
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
            self.blocks = nn.ModuleList(_TransformerBlock(dim, heads, hidden, width, **options) for width in windows)
        self.norm = _norm(dim, cfg.norm)
        self.head = nn.Linear(dim, vocab, bias=False)
        self._init_weights(self.tok)
        if cfg.untied_output:
            # A control: the output layer has weights of its own, as the output synapses of a stack have.
            self._init_weights(self.head)
        else:
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
