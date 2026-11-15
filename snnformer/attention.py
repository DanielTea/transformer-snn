"""Spiking self-attention variants.

All variants share the same projection scaffold (Meta-SpikeFormer style,
membrane-shortcut residual):

    x (real residual) -> SN -> {W_q, W_k, W_v} -> BN -> SN -> Q, K, V (spikes)
    Y = mix(Q, K, V) -> SN -> W_o -> BN -> (added to residual)

and differ only in ``mix``:

* ``ssa``      Spikformer (Zhou et al., ICLR 2023): Y = (Q K^T) V * s.  No
               normalisation -- in the causal case |Y_t| grows linearly with t.
               (SDSA-3 of Meta-SpikeFormer is the same function computed as
               Q (K^T V).)
* ``ssa_norm`` SSA divided by the number of visible keys.  Not from a paper;
               added as a *stronger* baseline so that gains of CSA cannot be
               attributed to normalisation alone.
* ``sdsa``     Spike-driven self-attention (Yao et al., NeurIPS 2023):
               Y = Q * SN(sum_j K_j * V_j)   (linear in N).
* ``softmax``  softmax(Q K^T / sqrt(d)) V on spike Q/K/V: needs exp/div, so it
               is *not* spike-driven; reference only (and used by the ANN).
* ``wta``      Winner-take-all spiking attention (Zhou et al. 2026,
               arXiv:2604.11321): binary top-K mask over Q K^T (K=1: hard WTA),
               softmax surrogate gradient.  Closest prior art to CSA; needs a
               top-K comparator tree over all N keys per query.
* ``csa``      Competitive Shunting Attention (this work), see
               :class:`CompetitiveShuntingAttention`.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from .config import InputNeuron, SeqBN, SNNConfig, make_neuron
from .neurons import heaviside
from .ops import OpCounter


def _causal_mask(n: int, device) -> torch.Tensor:
    return torch.ones(n, n, dtype=torch.bool, device=device).tril()


def _visible_keys(n: int, causal: bool, device, dtype) -> torch.Tensor:
    """Number of keys each query can see, shape [n, 1]."""
    if causal:
        return torch.arange(1, n + 1, device=device, dtype=dtype).unsqueeze(-1)
    return torch.full((n, 1), float(n), device=device, dtype=dtype)


class SpikingAttentionBase(nn.Module):
    gain_init = 1.0

    def __init__(self, dim: int, heads: int, cfg: SNNConfig, causal: bool = False) -> None:
        super().__init__()
        assert dim % heads == 0
        self.dim, self.heads, self.dh = dim, heads, dim // heads
        self.cfg, self.causal = cfg, causal
        self.in_sn = InputNeuron(dim, cfg)
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.q_bn, self.k_bn, self.v_bn = SeqBN(dim), SeqBN(dim), SeqBN(dim)
        qk_lv = cfg.qk_levels if cfg.neuron in ("ilif", "lif") else None
        self.q_sn, self.k_sn = make_neuron(cfg, levels=qk_lv), make_neuron(cfg, levels=qk_lv)
        self.v_sn = make_neuron(cfg)
        # learnable scalar gain in front of the output neuron (initialised to each
        # paper's constant), so that no variant is handicapped by a fixed scale
        self.gain = nn.Parameter(torch.tensor(float(self.gain_init)))
        self.out_sn = make_neuron(cfg)
        self.proj = nn.Linear(dim, dim)
        self.proj_bn = SeqBN(dim)

    @property
    def qk_unit(self) -> float | None:
        return getattr(self.q_sn, "unit", None)

    def _heads(self, x: torch.Tensor) -> torch.Tensor:  # [T,B,N,C] -> [T,B,H,N,dh]
        T, B, N, _ = x.shape
        return x.view(T, B, N, self.heads, self.dh).transpose(2, 3)

    def mix(self, q, k, v):  # pragma: no cover - abstract
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T, B, N, C = x.shape
        s = self.in_sn(x)
        q, k, v = self.qkv(s).chunk(3, dim=-1)
        q = self._heads(self.q_sn(self.q_bn(q)))
        k = self._heads(self.k_sn(self.k_bn(k)))
        v = self._heads(self.v_sn(self.v_bn(v)))
        y = self.mix(q, k, v) * self.gain
        y = y.transpose(2, 3).reshape(T, B, N, C)
        return self.proj_bn(self.proj(self.out_sn(y)))


class SSA(SpikingAttentionBase):
    """Spikformer spiking self-attention (softmax-free, unnormalised)."""

    # Spikformer: scale 0.125 then attn_lif with threshold 0.5. Equivalent (and
    # spike-count compatible with the other variants): gain 0.25, threshold 1.
    gain_init = 0.25
    normalize = False

    def mix(self, q, k, v):
        attn = q @ k.transpose(-2, -1)
        if self.causal:
            attn = attn.masked_fill(~_causal_mask(attn.shape[-1], attn.device), 0.0)
        y = attn @ v
        if self.normalize:
            y = y / _visible_keys(q.shape[-2], self.causal, q.device, q.dtype)
        if (c := OpCounter.active) is not None:
            c.spike_matmul(q, k.transpose(-2, -1), self.qk_unit)
            c.spike_matmul(v.transpose(-2, -1), attn.transpose(-2, -1))  # v-spikes gather scores
        return y


class SSANorm(SSA):
    gain_init = 1.0
    normalize = True


class SDSA(SpikingAttentionBase):
    """Spike-driven self-attention (Spike-driven Transformer V1): Q * SN(sum_j K_j * V_j)."""

    gain_init = 1.0

    def __init__(self, dim, heads, cfg, causal=False):
        super().__init__(dim, heads, cfg, causal)
        self.kv_sn = make_neuron(cfg, v_th=0.5)  # "talking-heads" LIF, threshold 0.5

    def mix(self, q, k, v):
        kv = k * v
        kv = kv.cumsum(-2) if self.causal else kv.sum(-2, keepdim=True)
        mask = self.kv_sn(kv)
        if (c := OpCounter.active) is not None:
            c.elementwise(float(c.spike_count(k, self.qk_unit)))  # k*v is an AND, accumulated over tokens
            c.elementwise(float(c.spike_count(q, self.qk_unit)))  # q masked by kv
        return q * mask


class SoftmaxAttention(SpikingAttentionBase):
    """softmax(t * QK^T/sqrt(d))V with a learnable per-head inverse temperature t.

    Spike inputs, but exp/div make it non-spike-driven.  The temperature matters
    for spiking Q/K: their entries are bounded in [0, 1], so with a fixed
    1/sqrt(d) scale the logit gap between a matching and a random key is ~1 and
    the softmax cannot become selective."""

    def __init__(self, dim, heads, cfg, causal=False):
        super().__init__(dim, heads, cfg, causal)
        self.log_temp = nn.Parameter(torch.full((heads, 1, 1), math.log(cfg.softmax_temp_init)))

    def mix(self, q, k, v):
        attn = (q @ k.transpose(-2, -1)) * (self.log_temp.exp() / math.sqrt(self.dh))
        if self.causal:
            attn = attn.masked_fill(~_causal_mask(attn.shape[-1], attn.device), float("-inf"))
        attn = attn.softmax(-1)
        if (c := OpCounter.active) is not None:
            c.spike_matmul(q, k.transpose(-2, -1), self.qk_unit)
            n = attn.shape[-1]
            c.nonlinear += attn.numel() * 2  # exp + div per score
            if c.levels:  # real weights accumulated wherever v spikes
                c.ac += float(c.spike_count(v)) * n
            else:
                c.mac += attn.numel() * self.dh
        return attn @ v


class CompetitiveShuntingAttention(SpikingAttentionBase):
    r"""Competitive Shunting Attention (CSA) -- this work.

    For each head, with spike tensors Q, K, V:

    1. *Coincidence scores*  :math:`S_{ij} = q_i \cdot k_j / \sqrt{d}` --
       with spikes this is an AND + pop-count (accumulates only).
    2. *Lateral (feed-forward) inhibition*: each query's selection threshold is
       set by an inhibitory unit that tracks the mean excitation it receives,
       :math:`\theta_i = \gamma_h\, q_i \cdot \bar k / \sqrt{d} + b_h`, where
       :math:`\bar k` is the mean key (prefix mean when causal).  Because
       :math:`\operatorname{mean}_j S_{ij} = q_i\cdot\bar k/\sqrt d`, this
       costs O(N d), not O(N^2).
    3. *Competitive selection*: the attention map is itself a spike train,
       :math:`a_{ij} = H(S_{ij} - \theta_i)` (surrogate gradient) -- a key is
       attended iff its coincidence exceeds the inhibition: a k-WTA without
       softmax.
    4. *Shunting (divisive) normalisation*:
       :math:`y_i = \sum_j a_{ij} v_j / \max(1, n_i)`, :math:`n_i=\sum_j a_{ij}`.
       On neuromorphic hardware the division is not computed: the output
       neuron's firing threshold is scaled by the count n_i (shunting
       inhibition), so only accumulates and comparisons are used.

    y_i is the fraction of selected keys whose value-neuron fired -- bounded in
    [0, 1] for every query and sequence length, which is what makes the same
    block usable for bidirectional vision and causal language modelling.
    """

    def __init__(self, dim, heads, cfg, causal=False):
        super().__init__(dim, heads, cfg, causal)
        self.gamma = nn.Parameter(torch.ones(heads, 1, 1))
        self.bias = nn.Parameter(torch.zeros(heads, 1, 1))

    def mix(self, q, k, v):
        cfg = self.cfg
        n = q.shape[-2]
        r = 1.0 / math.sqrt(self.dh)
        scores = (q @ k.transpose(-2, -1)) * r  # [T,B,H,N,N]
        if self.causal:
            kbar = k.cumsum(-2) / _visible_keys(n, True, k.device, k.dtype)
        else:
            kbar = k.mean(-2, keepdim=True)
        mean_exc = (q * kbar).sum(-1, keepdim=True) * r  # [T,B,H,N,1]
        if cfg.csa_compete:
            theta = self.gamma * mean_exc + self.bias
            a = heaviside(scores - theta, cfg.sel_alpha)
        else:  # ablation: soft, score-weighted selection (normalised linear attention)
            a = scores
        if self.causal:
            a = a.masked_fill(~_causal_mask(n, a.device), 0.0)
        y = a @ v
        if cfg.csa_normalize:
            y = y / a.sum(-1, keepdim=True).clamp_min(1.0)
        else:  # ablation: raw counts (grows with #selected keys)
            y = y / math.sqrt(n)
        if (c := OpCounter.active) is not None:
            c.spike_matmul(q, k.transpose(-2, -1), self.qk_unit)  # coincidence scores
            c.elementwise(float(c.spike_count(q, self.qk_unit)))  # q . kbar (accumulate kbar where q spikes)
            c.elementwise(float(a.numel()))  # threshold comparisons
            c.spike_matmul(v.transpose(-2, -1), a.transpose(-2, -1))  # masked value accumulation
        return y


class WTAAttention(SpikingAttentionBase):
    """Top-K winner-take-all spiking attention (Zhou et al. 2026)."""

    def __init__(self, dim, heads, cfg, causal=False):
        super().__init__(dim, heads, cfg, causal)
        self.k_win = cfg.wta_k
        self.log_temp = nn.Parameter(torch.zeros(heads, 1, 1))  # surrogate softmax temperature

    def mix(self, q, k, v):
        n = q.shape[-2]
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.dh)
        visible = _causal_mask(n, q.device) if self.causal else torch.ones(n, n, dtype=torch.bool, device=q.device)
        masked = scores.masked_fill(~visible, float("-inf"))
        kk = min(self.k_win, n)
        idx = masked.topk(kk, dim=-1).indices
        hard = torch.zeros_like(scores).scatter_(-1, idx, 1.0) * visible  # early causal rows: < K keys
        hard = hard / hard.sum(-1, keepdim=True).clamp_min(1.0)
        soft = (masked * self.log_temp.exp()).softmax(-1)
        a = hard + soft - soft.detach()  # forward: binary top-K mask; backward: softmax gradient
        if (c := OpCounter.active) is not None:
            c.spike_matmul(q, k.transpose(-2, -1), self.qk_unit)
            c.elementwise(float(scores.numel()))  # comparator tree, ~1 comparison per score
            c.spike_matmul(v.transpose(-2, -1), (hard > 0).to(v.dtype).transpose(-2, -1))
        return a @ v


ATTENTION = {
    "csa": CompetitiveShuntingAttention,
    "ssa": SSA,
    "ssa_norm": SSANorm,
    "sdsa": SDSA,
    "softmax": SoftmaxAttention,
    "wta": WTAAttention,
}


def make_attention(dim: int, heads: int, cfg: SNNConfig, causal: bool) -> nn.Module:
    return ATTENTION[cfg.attn](dim, heads, cfg, causal)
