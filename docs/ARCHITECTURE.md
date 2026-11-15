# ShuntFormer — design notes

ShuntFormer is a spike-driven transformer whose attention is built from two
classic cortical inhibition motifs instead of softmax:

* **lateral (subtractive) inhibition** — each query's selection threshold
  tracks the mean excitation it receives, so only keys that beat the
  competition are attended (a k-winner-take-all whose k adapts per query);
* **shunting (divisive) inhibition** — the attended values are divided by the
  number of winners, realised on neuromorphic hardware by scaling the output
  neuron's firing threshold, not by a division.

The same block works bidirectionally (vision) and causally (language), because
its output is bounded for every query and every sequence length.

## 1. Block

```
x  (real-valued residual / membrane-shortcut stream)
│
├─► BN ─► SN ─► W_q,W_k,W_v ─► BN ─► SN ─► Q, K, V     (integer spikes)
│                                         │
│                     CSA(Q, K, V)  ◄─────┘   (accumulates + compares only)
│                         │
│                  SN ─► W_o ─► BN ─────────────────────┐
x ◄──────────────────────────────────────────────── (+) ┘
│
├─► BN ─► SN ─► W_g ─► BN ─► H(·)  = g  (binary gate)
│           └─► W_u ─► BN ─► SN    = u  (integer)       CD-GLU
│                     g ⊙ u (masking) ─► W_o ─► BN ─────┐
x ◄──────────────────────────────────────────────── (+) ┘
```

*SN* = integer leaky integrate-and-fire neuron (I-LIF: one time step, up to
D = 4 spikes, emitted as D binary spikes over virtual sub-steps at inference).
Every weight layer receives spikes, so inference needs accumulates (AC) only;
every BN is folded into the adjacent weights or neuron thresholds.

## 2. Competitive Shunting Attention (CSA)

Per head, with spike tensors Q, K, V (N tokens, d channels):

| step | formula | spike-domain cost |
|---|---|---|
| coincidence scores | S_ij = q_i · k_j / √d | AC: each q-spike adds the stored k-counts (K is the KV cache) |
| lateral inhibition | θ_i = γ_h · q_i · k̄ / √d + b_h,  k̄ = mean key (prefix mean if causal) | AC, **O(N d)** — because mean_j S_ij = q_i · k̄ |
| competitive selection | a_ij = H(S_ij − θ_i)  (surrogate gradient) | 1 comparison per score |
| shunting normalisation | y_i = Σ_j a_ij v_j / max(1, n_i),  n_i = Σ_j a_ij | masked AC; the division is a threshold scaling of the output neuron by n_i |

`y_i` is the fraction of winning keys whose value neuron fired, bounded in
[0, 1] (in spike units) for every query, position and sequence length.

Why each part is needed (all measured, see README):

* **Without competition** (soft, score-weighted selection = normalised linear
  attention) the selectivity is limited by the score range of bounded spike
  vectors: associative recall collapses.
* **Without normalisation** the causal output grows with the number of
  winners, i.e. with position.
* **Spikformer's SSA** (QKᵀV·s) has neither; **SDSA** (Spike-driven
  Transformer) cannot select per query at all. Both fail associative recall.
* **Hard/top-K WTA** (Zhou et al. 2026) selects a fixed number of winners and
  needs a top-K comparator tree over all N keys; CSA's winner count adapts per
  query and its inhibition is one running mean.

## 3. The saturation problem of the membrane-shortcut stream (finding)

The Meta-SpikeFormer family keeps the residual stream real-valued and starts
every sub-block with a spiking neuron. The residual is an unbounded running
sum; an integer neuron saturates at D spikes. We found that once residual
channels drift into saturation, the superposed features they carry are
clipped (and their straight-through gradient is zero) — on associative recall
this alone keeps **every** spiking variant at chance, softmax included, while
the identical network with unbounded (but equally quantised) activations
solves the task. Controlled site-by-site test (all neurons unbounded except
one group):

| saturating neuron group | recall acc. |
|---|---|
| block-input neurons (read the residual) | 21.8 % |
| Q/K neurons | 21.0 % |
| V and attention-output neurons | 100 % |
| MLP neurons | 100 % |

Fix: batch-normalise the residual right before each block-input neuron
(`InputNeuron`). At inference this per-channel affine map folds into the
neuron's threshold and bias, so it costs nothing. Together with a neuron range
of 4 (I-LIF's "1 × 4" setting, spike unit 1) the spiking backbone learns
retrieval.

## 4. What is new and what is adopted

| component | status |
|---|---|
| mean-referenced, per-query adaptive threshold on the attention map (O(N d) inhibition; causal prefix-mean form) | **new** (closest: WTA Spiking Transformer 2026 — fixed K; Spike2Max/SpikedAttention — row-max; Shen et al. 2024 — attention shift) |
| normalisation by the number of binary winners, as the fix for causal spiking attention | **new use**; threshold-scaling-as-division itself is from SpikedAttention (NeurIPS 2024) |
| the combination in a causal LM trained from scratch | **new** |
| diagnosis of residual saturation + pre-neuron BN folded into thresholds | finding of this work (BN-before-SN itself is standard, e.g. tdBN) |
| coincidence-gated MLP (binary gate × spikes) | **adopted** from OST's SGFF (IJCAI 2024) / SGSAFormer's SGLU (2025); we use an integer value branch |
| integer-valued training / spike-driven inference (I-LIF, D=4) | adopted from SpikeYOLO / E-SpikeFormer |
| membrane-shortcut residual, BN instead of LN | adopted from Meta-SpikeFormer |

## 5. Inference equivalence

All weight layers are linear in their (integer) spike inputs, so expanding an
output of c spikes into c binary spikes over D sub-steps is exact. Inside CSA,
K and V are stored as integer spike counts (the KV cache); Q-spikes stream in
and accumulate scores; the binary mask a is computed by comparison and
everything after it is linear in V. CSA therefore does not suffer the
"temporal interaction mismatch" reported for integer-trained softmax-free
attention executed sub-step by sub-step without latching (PT-SSA, 2026).
