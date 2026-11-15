# Why spiking transformers fail at associative recall — the diagnostic trail

All runs: multi-query associative recall (MQAR), 8 key–value pairs drawn from
32 keys / 32 values, 8 queries, 2-layer causal model, d = 64, 2 heads, 2 000
steps, AdamW lr 3e-3, batch 64, single seed unless noted. Chance ≈ 3 %; picking
any value present in the context gives ≈ 12–20 %.

| # | configuration | acc. | conclusion |
|---|---|---|---|
| 1 | ANN reference (GELU, softmax), same scaffold | 100 % | task is learnable in budget |
| 2 | every spiking variant (CSA, SSA, SSA-norm, softmax-on-spikes), I-LIF D=4, range 1 | 19–23 % | spiking backbone fails regardless of attention |
| 3 | position embedding init 0.02 → N(0,1) | no change | (bug fixed anyway: a 0.02 signal is below the quantisation step) |
| 4 | softmax-on-spikes with 16 levels / range 4 / lr 1e-3 / learnable temperature / 6 000 steps / binary LIF T=4 | 18–21 % | not resolution, not lr, not training length, not the T-vs-D regime |
| 5 | softmax-on-spikes with **ReLU** neurons | 100 % | the attention rule is fine; neurons are the problem |
| 6 | CSA with ReLU Q/K/V | 90 % | CSA can do retrieval |
| 7 | neurons quantised to steps of 1/4, **no upper clamp** | softmax 100 %, CSA 99.96 % | quantisation is harmless |
| 8 | neurons clamped to [0, 1], **no quantisation** | softmax 19 %, CSA 19 % | **saturation is what breaks retrieval** |
| 9 | site test: unbounded everywhere except one group (clamped) | input neurons 21.8 %, Q/K 21 %, V/out 100 %, MLP 100 % | saturation of residual-reading and Q/K neurons |
| 10 | + BN on the residual before each input neuron (folded into thresholds at inference) | still ≈ 20 % | necessary, not sufficient — integer outputs were normalised by the range, shrinking Q·K |
| 11 | integer neurons in physical units (range 4, spike unit 1) + pre-BN, softmax-on-spikes | D=4: 81 %, D=8: 100 % | spiking backbone now learns retrieval |
| 12 | same backbone: **CSA (integer Q/K)** | D=4: 98.1 %, D=8: 99.7 % | spike-driven CSA solves it |
| 13 | same backbone: Spikformer SSA / SDSA | 19 % / 15 % | prior spike-driven attentions cannot select |
| 14 | CSA with binary Q/K | 20 % | binary Q/K too weak; use integer Q/K (KV cache stores counts) |

Take-aways

1. **Saturation, not quantisation, is the enemy of retrieval in spiking
   transformers.** Residual features are superposed additively; a saturating
   neuron clips the superposition and zeroes its straight-through gradient.
2. The membrane-shortcut backbone needs the residual normalised before each
   block-input neuron, and integer neurons need enough range (here 4 spike
   units at unit resolution — exactly I-LIF's 1×4 setting).
3. Even on a fixed backbone, softmax-free spike-driven attentions without
   per-query competition (SSA, SDSA) stay at chance, while CSA reaches ≈ 98 %.

Raw logs of these runs were kept out of the repository; the final sweeps in
`results/recall/` reproduce rows 1, 11–14 with three seeds.
