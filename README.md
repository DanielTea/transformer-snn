# ShuntFormer — a spike-driven transformer with competitive, shunting-normalised attention

ShuntFormer is a spiking transformer in the Spikformer / Meta-SpikeFormer
lineage. Its attention replaces softmax with two cortical inhibition motifs,
using only accumulates and comparisons:

* **Lateral inhibition.** Each query's firing threshold tracks the mean
  excitation it receives (`θ_i = γ·q_i·k̄ + b`, O(N·d)). Only keys that beat
  this threshold fire, so the number of winners adapts per query.
* **Shunting inhibition.** The attended values are divided by the number of
  winners. On neuromorphic hardware this is done by scaling the output
  neuron's threshold, not by a division.

Because the output is bounded for every query and every sequence length, the
same block runs bidirectionally (vision) and causally (language modelling).

Design notes: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).
How the residual-saturation failure was found:
[`docs/DIAGNOSTICS.md`](docs/DIAGNOSTICS.md).

<!-- RESULTS -->

## Repository layout

```
snnformer/            the library
  neurons.py          binary LIF and integer LIF (I-LIF) neurons with surrogate / STE gradients
  attention.py        CSA (ours) + baselines: Spikformer SSA, SSA-norm, Spike-driven SDSA,
                      WTA (Zhou et al. 2026), softmax-on-spikes
  mlp.py              coincidence-gated spiking MLP (CD-GLU) and the standard spiking MLP
  config.py           SNNConfig, neuron factory, residual-reading InputNeuron (pre-BN)
  models.py           SpikingViT (32x32 images) and SpikingLM (causal, byte level)
  ops.py              synaptic-operation counter and 45 nm energy estimate
experiments/
  common.py           the variant registry (every model compared in this README)
  recall.py           multi-query associative recall (MQAR)
  train_lm.py         enwik8 byte-level LM (90M/5M/5M split), bits per byte, learning curves
  train_vision.py     CIFAR-10
  summarize.py        results/*.json -> markdown tables;  plots.py -> results/figures
tests/                45 tests: causality, spike-driven inputs to every weight layer, CSA properties
results/              JSON for every run reported here (args, config, curves, energy)
```

## Reproduce

```bash
pip install -r requirements.txt
python -m pytest -q tests

cd experiments
python recall.py      --variant ours --steps 2000 --pairs 8 --keys 32 --queries 8 --lr 3e-3 --seed 0
python train_lm.py    --variant ours --steps 1500 --eval-every 250 --lr 2e-3      # needs enwik8 in /home/user/data
python train_vision.py --variant ours --dim 96 --depth 4 --heads 4 --epochs 5 --lr 2e-3   # needs CIFAR-10 python pickles
python summarize.py && python plots.py
```

`--variant` is one of `ours, csa, ours_nocomp, ours_nonorm, ours_binqk,
ours_nopbn, spikformer, spikformer_nopbn, spikformer_cdglu, ssa_norm, sdt, wta,
wta3, spk_softmax, ann` (see `experiments/common.py`). The `--neuron lif --T 4`
flags switch to binary multi-step LIF neurons.
