# Steer bypass pulses after the block first fires

The block's first residual spikes summarize the immediate paths,
fixed axons, and held-charge read. Those first spikes return through
the same steering synapses and can divert some of the immediate
bypass pulses to the coincidence path. The circuit uses one feedback
stage within each byte and a learned conductance per delay group.
The pair of immediate paths accounts for exactly the original content
pulses on each sign rail. Dendritic experiments have also measured
how a returning action potential can interact with a second input.
[Larkum, Zhu, and Sakmann](https://pubmed.ncbi.nlm.nih.gov/10192334/)

For the ordinary signed coincidence function $S(p,s)$, let
$u=p-S(p,s)$ be the immediate bypass. Let $C$ be the charge at the
block output and $z=\operatorname{fire}(C)$ its first spikes. The
steering synapses receive $z$. Their excitatory charge passes through
the learned gain and uses the block's pooled inhibitory threshold
to emit late steering spikes $\ell$. The redirected pulses are
$r=S(u,\ell)$. The final charge is

$$
C' = C + (W_{\mathrm{match}}-W_{\mathrm{bypass}})r.
$$

The block emits the signed correction $\operatorname{fire}(C')-z$
by subtracting the charge represented by the first spikes before its
second firing. Both the new steering spikes and the signed correction
count toward the pulse target. The axon rings and held charge get one
update from the original content and steering per byte. The seven
blocks share each of their steering synapses across the two stages.
The extra state is within the byte, so persistent storage is 19,536 B.
[Integer circuit](../snnformer/event_lm.py)

## Warm start and matched training

The 28 new conductances start at zero. Both models emitted exactly
1,288,006,522 pulses and the same vocabulary counts on 4,096 enwik8
validation bytes. Native CPU steps, CPU chunks and Metal agreed, and
all 28 parameters had nonzero gradients on 512 training targets.
The training comparison continues the 65.54M six-bit SNN using the
original-width control's seed 3307, 512-byte windows, 6,553,600
additional targets, and 310,000-pulse target.
[Warm-start audit](../results/event_lm/late_steering/warm_late_steering.json) ·
[Original-width control](GROWN_RESIDUAL.md#double-the-residual-stream-at-fixed-persistent-state)

```bash
export PYTORCH_ENABLE_MPS_FALLBACK=0
mkdir -p checkpoints/work
python experiments/audit_residual_growth.py --enwik8 data/enwik8 --device mps \
  --parent checkpoints/scale-6510k-next-data-study/trace_s2503.pt --late-steering \
  --json checkpoints/work/warm_late_steering.json
bash experiments/run_v4_continuation.sh late-steering
```

### Held-out scores

| Model at 72,089,600 targets | enwik8 bpb | PG-19 bpb | Raw PG-19 bpb | Native pulses per byte |
|---|---:|---:|---:|---:|
| Original-width SNN | 1.87259 | 2.81227 | 2.90738 | 309,048 |
| SNN with late steering | 1.87323 | 2.82081 | 2.92053 | 308,536 |
| Original-width full-attention Transformer | 1.77292 | 2.95995 | 3.16079 | — |

The circuit added 0.00064 bits per byte on enwik8 and 0.00854 on
calibrated PG-19 versus the matched SNN. Thirteen of the 28 gains
reached one quantized tick. The native audit measured 163 late
steering spikes and 175 signed correction spikes per byte across
the seven blocks. Zeroing the gains lowered the 65,536-target
enwik8 validation loss from 1.80002 to 1.79974. The held-out
comparison uses one seed.
[Both held-out corpora, native audit, and validation probe](../results/event_lm/late_steering)
