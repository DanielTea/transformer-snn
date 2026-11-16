# Redistributing inhibition across delay groups

The seven-block model uses one inhibitory pool per block.
It counts incoming pulses on both sign rails and raises
the firing thresholds of all four delay groups together.
The new circuit keeps that pool and lets pairs of groups
exchange threshold increments in response to the signed
spike pattern entering the block.

Let $b$ be the pooled spike count for a byte and $x$ its
incoming signed packet. Two signed relay neurons receive
integer synaptic charge from $x$. Their charge is bounded
by $b$ times the pool threshold, so each relay emits a signed
transfer $r_j$ in $[-b,b]$. The four groups receive these
inhibition counts.

$$
(b+r_0,\ b+r_1,\ b-r_1,\ b-r_0)
$$

The first group pairs with the last, and the two middle
groups form the other pair. Each pair retains a combined
count of $2b$. A group uses its own count to set both the
content and steering thresholds. The relay resets each
byte and its emitted pulses count toward the training pulse
budget. Divisive normalization also has recurrent circuit
formulations. [Decoder](../snnformer/event_lm.py) ·
[Heeger and Zemlianova](https://pubmed.ncbi.nlm.nih.gov/32843341/)

The transfer synapses start at zero. On 4,096 enwik8 validation
bytes the two models emitted identical vocabulary counts and
1,293,067,588 total pulses. Native CPU stepping and chunked
decoding matched the Metal forward pass. A next-byte loss
pass on 512 enwik8 training targets gave the new synapses
nonzero gradients in every block. The circuit adds 10,752
parameters to the 6,514,176-parameter parent and retains
19,536 bytes of persistent state.
[Initialization audit](../results/event_lm/inhibition_redistribution/warm_inhibition.json)

## Matched continuation

The run starts from the six-bit 49.15M checkpoint. It uses
seed 1801, 1,600 steps, 512-byte windows, batch size eight,
64 warmup steps, peak learning rate 0.0003, and the same
310,000-pulse target as the original-width control. The
native audit measures the transfer pulses. The validation
intervention zeros only the transfer synapses.
[Original-width control](EXPANSION_GROWTH.md#controlled-continuation)

```bash
PYTHON=.venv/bin/python bash experiments/run_v4_continuation.sh redistribute-inhibition
```

| Model at 55,705,600 targets | enwik8 bpb | PG-19 bpb | Raw PG-19 bpb | Native pulses per byte |
|---|---:|---:|---:|---:|
| Original-width held-charge SNN | 1.89057 | 2.84072 | 2.94263 | 309,545 |
| Redistribution of inhibition | 1.88938 | 2.83881 | 2.94020 | 308,919 |

The continuation improved loss by 0.00119 bits per byte on
enwik8 and 0.00192 on calibrated PG-19. It uses one seed.
On 65,536 enwik8 validation targets, zeroing the transfer synapses
increased loss from 1.82498 to 1.82604. Six of seven blocks emitted
transfer pulses on the native sample. The relay populations emitted
1.22 transfer pulses per byte across all seven blocks on average.
The forward rule conserves the sum for each pair of
groups on every byte.
[Training, held-out scores, and interventions](../results/event_lm/inhibition_redistribution)
