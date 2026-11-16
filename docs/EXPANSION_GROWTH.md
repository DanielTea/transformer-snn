# Growing content and steering neurons within each delay group

The wider residual stream added projection capacity with fixed
temporal state. This experiment grows the content and steering
populations from 187 to 256 neurons each. Each new content
channel adds its own delay slots and held-charge cell. The
seven residual populations stay at width 768.

The initializer places each original neuron at the start of its
delay group. Group widths change from 47, 47, 47, and 46 to four
groups of 64. It remaps the content and steering expansion rows
and the input columns of all four routed projections. The input
synapses of new neurons start at zero. Their outgoing synapses
start at their usual random values, which gives them gradients
through the routed projection.

Both widths use a 2,048-tick inhibitory threshold. The correction
bank keeps its independent 32-coordinate forecast geometry.
The added neurons use the same spike rules and the fixed taps
of their groups.
[Initialization and integer decoder](../snnformer/event_lm.py)

## Warm-start checks

Across 4,096 enwik8 validation bytes, the parent and expanded
SNN emitted identical vocabulary counts and 1,293,067,588 total
pulses each. The sample matched native stepping and chunks
and Metal batch execution. A real 512-target training batch
gave every block nonzero gradients for the new expansion rows.
Three subsequent full-batch Metal training steps completed.
[Warm expansion audit](../results/event_lm/expansion_8740k/warm_expansion.json) ·
[Training smoke](../results/event_lm/expansion_8740k/event_correction_code_s1801_expansion_8740k_smoke.json)

The Transformer grows its SwiGLU width from 1,756 to 2,481.
It preserves the original logits exactly on eight 512-byte
Metal validation windows. The added units' outgoing weights
received gradients in each block.
[Transformer warm-start audit](../results/event_lm/expansion_8740k/warm_expansion_transformer.json)

| Model | Parameters | Persistent state |
|---|---:|---:|
| Expanded held-charge SNN | 8,739,840 | 26,080 B |
| Grown full-attention Transformer | 8,740,352 | 4 MiB at 512 bytes |
| Original-width held-charge control | 6,514,176 | 19,536 B |

The expanded SNN stores 16,768 bytes of delay rings, 7,168 bytes
of local held charge, and 2,144 bytes of forecast bank state.
The integer footprint stays below 32 KiB per stream.

## Controlled continuation

All three arms start from the corresponding 49,152,000-target
checkpoint and see another 6,553,600 enwik8 targets. The
run uses seed 1801, 1,600 steps, batch size eight, 512-byte
windows, 64 warmup steps, and peak learning rate 0.0003.
The SNN arms share the six-bit charge grid, 16× firing
resolution, and a target of 310,000 pulses per byte.
[Parent results](GROWN_RESIDUAL.md#extend-training-exposure)

The 6.51M SNN controls for additional training exposure. The
8.74M Transformer matches the expanded SNN's parameter budget.
After training, zero the added expansion input synapses on a
validation sample to measure the effect of their spikes with
the trained weights held fixed. The probe records
`silence_added_expansion_bpb`.
[State probe](../experiments/probe_event_state.py)

| Model at 55,705,600 enwik8 training targets | enwik8 bpb | PG-19 bpb | Raw PG-19 bpb |
|---|---:|---:|---:|
| Expanded held-charge SNN | 1.88812 | 2.83753 | 2.94075 |
| Grown full-attention Transformer | 1.79731 | 3.01533 | 3.22392 |
| Original-width SNN | 1.89057 | 2.84072 | 2.94263 |

The expanded SNN led on all 16 PG-19 books. The Transformer
led enwik8 by 0.09081 bits per byte. The SNN emitted 309,457
pulses per byte and kept native parity across CPU step, CPU
chunks, and Metal. Silencing the added neurons on 65,536
enwik8 validation targets moved loss from 1.82737 to 1.82672.
Against the original-width training control, the expanded
model's loss was 0.00245 lower on enwik8 and 0.00319 lower
on calibrated PG-19. This comparison uses one seed.
[Score files and native audit](../results/event_lm/expansion_8740k)

```bash
PYTHON=.venv/bin/python bash experiments/run_v4_continuation.sh expansion-snn
PYTHON=.venv/bin/python bash experiments/run_v4_continuation.sh expansion-transformer
PYTHON=.venv/bin/python bash experiments/run_v4_continuation.sh expansion-control
```

## Training the added steering neurons

On the native 4,096-byte test audit, the added content neurons
emitted 577 pulses per byte. The added steering neurons emitted
zero pulses in every block. Their input weights were still
zero after 1,600 steps. The zero-current boundary in the
training rectifier passed zero gradient to those synapses.
[Per-population pulse counts](../results/event_lm/expansion_8740k/native_expanded_s1801.json)

The follow-up uses the right derivative of the rectifier at
zero current. The forward rule still emits the same pulses.
On the trained expanded checkpoint, changing the derivative
preserved all native counts on 4,096 validation bytes. On
512 training targets, new steering weights received gradients
in every block. The matched rerun changes this derivative and
repeats the original expansion from the same 49.15M checkpoint.
[Gradient diagnostic](../results/event_lm/expansion_8740k/steering_gradient_check.json)

```bash
PYTHON=.venv/bin/python bash experiments/run_v4_continuation.sh expansion-steering
```
