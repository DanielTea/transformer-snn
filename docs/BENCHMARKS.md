# Dendrite benchmark report

The [current spiking byte model leaderboards](LEADERBOARD.md) compare
integer spiking decoders with matched local and full-attention
Transformers on complete enwik8 and PG-19 test streams.

See [Temporal Spiking LM](TEMPORAL_SNN.md) for the separate byte-language
model with token-time spiking neurons, its reset ablation and the enwik8
training protocol.

See [Paired-pulse membrane memory](PULSE_MEMORY.md) for repeated recall,
enwik8 and the held-out state interventions.

Dendrite studies bounded memory in models with two different backbones.
The event model uses an integer-spike backbone to bind values to earlier
keys. The hybrid successor model adds verified byte suffixes and a spike
readout to a conventional local Transformer. The successor loss gains
measure that hybrid architecture. The spiking-backbone language model
scored worse than the conventional controls.
The [architecture report](DENDRITE.md) describes the mechanisms, early
experiments and ablations. The [reproduction guide](REPRODUCE_DENDRITE.md)
lists the commands and source files for the results below.
The [3.22M-parameter study](DENDRITE_SCALING.md) compares a plain
Transformer with continuous and spike readouts using the same cache.

## Recall across intervening tokens

![Recall accuracy for fixed-delay Dendrite, full RoPE attention and event Dendrite at matched sequence length, with a separate anchored extrapolation probe](../results/figures/dendrite_recall_distance.png)

The event model learned to capture a spike address and hold it until a
later spike commits the value. After training with zero to four
distractors between keys and values, it averaged **99.95%** accuracy at
a gap of 128 across three seeds. Full RoPE attention averaged **6.14%**
under the same 1,200-step protocol. The task uses distinct token groups
for keys, values and distractors, and presents one pending binding at a
time. Matched tests use four bindings and four queries in 524-token
sequences. [Matched runs](../results/recall_distance)

In a separate seed-0 probe, each value and query stayed at the same
position in an 8,204-token sequence. Moving the keys to set gaps of 0,
4, 128, 512 and 2,048 distractors gave **512 of 512 correct queries at
each gap**. This probe measures distance on the same typed task.
[Anchored probe](../results/recall_distance/dendrite_event_s0_anchored_extrapolation.json)

## Language modeling with the spiking event backbone

The event model scored **4.469 bits per byte** on 307,200 sampled enwik8
test bytes at context 256 in one seed. It uses one neuron step per byte,
quantized spike counts, and persistent address memory across bytes. The
conventional local Transformer control averaged **2.431 bits per byte**
over three seeds on the same test windows and byte budget. The typed
recall results above concern a different task.
[Language modeling controls and setup](DENDRITE.md#event-memory-in-language-models)

## Language modeling with the hybrid successor cache

![Bits per byte for the local Transformer, frozen-backbone successor readout and jointly trained successor model on enwik8 and PG-19 excerpts](../results/figures/dendrite_successor_loss.png)

The hybrid model adds a cache to the output of a conventional local
Transformer. Each suffix order uses a fixed table to track the byte
that last followed a suffix. The model verifies the exact suffix
before reading a record, and a learned readout emits graded spike
counts to weight its contribution.

| Continuous test stream | Target bytes | Local Transformer | Hybrid with frozen backbone | Hybrid trained from scratch |
|---|---:|---:|---:|---:|
| enwik8 test split | 4,999,999 | 2.457 | 2.074 | **2.025** |
| Eight PG-19 test excerpts | 1,048,568 | 3.416 | 3.052 | **2.965** |

Scores are bits per byte, with lower values better. The from-scratch
comparison uses one seed, the same 4.096 million enwik8 training bytes
and optimizer schedule, and 357,696 versus 357,937 learned parameters.
The continuous evaluator preserves memory across processing chunks.
Each PG-19 excerpt starts from an empty state, and neither readout was
fit on the books. The [excerpt manifest](../results/lm_successor/pg19_test_excerpt_manifest.json)
records the selection rule, byte offsets and checksums.
[Frozen-backbone scores](../results/lm_successor/scaled_v2_continuous.json) ·
[From-scratch scores](../results/lm_successor/end_to_end_continuous.json)

Using only cache records whose successors were observed at least 94
bytes earlier lowered enwik8 loss to **2.212** on the frozen backbone.
Those records are beyond the local Transformer's receptive field.
The test measures reuse of repeated text in compact byte models.
General language capabilities and hardware energy were outside the
evaluation. The results do not establish frontier language modeling
performance.

## Readout calibration

![Held-out loss for fixed, quantized and continuous cache readouts on the same frozen Transformer](../results/figures/dendrite_successor_calibration.png)

On a separate 524,288-byte enwik8 sample, suffix-scaled spike increments
scored **2.097 bpb** across three readout seeds. A fixed mixture fitted
on validation scored **2.119** and a learned continuous readout scored
**2.094**. These scores use the same frozen backbone and the same cache
capacity. With the independently trained context-256 backbone, the fixed
mixture scored 2.090 and the scaled readout scored 2.094.
[Calibration runs](../results/lm_successor) ·
[Architecture and ablations](DENDRITE.md)
