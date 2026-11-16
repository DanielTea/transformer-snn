# Scaling the spiking residual layers

The v9 selected-arrival SNN and full-attention reference each saw
121,241,600 enwik8 targets. They start this study at almost the
same parameter count. The selected-arrival circuit projects the
scheduled spikes admitted by steering into the existing residual
firing neurons. In the previous paired test its loss at predictions
17–32 was within 0.00001 bits per byte of the unmatched-arrival
circuit. The two widths in this study use the same selected-arrival
circuit. [v9 circuit comparison](FIXED_ARRIVALS.md)

The wide SNN grows the residual population from 768 to 1,408
neurons per block. Its four axon and held-charge populations and
the tagged prediction bank keep their previous geometry. The
eleven blocks and the input retain the trained synapses and start
the added residual neurons with zero incoming charge. Their
outgoing synapses can deliver gradients from the first update.
Both widths use 28,584 bytes of persistent tensors per stream.
Function-preserving growth has precedent. Here it is checked on
integer spike counts, network emissions, and stream state.
[Net2Net](https://arxiv.org/abs/1511.05641)

| Arm | Parameters | Continuation data |
|---|---:|---|
| 768-wide SNN | 14,066,688 | Expanded mixture |
| 1,408-wide SNN | 25,788,928 | Expanded mixture |
| 1,408-wide SNN | 25,788,928 | enwik8 only |
| Original full attention | 14,068,224 | Expanded mixture |
| Wider SwiGLU full attention | 25,790,976 | Expanded mixture |

The wide Transformer increases its SwiGLU population from 2,696
to 5,240 per block and keeps its 256-dimensional residual stream.
The added outputs start at zero. Metal reproduces the parent's
logits on the 4,096-byte enwik8 validation audit. The wide SNN
reproduces the parent's native integer outputs and total pulse
count on those bytes. Both have a gradient in the added units of
every block.
[Wide SNN warm audit](../results/event_lm/residual_scale/warm_snn.json) ·
[Wide Transformer warm audit](../results/event_lm/residual_scale/warm_transformer.json)

## Expanded training data

| Source | Eligible stored bytes | Documents | Share of continuation targets |
|---|---:|---:|---:|
| Canonical enwik8 train | 90,000,000 | 1 | 50% |
| English C4 train | 134,231,706 | 48,745 | 25% |
| PG-19 train | 68,196,337 | 105 | 25% |
| **Total** | **292,428,043** | | **100%** |

C4 uses the first accepted whole documents up to at least 64 MiB
from each of two pinned English training shards. PG-19 uses the
first accepted whole books up to at least 64 MiB in a fixed
candidate order. Its source manifest pins the object generations,
sizes, and published checksums. The mixture draws 513-byte
windows entirely inside individual documents, then predicts the
512 next bytes. Training records include the byte hash of the
ordered windows and the number of targets from each source.
[C4 corpus study](https://aclanthology.org/2021.emnlp-main.98/) ·
[PG-19 source](https://github.com/google-deepmind/pg19) ·
[Pinned sources](../results/event_lm/training_mixture/sources_v1.json) ·
[Prepared inventory](../results/event_lm/training_mixture/prepared_v1.json)

The added material was screened against both held-out enwik8
splits, the selected PG-19 calibration and test books, and the
entire scored byte-transfer suite including LAMBADA's prompts.
The screen used 386,405 exact 64-byte probes and 664,412
normalized 12-word probes. It rejected 44 C4 documents and 19
PG-19 candidates with matches. The probe lengths and strides
define the granularity of the screen. C4 and PG-19 are held-out
in-domain evaluations for the mixture-trained arms. The suite
continues to use the same pinned scoring inputs.
[Overlap counts and hashes](../results/event_lm/training_mixture/prepared_v1.json) ·
[Transfer protocol](BYTE_TRANSFER.md)

## Comparison

All arms add 9,830,400 targets in 2,400 updates of eight
512-byte training windows. They use seed 9631, 96 warmup
updates, and a peak learning rate of 0.00012. The SNNs use
the same 310,000-pulse soft budget. The four mixture arms
receive identical windows. The wide enwik8-only arm measures
the effect of the corpus choice at the same parameter count and
training exposure.
[Expected mixed-window receipt](../results/event_lm/training_mixture/windows_s9631.json) ·
[Study launcher](../experiments/run_v4_continuation.sh)

The primary comparison uses complete enwik8 test loss. At each
size, subtract the Transformer's loss from the SNN's loss. Then
subtract the smaller models' gap from the wider models' gap. A
negative change means the wide SNN closed part of the gap. The
leaderboard data also stratifies the gap by how long ago the
last matching eight-byte prefix occurred. The paired reset probe estimates the same change for predictions
1–32 and 17–32, with intervals resampled by window. The native
audit counts pulses in the added residual neurons. A fixed-weight
probe measures loss with those neurons silenced. PG-19 and the
five standard transfer evaluations measure the effect on other
corpora.
[Paired probe](../experiments/probe_early_context.py) ·
[Native pulse audit](../experiments/audit_spike_code.py) ·
[Fixed-weight probe](../experiments/probe_event_state.py)

## Fixed-weight validation of the enwik8 control

On 1,535 reset enwik8 validation windows, the trained wide
enwik8-only SNN scores 1.81145 bpb at predictions 17–32. Its
v9 parent scores 1.81698 with 9,830,400 fewer training targets.
Silencing the wide model's added neurons at fixed weights
raises loss by 0.00394 bpb. The paired 95% interval is
[+0.00095, +0.00687] bpb. The mute result measures how much
the wider model relies on its new neurons on this sample.
[Paired validation and interventions](../results/event_lm/residual_scale/enwik8_only_preview.json)

## Reproduce the data and launches

Prepare the canonical [enwik8 and PG-19 evaluation files](HANDOFF.md#datasets)
and the [five transfer datasets](BYTE_TRANSFER.md#reproduce) in
an active Python environment with the benchmark dependencies.

```bash
python experiments/prepare_byte_training_mixture.py \
  --sources data/training-v1/sources --output data/training-v1/prepared
cmp results/event_lm/training_mixture/prepared_v1.json data/training-v1/prepared/catalogue.json

study=checkpoints/residual-scale-study
for arm in enwik8-snn wide-snn snn wide-transformer transformer; do
  SNN_TRAIN_ONLY=1 bash experiments/run_v4_continuation.sh "residual-scale-$arm" "$study"
done
```

The launch expects the two verified v9 parents in
`checkpoints/trace-arrival-study`. The
[v9 handoff](V9_HANDOFF.md#restore-and-run) gives the restore
command and parent checksums. Omit `SNN_TRAIN_ONLY` to score
enwik8 and PG-19 and run the native audits. Start each mixed
replay from the exact prepared catalogue. Its checksum is
verified by [the replay command](../experiments/replay_event_run.py).
