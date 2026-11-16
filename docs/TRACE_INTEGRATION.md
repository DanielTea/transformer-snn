# Integrating spikes into held charge

The v6 decoder keeps one signed charge per content neuron. The
current content and steering counts define an interval. The cell moves
only if its stored charge lies outside that interval. Two changes to
the local circuit can be tested independently from the v6 weights.
[v6 model card](../models/spiking-byte-lm-v6/README.md)

## Where the enwik8 gap opens

Each checkpoint predicted the same 24,576 enwik8 validation targets,
in 1,536 disjoint tails of 16 bytes. The experiment reset both models
before each tail's warmup. The first target in a tail saw the number
of bytes in the left column. Subsequent targets also saw the earlier
bytes of that tail.

| Warmup bytes | v6 SNN bpb | Full attention bpb | Gap |
|---:|---:|---:|---:|
| 1 | 2.29514 | 2.28945 | 0.00569 |
| 8 | 1.92237 | 1.89783 | 0.02454 |
| 32 | 1.85264 | 1.79542 | 0.05722 |
| 96 | 1.83901 | 1.77220 | 0.06681 |
| 224 | 1.83709 | 1.76479 | 0.07230 |
| 497 | 1.83623 | 1.76326 | 0.07297 |

By 32 bytes of warmup, the gap reaches 78% of its size at the
longest matched history. The next test targets how the local router
uses the recent spikes. The probe reports behavior on its fixed
validation tails.
[History plot](../results/figures/event_lm_validation_history.png) ·
[Paired validation record and offsets](../results/event_lm/trace_integration/v6_validation_history.json)

The continued SNN and Transformer show the same pattern on these
exact targets. After another 9.83M training targets the gaps at 1,
32, and 497 bytes of warmup are 0.00578, 0.05609, and 0.07232 bpb.
[Continued checkpoint history probe](../results/event_lm/trace_integration/continued_validation_history.json)

## Learn which spikes move the cell

Let $p_t$ be the current content count, $d_t$ the incoming fixed-delay
count on the same channel, and $s_t$ the steering count. Two learned
synapses feed a signed integrate-and-fire cell. Its output moves the
held charge before the existing interval corrects it.

$$
a_t=\operatorname{FIRE}(w_p p_t+w_d d_t), \qquad
m_t=\operatorname{clip}(m_{t-1}+a_t,\,[p_t-s_t,p_t+s_t]\cap[-C,C]).
$$

The read still takes coincident pulses from $m_{t-1}$, and the
correction emits only the pulses needed to return the moved charge to
the allowed interval. Counts from both the input neuron and the
interval correction enter the pulse budget. Both synapses start at
zero, so they can open during training while reproducing the v6
predictions and pulse counts at initialization.

For chunk training, let $A_t=\sum_{j\le t} a_j$. In the translated
coordinate $m_t-A_t$ the update is the original interval clamp with
both bounds shifted by $-A_t$. This gives the same update as one
byte at a time and reuses the parallel interval scan. The sum is
rebased between chunks before reaching the exact integer limit of
the working dtype. A correction sends the state gradient to the
limiting bound. The gradient also reaches the pulses integrated
since the last correction. The inference state is
the original Int32 charge. The temporary cumulative sum is used only
inside a chunk.
[Interval scan](../snnformer/correction_memory.py) ·
[Exact parallel scans for spiking resets in prior work](https://arxiv.org/abs/2603.13283)

## Learn where the inhibitory pool permits finer firing

The second arm keeps the original held-charge update. For each delay
group $g$, a single learned parameter sets the base threshold for both
content and steering. Given $Q$ subdivisions of the original threshold
and $h_t$ spikes from the common pool, the threshold in charge ticks is

$$
\frac{\theta}{Q}\max\!\left(Q,\,b_g+\frac{Q}{4}h_t\right),
\qquad b_g\in\{Q/2,\ldots,2Q\}.
$$

$\theta=256$ router ticks at eight fractional bits. The quantized base
starts at $Q$, producing the same firing thresholds as v6. A group
can learn to use more of the ordinary pulse range under inhibition or
increase its threshold and emit fewer pulses. The floor preserves
the maximum count per neuron and the axon layout. The coarse arm
uses $Q=4$, a 64-tick step. The finer arm uses $Q=256$, one step
per router charge tick. One pool spike raises both thresholds by
64 ticks in either arm. Both use the same soft target of 310,000
generated pulses per byte.
[Firing and routing implementation](../snnformer/event_lm.py)

## Matched continuation

| Model | Parameters | Persistent state per stream |
|---|---:|---:|
| v6 SNN continued | 9,960,960 | 29,674 B |
| Signed input into held charge | 9,965,074 | 29,674 B |
| Learned group thresholds, coarse or fine | 9,961,004 | 29,674 B |
| Six-block full attention continued | 9,962,496 | 6 MiB at 512 bytes |

Both spiking changes preserve every vocabulary count and the total
pulse count on 4,096 real validation bytes at initialization. The
native CPU step and chunk paths agree with the Metal counts. All
eleven blocks have nonzero gradients to the new trace input synapses
on a 512-target training window. All 44 learned group thresholds
receive a gradient.
[Trace input warm start](../results/event_lm/trace_integration/warm_trace_integration.json) ·
[Coarse threshold warm start](../results/event_lm/trace_integration/warm_group_thresholds.json) ·
[Fine threshold warm start](../results/event_lm/trace_integration/warm_fine_group_thresholds.json)

Each arm adds 9,830,400 enwik8 targets for 91,750,400 cumulative
targets. All use seed 5801, 2,400 updates of eight 512-byte windows,
96 warmup steps, and peak learning rate 0.0002. The two v6 parents
come from the v6 release archive. The evaluator scores the full enwik8
test split and 16 complete PG-19 books. It fits PG-19 temperatures on
eight separate books.
[Dataset and checkpoint restore](HANDOFF.md)

## Held-out scores

| Same exposure at 91.75M targets | enwik8 bpb | PG-19 bpb | Raw PG-19 bpb | Native pulses/byte |
|---|---:|---:|---:|---:|
| Continued v6 SNN | 1.82999 | **2.78446** | **2.88327** | 307,913 |
| Learned trace input spikes | 1.82956 | 2.78972 | 2.89110 | 309,390 |
| Coarse group thresholds | 1.82918 | 2.78697 | 2.88461 | 308,616 |
| One-tick group thresholds | 1.82929 | 2.79350 | 2.89740 | 308,741 |
| Continued full attention | **1.74475** | 2.90840 | 3.10229 | — |

The ordinary SNN and full attention gained 0.01289 and 0.01238 bpb
respectively on enwik8 from the extra training. Their enwik8 gap is
0.08524 bpb. The four SNNs lead on each of the 16 PG-19 books.
Trace input changed test enwik8 loss by −0.00043 bpb and PG-19 loss by
+0.00525 bpb relative to the ordinary SNN. The results use one matched
continuation seed.
[Full-corpus records](../results/event_lm/trace_integration)

All 44 coarse threshold weights moved during training. The final
master weights range from 0.99946 to 1.06302. Every value rounds to
four quanta with $Q=4$, and the fixed-threshold intervention reproduces
the same validation loss. These observations motivated the one-tick
threshold grid.
[Coarse threshold audit](../results/event_lm/trace_integration/native_thresholds_s5801.json) ·
[Fixed-threshold intervention](../results/event_lm/trace_integration/probe_thresholds_s5801.json)

The one-tick grid produces an integer offset in 42 of the 44 delay
groups. The active offsets raise the base threshold by one to 14
charge ticks. The remaining two groups use the v6 base of 256 ticks.
The fine model lowers enwik8 loss by 0.00070 bpb relative to the
ordinary SNN and raises calibrated PG-19 loss by 0.00903 bpb. The
enwik8 gap to full attention is 0.08454 bpb. On the trained fine
weights, restoring the common base raises validation loss by
0.00034 bpb on 65,536 targets. On the paired history tails, its loss
at 32 warmup bytes is 0.00083 bpb higher than the ordinary SNN.
[Learned threshold offsets](../results/figures/event_lm_fine_group_thresholds.png) ·
[Fine threshold ablation](../results/event_lm/trace_integration/probe_fine_thresholds_s5801.json) ·
[Fine threshold history probe](../results/event_lm/trace_integration/fine_thresholds_validation_history.json)

## What the trace-input circuit used

On 16,384 enwik8 validation bytes, all eleven input populations
together emitted 23.96 signed pulses per byte. Of those pulses,
42.1% entered a cell whose interval admitted its moved charge.
12.11% of delivered read pulses came from cells whose last interval
correction was at least eight bytes earlier. The ordinary control
had a corresponding fraction of 12.14%. The age is evaluated per
cell at the time of each read.
[Trace input profile](../results/event_lm/trace_integration/trace_ages_transport_s5801.json) ·
[Control profile](../results/event_lm/trace_integration/trace_ages_control_s5801.json)

On the paired 24,576 validation targets, trace inputs changed loss
by +0.00339 bpb at 32 bytes of warmup and +0.00195 at 497. The trained
input circuit is active, with a small validation effect. The gain
intervention below scales its input synapses with the other trained
weights fixed.
[Trained model history probe](../results/event_lm/trace_integration/transport_validation_history.json)

| Gain multiplier | Validation bpb | Native trace input pulses/byte |
|---:|---:|---:|
| 0 | 1.75999 | 0.00 |
| 1 | **1.75867** | 24.32 |
| 2 | 1.78973 | 94.09 |
| 4 | 1.81411 | 326.88 |
| 8 | 1.85760 | 848.93 |

The loss uses 65,536 enwik8 validation targets. The native input
counts use their first 4,096 input bytes. These are frozen-weight
interventions.
[Gain sweep](../results/event_lm/trace_integration/gain_trace_input_s5801.json)

## Select among arrivals in the existing axons

On 16,384 validation bytes, the continued v6 model supplied 474,943
nonzero held-charge queries across the groups with multiple available
taps. Each candidate exposes its signed spikes through the current
steering aperture. The closest of up to four valid taps reduced total
pulse mismatch by 34.0% relative to the normal fixed tap and by
10.5% relative to the best fixed tap for each group on this sample.
It beat the shortest tap on 179,732 active queries. This measures
agreement between spike packets on real text.
[Tap comparisons by delay group](../results/event_lm/trace_integration/tap_choices_control_s5801.json)

The next study uses mismatch counts to delay each candidate's
firing. The earliest release spike selects its own packet. The
shortest-tap arm uses the same correction conductances and the two
arms share the original axon rings and pulse training target.
[Circuit and matched results](TRACE_TAP_READ.md)

## Reproduce

The five trained endpoints restore under `checkpoints/trace-transport-study/`
from the v7 release archive. Their digests bind the saved weights to
this comparison. The commands below replay the runs from the two v6
parents in fresh output directories.
[Restore the checkpoints](HANDOFF.md#checkpoints) ·
[Checkpoint checksums](../results/event_lm/trace_integration_checkpoints.sha256)

```bash
export PYTORCH_ENABLE_MPS_FALLBACK=0
python experiments/probe_lm_context.py --event-lm --device cpu --threads 4 \
  --checkpoint checkpoints/depth-growth-study/depth_s4603.pt checkpoints/depth-growth-study/transformer_s4603.pt \
  --data data/enwik8 --split val --histories 1 8 32 96 224 497 --tail-bytes 16 --tails 1536 \
  --batch-size 32 --json checkpoints/work/v6_validation_history.json
for study in trace-transport trace-group-thresholds trace-fine-group-thresholds trace-transport-control trace-transport-transformer; do
  bash experiments/run_v4_continuation.sh "$study"
done
python experiments/probe_event_state.py --device mps --enwik8 data/enwik8 --targets 65536 \
  --chunk-size 512 --trace-input-gains 0 1 2 4 8 \
  --checkpoint checkpoints/work/trace-transport/transport_s5801.pt \
  --json checkpoints/work/trace_input_gain.json
python experiments/profile_charge_trace.py --enwik8 data/enwik8 --split validation --bytes 16384 \
  --compare-recent-taps --checkpoint checkpoints/work/trace-transport-control/control_s5801.pt \
  --json checkpoints/work/tap_choices_control.json
```
