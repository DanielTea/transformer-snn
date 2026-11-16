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
content and steering. If the common pool emits $h_t$ spikes, the new
threshold in charge ticks is

$$
\frac{\theta}{4}\max(4,\,b_g+h_t),
\qquad b_g\in\{2,3,4,5,6,7,8\}.
$$

$\theta=256$ router ticks at eight fractional bits. The quantized base
starts at four, so both populations start with the same threshold as
v6. A group can learn to use more of the ordinary pulse range under
inhibition or increase its threshold and emit fewer pulses. The floor
preserves the original maximum count per neuron and the axon layout.
The two arms use the same soft target of 310,000 generated pulses per
byte, including the unchanged shared inhibitory pool.
[Firing and routing implementation](../snnformer/event_lm.py)

## Matched continuation

| Model | Parameters | Persistent state per stream |
|---|---:|---:|
| v6 SNN continued | 9,960,960 | 29,674 B |
| Signed input into held charge | 9,965,074 | 29,674 B |
| Learned group thresholds | 9,961,004 | 29,674 B |
| Six-block full attention continued | 9,962,496 | 6 MiB at 512 bytes |

Both spiking changes preserve every vocabulary count and the total
pulse count on 4,096 real validation bytes at initialization. The
native CPU step and chunk paths agree with the Metal counts. All
eleven blocks have nonzero gradients to the new trace input synapses
on a 512-target training window. All 44 learned group thresholds
receive a gradient.
[Trace input warm start](../results/event_lm/trace_integration/warm_trace_integration.json) ·
[Group threshold warm start](../results/event_lm/trace_integration/warm_group_thresholds.json)

Each arm adds 9,830,400 enwik8 targets for 91,750,400 cumulative
targets. All use seed 5801, 2,400 updates of eight 512-byte windows,
96 warmup steps, and peak learning rate 0.0002. The two v6 parents
come from the v6 release archive. The evaluator scores the full enwik8
test split and 16 complete PG-19 books. It fits PG-19 temperatures on
eight separate books.
[Dataset and checkpoint restore](HANDOFF.md)

```bash
export PYTORCH_ENABLE_MPS_FALLBACK=0
python experiments/probe_lm_context.py --event-lm --device cpu --threads 4 \
  --checkpoint checkpoints/depth-growth-study/depth_s4603.pt checkpoints/depth-growth-study/transformer_s4603.pt \
  --data data/enwik8 --split val --histories 1 8 32 96 224 497 --tail-bytes 16 --tails 1536 \
  --batch-size 32 --json checkpoints/work/v6_validation_history.json
for study in trace-transport trace-group-thresholds trace-transport-control trace-transport-transformer; do
  bash experiments/run_v4_continuation.sh "$study"
done
```
