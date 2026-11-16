# Confirming held-charge writes with delayed spikes

In the 49.15M-target SNN, 87.9% of held-charge read pulses on
16,384 enwik8 validation bytes came from charge changed within
the previous seven bytes. A charged cell with zero steering
was overwritten on 96.8% of such channel steps. At zero steering,
the usual tolerance interval is the current content alone.
[Charge ages and writes](../results/event_lm/scale_6510k_data/trace_ages_s1109.json) ·
[Effect of resetting held charge](GROWN_RESIDUAL.md#extend-training-exposure)

The new circuit asks the same fixed axon that serves the delayed
bypass whether current content should replace held charge. For
current content $p$, arriving delayed content $d$, and steering
count $s$, the base write interval is $[p-s,p+s]$. A neuron
receives the signed difference $d-p$, through a conductance $g$
shared by the delay group, and emits signed count $e$.

$$
e = \operatorname{fire}(g(d-p)),\qquad
I = [p-s + \min(0,e),\ p+s + \max(0,e)].
$$

The interval is clipped to the firing cap. The memory moves to
the nearest point of $I$. Disagreement extends only the side
toward the delayed packet, giving existing charge a chance to
survive. When both packets agree, the ordinary interval applies.
At full conductance the interval contains both packets within
the original tolerance. The read still uses pre-write charge and
the current steering spikes.

The seven blocks have four trainable conductances each. They
start silent and reuse the existing 19,536 bytes of stream
state. The sign of the incoming difference selects the rail
before firing. That preserves the surrogate gradient when
the emitted count is zero.
[Circuit and native step](../snnformer/event_lm.py) ·
[Related opponent-steering experiment](OPPONENT_STEERING.md)

The warm start preserved all output counts and the total
1,293,067,588 pulses on 4,096 enwik8 validation bytes. Native
step and chunked output matched Metal. All 28 conductances
received gradients in a next-byte loss pass on 512 enwik8
training targets. The trainer completed three full Metal
batches with the added rule.
[Warm audit](../results/event_lm/trace_write_confirm/warm_write_confirm.json) ·
[Training smoke](../results/event_lm/trace_write_confirm/event_correction_code_s1801_trace_write_confirm_smoke.json)

## Matched check

This arm uses the 49.15M six-bit checkpoint, seed 1801 and
6,553,600 additional enwik8 targets. It matches the original-width
arm of the expansion study, including the 310,000-pulse target,
with 28 additional learned conductances. The native audit
records the learned conductances and their pulse counts. The
validation probe zeros them while retaining all other weights.

```bash
PYTORCH_ENABLE_MPS_FALLBACK=0 python experiments/audit_residual_growth.py \
  --enwik8 data/enwik8 --parent checkpoints/scale-6510k-data-study/trace_s1109.pt \
  --trace-write-confirm --device mps --json checkpoints/work/warm_write_confirm.json
PYTHON=.venv/bin/python bash experiments/run_v4_continuation.sh trace-write-confirm
```

[Matched original-width control](EXPANSION_GROWTH.md#controlled-continuation)

| Model at 55,705,600 enwik8 training targets | enwik8 bpb | PG-19 bpb | Raw PG-19 bpb | Native pulses per byte |
|---|---:|---:|---:|---:|
| Original write interval | 1.89057 | 2.84072 | 2.94263 | 309,545 |
| Delayed write confirmation | 1.89069 | 2.83683 | 2.93664 | 309,026 |

Delayed confirmation improved calibrated PG-19 loss by 0.00389
bits per byte and increased enwik8 loss by 0.00012. Eleven
conductances in blocks four through seven learned a nonzero
value. They emitted 87 confirmation pulses per byte on the
native test audit. On 16,384 validation bytes, the fraction
of held-charge pulses read at least 32 bytes after the last
write rose from 4.46% to 4.64%. Zeroing the learned conductances
after training reduced the 65,536-target validation loss from
1.82714 to 1.82695. The held-out comparison uses one seed.
[Training and held-out records](../results/event_lm/trace_write_confirm) ·
[Control's age profile](../results/event_lm/expansion_8740k/trace_ages_control_s1801.json)
