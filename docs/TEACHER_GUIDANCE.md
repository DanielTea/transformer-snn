# Next-byte training with a frozen teacher

This experiment adds a training objective to the 6.51M
held-charge SNN. Its teacher is the 6.51M full-attention
checkpoint trained on 49,152,000 enwik8 targets. Both see the
same 512-byte windows and predict the same vocabulary. The
frozen teacher supplies its distribution at a position when
it assigns the observed next byte a higher probability than
the student does.

Let $p$ be the student's distribution from its signed output
spikes, $q$ the teacher distribution, and $y$ the observed
byte. At selected positions, the objective gives equal weight
to the observed-byte loss and the divergence from $q$ to $p$.
At other positions, it uses the observed-byte loss. Selection
and the teacher are detached during optimization. The
forecast-code objective and pulse budget carry over from
the matched continuation.
[Training code](../experiments/train_event_lm.py) ·
[Soft-target distillation](https://arxiv.org/abs/1503.02531)

The student trains on 6,553,600 additional observed bytes
with the windows, seed 1801, and optimizer settings of the
original-width control. This arm also uses the pretrained
Transformer's supervision and one extra forward pass per
training batch. Its checkpoint contains the teacher's digest
and exposure for reproducibility. The exported decoder uses
the student's integer spike architecture and 19,536 bytes of
persistent state.
[Equal-window SNN control](EXPANSION_GROWTH.md#controlled-continuation) ·
[Teacher checkpoint scores](../results/event_lm/scale_6510k_data)

On the third batch of the training smoke, the guidance rule
selected 58.0% of the targets. The checkpoint passed the
native parity audit on 4,096 enwik8 test bytes.
[Training smoke and native audit](../results/event_lm/teacher_guidance)

```bash
PYTHON=.venv/bin/python bash experiments/run_v4_continuation.sh teacher-guidance
```

| Model at 55,705,600 observed targets | enwik8 bpb | PG-19 bpb | Raw PG-19 bpb | Native pulses per byte |
|---|---:|---:|---:|---:|
| Original-width held-charge SNN | 1.89057 | 2.84072 | 2.94263 | 309,545 |
| Held-charge SNN with frozen teacher | 1.89537 | 2.84787 | 2.96148 | 308,918 |

The final measured training batch selected 60.6% of its targets.
Loss rose by 0.00480 bits per byte on enwik8 and by 0.00715 on
calibrated PG-19. The exported SNN passed integer count parity on
the 4,096-byte enwik8 test audit. The comparison uses one seed
and the teacher contributes additional training compute.
[Full training, held-out scores and native audit](../results/event_lm/teacher_guidance)
