# Held charge checks its own read

Opponent steering extended the lifetime of some held charge while
leaving enwik8 loss essentially tied with its paired control. This
experiment makes the old charge verify that it is still relevant
when it is about to emit a read. The same signed correction spikes
that move the cell also drive inhibition of the read.
[Opponent-steering results](OPPONENT_STEERING.md#results)

Let $m$ be the previous charge and let $\Delta m$ be the smallest
signed correction that moves it into the new content interval. The
veto neuron receives the magnitude of this correction through a
nonnegative quantized synapse. Its spikes remove overlapping pulse
slots from the old charge's coincident read.

```mermaid
flowchart LR
    Charge["Old held charge"] --> Coincidence["Read coincidence"]
    Steering["Steering spikes"] --> Coincidence
    Steering --> Interval["Content interval"]
    Content["Content spikes"] --> Interval
    Charge --> Correction["Correction spikes"]
    Interval --> Correction
    Correction --> Store["Move held charge"]
    Correction --> Veto["Inhibit incompatible read"]
    Coincidence --> Read["Delivered read spikes"]
    Veto --> Read
```

$$
v = \operatorname{IF}(\gamma |\Delta m|)
\qquad
r = \operatorname{sgn}(m)\max\bigl(0,\min(|m|,s)-v\bigr).
$$

The new read still takes its value from $m$. A charge already in
the interval emits no correction and therefore no veto. The gain
is shared within each axon delay group. Five blocks add 20 gains
and zero streaming state. All gains start at zero, so the new
decoder's native vocabulary counts, pulse totals, and 21,320-byte
streaming footprint match the released checkpoint at initialization.
Counts matched on 4,096 real enwik8 validation bytes on both CPU
and MPS. Gains received nonzero next-byte gradients in every
block on 512 enwik8 training targets.
[Decoder](../snnformer/event_lm.py) ·
[Initialization check](../results/event_lm/trace_read_veto/warm_identity.json)

The paired test starts both arms at the v2 checkpoint, uses seed
711 and 6,553,600 additional training targets, and trains both
with the chronological held-charge gradient. The candidate enables
`--trace-read-veto` and uses the same 310,000-pulse training target.
Scoring covers complete enwik8 and PG-19 test inputs. A zero-gain
intervention isolates the effect of read inhibition after training.
[Matched training rule](TRACE_WRITE_CREDIT.md)

## Matched continuation results

| Model | enwik8 bits/byte | PG-19 bits/byte | Native pulses per byte |
|---|---:|---:|---:|
| Held charge with chronological gradient | 1.98691 | 2.99097 | 307,270 |
| Correction spikes inhibit incompatible reads | 1.98405 | 2.98763 | 307,412 |

The read-veto run scored 0.00286 bits per byte lower on enwik8 and
0.00334 lower on calibrated PG-19. Raw PG-19 loss rose from 3.10893
to 3.11449. The learned gains were zero or 1/64, and zeroing them
after training lowered the enwik8 validation probe loss slightly,
from 1.919539 to 1.919396. Attribution of the small held-out gains
to the trained veto remains uncertain. The candidate kept
21,320 bytes of streaming state and its integer
counts matched MPS, including chunked decoding in the native audit.
[enwik8 score](../results/event_lm/trace_read_veto/enwik8_read_veto_s711.json) ·
[PG-19 score](../results/event_lm/trace_read_veto/pg19_read_veto_s711.json) ·
[Native audit](../results/event_lm/trace_read_veto/native_read_veto_s711.json) ·
[Zero-gain probe](../results/event_lm/trace_read_veto/probe_read_veto_s711.json)
