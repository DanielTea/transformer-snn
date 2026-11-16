# Spike differences across fixed delays

The delay filter adds a trainable correction to the coincident path of
the [five-block inhibited model](DEPTH_INHIBITION.md). It reuses packets
already in the axon ring. Each channel with an anchor delay $d>1$ uses the
shorter taps $1$ and $\lceil d/2\rceil$, with the shared tap used once at
$d=2$.

For channel $j$, the filter emits signed correction pulses from

$$
\sum_{\tau} w_{j,\tau}\bigl(p_{j,t-\tau}-p_{j,t-d}\bigr).
$$

The weights are quantized synaptic charges. The correction shifts the
anchor packet before it meets the steering spikes. The bypass still
carries the unmatched pulses from the original anchor. Equal counts at
all taps emit zero correction. Zero initialization reproduces every
parent output exactly, and the short taps fit inside the existing rings.
The variant shares the delayed coincidence path with the local held
charge and is evaluated independently of it.
[Decoder and integer export](../snnformer/event_lm.py) ·
[Related work on learnable spike delays](https://arxiv.org/abs/2306.17670)

## Paired continuation

Both arms started from the same inhibited seed-0 checkpoint, reset
AdamW, and used the same seed-610 windows for 4,915,200 additional
enwik8 targets. They used a 0.0002 peak learning rate, 48 warmup
steps, 1,200 optimizer updates, and a 310,000-pulse training target.
The filter adds 1,012 parameters and keeps the same 17,620-byte
streaming state. At initialization the integer CPU output matched the
parent for all 256 candidates on every byte of a 4,096-byte enwik8
validation sample. The added weights received a nonzero MPS gradient
on a batch of 512 enwik8 training targets.
[Initialization audit](../results/event_lm/delay_filter/warm_identity.json) ·
[Control training record](../results/event_lm/delay_filter/event_correction_code_s610_warm_delay_control.json) ·
[Filter training record](../results/event_lm/delay_filter/event_correction_code_s610_warm_delay_filter.json)

| Model | enwik8 bits/byte | PG-19 bits/byte | Native pulses/byte |
|---|---:|---:|---:|
| Inhibited parent before continuation | 2.07523 | 3.02885 | 303,469 |
| Continued control | 2.05743 | **3.01929** | 301,279 |
| Continued with delay filter | **2.05695** | 3.01999 | 301,781 |

The two continuations differ by less than 0.001 bits per byte on
either corpus, in opposite directions. The filter's final validation
loss was 2.10387 bits per byte and the control's was 2.10272.
Zeroing only the learned filter weights raised loss by 0.00026 bits
per byte on 65,536 enwik8 validation targets. Both continuations
improved on their parent with the additional training. The paired
comparison provides no consistent advantage for the filter.
[enwik8](../results/event_lm/delay_filter/enwik8_warm_pair.json) ·
[PG-19](../results/event_lm/delay_filter/pg19_warm_pair.json) ·
[Native audit](../results/event_lm/delay_filter/native_warm_pair.json) ·
[Zero-filter intervention](../results/event_lm/delay_filter/probe_warm_filter.json)

The [handoff](HANDOFF.md#train-and-continue) gives the exact replay
command and the parent checkpoint location.
