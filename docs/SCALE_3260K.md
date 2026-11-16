# A matched 3.26M-parameter scale comparison

The scaled spiking model uses seven blocks, 384 residual channels and
187 expansion channels per block. It keeps activity inhibition and
held charge. The matched full-attention model uses four 256-channel
blocks, four heads and an MLP expansion ratio of 4.078125. Both arms
use seed 0, a 512-byte training context and 26,214,400 enwik8 targets.
The spike path uses the 310,000-pulse training target. Native geometry
verification used 31,692 bytes of persistent state for the held-charge
model and completed forward and backward on a full MPS training batch.
[Exact commands](HANDOFF.md#scale-to-326m) ·
[Geometry checks](../results/event_lm/scale_3260k/geometry_smoke.json)

| Model | Parameters | Streaming tensor state | enwik8 bits/byte | PG-19 bits/byte |
|---|---:|---:|---:|---:|
| Five-block held charge | 1,630,992 | 21,320 B | 2.00577 | **2.98022** |
| Full attention, original width | 1,630,848 | 3 MiB | **1.81449** | 3.11725 |
| Full attention, scaled width | 3,256,832 | 4 MiB | 1.86093 | 3.05929 |
| Seven-block held charge | 3,257,088 | 31,692 B | Training | Training |

The scaled full-attention reference scored 0.04645 bits per byte
higher on enwik8 and 0.05795 lower on PG-19 than the smaller
reference at the same exposure. The two full-attention references
remain useful comparisons. Both see up to 512 bytes per scoring
window. Their state column counts FP32 keys and values at that length.
The PG-19 score covers the same 16 books with a
temperature chosen on the separate validation books.
[Scaled training](../results/event_lm/scale_3260k/transformer_s0_scale_3260k.json) ·
[Scaled enwik8](../results/event_lm/scale_3260k/enwik8_transformer_s0.json) ·
[Scaled PG-19](../results/event_lm/scale_3260k/pg19_transformer_s0.json) ·
[Original depth study](DEPTH_INHIBITION.md)
