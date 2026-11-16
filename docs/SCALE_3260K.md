# A matched 3.26M-parameter scale comparison

The scaled spiking model uses seven blocks, 384 residual channels and
187 expansion channels per block. It keeps activity inhibition and
held charge. The matched full-attention model uses four 256-channel
blocks, four heads and an MLP expansion ratio of 4.078125. Both arms
use seed 0, a 512-byte training context and 26,214,400 enwik8 targets.
The spike path uses the 310,000-pulse training target. The seven-block
model uses 31,692 bytes of persistent state, within the 32 KiB budget.
[Exact commands](HANDOFF.md#scale-to-326m) ·
[Geometry checks](../results/event_lm/scale_3260k/geometry_smoke.json)

| Model | Parameters | Streaming tensor state | enwik8 bits/byte | PG-19 bits/byte |
|---|---:|---:|---:|---:|
| Five-block held charge | 1,630,992 | 21,320 B | 2.00577 | 2.98022 |
| Full attention, original width | 1,630,848 | 3 MiB | **1.81449** | 3.11725 |
| Full attention, scaled width | 3,256,832 | 4 MiB | 1.86093 | 3.05929 |
| Seven-block held charge | 3,257,088 | 31,692 B | 1.99402 | **2.89822** |

The larger SNN improved by 0.01175 bits per byte on enwik8 and
0.08200 on PG-19 relative to the five-block SNN at the same
exposure. It led the 3.26M Transformer by 0.16107 bits per byte
on calibrated PG-19 and 0.27603 at raw scale, with lower loss on
all 16 books. The 3.26M Transformer led by 0.13309 on enwik8.
Both size comparisons use one seed.

The full-attention references see up to 512 bytes per scoring
window. Their state counts FP32 keys and values at that length.
The seven-block native audit matched CPU and MPS vocabulary counts
over 4,096 enwik8 test bytes and counted 308,389 pulses per byte.
[Scaled training, test scores and audit](../results/event_lm/scale_3260k) ·
[Original depth study](DEPTH_INHIBITION.md) ·
[Checkpoint inventory](../models/spiking-byte-lm-v3/manifest.json)
