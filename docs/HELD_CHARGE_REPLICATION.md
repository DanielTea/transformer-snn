# Held-charge replication

The five-block spiking decoder and the four-block full-attention
Transformer train from initialization on matched 512-byte enwik8
windows. Each seed sees 26,214,400 targets. The SNN uses 1,630,992
parameters, 21,320 bytes of stream state, and the 310,000-pulse
training target. The Transformer uses 1,630,848 parameters and
3 MiB of FP32 key and value state at 512 bytes.
[Depth controls](DEPTH_INHIBITION.md) · [Training and scoring commands](HANDOFF.md#repeat-the-held-charge-comparison)

| Seed | SNN enwik8 bpb | Transformer enwik8 bpb | SNN PG-19 bpb | Transformer PG-19 bpb | Books with lower SNN loss |
|---|---:|---:|---:|---:|---:|
| 0 | 2.00577 | 1.81449 | 2.98022 | 3.11725 | 16/16 |
| 1 | 2.00508 | 1.83666 | 2.96994 | 3.08604 | 16/16 |

PG-19 scores use temperatures fitted on eight separate
validation books. Seed 1's raw losses were 3.07107 bits per
byte for held charge and 3.24554 for full attention. Its native
audit measured 306,660 pulses per byte and exact CPU, MPS,
single-byte, and chunked decoding within the audited samples.
[Seed-0 scores](../results/event_lm/depth_inhibition) ·
[Seed-1 Transformer scores](../results/event_lm/depth_inhibition_replication) ·
[Seed-1 held-charge records](../results/event_lm/held_charge_replication)

The seed-2 SNN was training at release time. The seed-2
Transformer checkpoint and scores are included for the paired
comparison. [Checkpoint inventory](../models/spiking-byte-lm-v3/manifest.json)
