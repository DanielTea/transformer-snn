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
| 2 | 2.00814 | 1.83322 | 3.01923 | 3.08349 | 16/16 |
| **Mean** | **2.00633** | **1.82812** | **2.98979** | **3.09559** | — |

PG-19 scores use temperatures fitted on eight separate
validation books. The SNN led on each of the same 16 books in
all three seeds. At the raw output scale, average PG-19 loss was
3.10037 for held charge and 3.29465 for full attention. Full
attention led on enwik8 in each seed.

The native SNN audits counted 308,178, 306,660 and 306,200
pulses per byte across the three seeds. The audited CPU, MPS,
single-byte, and chunked spike counts matched. All six checkpoints
are included in the release.
[Seed-0 scores](../results/event_lm/depth_inhibition) ·
[Transformer scores for seeds 1 and 2](../results/event_lm/depth_inhibition_replication) ·
[Held-charge records for seeds 1 and 2](../results/event_lm/held_charge_replication) ·
[Checkpoint inventory](../models/spiking-byte-lm-v3/manifest.json)
