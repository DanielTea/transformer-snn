# ShuntFormer

ShuntFormer is an integer spiking byte language model. The v10
checkpoint has **25.79M parameters**, 11 spiking blocks and
**28,584 bytes of persistent tensor state per stream**. Its fixed
axons carry signed spike counts. Their local charge holds history,
and a tagged error bank drives a correction into the vocabulary
neurons. The residual stream grows from 768 to 1,408 neurons
per block. The added neurons start silent and learn from the
existing axon arrivals and held charge. The training pool contains
**292.4 million bytes** from enwik8, C4 and PG-19. The preparation
screens the added C4 and PG-19 text for benchmark overlap.

[Model card and circuit diagram](models/spiking-byte-lm-v10/README.md) ·
[Research handoff](docs/V10_HANDOFF.md) ·
[Checkpoint release](https://github.com/DanielTea/transformer-snn/releases/tag/spiking-byte-lm-v10)

![Held-out enwik8 and PG-19 loss at the two matched widths](results/figures/event_lm_residual_scale.png)

## Run the bundled model

From the repository root, set up Python 3.12 and run greedy
generation with the native CPU integer decoder.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python experiments/generate_event_lm.py models/spiking-byte-lm-v10/weights.pt \
  --prompt 'The first chapter' --bytes 128
```

## Matched benchmark

Every row saw **131,072,000 cumulative training targets**.
The 14.07M and 25.79M SNN–Transformer pairs received identical
512-byte windows for their final 9,830,400 targets. The wide
enwik8-only control received the same number of targets from
enwik8. The table shows bits per byte on the full enwik8 test
split and 16 pinned PG-19 test books. PG-19 temperatures are
fitted on eight separate calibration books.

<!-- event-lm-leaderboard:start -->

| Model | enwik8 | PG-19 calibrated | PG-19 raw | State |
|---|---:|---:|---:|---:|
| 14.07M SNN | 1.79789 | 2.04156 | 2.04156 | 28,584 B |
| 25.79M SNN | 1.79787 | 2.02699 | 2.02699 | 28,584 B |
| 25.79M SNN, enwik8 only | 1.78054 | 2.73942 | 2.83176 | 28,584 B |
| 14.07M full attention | 1.71414 | 1.98073 | 1.98073 | 6 MiB |
| 25.79M full attention | **1.71379** | **1.97644** | **1.97644** | 6 MiB |

<!-- event-lm-leaderboard:end -->

With the expanded mixture, the full enwik8 gap to full attention
is **0.08375 bpb at 14.07M** and **0.08408 bpb at 25.79M**.
On predictions 17–32 after reset, the estimated change is
+0.00378 bpb with a paired 95% interval of [−0.00075, +0.00819].
On the held-out PG-19 books, the gap shrinks from 0.06083
to 0.05055 bpb. The study uses one continuation seed.

State counts persistent tensors per stream. The full-attention
references evaluate with a 512-byte context. The paired enwik8
probe scores the first 32 predictions after reset.
[Width study and interventions](docs/RESIDUAL_SCALE.md) ·
[Full leaderboard](docs/LEADERBOARD.md) ·
[Browser report](docs/report.html#residual-scale)

## Byte transfer benchmarks

The fixed suite reports raw bits per byte on WikiText-103,
C4 validation, enwik9 beyond byte 100 million, and text8.
LAMBADA scores answer-byte loss and greedy word accuracy for
5,153 full passages and last-sentence prompts. Evaluations use
the same scoring windows. The preparation checks training text
for overlap, and the report also gives scores for Wikipedia
windows with zero hits. C4 validation and PG-19 test are
held-out in-domain evaluations for the mixed runs.
On C4, the wide mixed SNN scores 2.06397 bpb and the wide
enwik8-only control scores 2.20845 bpb.

![Corpus loss and LAMBADA accuracy for the width study](results/figures/event_lm_byte_transfer_residual_scale_131072k.png)

[Transfer scores](docs/LEADERBOARD.md#byte-transfer-at-the-same-training-exposure) ·
[Data, overlap audit and replay](docs/BYTE_TRANSFER.md)

## Resume the research

The v10 archive holds the five matched endpoints and both
parents. The manifest links weights, checksums, training
records and the exact training windows. The handoff includes
commands to reconstruct the corpus and replay each comparison.
[Restore and replay](docs/V10_HANDOFF.md#restore-the-checkpoints) ·
[Exact manifest](models/spiking-byte-lm-v10/manifest.json) ·
[Result file guide](results/README.md)

Regenerate the leaderboards, scientific figures and browser report
from the checked-in records.

```bash
python experiments/plot_event_lm_study.py
python experiments/plot_depth_inhibition.py
python experiments/make_report.py docs/report.html
```

[v9 research handoff](docs/V9_HANDOFF.md) ·
[Historical ShuntFormer and Dendrite experiments](docs/SHUNTFORMER_HISTORY.md)
