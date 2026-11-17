# ShuntFormer

ShuntFormer is an integer spiking byte language model. The v11
checkpoint has **25.79M parameters**, 11 spiking blocks and
**53,432 bytes of persistent tensor state per stream**. Its fixed
axons carry signed spike counts. Their local charge holds history,
and a tagged error bank drives a correction into the vocabulary
neurons. In v11 a population of forecast neurons predicts every
byte, and the spikes it gets wrong write **fast synapses** that
recall the byte which last followed the same context. They lower
held-out enwik8 loss from 1.790 to **1.645 bits per byte** without
a gradient step. The same error spikes also clocked four added
blocks. That clock did not lower held-out loss and is published
as a negative result with its controls.

[Model card and circuit diagram](models/spiking-byte-lm-v11/README.md) ·
[Research handoff](docs/V11_HANDOFF.md) ·
[Checkpoint release](https://github.com/DanielTea/transformer-snn/releases/tag/spiking-byte-lm-v11)

![Held-out enwik8 and PG-19 loss with the surprise clock and fast synapses](results/figures/event_lm_surprise_clock.png)

## v12: coasting read heads

The v12 study trains the 34.85M decoder from random weights
beside **read heads on a tape of the stream**. A head follows a
passage one byte at a time, gains charge with every confirmed
byte and is removed by the first wrong one. With its heads the
decoder scores **1.414 bits per byte** on the full enwik8 test
split. The same decoder trained beside the v11 fast synapses
scores 1.486, and the bundled v11 model scores 1.645.

Full attention is far ahead at this training budget. The
Transformer of the same size, trained from scratch on the same
windows, scores 1.207. The head decoders are scored by replaying
the heads over the decoder's recorded outputs. They have no
streaming or native evaluation yet and no row on the leaderboard.
Every arm is one seed. The v12 weights are release assets, and
the model that runs from a clone is v11.

[Coasting-head study](docs/COASTING_HEADS.md) ·
[Model card](models/spiking-byte-lm-v12/README.md) ·
[Research handoff](docs/V12_HANDOFF.md) ·
[Checkpoint release](https://github.com/DanielTea/transformer-snn/releases/tag/spiking-byte-lm-v12)

![enwik8 test loss of each decoder alone, with fast synapses and with read heads](results/figures/event_lm_coasting_heads.png)

## Run the bundled model

From the repository root, set up Python 3.12 and run greedy
generation with the native CPU integer decoder.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python experiments/generate_event_lm.py models/spiking-byte-lm-v11/weights.pt \
  --prompt 'The first chapter' --bytes 128
```

## Matched benchmark

Every row saw **140,902,400 cumulative training targets**.
The six trained models received identical 512-byte windows for
their final 9,830,400 targets. The two rows with fast synapses
add 24 fitted strengths to a trained decoder and change no
weight. The table shows bits per byte on the full enwik8 test
split and 16 pinned PG-19 test books. PG-19 temperatures are
fitted on eight separate calibration books.

<!-- event-lm-leaderboard:start -->

| Model | enwik8 | PG-19 calibrated | PG-19 raw | State |
|---|---:|---:|---:|---:|
| 25.79M SNN | 1.78973 | 1.97852 | 1.97852 | 28,584 B |
| 25.79M SNN with fast synapses | 1.64541 | 1.93961 | 1.93961 | 53,432 B |
| 34.85M SNN, every byte | 1.78633 | 1.96913 | 1.96913 | 40,714 B |
| 34.85M SNN, fixed stride | 1.78761 | 1.96842 | 1.96842 | 43,796 B |
| 34.85M SNN, surprise clock | 1.78973 | 1.97528 | 1.97528 | 43,796 B |
| 34.85M SNN, surprise clock and fast synapses | **1.64278** | **1.93610** | **1.93610** | 68,388 B |
| 25.79M full attention | 1.70433 | 1.93711 | 1.93711 | 6 MiB |
| 34.85M full attention | 1.70426 | 1.93678 | 1.93678 | 6 MiB |

<!-- event-lm-leaderboard:end -->

With fast synapses the 25.79M spiking decoder scores **1.64541
bpb** on enwik8, 0.0589 below the size-matched Transformer, and
1.93961 on PG-19, 0.0025 above it. Full attention reads 512
bytes and the synapses carry the whole stream. With the same
synapses on its output the Transformer scores 1.57753 on enwik8,
so the like-for-like gap is 0.0679 bpb. Without synapses it is
0.0854. The surprise clock scores 1.78973 on enwik8. The same
blocks score 1.78761 on a fixed stride and 1.78633 on every
byte. With fast synapses the three score 1.64278, 1.64281 and
1.64164, so the clock does not earn the lowest row of the table.
The study uses one continuation seed.

State counts persistent tensors per stream. The synapse strengths
are fitted on enwik8 validation and used unchanged on PG-19.
[Surprise study, controls and limits](docs/SURPRISE_CLOCK.md) ·
[Full leaderboard](docs/LEADERBOARD.md) ·
[Browser report](docs/report.html#surprise-clock)

## Byte transfer benchmarks

The fixed suite reports raw bits per byte on WikiText-103,
C4 validation, enwik9 beyond byte 100 million, and text8.
LAMBADA scores answer-byte loss and greedy word accuracy for
5,153 full passages and last-sentence prompts. Evaluations use
the same scoring windows. The preparation checks training text
for overlap, and the report also gives scores for Wikipedia
windows with zero hits. C4 validation and PG-19 test are
held-out in-domain evaluations for the mixed runs.

Every window starts from an empty stream, so the fast synapses
hold only what the window contains. With them the 25.79M SNN is
below its Transformer by 0.0743 bpb on WikiText-103, 0.0601 on
C4 and 0.0902 on text8, and 0.0221 above it on enwik9. From the
full LAMBADA passage it completes 174 of 5,153 final words.
Each Transformer completes one.

![Corpus loss and LAMBADA accuracy for the surprise study](results/figures/event_lm_byte_transfer_surprise_clock_140902k.png)

[Transfer scores](docs/LEADERBOARD.md#byte-transfer-at-the-same-training-exposure) ·
[Data, overlap audit and replay](docs/BYTE_TRANSFER.md)

## Resume the research

The v11 archive holds the eight study endpoints and both v10
parents. The manifest links weights, checksums, training
records, synapse fits and the exact training windows. The
handoff includes commands to replay each step of the study.
[Restore and replay](docs/V11_HANDOFF.md#restore-the-checkpoints) ·
[Exact manifest](models/spiking-byte-lm-v11/manifest.json) ·
[Result file guide](results/README.md)

The v12 archives hold the 21 checkpoints of the coasting-head
study. Its handoff has the restore commands, the launcher and
the replay that scores a head decoder.
[Restore and replay](docs/V12_HANDOFF.md#restore-the-checkpoints) ·
[Exact manifest](models/spiking-byte-lm-v12/manifest.json)

Regenerate the leaderboards, scientific figures and browser report
from the checked-in records.

```bash
python experiments/plot_event_lm_study.py
python experiments/plot_depth_inhibition.py
python experiments/plot_coasting_heads.py
python experiments/make_report.py docs/report.html
```

[v10 research handoff](docs/V10_HANDOFF.md) ·
[Historical ShuntFormer and Dendrite experiments](docs/SHUNTFORMER_HISTORY.md)
