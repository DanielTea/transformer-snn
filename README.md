# ShuntFormer

ShuntFormer v7 packages the 11-block integer spiking byte decoder
trained through **101,580,800 enwik8 targets**. The SNN uses
**29,674 bytes of persistent tensor state** per stream.
Inhibitory and steering neurons route signed counts through fixed
axons and held charge. The last four blocks forward arriving counts
and emit signed corrections. A tagged error bank inhibits mismatched
reads before the vocabulary neurons fire.

The v7 tap-read experiment derives a release spike's timing from
the mismatch between held charge and each candidate on an existing
axon ring. The first release selects its own spike packet. The
selector and its shortest-tap control both use the original rings.
[Architecture](models/spiking-byte-lm-v7/README.md) · [Tap read experiment](docs/TRACE_TAP_READ.md)

![Held-out loss for three spiking decoders and the matched full-attention reference](results/figures/event_lm_trace_tap_read.png)

## Run the model

Use Python 3.12 from the repository root. The checkout includes the
reference weights. Greedy generation runs on CPU.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python experiments/generate_event_lm.py models/spiking-byte-lm-v7/weights.pt \
  --prompt 'The first chapter' --bytes 128
```

| Where to start | What it contains |
|---|---|
| [v7 model card](models/spiking-byte-lm-v7/README.md) | End-to-end circuit, integer firing and memory rules, architecture and stream state |
| [Research handoff](docs/HANDOFF.md) | Dataset checksums, checkpoint restore, exact replay commands, and the next runs |
| [Ranked results](docs/LEADERBOARD.md) and [browser report](docs/report.html#spiking-byte-lm) | Matched comparisons, raw and calibrated loss, pulse counts, and charts |
| [Checkpoint release](https://github.com/DanielTea/transformer-snn/releases/tag/spiking-byte-lm-v7) and [manifest](models/spiking-byte-lm-v7/manifest.json) | 11 verified checkpoints, including both studies and their replay parents |

## Current benchmark

Each model below has about 9.96M parameters and saw **101,580,800
cumulative enwik8 training targets** in 512-byte windows, with seed
6899 for the final continuation. Scores are bits per byte on the full
enwik8 test split and 16 pinned PG-19 test books. PG-19 temperatures
are fitted on eight separate books.

<!-- event-lm-leaderboard:start -->

| Model | enwik8 | PG-19 | State |
|---|---:|---:|---:|
| Continued SNN | 1.81319 | **2.77756** | 29,674 B |
| Shortest eligible tap | 1.81655 | 2.78455 | 29,674 B |
| Held spike race | 1.81473 | 2.78521 | 29,674 B |
| 6-block full attention | **1.73380** | 2.89702 | 6 MiB |

<!-- event-lm-leaderboard:end -->

The bundled continued SNN leads the spiking arms on both test
sets and leads full attention on all 16 PG-19 books. Full
attention leads on enwik8.

State measures persistent tensors per stream, with a 512-byte
evaluation context for full attention. The result records include
the raw PG-19 scores and per-book losses.
[Full-corpus records and native pulse audits](results/event_lm/trace_tap_read) ·
[Evaluation protocol](docs/HANDOFF.md#score-checkpoints)

## Continue the research

Restore the v7 archive to replay any arm or start a new continuation
from its trained weights. A warm-start audit expands the local
populations from 187 to 243 channels under 32 KiB with exact
parent counts and pulses.
[Resume from v7](docs/HANDOFF.md#continue-from-the-v7-endpoints) ·
[Packed-width continuation and steering transfers](docs/PACKED_WIDTH.md) ·
[Held-charge inputs and group thresholds](docs/TRACE_INTEGRATION.md) ·
[Growth to 11 spiking blocks](docs/DEPTH_GROWTH.md) ·
[13.03M residual-width comparison](docs/GROWN_RESIDUAL.md#double-the-residual-stream-at-fixed-persistent-state)

Regenerate the leaderboards and figures from the checked-in results.

```bash
python experiments/plot_event_lm_study.py
python experiments/plot_depth_inhibition.py
python experiments/make_report.py docs/report.html
```

[Source map](docs/HANDOFF.md#source-and-release-map) ·
[Historical ShuntFormer and Dendrite experiments](docs/SHUNTFORMER_HISTORY.md) ·
[Result file guide](results/README.md)
