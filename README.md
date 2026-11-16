# ShuntFormer

ShuntFormer v4 is a seven-block spiking byte language model with
**6,514,176 learned parameters** and **19,536 bytes of persistent
tensor state** per stream. Its integer CPU decoder maps byte events to
signed vocabulary spike counts. A shared inhibitory pool regulates
firing, steering spikes route charge through fixed axons and local
memory, and an error bank checks learned context tags before delivering
correction spikes.

![Held-out loss for the grown spiking and full-attention models](results/figures/event_lm_grown_6510k.png)

## Run the model

Use Python 3.11 or 3.12 from the repository root. The checkout includes
the reference weights. Greedy generation runs on CPU.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python experiments/generate_event_lm.py models/spiking-byte-lm-v4/weights.pt \
  --prompt 'The first chapter' --bytes 128
```

| Where to start | What it contains |
|---|---|
| [v4 model card](models/spiking-byte-lm-v4/README.md) | Circuit diagram, firing and memory rules, training configuration, and the stream state layout |
| [Research handoff](docs/HANDOFF.md) | Dataset checksums, checkpoint restore, exact replay commands, and the next runs |
| [Ranked results](docs/LEADERBOARD.md) and [browser report](docs/report.html#spiking-byte-lm) | Matched comparisons, raw and calibrated loss, pulse counts, and charts |
| [Checkpoint release](https://github.com/DanielTea/transformer-snn/releases/tag/spiking-byte-lm-v4) and [manifest](models/spiking-byte-lm-v4/manifest.json) | Reference weights, variants, Transformer baselines, parents, and SHA-256 digests |

## Current benchmark

The models continued from their respective trained 3.26M parents. Each
saw **32,768,000 cumulative enwik8 training targets** with 512-byte
windows and continuation seed 711. Scores are bits per byte on the
full enwik8 test split and 16 pinned PG-19 test books. PG-19 temperatures
are fitted on eight separate books.

<!-- event-lm-leaderboard:start -->

| Model | enwik8 | PG-19 | State |
|---|---:|---:|---:|
| Grown SNN, finer charge | 1.95110 | **2.85083** | 19,536 B |
| Grown SNN, six-bit charge | 1.95646 | 2.86904 | 19,536 B |
| Grown full-attention Transformer | **1.84266** | 3.06318 | 4 MiB |

<!-- event-lm-leaderboard:end -->

The bundled finer-charge SNN scored lower than the matched
Transformer on all 16 PG-19 books. The Transformer led on enwik8.
The width-growth comparison uses one continuation seed. Pulse counts measure
emitted spikes on a fixed text sample. Stream state excludes weights
and temporary workspace.
[Per-book scores and growth ablation](docs/GROWN_RESIDUAL.md) ·
[Evaluation protocol](docs/HANDOFF.md#score-checkpoints)

## Continue the research

The research handoff points to the completed runs and the next
commands. The 49.15M-target continuation measures the effect of
more enwik8 exposure at fixed size. The next comparisons expand
the content and steering neurons and test whether delayed spikes
can help preserve held charge.
[Handoff](docs/HANDOFF.md#experiment-status) ·
[Training exposure](docs/GROWN_RESIDUAL.md#extend-training-exposure) ·
[Expansion study](docs/EXPANSION_GROWTH.md) ·
[Write confirmation](docs/TRACE_WRITE_CONFIRM.md)

Regenerate the leaderboards and figures from the checked-in results.

```bash
python experiments/plot_event_lm_study.py
python experiments/plot_depth_inhibition.py
python experiments/make_report.py docs/report.html
```

[Source map](docs/HANDOFF.md#source-and-release-map) ·
[Historical ShuntFormer and Dendrite experiments](docs/SHUNTFORMER_HISTORY.md) ·
[Result file guide](results/README.md)
