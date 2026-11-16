# ShuntFormer

ShuntFormer v6 is an 11-block spiking byte language model with
**9,960,960 parameters** and **29,674 bytes of persistent tensor
state** per stream. Its four added blocks forward the incoming spike
count and fire a signed correction. Across all blocks, inhibitory and
steering neurons route spikes through fixed axons and held charge.
A tagged error bank verifies reads before its correction neurons
contribute to the next-byte spike counts.

![Held-out loss for the deeper and wider SNNs and full attention](results/figures/event_lm_growth_9960k.png)

On the native enwik8 audit, the signed-correction circuit generated
15.3% fewer pulses than ordinary re-emission at the same weights and
identical next-byte spike counts. [Architecture and pulse audit](docs/DEPTH_GROWTH.md)

## Run the model

Use Python 3.12 from the repository root. The checkout includes
the reference weights. Greedy generation runs on CPU.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python experiments/generate_event_lm.py models/spiking-byte-lm-v6/weights.pt \
  --prompt 'The first chapter' --bytes 128
```

| Where to start | What it contains |
|---|---|
| [v6 model card](models/spiking-byte-lm-v6/README.md) | Circuit diagram, firing and memory rules, training configuration, and the stream state layout |
| [Research handoff](docs/HANDOFF.md) | Dataset checksums, checkpoint restore, exact replay commands, and the next runs |
| [Ranked results](docs/LEADERBOARD.md) and [browser report](docs/report.html#spiking-byte-lm) | Matched comparisons, raw and calibrated loss, pulse counts, and charts |
| [Checkpoint release](https://github.com/DanielTea/transformer-snn/releases/tag/spiking-byte-lm-v6) and [manifest](models/spiking-byte-lm-v6/manifest.json) | Reference weights, variants, Transformer baselines, parents, and SHA-256 digests |

## Current benchmark

Each model below has about 9.96M parameters and saw **81,920,000
cumulative enwik8 training targets** in 512-byte windows, with seed
4603 for the final continuation.
Scores are bits per byte on the full enwik8 test split and 16 pinned
PG-19 test books. PG-19 temperatures are fitted on eight separate books.

<!-- event-lm-leaderboard:start -->

| Model | enwik8 | PG-19 | State |
|---|---:|---:|---:|
| 11-block SNN | 1.84288 | **2.79854** | 29,674 B |
| 7-block wider SNN | 1.84496 | 2.80299 | 19,536 B |
| 6-block full attention | **1.75713** | 2.93008 | 6 MiB |

<!-- event-lm-leaderboard:end -->

The 11-block SNN leads full attention on all 16 PG-19 books.
Full attention leads on enwik8 by 0.08575 bits per byte. This
comparison uses one continuation seed. State measures persistent
tensors per stream, with a 512-byte evaluation context for full
attention. The seven-block output experiment has a separate 6.51M
matched group.
[Per-book scores and native pulse audits](results/event_lm/depth_growth) ·
[Evaluation protocol](docs/HANDOFF.md#score-checkpoints) ·
[6.51M output comparison](docs/DEPTH_GROWTH.md#seven-block-output-comparison)

## Continue the research

Start from the bundled 11-block model for depth work or the archived
seven-block counterparts for width and pulse-budget experiments.
The handoff has verified archive restoration, exact training replay
and recipes for the next continuation. The earlier width and circuit
studies remain part of the research record.
[Resume from v6](docs/HANDOFF.md#continue-from-the-v6-endpoints) ·
[13.03M comparison](docs/GROWN_RESIDUAL.md#double-the-residual-stream-at-fixed-persistent-state) ·
[Growth to 11 spiking blocks](docs/DEPTH_GROWTH.md) ·
[Charge at fixed width](docs/GROWN_RESIDUAL.md#refine-router-charge-at-fixed-width) ·
[Tag read competition](docs/TAG_READ_COMPETITION.md) · [Late steering](docs/LATE_STEERING.md) ·
[Training exposure](docs/GROWN_RESIDUAL.md#extend-training-exposure) ·
[Expansion study](docs/EXPANSION_GROWTH.md) ·
[Write confirmation](docs/TRACE_WRITE_CONFIRM.md) ·
[Inhibition study](docs/INHIBITION_REDISTRIBUTION.md)

Regenerate the leaderboards and figures from the checked-in results.

```bash
python experiments/plot_event_lm_study.py
python experiments/plot_depth_inhibition.py
python experiments/make_report.py docs/report.html
```

[Source map](docs/HANDOFF.md#source-and-release-map) ·
[Historical ShuntFormer and Dendrite experiments](docs/SHUNTFORMER_HISTORY.md) ·
[Result file guide](results/README.md)
