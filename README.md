# ShuntFormer

ShuntFormer is an integer spiking byte language model. The v8
checkpoint has **12.80M parameters**, 11 spiking blocks and
**31,257 bytes of persistent tensor state** per stream. Each
block routes counts through fixed axons and held charge. A cell's
positive and negative write corrections drive its read through
learned conductances. Neighboring cells can lend spare steering
pulses within their delay group. A tagged error bank inhibits
mismatched reads before the vocabulary neurons fire.

[Model card and circuit diagram](models/spiking-byte-lm-v8/README.md) ·
[Research handoff](docs/V8_HANDOFF.md) · [Checkpoint release](https://github.com/DanielTea/transformer-snn/releases/tag/spiking-byte-lm-v8)

![Held-out loss for the spiking read circuits and matched full-attention reference](results/figures/event_lm_trace_read_drive.png)

## Run the bundled model

From the repository root, set up Python 3.12 and run greedy
generation with the native CPU integer decoder.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python experiments/generate_event_lm.py models/spiking-byte-lm-v8/weights.pt \
  --prompt 'The first chapter' --bytes 128
```

## Matched benchmark

Every row below saw **116,326,400 cumulative enwik8 training
targets** in 512-byte windows. The continuation used seed 8237
and models have about 12.8M parameters. Scores are bits per byte
on the full enwik8 test split and 16 pinned PG-19 test books.
PG-19 temperatures are fit on eight separate validation books.

<!-- event-lm-leaderboard:start -->

| Model | enwik8 | PG-19 calibrated | PG-19 raw | State |
|---|---:|---:|---:|---:|
| Original held-charge read | 1.79577 | 2.75495 | 2.84136 | 28,584 B |
| Write input, six-bit reset | 1.79582 | 2.75706 | 2.84654 | 28,584 B |
| Present input, six-bit reset | 1.79466 | 2.75747 | 2.84860 | 28,584 B |
| Write input, eight-bit reset | 1.79449 | **2.75314** | 2.84192 | 28,584 B |
| Write input, eight-bit carry | 1.79533 | 2.75724 | 2.84850 | 31,257 B |
| Present input, eight-bit carry | 1.79525 | 2.75638 | 2.84361 | 31,257 B |
| Write input, shared steering | 1.79561 | 2.76102 | 2.85001 | 31,257 B |
| Two write rails, shared steering | 1.79464 | 2.75590 | 2.84506 | 31,257 B |
| Present input, shared steering | 1.79572 | 2.75322 | **2.84069** | 31,257 B |
| Full attention | **1.71363** | 2.88478 | 3.08081 | 6 MiB |

<!-- event-lm-leaderboard:end -->

The bundled two-rail model scores **1.79464 bpb on enwik8** and
**2.75590 bpb on calibrated PG-19**. The matched full-attention
control scores **1.71363** and **2.88478 bpb** respectively.
Zeroing the bundled read drive raises loss by 0.00035 bpb on
the 65,536-target validation intervention.
[Ranked results](docs/LEADERBOARD.md) · [Browser report](docs/report.html#spiking-byte-lm) ·
[Read circuit experiments](docs/TRACE_READ_DRIVE.md)

State counts persistent tensors per stream. The full-attention
reference uses a 512-byte evaluation context. The result records
include raw and calibrated PG-19 scores, per-book losses,
and native pulse measurements.
[Evaluation protocol](docs/V8_HANDOFF.md#evaluation) · [Result file guide](results/README.md)

## Resume the research

Restore the v8 archive for the exact continuation checkpoints,
matched full-attention baseline and replay parents. Each endpoint
links to its full training arguments, dataset checksum and
parent checkpoint hash. The handoff gives commands to rescore,
replay and train the 25.6M comparison. The widened SNN passed a
warm-start audit and one optimizer update at the same 31,257-byte
state budget.
[Restore and continue](docs/V8_HANDOFF.md#restore-checkpoints) ·
[Exact manifest](models/spiking-byte-lm-v8/manifest.json) ·
[Packed-width experiments](docs/PACKED_WIDTH.md)

Regenerate the leaderboards, scientific figures and browser report
from the checked-in records.

```bash
python experiments/plot_event_lm_study.py
python experiments/plot_depth_inhibition.py
python experiments/make_report.py docs/report.html
```

[Source map](docs/V8_HANDOFF.md#source-map) ·
[Historical ShuntFormer and Dendrite experiments](docs/SHUNTFORMER_HISTORY.md)
