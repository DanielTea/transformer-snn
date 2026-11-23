# ShuntFormer v16 research handoff

The release restores the checkpoints of the long-streams study:
the stack trained on windows of 2,048 bytes with forgetting slopes
that learn (`l2048-forget`, the model of this release), the same
stack with its slopes held at zero (`l2048-flat`), and the
Transformers trained on windows of 2,048 and 1,024 bytes
(`tf2048`, `tf1024`), each as its running average of weights.
The repository carries no v16 weights. The bundled model that
runs from a clone is still
[v11](../models/spiking-byte-lm-v11/README.md).
[Model card](../models/spiking-byte-lm-v16/README.md) · [Long streams](LONG_STREAMS.md) · [Trained plasticity](TRAINED_PLASTICITY.md)

v16 is a measurement release more than a new model. It trains
Transformers on the stack's own window lengths, and at equal
context the stack is behind by 0.032 to 0.046 bits per byte, each
read as it trained. Read with what only a stream has, a reach
beyond the window, forgetting slopes that learn and fast synapses,
the stack trained on 2,048 bytes scores 1.1199 on test, 0.013
behind its Transformer. The release also pilots cells that
multiply or divide charges, consolidating fast synapses, a loop of
tied levels and echo learning, under rules recorded before their
runs.

## Restore the checkpoints

Each archive unpacks under `checkpoints/`:

| Archive | File | SHA-256 |
|---|---|---|
| `spiking-byte-lm-v16-checkpoints-l2048-forget.tar.gz` | `long_streams/l2048-forget/model.pt.averaged` | `75c4c98b0c4ed3f5f69516b2f5ceb8859b108ea35cafb3bf73118c0798a46b8e` |
| `spiking-byte-lm-v16-checkpoints-l2048-flat.tar.gz` | `long_streams/l2048-flat/model.pt.averaged` | `9a2898fe68f6e8ddf06cc118557031244cefebb0b4c9f3435809228b37517174` |
| `spiking-byte-lm-v16-checkpoints-tf2048.tar.gz` | `long_streams/tf2048/model.pt.averaged` | `002771455560da83be426516bed552a079a34518d41086a20a26d9bea466ac34` |
| `spiking-byte-lm-v16-checkpoints-tf1024.tar.gz` | `long_streams/tf1024/model.pt.averaged` | `652a4c4e422efb3702f29c2d35abffac64844144ccdfd1babce5705a6e13126e` |

```bash
for name in l2048-forget l2048-flat tf2048 tf1024; do
  tar -xzf spiking-byte-lm-v16-checkpoints-$name.tar.gz -C checkpoints
done
shasum -a 256 -c results/event_lm/long_streams_checkpoints.sha256
```

The archives were replaced on 2026-11-22: each file is the scored file with two training arguments, `results_dir`
and `checkpoint`, rewritten to repository paths, and its weights are identical. The scored files' hashes are in
`models/spiking-byte-lm-v16/manifest.json` for `l2048-forget`, and were `5e9c46e1…`, `e72a6825…` and `2b0c4fcf…` for
`l2048-flat`, `tf2048` and `tf1024`.

`spiking-byte-lm-v16-bits.tar.gz` holds the bits of every piece
of every published scoring, for paired comparisons without
rescoring; it unpacks into `results/event_lm/long_streams/bits/`.
The second seeds of the stack and the Transformer (`l2048-forget-s2`,
`tf2048-s2`) are scored in the study but not released.

## Rescore

A stack as 16 unbroken streams of the enwik8 test split, as
trained, with a longer reach, with a fixed forgetting slope at
scoring, and with fast output synapses:

```bash
python experiments/eval_stream.py --checkpoint checkpoints/long_streams/l2048-flat/model.pt.averaged \
  --enwik8 data/enwik8 --split test --json test.json --bits test.npy
python experiments/eval_stream.py ... --reach 8192 --reach-levels 4 5 6 7 --piece 256
python experiments/eval_stream.py ... --reach 16384 --forget-slope -1 --piece 128
python experiments/eval_stream.py ... --teach-levels 0 1 2 3 4 5 6 7 8 9 10 --teach-threshold 16777216
```

A Transformer along the same streams, in windows of its training
length with 16 scored bytes after the rest, frozen and with fast
output synapses at the rate of v15:

```bash
python experiments/eval_taught_reference.py --checkpoint checkpoints/long_streams/tf2048/model.pt.averaged \
  --enwik8 data/enwik8 --split test --context 16 --history 2032 --rates 0 0.07 --json tf.json --bits tf.npy
```

Pair two scorings of the same streams, piece by piece and by
quarter of the stream, and pool the gain of fast synapses over
the pieces that follow the same amount of reading:

```bash
python experiments/compare_stream_scores.py --first tf.json tf.npy --first-row 0 --second test.json test.npy \
  --json pair.json
python experiments/stream_quarters.py --enwik8 data/enwik8 --split test --reference tf.json tf.npy \
  --arm stack test.json test.npy --json quarters.json
python experiments/reading_curve.py --help
```

## Regenerate the figures

```bash
python experiments/plot_long_streams.py
python experiments/plot_event_lm_study.py
```

The first draws the four figures of the [long-streams
page](LONG_STREAMS.md) and the [trained-plasticity
page](TRAINED_PLASTICITY.md) from the records in
`results/event_lm/long_streams/` and
`results/event_lm/trained_plasticity/arms.json`. The second
regenerates the [leaderboard](LEADERBOARD.md).

## Train

Every long-window arm is a fork. The stack of v15 trains on
windows of 512 bytes to 24,000 updates with `--read-forget`
(`--stop-at 24000`), and `experiments/stack/fork_state.py`
continues its saved state on windows of 2,048 bytes:

```bash
python experiments/stack/fork_state.py checkpoints/long_streams/p-forget/model.pt.resume \
  checkpoints/long_streams/l2048-forget/model.pt.resume --tag v16_l2048_forget \
  --set ctx=2048 bs=2 read_reach=2048
```

and the trainer continues it with the stack's flags and
`--read-forget --ctx 2048 --bs 2 --read-reach 2048`. The control
`l2048-flat` adds `read_forget_rate=0` to the fork and
`--read-forget-rate 0` to the trainer. The Transformers fork the
same way with `ctx=2048 bs=2`. [The stack's
flags](STREAM_FORM.md#reproduce)

| Flag | Sets |
|---|---|
| `--read-forget` | Every head of a read learns a slope, from zero, by which a cell beyond the window loses charge for every doubling of its age |
| `--read-forget-rate 0` | The slopes stay at zero: the control |
| `--forget-slope S` | At scoring: one slope for every head, beyond the window, in place of the learned ones |
| `--granule-gate 4 --granule-self-gate` | A cell whose gate hears the fibres through the cell's own synapses: k thresholds of charge fire k² spikes |
| `--granule-price 1` | The spikes of the cells carry the pulse price like every other spike |
| `--plastic-levels L` | Fast output synapses run in every training window from the cells after these levels, each with a gain learned from zero |
| `--plastic-rule`, `--plastic-chunk`, `--plastic-decay-shift`, `--plastic-shuffle` | The rule of those synapses: base-error (A), chunked (B) or normalized (N); the chunk; a decay of one part in 2^k a byte; the shuffled control |
| `--plastic-off`, `--teach-freeze-after` | At scoring: the trained rule silenced, or frozen after so many bytes |

**Watch the GPU memory, not only the clock.** A stack of 22
levels on windows of 2,048 bytes needs about 35 GB with
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, and a scorer
can hold 30 to 56 GB while it runs. The first start of `l2048-d22`
died after nine seconds beside one. Check every new run's log for
its first updates.

## Next pilots

- **The full-size trained-plasticity arm** (`l2048-plastic`, a
  fork of the same state with Rule A decaying on the last level)
  is queued on the server; its go/no-go is recorded in
  `results/event_lm/trained_plasticity/arms.json`.
- **The scaling pair at depth 22** (`l2048-d22`, 69.4M parameters,
  and `tf2048-d22`) is training and being scored. The Transformer
  at depth 22 scores 1.0704 on test.
- **Whole books as single streams**, with and without fast
  synapses, for the released stack and `l2048-forget`.
- **Depth with its energy.** On the test bed twelve levels gain
  0.130 over six at the same budget and 0.152 with it doubled; more
  pulses alone make the stack worse. A full-size stack whose budget
  grows with its levels is the cleanest test of how the stack
  scales.
- **Seeds before claims.** Two runs of one setting on the test bed
  differ by up to 0.03, as much as every cell and credit effect
  measured this weekend. A pilot needs four seeds, or a larger test
  bed, before it earns a full-size run.

## What not to try again

Each was one arm with one seed unless a second is named.

- Squared cells (`--granule-self-gate`) without a price on their
  spikes blow up; with one they collapse sooner. Gated cells with
  a price collapse from 3,000 updates. Products of charges are
  unstable at a rate of 0.0003.
- Fast synapses trained in the loop without a bound on a stream:
  Rule A with no decay, on one level or all six, and trained on
  stretches of 15 windows, all fire their fast neurons ever harder
  and score 8.8 to 22 bits per byte.
- The normalized rule (N): it learns no gain, and its stack
  trains worse.
- The trainer's stream check as a stand-in for long streams: its
  streams are about 1.5 KB.
- A reach beyond the window for a stack that trained on 2,048
  bytes with every slope at zero.
- Fast synapses that consolidate into slower stages
  (`--teach-cascade`): 0.022 worse than keeping every lesson.
- Echo learning, random or mirror (`--echo`): the random echo blows
  the pulses up, the mirror holds a gap of 0.33 at every depth.
- A doubled pulse budget for the stack of six levels: worse in two
  seeds.
- Shunting gated cells and memory-only credit as gains: each won at
  one seed and was level or behind at the next.

## Where things are

| Path | Holds |
|---|---|
| `snnformer/event_lm.py` | The stack with forgetting slopes (`read_forget`), squared cells (`granule_self_gate`), and trained fast output synapses (`_plastic_window`, `_teach` with `plastic_after`) |
| `experiments/train_event_lm.py` | The trainer with the flags above |
| `experiments/eval_stream.py` | The stream scorer, with `--forget-slope`, `--forget-slope-replaces`, `--teach-cascade`, `--plastic-off` and `--teach-freeze-after` |
| `experiments/stream_quarters.py` | Gains by quarter of every stream, paired |
| `experiments/reading_curve.py` | The gain of fast synapses against how much of a stream they have read |
| `experiments/stack/fork_state.py` | Continue a saved run as a new arm, with trained fast synapses added at zero |
| `experiments/plot_long_streams.py` | The figures of this release |
| `results/event_lm/long_streams/` | Score records, paired comparisons, training curves and the reading curve |
| `results/event_lm/trained_plasticity/arms.json` | The trained-plasticity study: its criteria, thresholds, results and decision |
| `results/event_lm/echo_learning/`, `cascade_synapses/`, `shunt_tied/`, `stack_loop/`, `v17_candidates/` | The weekend's pilots, each recorded before its runs, with its verdict |
