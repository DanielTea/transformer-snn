# ShuntFormer v17 research handoff

The release restores three checkpoints, each as its running average of weights:

- the best spike-only stack so far, v16's with 128 knowledge cells a head and trained on streams (`l2048-know-st`, the model of this release);
- the depth-22 stack (`l2048-d22`);
- its Transformer (`tf2048-d22`).

The repository carries no v17 weights. The bundled model that runs from a clone is still [v11](../models/spiking-byte-lm-v11/README.md).
[Model card](../models/spiking-byte-lm-v17/README.md) · [Knowledge-gap study](KNOWLEDGE_GAP.md) · [Roadmap](ROADMAP.md)

v17 is a diagnosis release.
- **Where the gap is:** split byte by byte by what each byte's context offers, the stack remembers better than the Transformer at every distance and is behind only on knowledge.
- **The best stack** scores 1.1170 on the enwik8 test split, 0.0103 behind the Transformer of about its size.
- **What was tried:** fifteen ideas, each under a rule recorded before its run.
- **What moves the gap:** more data reopens it on the test bed, because every part that helped knowledge helped early.

From here every design is judged by how its gap moves as the data doubles (`results/event_lm/knowledge_gap/data_scaling_long.json`).

## Restore the checkpoints

Each archive unpacks under `checkpoints/`:

| Archive | File | SHA-256 |
|---|---|---|
| `spiking-byte-lm-v17-checkpoints-l2048-know-st.tar.gz` | `knowledge_gap/l2048-know-st/model.pt.averaged` | `f8f3886e9c9501accf4820e7f114a0db12373339265181b6a662ff891aa5ea16` |
| `spiking-byte-lm-v17-checkpoints-l2048-d22.tar.gz` | `long_streams/l2048-d22/model.pt.averaged` | `99e8d4b181039b399c5653d3983b590f2c98596175ecf1552a5dafdde325d67e` |
| `spiking-byte-lm-v17-checkpoints-tf2048-d22.tar.gz` | `long_streams/tf2048-d22/model.pt.averaged` | `e10c9548d98524d706cd6e94814f8bd3ab4a4e3c63454c85ad51dd32c0bac106` |

```bash
for name in l2048-know-st l2048-d22 tf2048-d22; do
  tar -xzf spiking-byte-lm-v17-checkpoints-$name.tar.gz -C checkpoints
done
shasum -a 256 -c results/event_lm/knowledge_gap/v17_checkpoints.sha256
```

Each v17 file is the scored file with two training arguments, `results_dir` and `checkpoint`, rewritten to
repository paths; the weights are identical (`models/spiking-byte-lm-v17/manifest.json` keeps both hashes).
The v16 checkpoints (`l2048-forget`, `l2048-flat`, `tf2048`, `tf1024`) are assets of the v16 release.

## Rescore

To score the stack with its best reading (reach 8,192, fast output synapses on all levels at 2^24), and the depth-22 Transformer in windows of its training length:

```bash
python experiments/eval_stream.py --device cuda --enwik8 data/enwik8 --split test \
  --checkpoint checkpoints/knowledge_gap/l2048-know-st/model.pt.averaged --reach 8192 --piece 256 \
  --teach-levels 0 1 2 3 4 5 6 7 8 9 10 --teach-threshold 16777216 --json know.json --bits know.npy
python experiments/eval_taught_reference.py --device cuda --enwik8 data/enwik8 --split test \
  --checkpoint checkpoints/long_streams/tf2048-d22/model.pt.averaged --context 16 --history 2032 \
  --json tf.json --bits tf.npy
```

Pair two scorings piece by piece, and split the gap by what each byte's context offers, by kind of byte and by place in the word:

```bash
python experiments/compare_stream_scores.py --first tf.json tf.npy --second know.json know.npy --json pair.json
python experiments/gap_by_piece.py --enwik8 data/enwik8 --split test --first tf.json tf.npy \
  --second know.json know.npy --near 2032 --far 8192 --json split.json
```

## The test bed

Most of the v17 study runs on the test bed: 5.25M parameters, 24,000 updates (98.3M training bytes) or 6,000, scored on the first 2,000,000 validation bytes as 16 unbroken streams. Its recipes are in each record of `results/event_lm/knowledge_gap/`. The delayed test bed trained on streams, the base of the newest studies, is:

```bash
L=0,1,2,3,4,5
python experiments/train_event_lm.py --variant event_steered --no-carry --stack --dim 256 --hidden 4 --depth 6 \
  --spike-resolution 16 --router-charge-bits 8 --pulse-budget 113000 --read-after $L --read-width 256 \
  --read-heads 4 --read-window 256 --read-mode race --read-budget 256 --read-resolution 4 --read-cap 128 \
  --read-address-rate 64 --granule-after $L --granule-clusters 1 --granule-fan 256 --granule-cells 1022 \
  --granule-slope --granule-cap 128 --rotate-stacks --read-delays 4 --granule-delays 4 --stream 10 \
  --stream-groups 3 --data data/enwik8 --training-mixture data/training-v1/prepared/catalogue.json --ctx 256 \
  --bs 16 --steps 24000 --warmup 300 --lr 3e-4 --wd 0.05 --seed 20001 --eval-every 750 --eval-bs 16 \
  --eval-batches 16 --final-eval-batches 64 --validation-only --save-every 1500 --resume --device cuda \
  --rotated-adam 50 --weight-average 0.999 --tag bed --results-dir out/results --checkpoint out/model.pt
```

The test bed's Transformer is `--variant transformer --dim 256 --depth 6 --heads 4 --mlp-ratio 4` with the same training flags and no streams.

A run asked to stop with SIGTERM or SIGUSR1 trains on to its next stretch boundary, saves its state and exits with 75. With `--resume` it goes on exactly where it stopped.

| Flag | Sets |
|---|---|
| `--read-knowledge K` | K learned, ageless cells race beside the stream's cells in every read head |
| `--read-word-heads W --read-word-span S` | W heads of every read reach only the present word, weighing each cell by its place in the word, up to S places |
| `--granule-myelin D` | Every fibre of a cell layer learns its own delay, up to D−1 bytes back |
| `--granule-fan 512 --granule-cells 681` | The full size's cell shape on the test bed: every channel's present byte and one delayed copy |
| `--weight-dither` | Quantal release while training: every synapse rounds to a neighbouring tick of its charge grid at random, so that its expected charge is its stored size; scoring rounds to the nearest tick |
| `--granule-bits B`, `--router-charge-bits B` | The charge grid of the cells' synapses, or of all router synapses: 2^B ticks to a threshold |

## Next

1. **The data-scaling curves.** The test bed's Transformer, the delayed stack and the plain stack, each trained from scratch at 6,000 to 96,000 updates (24.6M to 393M bytes) and at its best learning rate. The base for all later work is the design whose gap shrinks fastest with data (`data_scaling_long.json`).
2. **Quantal release and finer charge grids** (`weight_precision_with_data.json`). Late in training most updates are smaller than one tick of a synapse's charge grid. Does that hold the stack back?
3. **The learning rate.** The test bed's Transformer gains 0.036 bits per byte from doubling its peak rate, and the stack gains 0.044 (one seed each). Every comparison now takes each model at its best rate (`learning_rate_with_data.json`).
4. Brain-grounded mechanisms for late learning, judged by their slope:
   - homeostatic synaptic scaling;
   - neurogenesis during training;
   - capacity designs that lost at 6,000 updates.

## What not to try again

Each of these was tested under a rule recorded before its runs, and failed:

| Idea | What happened | Record |
|---|---|---|
| Kept cells | Win on the trainer's short streams, lose over long ones | `kept_cells.json` |
| Keyed fast synapses | Fail at all four seeds | `knowledge_keyed_synapses.json` |
| Branching dendrites, apical bursts, racing experts | Fail on the test bed | `preregistered.json` |
| Eight fixed delays | Fail | `more_delays.json` |
| Narrower reads with more cells | Fail at 6,000 updates | `memory_for_knowledge.json` |
| Myelinated fibres | Their delays move toward the present; fail on both devices | `myelin_and_sheet.json` |
| The cerebellar sheet | Fail at 6,000 updates | `myelin_and_sheet.json` |
| Word heads | A memory gain that fades with data | `word_rhythm.json`, `word_heads_with_data.json` |
| More knowledge cells, a shared library | No gain over 128 a head | `laptop_screen_2.json` |

Two of these, narrower reads with more cells and the sheet, add capacity that may pay only late. They are re-tested by slope, not by their loss at one size.

## Where things are

- **Records and verdicts:** `results/event_lm/knowledge_gap/`. Every record is pre-registered, with `recorded_utc` taken from its commit.
- **Scorings and splits:** `results/event_lm/knowledge_gap/gap_by_piece_*.json`. Figures come from the `experiments/plot_*` scripts.
- **The roadmap and the rules for claims:** [ROADMAP.md](ROADMAP.md).
