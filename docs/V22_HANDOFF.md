# Volley v22 research handoff

The release restores one checkpoint, as its running average of weights: v21's lifted loop with power-law cells,
trained for 90,000 updates at 6e-4 beside v19's pair (`fsp-lift2`, the model of this release). The models it is read
against are assets of earlier releases: the lifted loop `fsp-lift` of [v21](V21_HANDOFF.md), the looped stack
`fsp-loop` of [v20](V20_HANDOFF.md), and the plain stack `fsp-st` and the Transformer `fsp-tf` of
[v19](V19_HANDOFF.md).

The repository carries no v22 weights. The bundled model that runs from a clone is still [v11](../models/spiking-byte-lm-v11/README.md).
[Model card](../models/spiking-byte-lm-v22/README.md) · [Scaling-law study](SCALING_LAW.md) · [Roadmap](ROADMAP.md)

v22 is a release about a measure, and about memory by lines.
- **Read as registered, power-law cells lose 0.003**: with fast output synapses the stack scores 1.0845 on test and
  1.1169 on validation, behind v21's stack and 0.0055 and 0.0121 behind the Transformer.
- **Read without fast output synapses, it is ahead of the Transformer on test and level on validation**: 1.0752
  against 1.0790 (0.0038 ahead, interval 0.0027 to 0.0047), 1.1054 against 1.1048 (0.0006 behind, within the
  interval). Not the registered measure, so not claimed; a second seed with that measure fixed in advance is
  training (`full_size_power_cells_second_seed.json`).
- **Heads that count lines**: a stack whose read heads reach only the present line recalls a binding 200 bytes back
  and writes 12-step chains unpractised (`results/event_lm/one_model/line_heads.json`).
- **Practice holds with rehearsal and breaks just past what is rehearsed**, for both models
  (`practice_cap_raised.json`, `practice_several_families.json`); learning from attempts below their task's mean is
  next (`practice_that_stands.json`).

## Restore the checkpoint

The archive unpacks under `checkpoints/`:

| Archive | File | SHA-256 |
|---|---|---|
| `spiking-byte-lm-v22-checkpoints-fsp-lift2.tar.gz` | `full_size_lifted_loop/fsp-lift2/model.pt.averaged` | `d1b82af9a3eada5e924edbf4413635ab6a2a3ebb4aff53a7778a2771025599bd` |

```bash
tar -xzf spiking-byte-lm-v22-checkpoints-fsp-lift2.tar.gz -C checkpoints
(cd checkpoints && shasum -a 256 -c ../results/event_lm/knowledge_gap/v22_checkpoints.sha256)
```

The file is the scored file with the training arguments that name folders rewritten to repository paths; the
weights are identical. Its state dictionary names every pass of the loop, so the synapses of levels 2 to 8 are stored
twice, as identical copies. The bits archive (`spiking-byte-lm-v22-bits.tar.gz`) holds the per-piece bits of every
scoring recorded since v21: this checkpoint's (every reading), the full-size pilots' (power-law cells at twice the
rate and with skip fibres, pools trained with the energy's gradient in the stack and in the Transformer) and the test
bed's (pools trained with the energy's gradient, cubic cells), so that every paired gap in the records can be
recomputed. Unpack it at the repository's root.

## Rescore

Without fast output synapses at the reach of 8,192 (the reading the second seed is registered on):

```bash
python experiments/eval_stream.py --device cuda --enwik8 data/enwik8 --split test \
  --checkpoint checkpoints/full_size_lifted_loop/fsp-lift2/model.pt.averaged --reach 8192 --piece 256 \
  --json lift2.json --bits lift2.npy
```

With `--teach-levels 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 --teach-threshold 16777216` it is scored with fast
output synapses, as its record registered. The paired gaps to v21's stack and to v19's Transformer (their scorings
are in the v21 and v19 bits archives) and the split by what each byte's context offers:

```bash
python experiments/compare_stream_scores.py --first lift.json lift.npy --second lift2.json lift2.npy --json lift2-lift.json
python experiments/compare_stream_scores.py --first tf.json tf.npy --first-row 0 --second lift2.json lift2.npy --json lift2-tf.json
python experiments/gap_by_piece.py --enwik8 data/enwik8 --split test --near 2032 --far 8192 \
  --first tf.json tf.npy --first-row 0 --second lift2.json lift2.npy --json gap.json
```

The cells' firing beside v21's stack, and the share of membrane entries at their cap:

```bash
python experiments/cell_activity.py --checkpoint checkpoints/full_size_lifted_loop/fsp-lift2/model.pt.averaged \
  checkpoints/full_size_lifted_loop/fsp-lift/model.pt.averaged --enwik8 data/enwik8 --split validation \
  --bytes 8192 --lead 256 --piece 256 --json activity.json
python experiments/membrane_saturation.py --checkpoint checkpoints/full_size_lifted_loop/fsp-lift2/model.pt.averaged \
  --enwik8 data/enwik8 --split validation --bytes 65536 --json saturation.json
```

## Train

fsp-lift2 is v21's recipe with `--granule-power 8` in both stages: the pair's stack (`full_size_best_rate_pair.json`)
with `--stack-loop 2,8,2 --pulse-budget 507000 --pool-lifts-ceiling --granule-power 8`, windows of 512 bytes in
batches of 8 for 24,000 updates, then unbroken streams of windows of 2,048 bytes for 66,000 more with
`--read-knowledge 128 --read-reach 2048`, at `--lr 6e-4`, seed 20001, trained by cell. The full flags are in the
[manifest](../models/spiking-byte-lm-v22/manifest.json). Its second seed trains the same recipe with seed 20002 on
the fast path: `--fused-optimizer --exact-products` for the windows and `--fused-reads --fused-optimizer
--exact-products` for the streams, on code 452f365, whose fused kernels are those checked on a418100
(`full_size_data_slope.json`, code_check_2), after a check that it scores this checkpoint to the same bits on the
first million validation bytes (`sched/bin/fast-lift2-check.sh` on the server, `full-pair5.sh` its runner).

New since v21:

| Flag | What it does |
|---|---|
| `--read-word-unit line` | Word heads that count lines: the letter neuron hears every byte but the newline, so a head reaches only the present line and weighs each byte by its place in it (with `--read-word-heads N --read-word-span 256`) |
| `--read-by-cell` | Every race read of a stream by cell, the same spikes as by age; the read path that word and line heads need |
| `practice_loop.py --families ... --practise ...` | Practice of several task families in one loop, each with its own staircase, every family rehearsed |
| `practice_loop.py --dips D` | The weight of attempts below their task's mean: 0 teaches from paid attempts alone, 1 is GRPO |
| `practice_loop.py --exact-products --fused-reads` | The trainer's speed switches in the practice loop |

The one-model track's tools: `experiments/task_families.py` (four families of written chains),
`experiments/recall_over_distance.py` (recall of one binding a set number of bytes back),
`experiments/read_age_offsets.py` (the reads' learned age offsets), `experiments/where_chains_break.py` and
`experiments/forms_families.py` (how written chains go wrong, and what a practised model writes).

## Open

- The second seed of the power-law stack, read without fast output synapses
  (`full_size_power_cells_second_seed.json`), on the fast path.
- The gap's slope with data at full size: the power-law stack and the Transformer at 45,000, 90,000 and 180,000
  updates (`full_size_data_slope.json`), the stack's runs with fused reads on code a418100, as its third amendment
  set out.
- Practice that stands: GRPO's dips with rehearsal, six seeds of each model (`practice_that_stands.json`).
- Line heads in practice and in the four-family corpus, as `line_heads.json` says.
- The fast output synapses' threshold for power-law cells, which no scan has measured.
- Cubic cells at full size (`fs-loop6-lift-pc3`), sleep with gates that start closed on the server's seeds
  (`sleep_closed_gates_with_data.json`).
- A second seed of the Transformer, which every full-size claim still lacks.
