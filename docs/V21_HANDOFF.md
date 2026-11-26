# Volley v21 research handoff

The release restores one checkpoint, as its running average of weights: the best spike-only stack so far, v20's
looped stack trained under the pool that lifts the ceiling for 90,000 updates at 6e-4 beside v19's pair (`fsp-lift`,
the model of this release). The models it is read against are assets of earlier releases: the plain stack `fsp-st`
and the Transformer `fsp-tf` of [v19](V19_HANDOFF.md), and the looped stack `fsp-loop` of [v20](V20_HANDOFF.md).

The repository carries no v21 weights. The bundled model that runs from a clone is still [v11](../models/spiking-byte-lm-v11/README.md).
[Model card](../models/spiking-byte-lm-v21/README.md) · [Scaling-law study](SCALING_LAW.md) · [Roadmap](ROADMAP.md)

v21 is a release about a ceiling that lifts.
- **The lifted loop is 0.003 behind the Transformer on test**: 1.0817 against 1.0790, and 1.1136 against 1.1048 on validation, 0.022 and 0.025 better than v20's loop at the same parameters, pulses and training time. It does not beat the Transformer.
- **Its gain is knowledge**: on new contexts it is 0.034 better than v20's stack, and the distance to the Transformer there falls from 0.078 to 0.044.
- **Power-law cells keep their gain with data**: 0.0125 on four seeds of the test bed, 0.0137 in a full-size pilot. Their full-length run is training.
- **What the pool costs**: in floats the price of the stack's pool is its division by the mean absolute activity; in spikes, pools that hear the energy fail, and pools that pass its gradient are on test.
- **Training with the same spikes in less time**: a stream update in 0.18 s for 0.81 s, from reads by age and fused race kernels.
- **A second track**: one model that reasons in writing and reads telemetry, with its first four screens.

## Restore the checkpoint

The archive unpacks under `checkpoints/`:

| Archive | File | SHA-256 |
|---|---|---|
| `spiking-byte-lm-v21-checkpoints-fsp-lift.tar.gz` | `full_size_lifted_loop/fsp-lift/model.pt.averaged` | `974d8a853fd57af2548b597a92cf8a983dd3d213f30fdaaf59c0aca03353fce6` |

```bash
tar -xzf spiking-byte-lm-v21-checkpoints-fsp-lift.tar.gz -C checkpoints
(cd checkpoints && shasum -a 256 -c ../results/event_lm/knowledge_gap/v21_checkpoints.sha256)
```

The file is the scored file with the training arguments that name folders rewritten to repository paths; the
weights are identical. Its state dictionary names every pass of the loop, so the synapses of levels 2 to 8 are stored
twice, as identical copies. The bits archive (`spiking-byte-lm-v21-bits.tar.gz`) holds the per-piece bits of every
scoring recorded since v20, with the runs they are paired against: the lifted loop's, the scan of the fast output
synapses' threshold on v20's stack, the full-size pilots' (the loop's third pass, rate and budget, decay on synapses
only, power-law cells, energy pools, and the Transformer's units, output and pools in floats) and the test bed's
screens' (power-law cells, energy pools, phases, the charge grid, the racing loop, the homeostatic gate's base at
length, sleep), so that every paired gap in the records can be recomputed. Unpack it at the repository's root.

## Rescore

The lifted loop at its registered reading (reach 8,192, fast output synapses on the cells of all 18 passes at 2^24):

```bash
python experiments/eval_stream.py --device cuda --enwik8 data/enwik8 --split test \
  --checkpoint checkpoints/full_size_lifted_loop/fsp-lift/model.pt.averaged --reach 8192 --piece 256 \
  --teach-levels 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 --teach-threshold 16777216 \
  --json lift.json --bits lift.npy
```

Without the teaching flags it is scored without fast output synapses, which is its best reading on validation
(1.1128). The paired gaps to v20's looped stack and to v19's Transformer (their scorings are in the v20 and v19
bits archives) and the split by what each byte's context offers:

```bash
python experiments/compare_stream_scores.py --first loop.json loop.npy --second lift.json lift.npy --json lift-loop.json
python experiments/compare_stream_scores.py --first tf.json tf.npy --first-row 0 --second lift.json lift.npy --json lift-tf.json
python experiments/gap_by_piece.py --enwik8 data/enwik8 --split test --near 2032 --far 8192 \
  --first tf.json tf.npy --first-row 0 --second lift.json lift.npy --json gap.json
```

The share of membrane entries at their bound, which for a lifted population counts the entries at the cap of
their count:

```bash
python experiments/membrane_saturation.py --checkpoint checkpoints/full_size_lifted_loop/fsp-lift/model.pt.averaged \
  --enwik8 data/enwik8 --split validation --bytes 65536 --json saturation.json
```

## Train

The lifted loop is the pair's stack recipe (`full_size_best_rate_pair.json`) with `--stack-loop 2,8,2
--pulse-budget 507000 --pool-lifts-ceiling` in both stages: windows of 512 bytes in batches of 8 for 24,000 updates,
then unbroken streams of windows of 2,048 bytes for 66,000 more, with `--read-knowledge 128 --read-reach 2048`, at
`--lr 6e-4`. The full flags are in the [manifest](../models/spiking-byte-lm-v21/manifest.json).

New since v20:

| Flag | What it does |
|---|---|
| `--granule-power P` | Power-law cells: round(k * k / P) spikes for k thresholds of charge (`--granule-power-exponent 3` for a cube) |
| `--pool-energy` | Pools that hear the stream's energy, the root mean square, in place of its summed activity |
| `--pool-energy-gradient` | Pools that fire as before and train with the energy's gradient |
| `--balanced-synapses` | Synapses that sum to zero over the stream's channels for every read, plain cell and output neuron; built and not run |
| `--stack-loop-exits` | A racing loop: the output hears the end of every pass and sums it as evidence, and each earlier pass is trained as a prediction of its own |
| `--stack-loop-phases` | Every pass of a looped level after its first has its own resting charge for each cell and its own offset to each read head's gain |
| `--transformer-norm cpool`, `spool`, `gpool` | The float controls: a centred pool, a pool dividing by the root mean square, and the pool trained through that division |

The trainer's speed flags (`experiments/speed_options.py`) come in two classes. `--exact-products`,
`--fused-optimizer` and `--cuda-graph` leave a run's weights as they were to the bit. `--fused-reads` and
`--compile-cells` fire the same spikes at the same weights and round the gradient differently: such a run repeats
itself to the bit and is another run than the one without, as another seed is. A trained model scores the same bits
by any read. `--fused-reads` needs Triton on a CUDA device and acts on the stream stage; its tests are
`tests/test_race_kernel.py`, and `tests/test_read_by_age.py` and `tests/test_whole_products.py` hold the reads by age
and the exact products to the bit. On an RTX PRO 6000 a full-size stream update takes 0.81 s by cell, 0.42 s by
age and 0.18 s fused, in 9 GiB of memory for 41.

The one-model track's screens are `experiments/hop_tasks.py` (the lookup lines, with or without written chains),
`experiments/hop_eval.py`, `experiments/practice_loop.py` (practice with reward on the answer) and
`experiments/pump_sim.py` with `experiments/telemetry_eval.py` (the simulated pump and the surprise detector).

## Open

- The lifted loop with power-law cells at full length (`full_size_lifted_loop_second.json`), read against this
  checkpoint.
- The gap's slope with data at full size: that stack and the Transformer at 45,000, 90,000 and 180,000 updates
  (`full_size_data_slope.json`); its two new stack runs train with fused reads.
- Pools that train with the energy's gradient, in floats, on the bed and at full size (`pool_gradient.json`).
- Power-law cells with a cube, with skip fibres and at the Transformer's rate in the loop
  (`cubic_cells_with_data.json`, `full_size_power_loop_variants.json`).
- Sleep with gates that start closed on the server's seeds and on the loop (`sleep_closed_gates_with_data.json`).
- Practice with reward from the models that write chains (`results/event_lm/one_model/practice_written_chains.json`).
- A second seed of the full-size pair, which every full-size claim still lacks.
