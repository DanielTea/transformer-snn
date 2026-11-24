# Volley v19 research handoff

The release restores three checkpoints, each as its running average of weights:

- the best spike-only stack so far, v17's stack trained at its best full-size rate, 6e-4, for 90,000 updates (`fsp-st`, the model of this release);
- its Transformer, trained the same way at its own best rate, 1.2e-3 (`fsp-tf`), so that the pair can be rescored and split without training;
- the full-size looped stack of the loop pilot, levels 2 to 8 run twice with the pulses for it, 24,000 updates at 6e-4 (`fs-loop6`).

The repository carries no v19 weights. The bundled model that runs from a clone is still [v11](../models/spiking-byte-lm-v11/README.md).
[Model card](../models/spiking-byte-lm-v19/README.md) · [Scaling-law study](SCALING_LAW.md) · [Roadmap](ROADMAP.md)

v19 is a release about where the distance to the Transformer lies.
- **Each model at its best rate** for 90,000 updates: the stack reaches its best score, and the gap widens to 0.034, because the faster rate pays the Transformer 0.028 and the stack 0.004.
- **The gap is knowledge**: the stack is ahead on every copy and 0.09 behind on new contexts.
- **Four fifths of the distance is the layout**: a float stand-in of reads and cells closes only a fifth of it at full size.
- **Depth by recurrence** holds with data on the test bed, on three seeds; its full-length run is training.

## Restore the checkpoints

Each archive unpacks under `checkpoints/`:

| Archive | File | SHA-256 |
|---|---|---|
| `spiking-byte-lm-v19-checkpoints-fsp-st.tar.gz` | `full_size_pair/fsp-st/model.pt.averaged` | `fc570db8e401e3fb7c7883da3be9be6628b880fe21b4c7c515398d3e6992492d` |
| `spiking-byte-lm-v19-checkpoints-fsp-tf.tar.gz` | `full_size_pair/fsp-tf/model.pt.averaged` | `4f915879339fcf9d77f7f07c39ec3f4b7b299913d4706d731b6f5ff909545f56` |
| `spiking-byte-lm-v19-checkpoints-fs-loop6.tar.gz` | `full_size_pilots/fs-loop6/model.pt.averaged` | `668e54f2a978dbf40d158dc3c64b3f3566803d827f16f04516a6494141261c23` |

```bash
for name in fsp-st fsp-tf fs-loop6; do
  tar -xzf spiking-byte-lm-v19-checkpoints-$name.tar.gz -C checkpoints
done
(cd checkpoints && shasum -a 256 -c ../results/event_lm/knowledge_gap/v19_checkpoints.sha256)
```

Each file is the scored file with the training arguments that name folders rewritten to repository paths; the
weights are identical. The bits archive (`spiking-byte-lm-v19-bits.tar.gz`) holds the per-piece bits of every
scoring in this release: the pair's, the full-size pilots', the test bed's screens' (loop, skip fibres, slope) and
the float unit diagnostic's, so that every paired gap in the records can be recomputed.

## Rescore

The stack at its best reading (reach 8,192, fast output synapses on all levels at 2^24):

```bash
python experiments/eval_stream.py --device cuda --enwik8 data/enwik8 --split test \
  --checkpoint checkpoints/full_size_pair/fsp-st/model.pt.averaged --reach 8192 --piece 256 \
  --teach-levels 0 1 2 3 4 5 6 7 8 9 10 --teach-threshold 16777216 --json st.json --bits st.npy
```

The Transformer as the pair reads it (windows of 2,048 bytes, 2,032 of history before 16 scored; the first row is
frozen, the second has taught output synapses at 0.07):

```bash
python experiments/eval_taught_reference.py --device cuda --checkpoint checkpoints/full_size_pair/fsp-tf/model.pt.averaged \
  --enwik8 data/enwik8 --split test --context 16 --history 2032 --rates 0 0.07 --json tf.json --bits tf.npy
```

The paired gap and its split by what each byte's context offers:

```bash
python experiments/compare_stream_scores.py --first tf.json tf.npy --first-row 0 --second st.json st.npy
python experiments/gap_by_piece.py --enwik8 data/enwik8 --split test --near 2032 --far 8192 \
  --first tf.json tf.npy --first-row 0 --second st.json st.npy --json gap.json
```

The split by place in the word is a least-squares fit over pieces: for this pair its first, second and third
letters and spaces after a word rise and fall together from piece to piece, so read only its narrow intervals.

## Train

The pair is v17's recipe (`full_size_best_rate_pair.json`): windows of 512 bytes in batches of 8 for 24,000 updates,
then unbroken streams of windows of 2,048 bytes for 66,000 more, with `--read-knowledge 128 --read-reach 2048` for the
stack, at `--lr 6e-4` for the stack and `--lr 1.2e-3` for the Transformer. The loop is `--stack-loop FIRST,LAST,PASSES`
with the pulse budget scaled by the level passes; skip fibres are `--stack-skips level` or `--stack-skips parts`; the
slope far below the threshold is `--granule-slope-tail 4` (and `--granule-slope-tail-local`); the float reference's
control gates are `--feed-forward rampglu`, `shuntglu`, `sigglu` and `bilinear`.

## Open

- The loop's run of 90,000 updates at full size (`full_size_loop_run.json`) and two pilots of loop variants
  (`full_size_loop_variants.json`).
- The looped bed with skip fibres on two seeds (`skip_fibres_with_data.json`) and the loop's energy controls on the
  server (`loop_with_data.json`).
- The two half stand-ins and the stand-in at the Transformer's rate (`full_size_spikes_late.json`).
- The bounded gates in floats (`bounded_gates_in_floats.json`).
- The slope far below the threshold on the server's two seeds and at full size (`subthreshold_slope_with_data.json`).
