# Probes of the state-read study

Short scripts that were written and run during the study of 17
to 18 November 2026. Each runs from the repository root, reads
`data/enwik8`, uses a GPU when it finds one and prints its
result. They are kept as they ran. They are diagnostics, not
part of the trainer or the evaluators.
[Study page](../../docs/STATE_READS.md) ·
[Outputs as printed](../../results/event_lm/state_reads/probes) ·
[Index of the outputs](../../results/event_lm/state_reads/probes/index.json)

Most probes compared snapshots that a run wrote during training.
Only the final checkpoints of four arms are released. `--averaged`
takes the running weight average out of a snapshot that holds
both sets of weights. A released `model.pt.averaged` is already
the average and is passed without it.
[Which probes run on the released files](../../docs/V13_HANDOFF.md#rerun-the-probes)

| Script | Asks | Outputs |
|---|---|---|
| `match_split.py` | Where one checkpoint is ahead of another, by the longest earlier match of the bytes just read and by whether the byte after it is the target | `match_split_*`, [`splits.json`](../../results/event_lm/state_reads/probes/splits.json) |
| `class_split.py` | The same by the kind of the target byte, its place in a word and the markup around it | `class_split_*` |
| `context_split.py` | The gap by the bytes in view and by the words that ended in view | [`context_split.json`](../../results/event_lm/state_reads/probes/context_split.json) |
| `expert_mix.py` | What a count table of the training text adds to a checkpoint's prediction | [`count_tables.json`](../../results/event_lm/state_reads/probes/count_tables.json) |
| `granule_stats.py` | How the cells of every layer fire | `cell_statistics_*`, [`cells.json`](../../results/event_lm/state_reads/probes/cells.json) |
| `ablate_granule.py` | The loss with the cells' output synapses cut | `cell_ablation_*` |
| `tap_use.py` | The synapse sizes by fibre delay, and the loss with one delay silenced | `delay_use_*` |
| `probe_silencing.py` | Which part of the training signal pushes threshold cells silent | `cell_silencing_*` |
| `probe_coherence.py` | How well the task gradient and the pulse gradient agree from batch to batch | `gradient_agreement_*` |
| `attribute_jitter.py` | Which weight groups carry the loss that averaging the weights removes | `averaging_attribution_*` |
| `sensitivity.py` | How far the loss moves when one weight group takes a random step | `sensitivity_*` |
| `probe_fidelity.py` | What one step along a batch's gradient buys on that batch and on others | `step_fidelity_*` |
| `probe_sharpness.py` | How far the gradient's promise holds when one weight group steps alone | `step_sharpness_*` |
| `probe_shared.py` | The same when several groups step together | `shared_step_*` |
| `common_mode.py` | The share of every synapse's input that is the same at all bytes | `common_mode_*` |
| `probe_dither.py` | What the rounding to whole spikes costs | [`dither.json`](../../results/event_lm/state_reads/probes/dither.json) |
| `quantize_transformer.py` | What the Transformer loses with weights on a grid as coarse as the decoder's | [`weight_grid.json`](../../results/event_lm/state_reads/probes/weight_grid.json) |
| `stream_stats.py` | The stream's activity before every block and the drives at a cap | `stream_statistics_*` |
| `headroom.py` | How much of the exact-integer range a decoder's synapse sums use | `headroom_*` |
| `copy_task.py` | Whether a small decoder with reads learns a sharp lookup | `copy_task_*` |
