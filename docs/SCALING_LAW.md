# A scaling law the other way around

*The study of Volley v18. Every number here is from a finished run. The verdicts are those of the records in
`results/event_lm/knowledge_gap/`, which were committed before their runs. What still runs is listed at the end.*

v17 found that the stack's gap to the Transformer is knowledge, and that on the test bed it reopens with more data:
the delayed stack trained on streams leads its Transformer after 6,000 updates and trails it after 24,000. A stack
that is to beat the Transformer must do the opposite and learn more from each added byte. Its gap must shrink as both
models see more data. This study therefore judges every design by **the slope of its gap per doubling of the training
data**, measured over separate runs at 6,000 to 96,000 updates (24.6M to 393M bytes), each model at its best learning
rate (`data_scaling_long.json`, analysis `experiments/data_scaling_long.py`).

## Both models wanted a higher learning rate

Every run up to v17 trained at a peak rate of 3e-4. On the test bed both models do better at higher rates, and the
stack gains a little more (`learning_rate_with_data.json`, two seeds):

| 24,000 updates, 2M validation bytes as streams | 1.5e-4 | 3e-4 | 6e-4 | 1.2e-3 | 2.4e-3 |
|---|---:|---:|---:|---:|---:|
| Spiking stack, seed 20001 | 1.6329 | 1.5475 | 1.5036 | **1.4864** | 1.4924 |
| Spiking stack, seed 20002 | 1.6379 | 1.5580 | 1.5130 | **1.4926** | 1.4950 |
| Transformer, seed 20001 | 1.5917 | 1.4927 | 1.4569 | **1.4442** | 1.4455 |
| Transformer, seed 20002 | 1.6066 | 1.4955 | 1.4575 | 1.4460 | **1.4368** |
| Gap of the two-seed means | 0.036 | 0.059 | 0.051 | 0.044 | 0.053 |

The stack is best at 1.2e-3 on both seeds, the Transformer at 2.4e-3 on their mean. From 3e-4 to its best rate the
stack gains 0.063 and the Transformer 0.053. The difference of 0.010 is under the 0.020 that the record set for calling
the flattening an artefact of the learning rate: **it is not one.** At each model's best rate the gap is 0.048, and it
is on new contexts on both seeds (+0.056 and +0.075). The data-scaling curves now read the Transformer at 2.4e-3.

![Both models wanted a higher learning rate](../results/figures/event_lm_learning_rate.png)

Split by what each byte's context offers, the better rate helps the stack's knowledge almost twice as much as the
Transformer's (seed 20001, `learning_rate_by_context.json`, a diagnostic made after the runs):

| From 3e-4 to 1.2e-3, bits per byte | New contexts (76% of bytes) | Copyable from the last 240 bytes (7%) |
|---|---:|---:|
| Spiking stack | **−0.073** | −0.034 |
| Transformer | −0.039 | **−0.177** |

At 3e-4 the stack's knowledge was starved, and the Transformer's copying. At each model's best rate the gap on new
contexts falls from 0.087 to 0.061, and it is still the whole gap.

The stack gaining more from bigger steps fits a reading of where its curve flattens. Every synapse sends whole ticks
of charge, and every cell whole spikes, so an update smaller than a tick changes nothing the stack fires. Two
spike-only remedies are under test: quantal release (synapses that round at random while training, so that their
expected charge is their stored size, `weight_precision_with_data.json`) and stochastic firing (threshold cells that
fire their last spike at random, `cell_dither_with_data.json`). Both are exact at scoring.

The neighbouring rates show that the band of good rates is narrow for the stack's knowledge and wide for the
Transformer's (seed 20001, the extension of `learning_rate_by_context.json`):

| Bits per byte | New contexts | Fourth letter of a word on | Copyable from the last 240 bytes |
|---|---:|---:|---:|
| Stack, 6e-4 to 1.2e-3 | −0.017 | level | **+0.048** |
| Transformer, 6e-4 to 1.2e-3 | −0.010 | −0.013 | −0.033 |
| Stack, 1.2e-3 to 2.4e-3 | **+0.012** | **+0.031** | level |
| Transformer, 1.2e-3 to 2.4e-3 | +0.004 | level | −0.021 |

Below its best rate the stack's knowledge starves; above it, its knowledge is what breaks first, the completion of
words most of all. Three other readings of the flattening were ruled out on the way:

- **The pulse budget.** Its price falls to 0.00–0.02 by the end of every 24,000-update run, while the stack spends
  its budget to within 2%. A looser budget would buy almost nothing. Where a budget binds, its price shows it: a
  12-level test bed held to the 6-level budget ran at a price of 0.10 to 0.24, and gained 0.067 when its budget
  doubled. At full size the price stays at 0.00–0.02 at depth 11 and at depth 22 alike, so the wider gap at depth
  22 is not the budget's either.
- **Noisy late weights.** Averaging the weights helps the stack about as much as it helps the Transformer, and by
  the end of a run it helps neither.
- **A dense code that grows denser.** 35% to 56% of a level's cells fire at every byte, at 6,000 and at 24,000 updates
  alike, in the plain bed, which learns late better, as in the delayed one (`experiments/cell_activity.py`).

The largest step that v15's walk from the Transformer to the stack priced was the unit: plain rectifying units in place
of gated ones cost 0.025, against 0.001 for whole spikes (`docs/STREAM_FORM.md`). The stack's cells are plain units.
Gated cells were only judged at 6,000 updates. A screen on the laptop now measures them by their slope
(`gated_cells_with_data.json`). A second screen measures the other half of a smooth unit, in spikes: noisy thresholds
(`noisy_thresholds_with_data.json`). A real neuron's membrane is noisy, so near its threshold it fires by chance, and
its average rate is a smooth curve of its input. Under logistic noise of half a threshold, a cell's average count is
within 0.025 spikes of 0.5 · softplus(2 · charge / threshold). The cell learns from that curve's slope, where a plain
cell learns from a stand-in. The noise falls with the learning rate, so training ends on the whole counts that scoring
fires.

## At full size the membranes cap the rate

The full-size pilot (`full_size_rate_pilot.json`, 24,000 updates on windows of 512, one seed) did not repeat the test
bed's finding:

| 2M validation bytes as streams | 3e-4 | 6e-4 | 1.2e-3 |
|---|---:|---:|---:|
| Spiking stack | 1.3416 | **1.3199** | 1.3475 |
| Transformer | 1.3057 | 1.2837 | **1.2762** |
| Gap | 0.036 | 0.036 | 0.071 |

At each model's best rate the gap is 0.044, wider than at 3e-4. The Transformer gains more from its best rate than the
stack does (0.029 against 0.022). On new contexts the two gain alike (the stack 0.029, the Transformer 0.028); the
Transformer's extra gain is on near copies. The stack is best at a lower rate, and at the higher one its membranes saturate:

![Saturated membranes cap the full-size stack's rate](../results/figures/event_lm_membrane_saturation.png)

At 1.2e-3, 2.7% of the reads' entries and 2.0% of the cells' sit at the membrane's bound, up to 9.7% and 11.8% at the
top level. A neuron there fires its largest count whatever more it hears, and learns nothing through its charge. The
test bed barely saturates even at 2.4e-3, which is why it never showed this (`membrane_saturation.json`,
`experiments/membrane_saturation.py`). v15's pool that lifts the ceiling divides the charge before the bound. In a
continued run at 1e-3 it was 0.019 ahead, and no release used it, because all of them trained at 3e-4, where nothing
saturates. A full-size pilot of the lifted stack (`full_size_lifted_ceiling_pilot.json`):

| 2M validation bytes as streams | 6e-4 | 1.2e-3 |
|---|---:|---:|
| Plain stack | 1.3199 | 1.3475 |
| Lifted stack | 1.3134 | **1.3126** |
| Transformer | 1.2837 | 1.2762 |

![At full size, lifting the ceiling stops the stack breaking at a high rate](../results/figures/event_lm_full_size_rate.png)

Lifting removes the saturation entirely, at the same energy (about 308,000 pulses a byte in training): no read and
almost no cell reaches its cap. At 1.2e-3 it is worth 0.035, and the stack is level across 6e-4 and 1.2e-3 instead of
breaking at the higher rate. At each model's best rate the lifted stack is 0.036 behind the Transformer, as the plain
stack was at 3e-4. Lifting removes the widening, and the gap that remains is the stack's knowledge. Its best arm was
0.0073 ahead of the plain stack's best, short of the 0.010 that the record asked, so the full-size pair trains the
plain stack at 6e-4 as recorded.

## The unit at full size, and word heads

The largest step that v15's walk from the Transformer to the stack priced was the unit (0.025). One 24,000-update
run each at full size, against the plain stack at 6e-4 (`full_size_unit_screen.json`):

| 2M validation bytes as streams | Score | Against the plain stack | On new contexts |
|---|---:|---:|---:|
| Plain stack | 1.3199 | | |
| Gated cells, equal parameters | 1.4381 | +0.118 | +0.162 |
| Noisy thresholds | 1.3157 | −0.004 | level |

Gated cells lose badly: their gate dendrites sit at their bound a fifth of the time. Noisy thresholds train level at
the same energy, with a small gain on near copies. Neither closes the knowledge gap. The self-gated arm runs next.

Word heads at full size, at 90,000 updates with v15's fast synapses at reach 8,192, gain 0.0009 on test over v17's
stack (paired interval 0.0004 to 0.0013), under the 0.003 their record asked (`full_size_word_heads.json`). The run is
the best spike-only stack so far, 1.11615 on test and 1.14908 on validation: 0.0095 behind the Transformer as it
reads, and 0.013 behind the Transformer with taught output synapses. Its gap keeps v17's shape, all of it knowledge.

## Still running

| Record | Question |
|---|---|
| `data_scaling_long.json` | The curves: the Transformer, the delayed stack and the plain stack at 6,000 to 96,000 updates, at each model's best rate |
| `weight_precision_with_data.json` | Finer charge grids and quantal release, at 6,000 and 24,000 updates |
| `cell_dither_with_data.json` | Stochastic firing of the threshold cells, alone and with quantal release |
| `cell_shape_with_data.json` | The full size's cell shape at 24,000 updates, four seeds |
| `randomness_at_best_rate.json` | Quantal release and stochastic firing at the best rate, 1.2e-3; a third seed on the laptop |
| `gated_cells_with_data.json` | Gated and self-gated cells at 6,000 and 24,000 updates at the best rate, a screen on the laptop |
| `noisy_thresholds_with_data.json` | Noisy thresholds at 6,000 and 24,000 updates at the best rate, a screen on the laptop |
| `full_size_unit_screen.json` | Its third arm, self-gated cells at full size |
| `full_size_best_rate_pair.json` | The stack and its Transformer at full size, each at its best full-size rate, 90,000 updates: the plain stack at 6e-4 training since 2026-11-23 00:35 UTC, the Transformer at 1.2e-3 next |
