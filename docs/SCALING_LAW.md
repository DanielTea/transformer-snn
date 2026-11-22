# A scaling law the other way around

*A study in progress for the next release. Every number here is from a finished run. The verdicts are those of the
records in `results/event_lm/knowledge_gap/`, which were committed before their runs.*

v17 found that the stack's gap to the Transformer is knowledge, and that on the test bed it reopens with more data:
the delayed stack trained on streams leads its Transformer after 6,000 updates and trails it after 24,000. A stack
that is to beat the Transformer must do the opposite and learn more from each added byte. Its gap must shrink as both
models see more data. This study therefore judges every design by **the slope of its gap per doubling of the training
data**, measured over separate runs at 6,000 to 96,000 updates (24.6M to 393M bytes), each model at its best learning
rate (`data_scaling_long.json`, analysis `experiments/data_scaling_long.py`).

## Both models wanted a higher learning rate

Every run up to v17 trained at a peak rate of 3e-4. On the test bed both models do better at higher rates, and the
stack gains more (`learning_rate_with_data.json`; seed 20001 complete, seed 20002 running):

| 24,000 updates, 2M validation bytes as streams | 1.5e-4 | 3e-4 | 6e-4 | 1.2e-3 | 2.4e-3 |
|---|---:|---:|---:|---:|---:|
| Spiking stack, seed 20001 | 1.6329 | 1.5475 | 1.5036 | **1.4864** | 1.4924 |
| Transformer, seed 20001 | 1.5917 | 1.4927 | 1.4569 | **1.4442** | 1.4455 |
| Gap | 0.041 | 0.055 | 0.047 | **0.042** | 0.047 |

From 3e-4 to its best rate the stack gains 0.061 and the Transformer 0.048. The difference is 0.013, under the 0.020
that the record set for calling the flattening an artefact of the learning rate. Seed 20002 agrees so far: from 3e-4
to 6e-4 it gains the stack 0.045 (seed 20001: 0.044).

![Both models wanted a higher learning rate](../results/figures/event_lm_learning_rate.png)

The stack gaining more from bigger steps fits a reading of where its curve flattens. Every synapse sends whole ticks
of charge, and every cell whole spikes, so an update smaller than a tick changes nothing the stack fires. Two
spike-only remedies are under test: quantal release (synapses that round at random while training, so that their
expected charge is their stored size, `weight_precision_with_data.json`) and stochastic firing (threshold cells that
fire their last spike at random, `cell_dither_with_data.json`). Both are exact at scoring.

## Still running

| Record | Question |
|---|---|
| `data_scaling_long.json` | The curves: the Transformer, the delayed stack and the plain stack at 6,000 to 96,000 updates, at each model's best rate |
| `weight_precision_with_data.json` | Finer charge grids and quantal release, at 6,000 and 24,000 updates |
| `cell_dither_with_data.json` | Stochastic firing of the threshold cells, alone and with quantal release |
| `cell_shape_with_data.json` | The full size's cell shape at 24,000 updates, four seeds |
| `full_size_rate_pilot.json` | Which rate is best at full size, before the next full-size pair |
| `full_size_word_heads.json` | Word heads at full size |
