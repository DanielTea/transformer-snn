# Roadmap: a spiking, streaming language model that beats the Transformer

The aim of this project, Volley (formerly ShuntFormer), is a language model
that is **spiking and streaming from end to end** and that predicts text **better than a Transformer of the
same size trained on the same data**. Spiking means integer spike counts
between every pair of neuron populations. Streaming means no windows: the
model reads a stream byte by byte and carries its state for as long as the
stream runs. The work is open, so that anyone can check it, rerun it and
build on it.

This page says where the work stands, what has been learned, by which rules
claims are made, and what to try next. Every number links to a record in
`results/` or a study page in `docs/`.

## Where it stands

| Scale | Stack | Transformer | Gap (bits per byte) |
|---|---|---|---|
| Full size, the lifted loop with power-law cells read without fast output synapses (v23, fsp-lift2-s2, second seed; 36.3M parameters, 507,000 pulses a byte), enwik8 test | 1.0751 | 1.0790 (34.75M, at 1.2e-3, frozen) | **0.004 ahead** on test, level on validation (+0.0003, interval -0.0007 to +0.0013); not beating by the registered rule |
| Full size, the looped stack under the pool that lifts the ceiling (v21, fsp-lift: levels 2 to 8 run twice, 36.3M parameters, 1.63 times the pulses), enwik8 test | 1.0817 | 1.0790 (34.75M, at 1.2e-3) | **0.003 behind** (0.009 on validation) |
| Full size, the looped stack (v20, fsp-loop: the same without the lifting pool), enwik8 test | 1.1039 | 1.0790 (34.75M, at 1.2e-3) | 0.025 behind |
| Full size, each model at its best rate (v19: fsp-st at 6e-4, 36.3M parameters), enwik8 test | 1.1129 | 1.0790 (34.75M, at 1.2e-3) | 0.034 behind |
| The same pair against the Transformer with output synapses that learn while it reads | 1.1129 | 1.0771 | 0.036 behind |
| Full size, both at 3e-4 (v18: l2048-word-st, 36.3M), enwik8 test | 1.1162 | 1.1067 (34.75M) | 0.0095 behind |
| The same stack against the Transformer with output synapses that learn while it reads | 1.1162 | 1.1032 | 0.013 behind |
| Full size, v17 (l2048-know-st, 36.3M), enwik8 test | 1.1170 | 1.1067 | 0.010 behind |
| Full size, v16 (34.8M), enwik8 test | 1.1199 | 1.1067 | 0.013 behind |
| Test bed (5.2M), 24.6M training bytes, 2M validation bytes as long streams | 1.7724 | 1.8190 | 0.047 ahead |
| Test bed (5.2M), 98.3M training bytes, the same bytes | 1.5468 | 1.4927 | **0.054 behind** |
| Test bed, 98.3M training bytes, each model at its best learning rate, mean of two seeds | 1.4895 | 1.4412 | 0.048 behind |

The full-size stacks read up to 8,192 cells back and have fast synapses that
learn while they read; the best one has 128 knowledge cells in every read,
trained on streams at its best rate, 6e-4, and runs its levels 2 to 8 twice with
tied synapses under the inhibitory pool that lifts the membranes' ceiling
(`full_size_lifted_loop_run.json`): 1.0817 on test and 1.1136 on validation,
0.0028 and 0.0088 behind the Transformer with each interval above zero, for
1.63 times the plain stack's pulses. The same loop with power-law cells, read
without its fast output synapses, is level with the Transformer or ahead of it
with two seeds alike (`full_size_power_cells_second_seed.json`): fsp-lift2-s2
scores 1.0751 on test and 1.1051 on validation, 0.0039 ahead (interval 0.0029
to 0.0048) and 0.0003 behind (0.0007 ahead to 0.0013 behind). The first seed
scored 1.0752 and 1.1054; the two lie 0.0001 and 0.0003 apart. Ahead on one
split and level on the other is not a win by the rule the record set, so the
claim is not made. The full-size Transformer reads windows of
2,048 bytes. The test-bed stack has delayed fibres and trained on streams.
Sources: `results/event_lm/knowledge_gap/full_size_lifted_loop_run.json`,
`results/event_lm/knowledge_gap/full_size_loop_run.json`,
`results/event_lm/knowledge_gap/full_size_best_rate_pair.json`,
`results/event_lm/long_streams/leaderboard.json`,
`results/event_lm/knowledge_gap/full_size.json`,
`results/event_lm/knowledge_gap/full_size_word_heads.json`,
`results/event_lm/knowledge_gap/gap_with_data.json` and
`results/event_lm/knowledge_gap/learning_rate_with_data.json`; figure
`results/figures/event_lm_best_model.png`.

### Running now: the server around the clock until Thursday 3 December

Daniel opened the server's two GPUs around the clock from Saturday 28 November until Thursday 3 December,
07:45 Berlin. Until then DeepSeek does not run on the server. The order of the work:

- **Saturday:** the data slope (`full_size_data_slope.json`). The Transformer trains at 45,000 and 180,000
  updates and the stack at 45,000; the stack at 180,000 has trained. The verdict comes on Saturday evening.
  Beside it run the threshold scan of the fast output synapses for power-law cells
  (`fast_synapse_threshold_power.json`) and the cubic-cells pilot at full size (`cubic_cells_with_data.json`).
- **Saturday night to Sunday morning:** second seeds (`full_size_second_seeds.json`). The Transformer trains
  again at 90,000 updates, as every full-size claim needs, and both models at 180,000. That record fixes the
  win rule at 180,000 updates before any Transformer run at that length has trained.
- **Sunday and Monday:** the test-bed studies that have waited since 22 to 26 November, about 130 runs. They
  cover sleep with closed gates, subthreshold slopes, weight precision, cell dither, randomness at the best rate
  and the data-scaling curves of `data_scaling_long.json`.
- **Monday to Thursday:** what the weekend's verdicts call for, each registered before it runs. That means the
  cube in the next full-length run if its pilot passes, the replay study of the one-model track, and two seeds
  of the full-size candidate the verdicts name. If the stack beats the Transformer at 180,000 updates, the next
  step is more data for both models. Nothing is downloaded for that without Daniel's consent.
- **Whenever nothing else waits:** both models train at 360,000 updates (`full_size_data_slope_360k.json`), a
  fourth length for the data slope. They have the lowest priority in the queue. At that length the mixture's
  enwik8 bytes repeat about 8.2 times, so the record reads it as longer training on repeated data.

## The diagnosis: memory is solved, knowledge is not

`experiments/gap_by_piece.py` splits the paired bits of two scorings by what
each byte's context offers: whether the eight bytes before it ran earlier in
the stream, and whether the byte that followed them then is the byte to
predict. At full size:

| Context | Share of bytes | Stack minus Transformer |
|---|---|---|
| Copyable from the last 2 KB | 13 to 17% | **-0.09** (the stack is better) |
| Copyable only from 2 to 8 KB back | 4% | **-0.31 to -0.32** |
| New context | 74 to 77% | **+0.057 to +0.066** |

The stack remembers better than the Transformer. The whole gap is
**knowledge**: predicting a byte from what the model learned in training
rather than from what it can copy. The best stack today, fsp-lift, has the
same shape with a smaller last row: -0.087 on near copies, -0.40 to -0.44 on
far ones, and +0.044 (test) to +0.052 (validation) on new contexts
(`full_size_lifted_loop_run.json`). With power-law cells, read without fast
output synapses (fsp-lift2-s2), the last row shrinks again: -0.054 (test) and
-0.070 (validation) on near copies, -0.40 and -0.38 on far ones, and +0.025 and
+0.030 on new contexts. Validation holds fewer copyable bytes than test (13.5%
near against 16.6%), so the same model is ahead on test and level on
validation. Spikes suit memory, because copying is
exact matching. Knowledge is many fine-grained patterns held in synapses on
top of each other, and that is where the work has to go.

How the gap moves with data depends on the stack. For the plain test bed,
trained in windows, it halves with four times the data (0.148 to 0.077). The
delayed test bed trained on streams learns much faster at first: at 6,000
updates it is level with its Transformer on new contexts and 0.047 ahead in
all. With four times the data it falls 0.054 behind, 0.083 on new contexts,
while its copying stays far better. Its learning curve flattens where the
Transformer's keeps falling (`gap_with_data.json`, figure
`results/figures/event_lm_gap_with_data.png`).

**What this says about the next step.** Every part that helped knowledge so
far helped early: delayed fibres, training on streams, knowledge cells (whose
lead shrinks within a run from 0.022 to 0.011, `knowledge_data_scaling.json`).
None changed the slope. What closes the gap must keep the stack learning
knowledge late in training: how its synapses learn through spikes, not one
more part beside them.

**What v18 and the day after added.** Three findings narrow the search to the
cell itself.

- The cell's shape is the first change whose gain on knowledge did not fade.
  The full size's shape is 0.020 better on new contexts after 24,000 updates,
  at each of four seeds (`cell_shape_with_data.json`).
- The Transformer's code is no sparser than the stack's. At 3e-4, 40 to 57%
  of its units are open at a byte, against 35 to 49% of the stack's cells
  firing (`unit_density.json`, a diagnostic). Sparse codes are not the lever.
  At the rates where the models train best the stack's code is the thinner
  one: 16% of a level's cells fire at full size at 6e-4 and 7% at 1.2e-3,
  where 30% and 26% of the Transformer's units are open, and half of its
  units carry nine tenths of what a block passes on at every rate
  (`unit_density_by_rate.json`). A closed unit still passes a share. A
  silent cell passes nothing.
- The largest priced difference between the two models is still the unit.
  The Transformer's multiplies a gate by a value; the stack's cells add and
  threshold (`docs/STREAM_FORM.md`, 0.025 at 9,000 updates).

## The target: a scaling law the other way around

A stack that beats the Transformer must learn more from each added byte, not less: its gap must shrink as both see
more data. From 2026-11-22 every design is judged by the **slope of its gap per doubling of the training data**, over
separate runs at 6,000 to 96,000 updates of the test bed (24.6M to 393M bytes), each model at its best learning rate
(`results/event_lm/knowledge_gap/data_scaling_long.json`, analysis `experiments/data_scaling_long.py`, figure
`results/figures/event_lm_data_scaling_long.png`). Screens at 6,000 updates chose early learners: delayed fibres,
training on streams and knowledge cells all helped most early and faded with data. Two facts point the way:

- The plain test bed's gap on long streams halves from 6,000 to 24,000 updates (0.148 to 0.077); the delayed bed
  trained on streams goes from 0.047 ahead to 0.054 behind (`gap_with_data.json`).
- At 24,000 updates the stack's layout without spikes, a diagnostic stand-in, is 0.022 ahead of the spiking stack on
  the trainer's windows, where at 6,000 updates of the same runs the two were level (`data_scaling.json`): part of the
  flattening comes from the spikes' grid. Every synapse sends whole ticks of charge, and late in training most
  updates are smaller than a tick. Finer grids and quantal release, synapses that round at random while training so
  that each step moves their expected charge (`--weight-dither`), are tested in `weight_precision_with_data.json`.

Both models gain from a higher learning rate, and about alike: over two seeds the stack gains 0.063 at 1.2e-3 and the
Transformer 0.053 at 2.4e-3, a difference of 0.010 where 0.020 was asked. The flattening is not an effect of the rate
(`learning_rate_with_data.json`), and every comparison now takes each model at its best rate.

Until now every full-size comparison has been at one length, 90,000 updates, which shows the gap's level but not
its slope. The weekend of 2026-11-28 measures the slope at full size: fsp-lift2's stack and fsp-tf's Transformer,
each trained for 45,000, 90,000 and 180,000 updates with nothing else changed (`full_size_data_slope.json`). If the
gap shrinks by at least 0.005 per doubling, more data is the way past the Transformer; if it grows, the stack must
change how it learns before it is scaled.

The stack trained about ten times slower than the Transformer (16 hours against 1.6 for 90,000 updates), almost all
of it in the stream stage, where each race read compared every byte with every cell of its table. A read by age
(`snnformer/event_lm.py`, `tests/test_read_by_age.py`) works each byte out only against the cells within its reach:
the same pulses, spike for spike, at half the work. On the server's GPU a stream-stage update takes 0.42 s instead of
0.81 s, and a trained model is scored to the same bits.

**Fused reads** (`snnformer/race_kernel.py`, the trainer's `--fused-reads`, `tests/test_race_kernel.py`) go further.
Four GPU kernels of our own, written in Triton, run a read's whole race in blocks, forward and backward, and never
hold the table of every byte against every cell in reach. At every weight they fire the read by age's spikes, count
for count; the gradient agrees within 7e-6 of each tensor's largest entry, and two runs agree to the bit. A read with
its backward pass takes 3.2 ms for 15.9, a full-size stream update 0.18 s for 0.42, in 9 GiB of memory for 41, so a
full-size run of 90,000 updates trains in under five hours where it took 16. Four smaller options stand
beside them (`experiments/speed_options.py`): the products of whole spike counts and whole synapse ticks on the
GPU's tensor cores, to the bit (`--exact-products`, 1 to 5% of an update); the optimizer's steps over all tensors
at once; the window stage replayed from a CUDA graph; and compiled cells. They fall into two classes. Exact
products, the fused optimizer and the graph leave a run's weights as they were to the bit. Fused reads and compiled
cells fire the same spikes at the same weights but round the gradient differently, and that is enough to make
another run: from the same weights and bytes, two reads that differ only in rounding hold different weights after two
updates and are as far apart after 14 as the updates moved them, while each repeats itself to the bit. A model
trained with them is therefore another run of the same recipe, as another seed is, and any trained model scores
the same bits by either read. The scaling test's two new stack runs train with fused reads: a check on the deployed
code met the six conditions registered before it ran (`full_size_data_slope.json`, its third amendment and second
code check).

On the test bed the same options give less, because a small model's update is spent launching kernels more than in
them (2026-11-27, 600 updates on streams, each way twice): 0.056 s an update with reads by age, 0.053 with exact
products (the weights the same to the bit), 0.050 with fused reads beside them, and 0.036 with fused reads and
compiled cells; the looped, lifted bed takes 0.106, 0.095 and 0.084. From that day every newly registered run trains
on the fast path of its size, the base of its comparison included, since a run with fused reads or compiled cells is
another run; runs registered earlier keep the path they were registered on. The practice loop of the one-model plan
takes exact products and fused reads as well and has moved to the server's GPU, where a round of the bed's stack
takes 0.65 s against 1.1 s on the laptop and several of the stack's runs share the GPU (six side by side took 0.8
to 1.8 s a round each, nine 1.4 to 3.1 s). The two models take turns there: with the Transformer's runs beside
them the stack's runs took 8 to 40 s a round.

## Rules for claims

1. **Spiking and streaming only.** The stack is built, trained and scored as
   a stream. The Transformer reference may read windows, as it was trained.
   Non-spiking models may help train it, as teachers, task generators, verifiers
   or graders (Daniel, 2026-11-25), but the model that is released and scored is
   purely spiking, with no float layer, twin or hybrid in it.
2. **Equal parameters and equal training data** for every comparison with
   the Transformer; whatever a teacher gives one of them, the other gets too. Pulses per byte, the stack's energy, are reported beside
   it. The budget may be raised where the evidence says that is what closes
   the gap (Daniel, 2026-11-22), but only as an arm of its own against the
   old budget, never hidden inside another change, and every claim states
   what the extra pulses cost.
3. **Pre-register every test.** Write the question, the arms, the measure and
   the pass mark, and commit them before the first result. Take the
   record's time from the commit. Record every verdict, failures included.
4. **Judge on long unbroken streams**: 2 million bytes as 16 streams of about
   125 KB. The trainer's streams of about 1.5 KB have misled before: kept
   cells won there by 0.013 and lost by 0.014 over long streams. Split every
   gain by what the context offers.
5. **Seeds before claims.** On the test bed, seeds differ by up to 0.03 and
   copying strength by about a bit a byte. A screen is one seed, or one seed
   on each of two devices. A screen that passes earns four seeds, and four
   seeds earn a full-size run, which decides.
6. **Test-bed results about context do not carry to full size by
   themselves.** The test bed reaches 256 cells back, the full size 8,192.
   Knowledge cells helped the test bed copy and did nothing at full size.
   Check a result on knowledge at four times the data (24,000 updates) before
   spending a full-size run on it.

## What has been tried, and what it taught

| Idea | Verdict | Lesson | Record |
|---|---|---|---|
| Delayed fibres (cells and reads hear the last 4 bytes side by side) | **0.11 better** on the test bed | Precise access to recent bytes is knowledge | `v17_candidates/test_bed_with_delays.json` |
| Training on streams | **0.029 better** on long streams (test bed) | Train as the model is used | `knowledge_gap/window_and_stream_training.json` |
| Knowledge cells (learned ageless cells in the race) | Pass on the test bed, none at full size | **One cell per head gives the whole copying gain: an attention sink** | `knowledge_and_plasticity.json`, `full_size.json` |
| Kept cells (keep winning cells beyond the reach) | Fail | Help in the first 8 KB, cost later: they never age, and the stack never trained that long | `kept_cells.json` |
| Eight fixed delays | Fail | A thinner fixed share of the same fibres | `more_delays.json` |
| Narrower reads with more cells | Fail | Knowledge needs the reads as much as the cells | `memory_for_knowledge.json` |
| Fast synapses keyed by knowledge cells | Fail at all four seeds | The keys swamp the shared pool | `knowledge_keyed_synapses.json` |
| Trained plasticity with knowledge cells | Fail | They get in each other's way | `knowledge_and_plasticity.json` |
| Branching dendrites, apical bursts, racing experts | Fail | Splitting or gating cells at equal parameters does not add knowledge | `knowledge_gap/preregistered.json` |
| Looped levels | A tied pass is worth about a third of a level | Depth needs its own synapses | `results/event_lm/stack_loop/preregistered.json` |
| Looped levels at full size (levels 2 to 8 run twice, tied synapses, the budget scaled with the passes) | **Pass**: 0.019 ahead of the plain stack given the same pulses, 0.026 on new contexts; 0.010 ahead of the plain stack at the old budget | The first part at equal parameters that adds knowledge at full size, for 1.63 times the pulses and 1.4 times the training time | `knowledge_gap/full_size_loop_pilot.json` |
| A raised pulse budget alone at full size (507,000 for 310,000 a byte) | Fail: 0.009 behind | More pulses in the same levels saturate the membranes | `knowledge_gap/full_size_loop_pilot.json` |
| Echo learning, consolidating fast synapses, doubled pulse budget, squared cells | Fail | See `docs/V16_HANDOFF.md`, "What not to try again" | |
| The full size's cell shape on the test bed (every channel's present byte and one delayed copy) | **Pass at four seeds**: 0.015 ahead at 24,000 updates, 0.020 on new contexts | The first gain on knowledge that did not fade with data, and it is in the cell | `knowledge_gap/cell_shape_with_data.json` |
| Word heads at full size | Fail by their mark: 0.0009 ahead where 0.003 was asked; the best spiking score so far | Memory again, not knowledge | `knowledge_gap/full_size_word_heads.json` |
| Gated cells at full size (a gate dendrite as a shunt) | Fail: 0.118 behind | Their gate dendrites sit at the membrane's bound a fifth of the time | `knowledge_gap/full_size_unit_screen.json` |
| Noisy thresholds at full size | Level: 0.004 better, none of it on new contexts | A smooth slope alone is not the unit's lead | `knowledge_gap/full_size_unit_screen.json` |
| Self-gated cells at full size | Level: 0.001 behind, 0.006 behind on new contexts | A cell that gates itself is a plain cell | `knowledge_gap/full_size_unit_screen.json` |
| Gated cells under the pool that lifts the ceiling, at full size | Fail: 0.113 behind the lifted plain stack | Saturation was not why gated cells lose: with no cell at its cap the loss is the same | `knowledge_gap/full_size_lifted_gate.json` |
| A proportional gate under the lifting pool, on the test bed by slope, two seeds | Fail: 0.065 and 0.048 behind at 6,000 updates, 0.802 and 0.653 at 24,000 | With data the graded gate turns into a switch, and its synapses grow until the range rule cuts thousands of neurons; nothing is at a cap | `knowledge_gap/proportional_gates_with_data.json` |
| Noisy thresholds on the test bed, by slope | Reported only: 0.0036 ahead at 6,000 updates, 0.0019 at 24,000 | A true slope for the threshold cell is worth 0.002 to 0.004 and fades with data | `knowledge_gap/noisy_thresholds_with_data.json` |
| Output spikes of 1/16 bit in place of 1/128 (`--output-step 8`), on trained weights | 34% fewer pulses on the test bed and 21% fewer at full size, for 0.00004 and 0.00002 bits per byte | The output's spike was the one spike whose weight no synapse learns, and it was finer than the score needs: an energy saving, no gain in bits | `knowledge_gap/output_step_on_trained_weights.json` |
| The pool that lifts the ceiling, at full size | Removes saturation at the same energy; 0.007 ahead of the plain best where 0.010 was asked | The membranes cap the stack's learning rate; lifting frees the rate but adds no knowledge | `knowledge_gap/full_size_lifted_ceiling_pilot.json` |
| The lifting pool under the loop, at full length (fsp-lift) | **Pass**: 0.022 better than the loop on test and 0.025 on validation at the same pulses, 0.034 of it on new contexts; 0.003 and 0.009 behind the Transformer | With no membrane at a bound the loop keeps learning knowledge late: the pilot's gain of 0.0095 more than doubled with length | `knowledge_gap/full_size_lifted_loop_run.json` |
| A higher learning rate as the cause of the flattening | Ruled out over two seeds | Both models gain alike from their best rates | `knowledge_gap/learning_rate_with_data.json` |
| A sparser code | Not built | At 3e-4 the Transformer's units are as dense as the stack's cells, and at the best rates the stack's code is already the thinner one | `knowledge_gap/unit_density.json`, `knowledge_gap/unit_density_by_rate.json` |
| A census of cells and units in use at each peak rate, on trained weights | The stack's code thins as the rate rises (30%, 16% and 7% of a level's cells fire at full size at 3e-4, 6e-4 and 1.2e-3) and with data; half of the Transformer's units carry nine tenths of a block's output at every rate, against 11% of the stack's cells at its best rate; no cell dies | A closed unit still passes a share and a silent cell passes nothing; not shown to cost bits, since the gap is no smaller at 3e-4 | `knowledge_gap/unit_density_by_rate.json` |

## Open bets, in order

1. **The full-size pair, each model at its best learning rate** (`full_size_best_rate_pair.json`, read): after
   90,000 updates the plain stack at 6e-4 scores 1.1129 on test, the best spike-only score so far, and the
   Transformer at 1.2e-3 scores 1.0790. The gap at each model's best rate is 0.034 on test and 0.042 on
   validation, against 0.010 and 0.019 at 3e-4: the faster rate pays the Transformer 0.028 and the stack 0.004
   once both have seen all their data, where after 24,000 updates the two gains were alike. The stack stays ahead
   on every copy and is 0.09 behind on new contexts. A float stand-in at full size closes a fifth of the distance
   and leaves four fifths (`full_size_spikes_late.json`), so what closes the gap must change what the blocks
   compute and how they keep learning with data, not only their spikes. Next at full size: the loop's run of
   90,000 updates (bet 5), the part that adds knowledge at equal parameters.
2. **The unit, in spikes: a cell that multiplies.** The Transformer's unit multiplies a gate by a value; the stack's
   cells add and threshold. v15's priced walk put plain units in place of gated ones as the largest step from the
   Transformer to the stack (0.025, against 0.001 for whole spikes). Of that, a hard gate recovers 0.013 and a
   smooth plain unit 0.016 (`docs/STREAM_FORM.md`). At 3e-4 the Transformer's units are as dense as the stack's cells
   (`unit_density.json`), so the difference is not a sparser code. At the rates where the models train best the
   stack's code is the thinner one, and it differs in what a unit below its threshold does: half of the
   Transformer's units carry nine tenths of what a block passes on, because a smooth gate lets a share through below
   zero, where a silent cell passes nothing (`unit_density_by_rate.json`). The spike forms so far:
   - a gate dendrite as a shunt, where k gate spikes let in the share k / (k + 4) of the value dendrite's charge:
     0.118 behind at full size, with its gate dendrites at the membrane's bound a fifth of the time
     (`full_size_unit_screen.json`). On the laptop's seed of the test bed it is 0.067 behind at 6,000 updates and
     0.108 behind at 24,000 (`gated_cells_with_data.json`). Its gates' own steps were a third of its pulses at
     6,000 updates and a fifth at 24,000 (23,454 of 112,207 a byte, `experiments/gate_state.py`), where the reads
     lost 30% of their spikes and the output 17%. A gate whose steps stay inside the cell (`--granule-inner-gate`)
     spends pulses as the plain stack does, but in a check of 2,250 updates it is still 0.066 behind, so that
     energy tax is the smaller part of the loss (`proportional_gate_check.json`);
   - the same under the pool that lifts the ceiling: 0.113 behind the lifted plain stack at full size, as far
     as without the lift, although the cells' bodies no longer reach their cap and the saturated share among
     the gates' populations falls from 21% to 6.5% (`full_size_lifted_gate.json`). Saturation was not the
     reason. On the laptop's seed of the test bed it is 0.078 behind at 6,000 updates; its longer run was
     dropped (`lifted_gates_with_data.json`);
   - self-gated cells: level on the test bed at 6,000 and at 24,000 updates (0.0014 and 0.0016 behind), with no
     growth, so they earn no seeds (`gated_cells_with_data.json`), and level at full size (0.001 behind, 0.006
     behind on new contexts, `full_size_unit_screen.json`);
   - noisy thresholds, where membrane noise makes a cell smooth on average and it learns from that true slope:
     level at full size (0.004 better, none of it on new contexts), and on the test bed 0.0036 ahead at 6,000
     updates and 0.0019 at 24,000, a lead that shrinks with data, so they earn no seeds
     (`noisy_thresholds_with_data.json`).

   Built, checked, and failed once it had data: **a proportional gate** (`--granule-ramp 16 --granule-inner-gate`). The gate dendrite fires
   four steps for every threshold of its charge above the rest, and k steps let in k / 64 of the value dendrite's
   charge. Every step lets in the same share, the gate is wide open at 16 thresholds, and it never lets in more than
   the whole. Its steps stay inside the cell, as a dendritic spike sends nothing down the axon. In a check of 2,250
   updates (`proportional_gate_check.json`) it is the first gated cell that keeps pace with plain cells: under the
   lifting pool it leads by 0.088 at 750 updates and is level at 2,250, where the shunting gates trail by 0.063 to
   0.080. At that point its gates are graded. Sorted by the share of the value that an open gate lets in (up to a quarter, a
   half, three quarters, or more), they spread 40 / 27 / 16 / 17% at the first level and 28 / 23 / 17 / 32% at the
   last (`experiments/gate_state.py`). Every gate that lost has half to all of its open states in the last quarter
   (48% to 100%, the shunting gates 66% to 79%): it works as a switch. A narrow ramp (4 thresholds), or any ramp
   under the bounded pool, saturates into a switch and runs away. The inner steps are not pulses and the budget
   counts none, but they are taken: 47,915 a byte beside the stack's 112,663 pulses, a cost that every report of
   this gate states.

   The screen by slope failed on both seeds of the server's fan2 bed (`proportional_gates_with_data.json`): the
   gate is 0.065 and 0.048 behind at 6,000 updates and 0.802 and 0.653 behind at 24,000. With data it does not
   stay graded. An open gate lets in 0.57 to 0.74 of its value at 6,000 updates and 0.82 to 0.87 at 24,000, where
   three quarters or more of the open gates stand in the last quarter: it turns into the switch. Its synapses grow
   until the range rule shrinks 9,000 to 14,000 neurons, the stream's neurons fire twice as much, and the output
   is left with a fifth of its pulses. Nothing is at a cap, so this is not saturation, and it starts after the
   rate has fallen below its peak. The full-size run was not released (`full_size_proportional_gate.json`), and
   the laptop's seed was not run. Untried: a lower rate or a decay on the gate's drive, and a set point, a resting
   level that keeps each gate open at about half of the bytes as cortical neurons keep their excitability in
   range. They wait for the float diagnostic below: if a gated unit earns nothing at this size in floats, no spike
   form of it can. Built for v20 (`--granule-gate-home SHARE`, c750cac): homeostatic gates. The
   proportional gate failed because its synapses grew until the range rule shrank 9,000 to 14,000 neurons, so a set
   point on the rest alone would chase that growth. Cortex holds a neuron in range by two mechanisms at once:
   synaptic scaling, which keeps the total strength of its synapses fixed, and intrinsic plasticity, which moves
   its excitability toward a target rate. Here a ramp gate's synapses keep the total they had at the first byte,
   and a local rule moves its rest, outside the gradient and only while training, toward a let-in share of one
   half on the mean, so that a gate can only be graded. Its screen on the bed where the proportional gate turned
   into a switch is pre-registered and queued, held until `bounded_gates_in_floats.json` has been read
   (`homeostatic_gates_with_data.json`); `full_size_units_in_floats.json` prices what a working cell could win.
   **Result:** the homeostatic gate is the first spiking gate that caught up with data (0.020 behind the plain cell
   after 6,000 updates, 0.005 after 24,000), but by 96,000 updates it falls behind again, 0.0097 on the mean of two
   seeds (`homeostatic_gate_at_length.json`), with gates that stay graded and the base's pulses. The signed gate,
   a product of two drives of either sign, fails on three seeds (`signed_gates_with_data.json`). No form of the
   gated cell is open on the bed; depth by recurrence is the line that continues.

   Two controls in floats would say whether the unit also explains the gap late in training, since the walk priced
   it at 9,000 updates: a Transformer with plain units on the test bed at 6,000 and 24,000 updates (does it flatten
   as the stack does?), and the stack in floats at full size at 24,000 updates (do spikes cost anything late, on a
   charge grid 16 times finer than the test bed's?). They are diagnostics and never a path. The first is
   pre-registered and runs on the server's CPU with all four kinds of unit, plain or gated, hard or smooth
   (`plain_units_with_data.json`). Along the walk's full-size curves the two hard kinds fell further behind as
   training went on (the plain one from 0.008 to 0.027, the gated one from 0.001 to 0.014) and the smooth plain
   unit did not (0.008 to 0.011), one seed with unequal schedules.
   The second control is pre-registered as well (`full_size_spikes_late.json`): at 3e-4 and 24,000 updates, the
   full-size stack beside a stand-in whose reads are softmax shares and whose cells answer in real numbers. The
   stand-in's switches are not wired to the pool of fine quanta, so a spiking arm on the coarse pool trains
   beside it, and both are read against the model's own pilot at that rate (1.3416) and the Transformer's
   (1.3057). If the stand-in is less than 0.005 ahead of the model, spikes in reads and cells cost nothing late
   and the distance is the layout's; if it is 0.015 or more ahead, stand-ins with only the reads or only the
   cells in floats say which part's learning through spikes to work on. A third arm asks whether the stand-in
   gains from the Transformer's rate where the spiking stack cannot.
   Read for v19: the stand-in is 0.0069 ahead of the model after 24,000 updates, between the marks, and 0.0290
   behind the Transformer (0.033 on new contexts); on one pool spikes in reads and cells cost 0.0128. Four
   fifths of the distance belong to the layout. The half stand-ins and the third arm run next.
   Read for v19 (`plain_units_with_data.json`): the plain hard unit, the stack's cell in floats, falls behind the
   gated reference as data grows (growth 0.029 on the mean of two seeds); a smooth plain unit
   and a hard gated unit each end within about 0.005 on the mean, with seeds that disagree (growth
   0.007 and 0.008, unclear). A smoother cell and a cell that multiplies both
   remain candidates; `bounded_gates_in_floats.json` asks whether a gate may be bounded.
   Read for v20 (`full_size_units_in_floats.json`): at full size the Transformer with the stack's kind of unit is
   0.0077 behind its own after 24,000 updates, under a quarter of the spiking stack's distance, so the unit is a
   small part of what the layout costs. A unit that multiplies is still 0.015 better than the plain hard one, with
   a code as thin. The rest of the layout is walked next in floats (`full_size_layout_in_floats.json`, queued): the
   stack's kind of pool in place of the Transformer's LayerNorms, and an output layer with weights of its own.
   **Read: the pool is the largest part found.** In place of every LayerNorm the stack's pool costs the
   Transformer 0.069 after 24,000 updates at full size (new contexts +0.084, far copies +0.075), three times
   what the layout was thought to hold; an output with weights of its own makes it 0.008 better, so that part is
   none of the gap. The pool divides by the stream's activity but takes no mean out. A float control with a
   centred pool (`full_size_centring_in_floats.json`) says which half costs. Built beside it: **balanced
   synapses** (`--balanced-synapses`, aa37176, `balanced_synapses.json`). In cortex the excitation and inhibition
   onto a neuron balance, so it answers the pattern of its inputs and not their level. Here every synapse onto a
   read's probe, tag and payload neurons, a plain cell and an output neuron is held at the neuron's mean synapse
   less, in whole ticks, so each neuron's synapses sum to zero over the stream's channels: a level added to every
   channel moves no charge, the charge stays exact, the parameters are the same, and the pool still divides. Two
   bed seeds and a full-size pilot run if the control reads centring. **The control reads the division**: the
   centred pool still costs the Transformer 0.064, so centring recovers 8% and balanced synapses wait. The pool
   divides by the stream's mean absolute activity where a norm divides by its root mean square; the two part where
   a few channels are loud. Cortex divides by energy: in the normalization model a neuron's response is divided by
   the pooled squared activity of its neighbours. **Energy pools** (`--pool-energy`, 91079d6, `energy_pool.json`)
   let every pool of the stack hear the root of the stream's summed squared counts times its width, the summed
   counts where the active channels fire alike and more where a few are loud: whole counts, the same spikes
   however a stream is cut, no new parameters. A float control that divides by the root mean square, a full-size
   pilot and two bed seeds run in the night of 2026-11-25. **The float control finds the price**: a pool that
   divides by the root mean square costs the Transformer only 0.0020 against its LayerNorm, so 97% of the pool's
   0.069 is dividing by the mean absolute activity in place of the energy; no centring is needed. **The spiking
   energy pool fails on the bed** (after 24,000 updates one seed 0.0105 behind, one 0.0045 ahead, at equal pulses)
   **and at full size**: with the membranes' ceiling lifted, so that a stronger pool takes no range from the
   neurons, the lifted loop with energy pools is 0.0040 behind the same loop (new contexts 0.008, far copies 0.012),
   and energy pools stay off. What a Transformer loses with the stack's pool is not what the stack loses with it,
   so the layout walk's largest part is not yet a recipe for the stack. The guess that a Transformer's stream has a
   few loud channels the stack's capped counts lack is wrong (`stream_loudness.json`): both trained streams are
   dense and close to Gaussian, and the ratio of their root mean square to their mean absolute value stays within a
   few percent of 1.25 from token to token, so after training a division by mean activity is a division by the
   root mean square times a constant. The pool's price lies in how it trains. A division by mean activity passes
   the stream's sign pattern back through its divisor, a division by the root mean square the stream itself.
   **Pools that train with the energy's gradient** (`--pool-energy-gradient`, `pool_gradient.json`) hear the summed
   counts as before, so the stack fires the same spikes, and pass the energy's gradient: a float control with the
   pool's forward and the root mean square's gradient (`gpool`), a full-size pilot and two bed seeds run on GPU 1
   once fsp-lift is scored. **The float control is read: the gradient is the pool's whole price.** The Transformer
   whose every norm is the stack's pool, trained with the energy's gradient, scores 1.2973: 0.0084 better than with
   its LayerNorms (0.0075 to 0.0092), where the same pool trained through its own divisor is 0.0691 worse. Both
   pools compute the same division, so the 0.078 between them is the gradient that the divisor passes and nothing
   else. That was the reason to expect a gain in the stack, and **the stack gains nothing**: at full size the lifted
   loop whose pools train with the energy's gradient is 0.0011 behind the same loop (0.0003 to 0.0019), and on the
   bed after 24,000 updates one seed is 0.0027 behind and one 0.0018 ahead, a fail by the shared reading, so these
   pools stay off. Twice now, with energy pools and with their gradient, what the stack's pool costs a Transformer
   has not been what it costs the stack: the walk of the layout in floats explains the pool's price in a
   Transformer in full, and it is no recipe for the stack.
   **Power-law cells** (`--granule-power P`, `power_cells_screen.json`, on the laptop tonight): a cortical neuron's
   rate grows with a power of about two of how far its membrane stands past the threshold. A power-law cell fires
   round(k * k / P) spikes for k thresholds of charge, up to its cap, as a plain cell does at P thresholds: the
   spike form of a squared rectifier, which in floats recovered what plain units lose (v15's walk: 1.5087 against
   GELU's 1.5094 and the plain rectifier's 1.5249) without a second drive. It has a plain cell's synapses and no
   gate, where every spiking gate has failed; the squared cell tried in v15 fired k * k spikes with no scale and
   collapsed. On the laptop's seed it led the plain bed by 0.0262 after 6,000 updates at equal parameters and
   pulses, but by **0.0045 after 24,000** (`power_cells_screen.json`): it learns faster early and the plain cells
   catch up as the rate decays. At 24,000 updates it is still 0.010 better on new contexts and 0.040 on far copies,
   and loses that on contexts seen before with another follower (0.095 worse on 5% of bytes). Under the screen's
   mark; the server seeds (`power_cells_with_data.json`) run behind the night's recipe pilots. A cell whose count
   grows more gently than a square, or a smaller P, is the open question. **The two server seeds read a constant
   gain**: after 24,000 updates power-law cells lead the bed by 0.0100 and 0.0156 (0.0128 on the mean, both seeds
   ahead) at equal parameters and pulses, most of it on new contexts, after 0.0156 at 6,000 updates. The full-size
   pilot (`full_size_power_cells.json`) and two more seeds run the same night; power-law cells join the next
   full-length run only if four seeds hold and the pilot is 0.005 better than the lifted loop. **The full-size
   pilot passes**: after 24,000 updates the lifted loop with power-law cells scores 1.2870, 0.0137 better than the
   same loop with plain cells (interval -0.0145 to -0.0129) at equal parameters and 0.4% more pulses, on new
   contexts (0.018) and far copies (0.019), at no cost in parameters and almost none in pulses. Its code is
   sparser: in every level about half as many cells fire at a byte (5 to 25% against 10 to 44%), and the cells that
   fire have a longer tail. That is 0.0033 behind the rate pilot's Transformer of the same length at the same rate,
   and 0.0108 behind it at its best rate. **Four seeds hold**: after 24,000 updates power-law cells lead the bed on
   every seed (0.0100, 0.0156, 0.0100, 0.0143; 0.0125 on the mean) at equal pulses, a constant gain. After 6,000
   updates they led by 0.0176, so the plain cells catch up part of the way with data. Power-law cells therefore join
   the next full-length run, **fsp-lift2** (`full_size_lifted_loop_second.json`): the lifted loop with power-law
   cells, on GPU 0 alone from the evening of 2026-11-26, read against fsp-lift, fsp-loop and the Transformer.
   **By its registered measure it fails; without the fast output synapses it is the best stack yet.** Read with
   fast output synapses at 2^24, as every full-size stack since v19, fsp-lift2 is 0.0027 worse than fsp-lift on
   test and 0.0034 on validation, and 0.0055 and 0.0121 behind the Transformer. The synapses cost this run 0.0092
   and 0.0115: they still help copies, and they cost new contexts 0.026 to 0.028. Read without them, at the reach
   of 8,192, fsp-lift2 scores 1.0752 on test and 1.1054 on validation. Power-law cells are then 0.0074 better than
   plain cells on both splits, mostly on new contexts, so the bed's gain did carry to full length. Against the
   Transformer, neither model using fast synapses, the stack is 0.0038 ahead on test (interval 0.0027 to 0.0047)
   and 0.0006 behind on validation (0.0004 ahead to 0.0016 behind). That is not the measure the record named, so
   it decides nothing. A second seed is registered with this measure fixed in advance
   (`full_size_power_cells_second_seed.json`), trained on the fast path. **The second seed repeats the first,
   and it is not a win by the rule.** fsp-lift2-s2 (seed 20002, 4.99 hours of training on the fast path against
   16.5 for the first seed) scores 1.0751 on test and 1.1051 on validation without fast output synapses: 0.0039
   ahead of the Transformer on test (interval 0.0029 to 0.0048) and 0.0003 behind on validation (0.0007 ahead to
   0.0013 behind). The two seeds' mean says the same (0.0038 ahead, 0.0004 behind). The seeds lie 0.0001 and 0.0003
   apart, the first measure of the spread at full size, so the lead on test is the recipe's, not luck. The fast
   output synapses cost it 0.0097 and 0.0119 again, so power-law stacks are read without them from here on, and a
   scan of their threshold is owed. What is left is new contexts: 0.025 to 0.030 a byte behind on three quarters
   of the bytes, against 0.05 to 0.40 ahead on the bytes the context offers. **In the loop the gain is larger**: on the
   laptop's seed, after 24,000 updates, the looped bed with power-law cells leads the looped bed with plain cells
   by 0.0095 (interval 0.0086 to 0.0105) at equal pulses (`power_cells_in_loop_screen.json`), twice the same seed's
   lead on the plain bed, again on new contexts (0.015 better on 76% of bytes) with near copies 0.029 worse.
   **Next on this line.** Cortical rates grow with powers of about 2 to 4, so the square is one point of a family:
   cubic cells (`--granule-power-exponent 3`, round(k^3 / 64) spikes for k thresholds, the same as a square cell at 8
   thresholds; `cubic_cells_with_data.json`) keep every cell under 4 thresholds silent. They run on two bed seeds, a
   full-size pilot against the lifted loop with square cells (1.2870), and the laptop's seed. **On the bed the cube
   is worse**: after 24,000 updates it trails the square by 0.0308 and 0.0272 on the server's two seeds (intervals
   0.0297 to 0.0320 and 0.0262 to 0.0283) and by 0.0278 on the laptop's, at equal pulses, mostly on new contexts
   (0.035 a byte on three quarters of the bytes) and on far copies. The steeper power silences too much. Its
   full-size pilot is still to be scored. Two cheap full-size
   pilots on the same loop needed no new code (`full_size_power_loop_variants.json`): skip fibres, which gave the
   looped bed a steady 0.006 under their own mark, and twice the rate, which gave the lifted loop 0.003. **Neither
   passes.** Skip fibres are 0.0011 better (interval 0.0003 to 0.0019), under the mark of 0.005, so they stay out.
   Twice the rate is 0.0128 worse (0.0119 to 0.0136): plain cells gained 0.003 from it, and power-law cells, which
   gain 0.0137 at 6e-4, cost 0.0022 at 1.2e-3. The cells' gain is tied to the rate, and the next full-length run
   keeps fsp-lift2's recipe as it is.
3. **What holds late learning back: the charge grid and weight decay.** Does the charge grid limit late learning
   (`weight_precision_with_data.json`; at the best rate `randomness_at_best_rate.json`)? A typical full-size synapse
   is about 8 ticks from zero, so a tick is an eighth of it, and on the test bed a float stand-in that was level
   with the spiking bed after 6,000 updates was 0.022 ahead after 24,000. A laptop screen puts the reads' and cells'
   synapses on a grid four times finer at the bed's best rate (`charge_grid_at_best_rate.json`); **it is no limit at
   this length**: on the finer grid the bed is 0.0066 behind after 24,000 updates and 0.0086 after 6,000. Quantal release,
   which rounds at random instead, was behind on the laptop's seed. **Weight decay on synapses only**
   (`--decay-synapses-only`, `decay_synapses_only_with_data.json` and `full_size_decay_synapses_only.json`): the
   trainer has decayed every parameter of two dimensions, a read's gain, grain and offsets by age and a cell's resting
   charge among them, pulling each toward its starting value as if it were a synapse, and the more so at a higher
   rate. A Transformer never decays its norms or temperatures. In the released looped stack the read gains settled at
   -1.3 to -3.2 and the rests sank from 0.5 to about 0.2, both against that pull. The screen on the bed and a
   full-size pilot run the same night; at no cost in parameters or pulses, a lead of 0.003 adopts it. **The bed's
   screen fails**: after 24,000 updates one seed is 0.0034 behind and the other 0.0040 ahead, a mean of 0.0003, and
   the lead shrank from 6,000 updates, so it is not made a default. The full-size pilot keeps its own reading.
4. **The data-scaling curves** (`data_scaling_long.json`, running): which design's gap shrinks with data. It becomes
   the base for the rest.
5. **Depth by recurrence at full size** (`full_size_loop_pilot.json`, passed): a looped stack runs its middle levels
   twice with tied synapses, more computation per parameter, as cortex iterates. On the test bed a loop gained 0.047
   and 0.062 over depth 6 at equal parameters, with a budget scaled to its passes. At full size, after 24,000
   updates, the looped stack scores 1.3102: 0.019 ahead of the plain stack given the same pulses (0.026 on new
   contexts) and 0.010 ahead of the plain stack at the old budget (0.010 on new contexts), the first part at equal
   parameters that adds knowledge at full size. It spends 1.63 times the pulses and 1.4 times the training time.
   On the test bed the loop holds with data (`loop_with_data.json`): 0.025 ahead of the plain bed given the same
   pulses after 6,000 updates and 0.017 after 24,000 (0.013 on new contexts), and on two more seeds 0.020 and
   0.025 ahead of the plain bed. **At full length the gain holds** (`full_size_loop_run.json`): after 90,000
   updates beside the v19 pair the looped stack scores 1.1039 on test and 1.1387 on validation, 0.0090 and
   0.0086 better than the plain stack, more than nine tenths of it on new contexts, and 0.025 behind the
   Transformer on test where the plain stack is 0.034 behind. Its top membranes sit at their bound twice as often
   as the plain stack's (27% of the top cells' entries), which the lifting pool removes. The loop is the stack's
   depth from here on.
   The raised budget alone makes the plain stack 0.009 worse, because its membranes saturate. The loop has earned
   a run of 90,000 updates beside the pair, and that run is pre-registered (`full_size_loop_run.json`): the
   pair's stack arm with the loop and its budget, queued for the night of 2026-11-23 and read against the pair's
   plain stack and its Transformer. It holds if the looped stack is at least 0.005 better than the plain one on
   test and validation. Still open, and piloted beside it (`full_size_loop_variants.json`, pre-registered): whether
   the loop needs the raised budget at full size, and whether the pool that lifts the ceiling adds to it. The
   pilot picks the form of the loop's next full-length run.

   Built and pre-registered beside it: **sleep** (`--stack-sleep 64`, 79cf0a4, `sleep_with_data.json`), depth
   for the memory instead of for every byte. After every 64 bytes of a stream the stack replays the window that
   has just ended, as the hippocampus replays recent experience at rest: the stream as it left the top of the stack
   enters the first level again, every level runs once more over the window, and the replay's reads reach the
   whole window, the later bytes too, since the window is over. Gate cells in every read then decide, cell by cell
   and in quarters, how much of each stored cell the replay rewrites, so the bytes after a window read it as the
   replay saw it. It follows Lee et al. 2026 ("Sample-Efficient In-Context Adaptation via Sleep"), which reached a
   Transformer's loss with 2.1 to 2.4 times fewer context tokens. Integer spikes, exact under any cut, and with
   gates that keep every cell it is the stack without sleep. The screen runs sleep alone and on top of the loop,
   two seeds at 6,000 and 24,000 updates, against the plain and looped beds; the replay doubles the pulses a byte,
   which the budget is scaled for, and costs 2.3 times the training time. **Sleep alone fails its reading**
   (`sleep_with_data.json`): after 24,000 updates it is 0.0099 ahead of the plain bed on one seed and 0.0031 behind
   on the other, a mean of 0.0034, but its lead grew by 0.0070 from 6,000 updates, which earns it a pair at 96,000
   updates if the loop with sleep fails as well. Where it loses, the cost sits right after each window boundary
   (the first byte after a replayed window 0.011 worse), so a screen with gates that start closed, where sleep
   starts as the stack without it, runs on the laptop (`sleep_closed_gates_screen.json`, `--stack-sleep-keep`).
   **Closed gates pass the laptop's screen**: after 24,000 updates sleep with gates that start closed leads the plain
   bed by 0.0061 (interval 0.0052 to 0.0070; 0.0080 after 6,000 updates), with 0.0042 of gain on new contexts,
   where the open gates lost 0.009, for twice the pulses. Two server seeds, alone and on the looped bed, are queued
   (`sleep_closed_gates_with_data.json`); a pass by the shared reading sends sleep to a full-size pilot.

   Built and pre-registered next: **phases of a loop** (`--stack-loop-phases`, 70774fe,
   `loop_phases_with_data.json`). The passes of a loop share every synapse, so a level cannot tell its first pass
   from its second. In cortex the phases of a rhythm change a circuit's excitability, not its wiring; here every
   later pass gets a resting charge of its own for each cell and an offset of its own to each read head's gain,
   starting at zero, where the stack is the loop without them: 0.16% more parameters on the looped bed and 0.02% at
   full size, no pulses. A mean lead of 0.005 after 24,000 updates on two seeds earns a full-size pilot. **Phases
   fail**: after 24,000 updates they are behind the looped bed on both server seeds (by 0.0023 and 0.0043) and on
   the laptop's (by 0.0037), so they stop here. Whatever lets a level tell its passes apart, a resting charge and a
   gain offset of its own are not it.

   **Next full-length run** (`full_size_lifted_loop_run.json`, pre-registered): the looped stack under the pool that
   lifts the ceiling, with a third pass and the Transformer's rate if their pilots pass; a job on the server picks
   the form and queues the run as soon as the pilots are scored. Beside it a pilot asks whether the lifted loop is
   still held back by its pulses (`full_size_loop_budget.json`: twice the budget, its cost stated). **Neither pilot
   passed**, so the run trains with two passes at 6e-4 and has trained since 2026-11-25 17:15 UTC: a third pass is
   0.0008 better (`full_size_loop_passes.json`) for 1.39 times the pulses, and the Transformer's rate 0.0031 better
   (`full_size_loop_variants.json`), where the Transformer gains 0.0075 from the same step. **Nor do more pulses
   help**: given twice the budget the lifted loop fires 1.51 times the pulses and is 0.0005 worse, so the budget
   stays at 507,000 a byte. **Read: the lifted loop is the best spiking model so far, 0.003 behind the Transformer
   on test** (`full_size_lifted_loop_run.json`). After 90,000 updates fsp-lift scores 1.0817 on test and 1.1136 on
   validation with fast output synapses, 0.0222 and 0.0251 better than fsp-loop at the same pulses (506,746 a byte)
   and the same training time. The pilot's gain after 24,000 updates was 0.0095, so the lifting pool's gain did not
   fade with length but more than doubled. Against the Transformer 0.0028 are left on test (0.0017 to 0.0038) and
   0.0088 on validation (0.0077 to 0.0098), a ninth and a quarter of fsp-loop's gap, so it does not beat it. By
   context, on test: 0.087 better than the Transformer where the answer can be copied from the last 2 KB, 0.44
   better where only from further back, and 0.044 behind on new contexts, where fsp-loop was 0.078 behind. No
   membrane is held at a bound any more (0.04% of the top cells' entries reach their cap, where 27% of fsp-loop's sat
   at the bound). The fast output synapses now add little: 0.0009 on test, and on validation they cost 0.0007 (a
   scan on fsp-loop kept their threshold at 2^24, `fast_synapse_threshold_loop.json`). The lifted loop is the
   stack's recipe from here on. fsp-lift2, the same loop with power-law cells, is worse than it by the registered
   measure. Without fast output synapses it is ahead of it and of the Transformer on test (see power-law cells
   above); a second seed tests that with the measure fixed in advance.

   Built and pre-registered next: **a racing loop** (`--stack-loop-exits`, d837b6c, `racing_loop_screen.json`).
   A uniform third pass helps hard bytes and hurts copies, so it nets nothing. In cortex a decision is a race:
   neurons for each option sum evidence, and the choice is made when one leads by enough, so easy choices are fast
   and hard ones slow. Here the output neurons hear the end of every pass of the loop and sum it as evidence; each
   pass's sum learns to predict the byte by itself, and a byte can be decided at the first pass whose evidence
   leads by a margin. Two server seeds ask what summing costs at the top and how early bytes settle
   (`experiments/loop_exits.py`); if they settle early, the loop that stops is built and screened at equal mean
   pulses. **It fails in this form**: summing the passes costs the top 0.031 bits a byte on both seeds at equal
   pulses, ten times what the screen allowed. Bytes do settle early (at a margin of 6 bits a third of them before the
   last pass, 2.38 passes on the mean, for 0.0013 bits), but the loop that stops waits for both answers. The passes
   know different bytes: picking each byte's best pass would beat the loop without exits by 0.013, so the price is in
   summing every pass into one prediction and in making early passes predict alone. A loop whose top reads the last
   pass and whose exits only decide when to stop is the form left to try.
6. **Skip fibres** (`skip_fibres_with_data.json`, pre-registered): every part of the stack hears one stream, the
   running sum of all that came before, fired again as whole counts after every read and cell layer.
   DeepCrossAttention (arXiv 2502.06785) finds that a Transformer gains when each layer hears the earlier layers
   through learned weights for every channel, and gains more when queries, keys and values have their own. In
   spikes those weights are skip fibres (`--stack-skips`): a read or a cell layer also hears the stream as it stood
   at the input and after every earlier level, through a gain for every channel, as cortical areas receive
   projections that skip levels. A fibre fires nothing, so the pulse budget is untouched. The gains are whole
   sixteenths, start at zero, where the stack trains bit for bit as before, and add 0.2% to 0.3% parameters; the
   charge stays exact. The screen runs the plain test bed with one set of fibres for a level's read and cells, and
   with fibres of their own for the read's probes, tags and payloads, on two seeds; and the looped bed with gains
   that every pass has for itself, on two seeds on the server and a third on the laptop. The same idea helps a Transformer, which any comparison
   that uses it will say.
7. **A learned normalization pool**: divisive normalization with learned weights per channel, as cortical
   normalization weighs its pool, in place of one global sum. The pool in place of LayerNorm was v15's second
   largest step (0.012), but that step was an effect of the learning rate: at 3e-4 the fine pool is ahead of
   LayerNorm (`docs/STREAM_FORM.md`). It ranks below the unit.
8. **Homeostatic synaptic scaling and intrinsic plasticity**: neurons keep their synapses and thresholds in the range
   where they still learn, as cortical neurons do. The census by rate (`unit_density_by_rate.json`) says what there
   is to keep: no cell is dead, so there is none to revive, but the share of cells in use falls from 30% to 7% at
   full size as the rate rises from 3e-4 to 1.2e-3, and the stack stops gaining from the rate there. A cell's
   resting charge cannot hold that share, since the charge that its synapses bring is below zero at most bytes. Two
   forms are open: a set point that scales a cell's synapses to hold its share of bytes, and a cell with a soft
   threshold that fires a few fine spikes below it, as a pool of neurons with scattered thresholds does, so that a
   cell near its threshold passes a share as a closed unit of the Transformer does. Neither is built. The charge
   count on every stack of the census (its amendment) narrows the second form: at the best rates only 1% to 3.5%
   of cells and bytes lie within one threshold below firing at full size, and 63% to 86% lie more than eight
   thresholds below zero, so a threshold soft over one threshold would reach almost nothing. A soft threshold has to
   be as wide as the charge's spread, or the set point has to hold the share.

   Built and pre-registered: **a slope far below the threshold** (`--granule-slope-tail 4`,
   `subthreshold_slope_with_data.json`). It is a third form, and it changes no spike. A plain cell learns from a
   smooth threshold one spike wide (at full size a fine spike, a sixteenth of a threshold), so a silent cell learns
   only where it almost fires, and nothing holds its charge at the bytes where it is silent. With the tail its slope is one above zero charge and falls with the
   square of the distance below it, to a quarter at four whole thresholds, as plasticity in a neuron depends on
   how far the membrane stands from its threshold and does not wait for a spike. A cell eight thresholds below
   zero then still learns at a ninth of the full slope. In its local form the slope below zero teaches only the
   cell's own synapses and its rest, a rule of three factors (the input's spikes, the membrane's distance, the
   error that the cell's output synapses bring back), and the stream never learns from spikes that were not fired.
   The smooth threshold a whole threshold wide failed at full size (0.0145 behind after 6,000 updates at 1e-3,
   0.074 after 15,000): it dies within three thresholds, is negative from 0.75 to 3 thresholds below zero and
   halves the slope of a cell that has just begun to fire. The tail has none of the three. Both forms are screened
   by slope on the test bed (the laptop's seed first, two seeds on the server) and run at full size at 6e-4 and at
   1.2e-3, where the question is whether the stack can then use the Transformer's rate.
9. **Neurogenesis**: fresh, more plastic cells join during training at equal final parameters, as new granule cells do
   in the dentate gyrus, to take up what the old ones cannot.
10. **Capacity that pays late**: designs that lost at 6,000 updates because they start slowly (the cerebellar sheet,
   more cells, narrower reads), tested by their slope.
11. **Surprise-gated fast synapses**, learning while reading only when the prediction was wrong.
12. **More data at full size**, once a design's slope is negative on the test bed.

Myelinated fibres and the cerebellar sheet failed at 6,000 updates (`myelin_and_sheet.json`); the sheet is re-tested
by slope under 10. Ruled out as causes of the flattening: the pulse budget, noisy late weights and a code that grows
denser with data (`learning_rate_by_context.json`), and the learning rate (`learning_rate_with_data.json`).

## A second track: one model for language, reasoning and telemetry

The plan (Volley-One) is one spiking model that writes and reasons in its own words with chains of thought and reads
machine telemetry as a second sense, without template-written answers. Non-spiking teachers such as DeepSeek may
help train it, but the model that ships is purely spiking, and the race against the Transformer stays free of any
teacher. Knowledge is
read from documents into the stream rather than memorized; reasoning is learned as people learn it, from human
worked solutions, then by practice with feedback on checkable tasks kept at the edge of what the model can do,
replayed in sleep, and internalized into silent passes of the racing loop. The first screens, pre-registered in
`results/event_lm/one_model/`:

- **How many steps of lookup a model chains without writing them** (`hop_generalization.json`,
  `experiments/hop_tasks.py`, `experiments/hop_eval.py`): the test bed, its loop and its Transformer trained on
  chains of 1 to 4 hops and tested on up to 16. None learned it, at 12,000 updates or after training again to
  24,000 (mean accuracy over 1 to 4 hops at 24,000: the stack 0.63, the loop 0.10, the Transformer 0.10, against a
  mark of 0.80), so the screen ends without a comparison. The stack looks up one hop in every line, two in most
  (0.89) and is starting on three (0.38); the loop answers 0 on every line and the Transformer gives every digit
  the same probability, though both learned the corpus's other bytes about as well as the stack. A probe that
  first teaches one lookup (shorter lines, or a curriculum from one hop) would need its own registration.
- **Telemetry** (`telemetry_screen.json`, `experiments/pump_sim.py`, `experiments/telemetry_eval.py`): a simulated
  centrifugal pump with nine faults whose truth is kept, eight bytes a second; the stack's surprise as a fault
  detector beside a per-channel z-score detector (AUC 0.632 on the test pumps). The surprise fails the mark of 0.80
  (AUC 0.715). It is nearly certain for faults that add vibration (bearing wear, imbalance, misalignment: 0.96) but
  at chance for the seal leak and the stuck valve, which move the pump to a new steady state that soon becomes as
  predictable as the old one. So the telemetry sense must add three things: a slow memory of each pump's normal,
  the relations between channels that the physics fixes, and a test that also marks a stream that grows too
  predictable (a frozen sensor).
- **Chains of thought learned from reward alone** (`practice_hops.json`, `experiments/practice_loop.py`): the
  models practise hop tasks with room to write before the answer; only the answer is checked; the difficulty
  follows the solve rate so that a quarter to a half of the tasks are solved. Since no base learned the whole hop
  task, an arm is read if its base looks up one hop (amended before any practice ran): the stack is read, the
  Transformer practises as a control. It fails: after 300 rounds the stack's accuracy at 8 hops is 0.23 (0.24
  before) and none of its answers writes a chain. It wrote a name in 0.5% of its attempts, so reward had almost
  nothing to reinforce; practice only sharpened its direct answers at 3 hops (0.42 to 0.64). A chain of thought
  did not arise from reward alone: the model needs written chains to start from (in its training text, or from a
  teacher) before practice can lengthen them.
- **Written chains** (`written_chains.json`, `hop_tasks.py corpus --written-chains`): the same hop lines with every
  chain written out before the answer (?zt=mx=qa=7), so that each byte after the query is one lookup of the name
  just written. The stack and the Transformer train on them for 24,000 updates and answer greedily with room to
  write; the screen asks whether they learn to chain in writing and whether the chain carries them past the 4 hops
  they trained on (accuracy at 8 hops of at least 0.50). Both learn to chain in writing, where neither learned the
  task from direct answers: over 1 to 4 hops the stack answers 0.995 of lines right and the Transformer 0.991, each
  writing the whole chain right in at least 0.985 of them. Neither chains past what it read: at 6 hops the stack
  writes 2% of the chains right and the Transformer none, and from 8 to 16 hops both answer as often as a guess
  among ten digits (0.09 to 0.12). The step is the same lookup at every length, but both learned it together with
  the lengths they saw.
- **Practice from written chains** (`practice_written_chains.json`): the two models practise 2,000 rounds from 4
  hops, with only the answer checked, as in the practice screen. Now every attempt writes a chain for the reward to
  grade, so the question is whether reward at the edge of what a model solves makes its chains longer than any it
  has read: an accuracy of at least 0.50 at 6 hops with the whole chain written in half of the answers, the same at
  8 hops for twice the trained length, and the trained lengths kept. The Transformer's chains grew: 0.71 right at 6
  hops with 0.63 whole chains, 0.61 at 8 hops with 0.47 whole chains, just under the second mark. But it lost the
  short ones (0.51 over 1 to 4 hops, from 0.99). The stack lost everything (0.05 at 6 hops, 0.11 over 1 to 4). The
  rounds show two faults of the loop itself. The staircase climbed past 20 hops within 150 rounds, where a guess
  pays: a line holds 24 assignments whatever its hops, so a long chain leaves few other digits (`hop_tasks.py
  floors`). And GRPO's attempts below the mean push the likelihood of the model's own bytes down without a floor.
- **Practice within reach** (`practice_within_reach.json`, on the laptop): the same practice with the staircase
  capped at 8 hops, with only rewarded attempts teaching, and with both, from both models. Its reading is taken
  after 2,000 rounds. The stack's three arms are done and none passes. The arm with both changes failed in a way
  worth more than a pass would have told: within 100 rounds it answered 0.96 of the tasks of 8 hops right with the
  whole chain in 0.93, and it held that for 700 rounds. The model kept at round 700 writes whole chains of 16 hops
  right in 0.72 of its answers, four times the length it read and twice the length it practised. Then, within
  about 50 rounds, it lost the stop of a chain (it writes names until the room ends) and never returned. The
  capped arm that keeps GRPO's dips lost its chains between rounds 300 and 400. The uncapped arm in which only
  rewarded attempts teach lost them between rounds 100 and 200, after its staircase had climbed to 24 hops, where
  a guess pays. Reward on the answer alone does lengthen a spiking model's written chains; without rehearsal the
  loop does not hold what it wins. The Transformer's three arms fail the same way: each reached 6 hops along the
  way and lost it (the capped arm with dips at 10 of 21 evaluations up to round 1,400, the other two at round 200
  only), and after 2,000 rounds none answers more than 0.10 at 6 hops. Neither model practises further.
- **Practice with rehearsal** (`practice_with_rehearsal.json`, `practice_loop.py --rehearse`): the capped,
  reward-only practice with the written chains of 16 fresh tasks of 1 to 4 hops rehearsed in every round, as the
  brain replays old memories among new ones. The stack's arm passes: it meets the three marks after 2,000 rounds
  and held them at every one of the 20 evaluations on the way. After practice it answers 0.99 of the tasks of 8
  hops right with the whole chain written right in 0.985, and it writes chains it never practised: whole chains
  in 0.975 of its answers at 12 hops, 0.875 at 16 and 0.345 at 20, from a corpus that showed it 1 to 4. Rehearsal
  costs nothing on the way up (0.955 at 8 hops by round 100). The Transformer's arm meets the three marks after
  practice but not the fourth: at round 100 it answered 0.43 of the 8-hop tasks and at round 200 0.13, then 0.985
  to 1.000 at every evaluation from round 300. It lost its chains early and found them again, which the record
  counts as no pass. After 2,000 rounds the two models are level at 6 and 8 hops.
- **The cap raised** (`practice_cap_raised.json`, on the server's GPU): the same loop with its cap at 8, 12 and
  16 hops, from both models, with three seeds each. The stack's nine runs are in. Every one holds its chains at
  every evaluation, so rehearsal keeps what practice wins with each seed and each cap. The cap at 12 reaches
  furthest: from a corpus of 1 to 4 hops, two seeds of three write chains of 20 hops whole in 0.92 and 0.905 of
  their answers (0.605 in the third), against 0.645, 0.35 and 0.22 with the cap at 8. The cap at 16 does not beat
  the cap at 8: at 16 hops a guess is right in 0.37 of the tasks, one attempt in seven is rewarded without its
  chain, and the reward teaches that too. Two runs began to lose their chains as the run without rehearsal had
  lost them for good; the staircase took them down to 6 and 7 hops and both stood at their cap again within 40
  rounds. The Transformer's three runs with the cap at 8 are in as well. At 6 and 8 hops neither model is ahead.
  Past the practised length the Transformer carries further: after practice up to 8 hops it answers 0.975, 0.995
  and 0.92 of the 20-hop tasks, the stack 0.705, 0.44 and 0.38 (the stack's best cap, 12, takes it to 0.945,
  0.935 and 0.805). It got there later and less evenly. The stack stands at 8 hops within 28 rounds with every
  seed and holds at every evaluation; the Transformer needed 71, 171 and 712 rounds, one seed writing four steps
  and stopping for 700 of them. One of its three seeds left the corpus's form: it writes the right chain with a
  fresh question mark before every fourth name, turning a long chain into chains of the length it was taught, so
  its answers are right and none of its chains counts as whole. With the cap at 12 and 16 it is the same picture.
  Neither model practises further at 6 and 8 hops under any cap. Past it, five of the Transformer's nine runs
  write 0.91 to 0.975 of the 20-hop chains whole, against the stack's best 0.92 (cap 12), but four of its nine
  runs fall short: two write the chain with question marks in it, one loses its chains, and one reaches 8 hops
  only at the end.
  Every one of the stack's nine runs holds. Why the stack's chains thin out with length where the Transformer's do not: a
  diagnostic of the practised models says it is recall over distance, not the chain's step. At 20 hops the stack
  writes a step right almost always while the assignment it needs stands within about 150 bytes back, and takes a
  neighbouring name in 16% of the steps (cap 8) when it stands 150 to 200 bytes back. The Transformer misses 1.2%
  there. With the cap at 16 this holds at every place in the chain alike; the cap at 12 holds far assignments best
  (3.4%). The stack's own weights say why. Its memory reads weigh every earlier byte by its age as well as by
  what it holds, and every head learned to turn against bytes older than about 120, most of all at 140 to 150
  bytes back: in the corpus that is where the line before ends (a line is 131 to 141 bytes up to its question).
  The reads tell the present line from the one before by age alone, and a long chain written after the line
  pushes the line's first assignments past that age. Practice did not move these weights. A probe of recall by
  distance bears it out (`recall_over_distance.json`, `experiments/recall_over_distance.py`): asked for one
  binding in lines longer than any it read, the stack recalls it almost always up to 150 bytes back, but in only
  0.31 and 0.19 of the lines at 150 to 175 and 175 to 200 bytes, where the Transformer recalls 1.00 and 0.97.
  Practice with the cap at 8 or 12 had taught the stack to reach that far (0.92 to 0.98) without moving its age
  offsets, so the practised runs' long chains break partly over the chain they write themselves.
- **Line heads** (`line_heads.json`, `--read-word-unit line`): two of every read's four heads count lines instead
  of bytes. They are the read's word heads, which reach only the present word, made to reach only the present
  line, as the brain cuts experience into events and recalls within them. **All three readings pass.** Asked for
  one binding 150 to 200 bytes back, the stack with line heads answers 0.965 and 0.975, where the same recipe
  without them answers 0.195 and 0.29. It keeps 1 to 4 hops (0.999), and without any practice it writes 0.975 of
  8-hop chains whole (without line heads 0.655), 0.785 of 12-hop chains (0.205) and 0.455 of 16-hop chains
  (0.015). Validation on the hop corpus is level, so the line heads cost nothing. Its plain heads still learned
  the cliff at about 120 bytes; the far recall is the line heads'. As registered, line-stack goes into the
  practice loop with rehearsal next, and line heads go into the four-family corpus. Records of telemetry are lines
  too.
- **Four families in one model** (`written_chains_four_families.json`, `experiments/task_families.py`, training on
  the laptop): before the loop can practise more than one kind of reasoning, a model must have learned to write
  each kind's chain. One corpus, four families, 1 to 4 steps each: the lookups along names; lookups that add as
  they go (`?zt=mx2=qa5=2`), whose answer cannot be guessed from the line; the running sum of a channel's readings
  (`+mx=7=4=8`), found by its place in a row as telemetry is; and the same sum written as a log that copies each
  reading (`#mx=77=74=48`). It asks which families the stack and the Transformer learn, what sharing a model costs
  the lookups, and whether a chain that keeps its place silently is learned as well as one that writes down what
  it reads. Every learned family goes into the practice loop next (`practice_loop.py --families`). The
  Transformer's arm is in: with the 24,000 updates that teach it the lookups alone, it learned none of the four
  families, the lookups included. It writes each family's chain in its form, with names taken from the line, and
  follows no binding. The stack learned two of the four: the lookups (0.9925 over 1 to 4 steps, no worse than the
  stack trained on them alone) and the lookups that add (0.979). It learned neither running sum; there it writes a
  digit or two and runs on into assignments instead of ending the line. And without any practice its lookups now
  run past the trained length: 0.93 right at 8 steps and 0.775 at 12, where the stack trained on lookups alone
  managed 0.105 at 8 and needed practice to get further. The two learned families go into the practice loop
  together this weekend; the running sums need another way of writing their chains first.
- **Practice of several families** (`practice_several_families.json`, on the server's GPU on 2026-11-27): the
  stack's two learned families in one practice loop with rehearsal, capped at 12 steps: both practised in the
  same rounds, and each practised alone while the other is only rehearsed, three seeds each. **Four of the nine
  stack runs fall.** The five that hold write 0.78 to 0.985 of their practised 12-step chains whole (the base
  wrote 0.715 for the lookups and 0.105 for the lookups that add), the run that practised both in seed 0 writes
  0.98 and 0.895 of its 16-step chains, and practice in one family lengthened the other's chains too. But two of
  the three runs that practised both families, and one run of each single arm, lost everything past the
  rehearsed lengths. They write the first steps right, then fill the room with one name repeated or with equals
  signs, and end on one digit whatever the task. Once a run has fallen, its staircase goes between 4 and 5 steps,
  and nothing in the loss makes the filler less likely, because only paid attempts teach. The Transformer
  practised its one learned family, the lookups, with three seeds. Two runs write every chain of 5 to 12 steps
  whole and 0.99 of those of 16 steps. The third never got past the corpus's longest form: it writes three names
  and a guessed digit at every length. Neither model practises further by the reading. Every run that fell, of
  either model, broke just past the longest chain the rehearsed corpus writes. So the choice between practising
  families together and in turns must wait: first the loop has to keep every run standing.
- **The Transformer again** (`four_families_transformer_again.json`, on the laptop): a comparison of practice
  needs a Transformer that has learned the chains, so it gets two further chances at the same corpus: four times
  the updates, and a start from the model that already follows names. Each learned the lookups and nothing else
  by the mark. The one with four times the updates came close on the lookups that add (0.91 over 1 to 4 steps);
  the one started from the lookup model managed the running sum up to three readings. Neither reaches past the
  trained length without practice, where the stack's lookups reach 8 steps. By the registered tie rule the
  Transformer practises the lookups this weekend from the arm that started from the lookup model.
- **Practice that stands** (`practice_that_stands.json`, on the server's GPU on 2026-11-28): the same loop with
  GRPO's dips at full weight (`--dips 1`), so that attempts below their task's mean are made less likely, as the
  pause of dopamine neurons teaches the indirect pathway what not to do again. Six seeds a model, paired with six
  runs without dips. **The stack's half is in, and it does not stand by the record's rule.** Four of the six runs
  with dips hold. No run with dips fills the room, where five of the six runs without them do. The two that do
  not hold keep 1 to 4 steps and lose reach slowly: one does not keep it in the offset family at every late
  evaluation, the other drops to 0.19 whole chains at 8 steps there. The runs that hold reach less far than the
  one run without dips that held (lookups at 16 steps 0.43 on the mean against 0.98). So dips end the filler
  falls and cost reach. The Transformer's runs finish today.

## How to continue

- The newest handoff (`docs/V22_HANDOFF.md`) says how to restore checkpoints,
  rescore and retrain.
- `experiments/train_event_lm.py` trains every arm; `experiments/eval_stream.py`
  scores streams, and `experiments/gap_by_piece.py` splits two scorings.
- Pre-registrations and verdicts are in `results/event_lm/*/`. Read a study's
  record before rerunning it.
- `tests/` holds the exactness tests: a stream must give the same spikes
  under any cut, and every new part must start as the stack it joins.
