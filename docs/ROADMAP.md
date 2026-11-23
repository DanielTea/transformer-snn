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
| Full size, best so far (v18: l2048-word-st, 36.3M parameters), enwik8 test | 1.1162 | 1.1067 (34.75M) | **0.0095 behind** |
| The same stack against the Transformer with output synapses that learn while it reads | 1.1162 | 1.1032 | 0.013 behind |
| Full size, v17 (l2048-know-st, 36.3M), enwik8 test | 1.1170 | 1.1067 | 0.010 behind |
| Full size, v16 (34.8M), enwik8 test | 1.1199 | 1.1067 | 0.013 behind |
| Test bed (5.2M), 24.6M training bytes, 2M validation bytes as long streams | 1.7724 | 1.8190 | 0.047 ahead |
| Test bed (5.2M), 98.3M training bytes, the same bytes | 1.5468 | 1.4927 | **0.054 behind** |
| Test bed, 98.3M training bytes, each model at its best learning rate, mean of two seeds | 1.4895 | 1.4412 | 0.048 behind |

The full-size stacks read up to 8,192 cells back and have fast synapses that
learn while they read; the best one has 128 knowledge cells and a word head in
every read and trained on streams. The full-size Transformer reads windows of
2,048 bytes. The test-bed stack has delayed fibres and trained on streams.
Sources: `results/event_lm/long_streams/leaderboard.json`,
`results/event_lm/knowledge_gap/full_size.json`,
`results/event_lm/knowledge_gap/full_size_word_heads.json`,
`results/event_lm/knowledge_gap/gap_with_data.json` and
`results/event_lm/knowledge_gap/learning_rate_with_data.json`; figure
`results/figures/event_lm_best_model.png`.

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
rather than from what it can copy. Spikes suit memory, because copying is
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

## Rules for claims

1. **Spiking and streaming only.** The stack is built, trained and scored as
   a stream. The Transformer reference may read windows, as it was trained.
2. **Equal parameters and equal training data** for every comparison with
   the Transformer. Pulses per byte, the stack's energy, are reported beside
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
| A higher learning rate as the cause of the flattening | Ruled out over two seeds | Both models gain alike from their best rates | `knowledge_gap/learning_rate_with_data.json` |
| A sparser code | Not built | At 3e-4 the Transformer's units are as dense as the stack's cells, and at the best rates the stack's code is already the thinner one | `knowledge_gap/unit_density.json`, `knowledge_gap/unit_density_by_rate.json` |
| A census of cells and units in use at each peak rate, on trained weights | The stack's code thins as the rate rises (30%, 16% and 7% of a level's cells fire at full size at 3e-4, 6e-4 and 1.2e-3) and with data; half of the Transformer's units carry nine tenths of a block's output at every rate, against 11% of the stack's cells at its best rate; no cell dies | A closed unit still passes a share and a silent cell passes nothing; not shown to cost bits, since the gap is no smaller at 3e-4 | `knowledge_gap/unit_density_by_rate.json` |

## Open bets, in order

1. **The full-size pair, each model at its best learning rate** (`full_size_best_rate_pair.json`, running): the plain
   stack at 6e-4 and the Transformer at 1.2e-3, 90,000 updates each. The pilot behind it
   (`full_size_rate_pilot.json`): after 24,000 updates the stack is best at 6e-4 (1.3199) and the Transformer at
   1.2e-3 (1.2762). The gap at the best rates is 0.044 against 0.036 at 3e-4, and both gain alike on new contexts.
   The stack's membranes saturate at the higher rate: at 1.2e-3, 2.7% of the reads' entries and 2.0% of the cells'
   sit at their bound, up to 11.8% at the top level, where a neuron learns nothing through its charge
   (`experiments/membrane_saturation.py`). The pool that lifts the ceiling removes that bound at the same energy and
   is worth 0.035 at 1.2e-3. Its best arm was 0.007 ahead of the plain best where 0.010 was asked, so the pair's
   stack trains plain (`full_size_lifted_ceiling_pilot.json`).
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
   form of it can.

   Two controls in floats would say whether the unit also explains the gap late in training, since the walk priced
   it at 9,000 updates: a Transformer with plain units on the test bed at 6,000 and 24,000 updates (does it flatten
   as the stack does?), and the stack in floats at full size at 24,000 updates (do spikes cost anything late, on a
   charge grid 16 times finer than the test bed's?). They are diagnostics and never a path. The first is
   pre-registered and runs on the server's CPU with all four kinds of unit, plain or gated, hard or smooth
   (`plain_units_with_data.json`). Along the walk's full-size curves the two hard kinds fell further behind as
   training went on (the plain one from 0.008 to 0.027, the gated one from 0.001 to 0.014) and the smooth plain
   unit did not (0.008 to 0.011), one seed with unequal schedules.
3. **Quantal release and finer synaptic grids** (`weight_precision_with_data.json`; at the best rate
   `randomness_at_best_rate.json`): does the charge grid limit late learning?
4. **The data-scaling curves** (`data_scaling_long.json`, running): which design's gap shrinks with data. It becomes
   the base for the rest.
5. **Depth by recurrence at full size** (`full_size_loop_pilot.json`, passed): a looped stack runs its middle levels
   twice with tied synapses, more computation per parameter, as cortex iterates. On the test bed a loop gained 0.047
   and 0.062 over depth 6 at equal parameters, with a budget scaled to its passes. At full size, after 24,000
   updates, the looped stack scores 1.3102: 0.019 ahead of the plain stack given the same pulses (0.026 on new
   contexts) and 0.010 ahead of the plain stack at the old budget (0.010 on new contexts), the first part at equal
   parameters that adds knowledge at full size. It spends 1.63 times the pulses and 1.4 times the training time.
   The raised budget alone makes the plain stack 0.009 worse, because its membranes saturate. The loop has earned
   a run of 90,000 updates beside the pair, and that run is pre-registered (`full_size_loop_run.json`): the
   pair's stack arm with the loop and its budget, queued for the night of 2026-11-23 and read against the pair's
   plain stack and its Transformer. It holds if the looped stack is at least 0.005 better than the plain one on
   test and validation. Still open, and piloted beside it: whether the loop needs the raised budget at full size,
   and whether the pool that lifts the ceiling adds to it.
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
9. **Neurogenesis**: fresh, more plastic cells join during training at equal final parameters, as new granule cells do
   in the dentate gyrus, to take up what the old ones cannot.
10. **Capacity that pays late**: designs that lost at 6,000 updates because they start slowly (the cerebellar sheet,
   more cells, narrower reads), tested by their slope.
11. **Surprise-gated fast synapses**, learning while reading only when the prediction was wrong.
12. **More data at full size**, once a design's slope is negative on the test bed.

Myelinated fibres and the cerebellar sheet failed at 6,000 updates (`myelin_and_sheet.json`); the sheet is re-tested
by slope under 10. Ruled out as causes of the flattening: the pulse budget, noisy late weights and a code that grows
denser with data (`learning_rate_by_context.json`), and the learning rate (`learning_rate_with_data.json`).

## How to continue

- The newest handoff (`docs/V18_HANDOFF.md`) says how to restore checkpoints,
  rescore and retrain.
- `experiments/train_event_lm.py` trains every arm; `experiments/eval_stream.py`
  scores streams, and `experiments/gap_by_piece.py` splits two scorings.
- Pre-registrations and verdicts are in `results/event_lm/*/`. Read a study's
  record before rerunning it.
- `tests/` holds the exactness tests: a stream must give the same spikes
  under any cut, and every new part must start as the stack it joins.
