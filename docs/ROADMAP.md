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
| Full size, best so far (l2048-know-st, 36.3M parameters, 368.6M training bytes), enwik8 test | 1.1170 | 1.1067 (34.75M) | **0.010 behind** |
| Full size, v16 (34.8M), enwik8 test | 1.1199 | 1.1067 | 0.013 behind |
| Test bed (5.2M), 24.6M training bytes, 2M validation bytes as long streams | 1.7724 | 1.8190 | 0.047 ahead |
| Test bed (5.2M), 98.3M training bytes, the same bytes | 1.5468 | 1.4927 | **0.054 behind** |

The full-size stacks read up to 8,192 cells back and have fast synapses that
learn while they read; the best one has 128 knowledge cells a head and trained
on streams. The full-size Transformer reads windows of 2,048 bytes. The
test-bed stack has delayed fibres and trained on streams. Sources:
`results/event_lm/long_streams/leaderboard.json`,
`results/event_lm/knowledge_gap/full_size.json` and
`results/event_lm/knowledge_gap/gap_with_data.json`; figure
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

The test bed's Transformer also gains from a higher learning rate (0.036 at 6e-4 on one seed,
`learning_rate_with_data.json`), so every comparison now takes each model at its best rate.

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
| Echo learning, consolidating fast synapses, doubled pulse budget, squared cells | Fail | See `docs/V16_HANDOFF.md`, "What not to try again" | |

## Open bets, in order

1. **Each model at its best learning rate at full size** (`full_size_rate_pilot.json`, running, then
   `full_size_best_rate_pair.json`): on the test bed the stack gains 0.013 more than the Transformer from its best rate,
   most of it on new contexts; v17's full-size gap is 0.010. The pilot did not repeat that: after 24,000 updates the
   stack is best at 6e-4 (1.3199), the Transformer at 1.2e-3 (1.2762), and the gap at the best rates is 0.044 against
   0.036 at 3e-4. Both gain alike on new contexts. Its membranes saturate at the higher rate:
   at 1.2e-3, 2.7% of the reads' entries and 2.0% of the cells' sit at their bound, up to 11.8% at the top level,
   where a neuron learns nothing through its charge (`experiments/membrane_saturation.py`). v15's pool that lifts the
   ceiling removes that bound and was never used in a release. A full-size pilot of it at 6e-4 and 1.2e-3 decides
   whether the pair's stack trains lifted (`full_size_lifted_ceiling_pilot.json`).
2. **The unit, in spikes** (`gated_cells_with_data.json`, `noisy_thresholds_with_data.json`, laptop screens running).
   v15's priced walk put plain units in place of gated ones as the largest step from the Transformer to the stack
   (0.025, against 0.001 for whole spikes). Two spike-native halves of a gated unit are screened by their slope at the
   best rate:
   - dendritic gating: a gate dendrite lets in a share of what a value dendrite collects;
   - noisy thresholds: membrane noise makes a cell smooth on average, and it learns from that true slope. The noise
     falls with the learning rate, so scoring stays exact.
   At full size (`full_size_unit_screen.json`) gated cells lost by 0.118 and noisy thresholds were level (0.004
   better, none of it on new contexts); the self-gated arm and the test-bed screens by slope still run.
3. **Quantal release and finer synaptic grids** (`weight_precision_with_data.json`; at the best rate
   `randomness_at_best_rate.json`): does the charge grid limit late learning?
4. **The data-scaling curves** (`data_scaling_long.json`, running): which design's gap shrinks with data. It becomes
   the base for the rest.
5. **A learned normalization pool**: divisive normalization with learned weights per channel, as cortical
   normalization weighs its pool, in place of one global sum. The pool in place of LayerNorm was v15's second
   largest step (0.012).
6. **Homeostatic synaptic scaling and intrinsic plasticity**: neurons keep their synapses and thresholds in the range
   where they still learn, as cortical neurons do.
7. **Neurogenesis**: fresh, more plastic cells join during training at equal final parameters, as new granule cells do
   in the dentate gyrus, to take up what the old ones cannot.
8. **Capacity that pays late**: designs that lost at 6,000 updates because they start slowly (the cerebellar sheet,
   more cells, narrower reads), tested by their slope.
9. **Surprise-gated fast synapses**, learning while reading only when the prediction was wrong.
10. **More data at full size**, once a design's slope is negative on the test bed.

Myelinated fibres and the cerebellar sheet failed at 6,000 updates (`myelin_and_sheet.json`); the sheet is re-tested
by slope under 8. Ruled out as causes of the flattening (`learning_rate_by_context.json`): the pulse budget, noisy
late weights, and a code that grows denser with data.

## How to continue

- The newest handoff (`docs/V16_HANDOFF.md`, and v17's when released) says
  how to restore checkpoints, rescore and retrain.
- `experiments/train_event_lm.py` trains every arm; `experiments/eval_stream.py`
  scores streams, and `experiments/gap_by_piece.py` splits two scorings.
- Pre-registrations and verdicts are in `results/event_lm/*/`. Read a study's
  record before rerunning it.
- `tests/` holds the exactness tests: a stream must give the same spikes
  under any cut, and every new part must start as the stack it joins.
