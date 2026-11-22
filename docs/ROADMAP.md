# Roadmap: a spiking, streaming language model that beats the Transformer

The aim of this project is a language model that is **spiking and streaming
from end to end** and that predicts text **better than a Transformer of the
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
| Full size, v16: 34.8M parameters, 368.6M training bytes, enwik8 test | 1.1199 | 1.1067 | **0.013 behind** |
| Test bed: 5.2M parameters, 24.6M training bytes, 2M validation bytes as long streams | 1.7724 | 1.8190 | **0.047 ahead** |

The full-size stack reads up to 8,192 cells back and has fast synapses that
learn while it reads. The full-size Transformer reads windows of 2,048 bytes.
The test-bed stack has delayed fibres and was trained on streams. Sources:
`results/event_lm/long_streams/leaderboard.json` and
`results/event_lm/knowledge_gap/gap_with_data.json`.

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

On the test bed the gap halves with four times the data (0.148 to 0.077 for
the plain stack). The stack overtakes the Transformer on copying first, and
its knowledge gap shrinks as the data to the power -0.44
(`gap_with_data.json`, figure `results/figures/event_lm_gap_with_data.png`).

## Rules for claims

1. **Spiking and streaming only.** The stack is built, trained and scored as
   a stream. The Transformer reference may read windows, as it was trained.
2. **Equal parameters and equal training data** for every comparison with
   the Transformer. Pulses per byte, the stack's energy, are reported beside
   it.
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

1. **Myelinated fibres** (`--granule-myelin`, built and running). In the
   brain, activity-dependent myelination tunes how late each axon delivers
   its spikes. Here every fibre learns which earlier byte it brings, up to 31
   back, at one number a fibre. The spikes stay exact. Record:
   `myelin_and_sheet.json`.
2. **The cerebellar sheet** (running). Four times the cells on a quarter of
   the stream each, at equal parameters, as the granule layer of the
   cerebellum is built.
3. **Both together**, if either passes.
4. **Myelinated reads**, and myelinated fibres at full size. At full size
   every channel's delayed copy can learn its delay at no cost in
   parameters (built, `tests/test_myelin.py`).
5. **Dendritic plateaus on sparse random inputs.** Segments that fire only
   when enough of their few inputs coincide, as NMDA spikes do. These differ
   from the branches that failed, which split one dense fan.
6. **Surprise-gated fast synapses**, learning only when the prediction was
   wrong, as dopamine gates plasticity.
7. **Scaling at a fixed energy.** Grow the sheet's cells and keep the pulses
   per byte. Neurons that do not fire cost no energy, so a spiking network
   can hold far more parameters per joule than a Transformer can.
8. **More data at full size.** A pair trained twice as long. By the
   test-bed trend the gap would fall to about 0.0095.

## How to continue

- The newest handoff (`docs/V16_HANDOFF.md`, and v17's when released) says
  how to restore checkpoints, rescore and retrain.
- `experiments/train_event_lm.py` trains every arm; `experiments/eval_stream.py`
  scores streams, and `experiments/gap_by_piece.py` splits two scorings.
- Pre-registrations and verdicts are in `results/event_lm/*/`. Read a study's
  record before rerunning it.
- `tests/` holds the exactness tests: a stream must give the same spikes
  under any cut, and every new part must start as the stack it joins.
