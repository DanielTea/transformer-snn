# Echo learning

The parts of a spiking stack learn from an echo of the error at the output instead of from the error passed back
through every level above them. Two echoes were built: one through fixed random synapses, and a mirror, the error of
the stream that the output hears, sent back to every part in one hop. A ladder of test-bed runs was recorded before
any of them trained.

In short: a null result, and a finding. Neither echo trains the
stack: the random echo blew the pulses up fourfold, and the mirror
held a gap of 0.33 bits per byte at every depth. Gradient runs with
the error cut on one path at a time showed why: the credit a part of
the stack needs runs back in time along the reads, from the bytes
that read what it wrote.

## Limits

- Within a part, its response to its echo is computed by the part's own surrogate spikes, as in gradient training.
  The echo replaces only the error that would have come back through the parts above it.
- The mirror echo is not gradient-free. It is the error of the loss at the stream that the output hears, passed back
  one hop through the output's own synapses: a truncated gradient along the stream's identity path. It passes through
  no other part or level. The random echo carries no error through any part.
- The ladder runs on the small test bed (6 levels, 256 channels, 4.86 million parameters, windows of 256 bytes), one
  seed per arm, on a GPU. It decides whether echo learning earns a full-size run.

## The question

Gradient training passes the error of every byte back through all levels of the stack and through the surrogate
derivative of every spiking neuron on the way. Cells that multiply two learned charges blew up under it this weekend.
Biological circuits are thought to do without that backward pass, and a family of local rules shows how:
fixed random feedback (Lillicrap et al., 2016), sent straight to every layer (Nøkland, 2016), in two-compartment
spiking neurons (event-driven random back-propagation, Neftci et al., 2017), with eligibility traces (e-prop, Bellec
et al., 2020). None has been shown to train a spiking language model. Can the stack learn without the backward pass
through its levels, and does that keep multiplying cells stable?

## Two echoes

Every part of the stack, the input synapses, each state read and each cell layer, adds what it fires to one stream,
and the output neurons hear the stream. In echo learning a part hears the stream without passing any error back into
it, and what it adds is kept (`EchoTape`). The stream fires exactly as without echoes.

- **Random echo.** Each part has fixed random synapses of sign ±1 from the 256 byte values to its channels, drawn from
  a seed and the part's name and never trained. Its echo is the error of every byte, its softmax less the byte that
  came, through those synapses.
- **Mirror echo.** Every part is told the same thing: the error of the loss at the stream that the output hears.
  That is what the output would have wanted the stream to be, and it reaches each part in one hop through the
  output's own synapses, as if that part alone had written into the stream.

The output neurons learn from their own error. The pulse budget teaches each part its own pulses. The optimizer, the
schedule and the averaging of weights are those of the gradient runs (`--echo all|cells`,
`--echo-feedback random|mirror`; `tests/test_echo_learning.py`).

## A first look

Tiny runs on enwik8, a stack of 2 levels and 64 channels, windows of 64 bytes, 400 updates at 3e-3, a budget of
20,000 pulses a byte; validation in windows at 400 updates:

| Training | Validation | Pulses a byte |
|---|---:|---:|
| Gradient | 3.7808 | 19,770 |
| Mirror echo, every part | 3.8362 | 19,851 |
| Mirror echo, cells only | 3.8094 | 19,783 |
| Random echo, every part | 8.3420 | 82,757 |
| Random echo, cells only | 7.7826 | 54,405 |

The random echo drives the pulses to four times the budget and learns nothing; the mirror stays within the budget and
within 0.06 of the gradient.

## The ladder

Recorded at 13:26:59 UTC on 2026-11-21 (`results/event_lm/echo_learning/preregistered.json`): the small test bed
trained six ways for 6,000 updates on the server's GPU 1.

Averaged validation in windows at 6,000 updates (64 batches), and
pulses a byte at the last evaluation:

| Arm | Training | Validation | Pulses a byte |
|---|---|---:|---:|
| g6 | Gradient | 1.9860 | 114,576 |
| m6 | Mirror echo, every part | 2.3180 | 115,401 |
| mc6 | Mirror echo, cells only | 2.0687 | 114,628 |
| r6 | Random echo, every part | 8.0341 | 435,960 |
| gate-g | Gated cells, gradient | 1.9897 | 115,807 |
| gate-m | Gated cells, mirror echo | 2.3389 | 114,457 |

Every echo arm fails its mark. The random echo got worse as it
trained (5.97 at 750 updates, 8.03 at 6,000). The mirror opened its
gap within 750 updates and held it: a worse solution, not a slower
one. The gated cells did not blow up under the gradient on this
test bed, so the echoes had nothing there to save.

## Why: credit that runs back in time

Diagnostics recorded before their runs
(`results/event_lm/echo_learning/diagnostics.json`):

| Diagnostic | Gradient | Mirror echo | Gap |
|---|---:|---:|---:|
| 2 levels | 2.315 | 2.699 | 0.384 |
| 4 levels | 2.069 | 2.429 | 0.361 |
| 6 levels | 1.995 | 2.329 | 0.334 |

(16 batches at 6,000 updates.) The gap does not grow with depth, so
the missing credit is not the credit between levels. A read passes
its error down to the stream two ways: through its probe, the byte
it reads for, and through the tags and payloads of the earlier bytes
it reads. Gradient runs that keep one path:

| Reads pass their error down through | Validation |
|---|---:|
| Every path (g6) | 1.9860 |
| Probes alone | 2.0650 |
| Tags and payloads alone | 1.9712 |

A part needs to learn how what it wrote at a byte was read at later
bytes. **Read receipts** carry exactly that: the error of a reading
byte returns along the race to the cells it read, and enters the
stream where they were written (`results/event_lm/echo_learning/receipts.json`).
In its exact form, the gradient restricted to receipts, it scored
2.0534, against 2.3379 for the same restriction without receipts:
four fifths of the mirror's gap recovered, and 0.067 short of the
gradient, which fails its mark of 0.02. The rest is the credit that
the cells pass to the stream they hear.

Memory-only credit beat the full gradient at seed 20001 (1.9712
against 1.9860) but not at seed 20002 (1.9965 against 1.9855); two
runs of one setting on this test bed differ by up to 0.03, so it is
not a gain (`results/event_lm/v17_candidates/preregistered.json`).

## Reproduce

```bash
python experiments/train_event_lm.py TEST_BED_FLAGS --echo all --echo-feedback mirror --device cuda
python experiments/train_event_lm.py TEST_BED_FLAGS --echo all --device cuda
python experiments/train_event_lm.py TEST_BED_FLAGS --read-credit memory --cell-credit local --device cuda
```

`TEST_BED_FLAGS` are those of `results/event_lm/stream_form/small_test_bed.json`.
