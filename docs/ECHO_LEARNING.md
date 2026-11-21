# Echo learning

The parts of a spiking stack learn from an echo of the error at the output instead of from the error passed back
through every level above them. Two echoes were built: one through fixed random synapses, and a mirror, the error of
the stream that the output hears, sent back to every part in one hop. A ladder of test-bed runs was recorded before
any of them trained.

<!-- SUMMARY: the ladder's result in two sentences -->

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

<!-- results: g6, m6, mc6, r6, gate-g, gate-m; pulses; decision -->

## Reproduce

<!-- commands -->
