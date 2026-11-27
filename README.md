# Volley

Volley (formerly ShuntFormer) is an integer spiking byte
language model. Its neurons exchange signed spike counts over
fixed axons, and local charge holds history. It reads text as
one unbroken stream, byte by byte, and its memory is a race of
spikes over what it has read. Every release is measured against
a full-attention Transformer of the same size and training.

The name comes from the volley principle of neuroscience: groups
of neurons that fire in turn carry together what no single neuron
can. In the game, a volley answers the ball before it bounces, as
the model answers each byte when it comes. Releases up to v17 were
published as ShuntFormer; their tags and files keep that name.

The newest release, **v22**, trains v21's lifted loop with
power-law cells, which fire about the square of how far their
membrane stands past threshold, for the full 90,000 updates.
Read as its record registered, with fast output synapses, it is
0.003 worse than v21's stack and the Transformer stays ahead. But
those synapses cost this stack 0.009 to 0.012. Read without them,
the stack scores **1.0752 bits per byte** on the enwik8 test and
1.1054 on validation with the same 36.26M parameters: 0.0074
better than v21's read the same way, **0.0038 ahead of the
Transformer on test** (1.0790; paired interval 0.0027 to 0.0047)
and level with it on validation (1.1048; 0.0006 behind, within the
interval). That measure was chosen after the scores were seen, so
nothing is claimed: a second seed, read without fast synapses as
fixed in advance, is training. v22 also gives the stack heads that
count lines instead of bytes, so that it recalls a binding as far
back as the Transformer and writes chains of 12 steps unpractised.
It finds where its practice of written chains breaks: just past
the lengths it rehearses. The v22 weights are release assets,
and the model that runs from a clone is v11.

![Release by release, the spiking stack against the Transformer](results/figures/event_lm_full_size_line.png)

| Release | Adds | enwik8 test | Transformer | Scored in |
|---|---|---:|---:|---|
| [v22](#v22-power-law-cells-at-full-length-and-practice-that-has-to-stand) | Power-law cells at full length (1.075 on test without fast output synapses, not the registered reading), heads that count lines, where practice of written chains breaks | 1.084 | **1.079** | Unbroken streams; windows of 2,048 bytes |
| [v21](#v21-the-ceiling-lifted-at-full-length) | The ceiling lifted at full length, power-law cells, what the pool costs, fused reads, first screens of one model that reasons in writing | 1.082 | **1.079** | Unbroken streams; windows of 2,048 bytes |
| [v20](#v20-depth-by-recurrence-at-full-length) | Depth by recurrence at full length, the loop's form, gated cells that catch up, sleep | 1.104 | **1.079** | Unbroken streams; windows of 2,048 bytes |
| [v19](#v19-each-model-at-its-best-rate) | Each model at its best rate for 90,000 updates, where the distance lies, depth by recurrence with data | 1.113 | **1.079** | Unbroken streams; windows of 2,048 bytes |
| [v18](#v18-membranes-that-saturate-and-a-ceiling-that-lifts) | Word heads, the learning rate at full size, membranes that saturate and the ceiling that lifts | 1.116 | **1.107** | Unbroken streams; windows of 2,048 bytes |
| [v17](#v17-the-knowledge-gap) | Knowledge cells in the race, training on streams, where the gap is | 1.117 | **1.107** | Unbroken streams; windows of 2,048 bytes |
| [v16](#v16-long-streams) | Transformers on the stack's window lengths, forgetting slopes, fast synapses trained in the loop | 1.120 | **1.107** | Unbroken streams; windows of 2,048 bytes |
| [v15](#v15-a-stack-that-reads-a-stream) | A stack of state reads and threshold cells that reads a stream | **1.141** | 1.144 | Unbroken streams |
| [v14](#v14-read-clocks-and-the-address-rate) | A faster address rate and clock cells for the read heads | 1.244 | 1.182 | 512-byte windows |
| [v13](#v13-state-reads-and-race-cells) | State reads and race cells on delayed fibres | 1.262 | 1.182 | 512-byte windows |
| [v12](#v12-coasting-read-heads) | Read heads on a tape of the stream | 1.414 | 1.182 | One continuous stream |
| [v11](#v11-surprise-clock-and-fast-synapses) | Fast synapses written by error spikes | 1.645 | 1.704 | One continuous stream |

Each Transformer has the size and the training of its row. The
Transformers of v22, v21, v20, v19, v18, v17 and v16 trained on windows
of 2,048 bytes from 24,000 updates, as their stacks did, and read
2,048; every other one reads 512 bytes. v22, v21, v20 and v19 share
their Transformer, and their stacks and it trained at each model's
best peak rate, the stacks at 6e-4 and the Transformer at 1.2e-3;
the earlier ones at 3e-4. The stacks of v22, v21, v20, v19, v18 and v17
have 36.26M parameters, 4.2% more than their Transformer, for
their knowledge cells; v18's word heads add 352 more. The stacks
of v22, v21 and v20 run their levels 2 to 8 twice and fire 1.63 times
v19's pulses. v16 to v12 have about 34.8M parameters and
368.64M training targets from random weights. The Transformer of
v15 has the standard shape and trained under the rotated
optimizer of the spiking models. It reads a stream in windows of
496 bytes of history before 16 scored ones, and in the windows
of v13 and v14, 256 before 256, it scores 1.156. The rows of v14
to v12 name the Transformer of the standard shape that v14
trained with AdamW. The 256-wide Transformer that v12 and v13
published scores 1.207. In the windows of v13 and v14 the v12
decoder scores 1.467, because it gives up the stream it carries.
v11 has 25.79M parameters and 140.90M targets. Its fast synapses
carry the whole stream, and with the same synapses on its output
the v11 Transformer scores 1.578. Every arm is one seed.

[Knowledge-gap study](docs/KNOWLEDGE_GAP.md) ·
[Roadmap](docs/ROADMAP.md) ·
[Scaling-law study](docs/SCALING_LAW.md) ·
[Model card](models/spiking-byte-lm-v22/README.md) ·
[Leaderboard](docs/LEADERBOARD.md) ·
[Research handoff](docs/V22_HANDOFF.md) ·
[Latest release](https://github.com/DanielTea/transformer-snn/releases/tag/spiking-byte-lm-v22)

## How Volley works

```mermaid
flowchart LR
    B["Byte"] --> I["Input synapses"]
    I --> S["Stream: 512 channels of signed spike counts"]
    S --> R["State read: 8 heads race for 256 spikes over the stored cells of the stream and 128 knowledge cells"]
    R --> C["Threshold cells: 1,188 a level, under an inhibitory pool"]
    C -->|"back into the stream; 11 levels, levels 2 to 8 passed twice"| S
    S --> O["256 output neurons and their pool cell"]
    T[("Tape: every read's cells of the last 8,192 bytes")] -.-> R
    C -.->|"fast output synapses, taught by every arriving byte"| O
```

Every neuron sends whole spike counts, every synapse is a whole
number of charge ticks, and every sum stays within the range in
which floating point adds exactly, so the same spikes come out
however a stream is cut into pieces. A level is a state
read and a layer of threshold cells. The read sends a probe; the
cells that earlier bytes left on its tape answer with tags, and
those whose tags match it best win a race for a fixed number of
spikes and pass their payloads on: attention as a competition of
spikes, with an offset by age and a forgetting slope that lets it
reach 8,192 bytes back. The threshold cells hear the stream and
its last few bytes through delayed fibres, and a pool of
inhibitory cells raises every threshold with the stream's
activity, the role a norm plays in a Transformer. Since v20 the
middle levels run twice through the same synapses, more
computation for the same parameters, and since v21 the pool
divides a neuron's charge before its membrane bounds it, so no
neuron is held at its bound. Fast output synapses learn
from every byte after it was predicted, so the model adapts while
it reads. Training uses gradients through surrogate slopes; the
trained network fires whole counts, and its pulses are counted
against a budget.

## v22: power-law cells at full length, and practice that has to stand

v21 found a cell that keeps its gain with data on the test bed, the
power-law cell, which fires round(k * k / 8) spikes for k
thresholds of charge (`--granule-power 8`). v22 trains v21's lifted
loop with power-law cells for the full 90,000 updates, scored as
v21's stack was (`full_size_lifted_loop_second.json`). It also
takes the one-model track's practice of written chains further:
what keeps it, where it breaks, and why the stack's long chains
thinned out. Every test was recorded before its runs
(`results/event_lm/`).

**Read as registered, the power-law cells lose 0.003.** The record
read the stack as every full-size stack since v19: at a reach of
8,192 bytes with fast output synapses on all 18 passes. Read that
way the new stack scores 1.0845 bits per byte on the enwik8 test
and 1.1169 on validation, 0.0027 and 0.0034 worse than v21's
(paired 95% intervals 0.0022 to 0.0033 and 0.0028 to 0.0040), and
0.0055 and 0.0121 behind the Transformer. By the record, v21's
lifted loop stays the recipe.

**Without fast output synapses it is ahead of the Transformer on
test and level on validation.** The fast output synapses, which
learn from each byte after it was predicted, cost this stack 0.0092
on test and 0.0115 on validation: they still help copies and cost
new contexts 0.026 to 0.028. Read without them, the stack scores
1.0752 on test and 1.1054 on validation. That is 0.0074 better
than v21's stack read the same way, on both splits (0.0069 to
0.0080), so the power-law cells' gain carried to full length, most
of it on new contexts. Against the Transformer, with neither model
using fast synapses, the stack is 0.0038 ahead on test (0.0027 to
0.0047) and 0.0006 behind on validation (0.0004 ahead to 0.0016
behind). On new contexts the distance falls to 0.026 and 0.029,
from v21's 0.044 and 0.052. It spends 506,607 pulses a byte, 1.63
times the plain stack's.

**This is reported, not claimed.** The measure without fast
synapses was chosen after the scores were seen, and on validation
the stack is level, not ahead. A second seed of the same recipe is
registered with that measure fixed in advance
(`full_size_power_cells_second_seed.json`). It beats the
Transformer only if it is ahead on test and on validation, each
interval clear of zero, and the mean of the two seeds must agree.
It trains now on the fast path, the fused race kernels of v21,
after a check that the fast code scores this checkpoint to the same
bits.

**Practice holds with rehearsal, and breaks just past what is
rehearsed.** The second track trains one model to write its own
chains of thought. Trained on chains of 1 to 4 lookups written out,
a model practises longer ones and is paid only for the right
answer. Without rehearsal of the written corpus every arm of both
models lost its chains (`practice_within_reach.json`). With 16
written tasks rehearsed every round, all nine of the stack's runs
held under caps of 8, 12 and 16 steps (`practice_cap_raised.json`).
Up to 8 steps the Transformer practised as far. Past that, five of
its nine runs wrote 0.91 to 0.975 of 20-step chains whole, against
the stack's best 0.92, and four fell short: two found a form with
question marks, one lost its chains, one arrived late. With two
families of tasks in one loop, four of the stack's nine runs fell
(`practice_several_families.json`). Each kept 1 to 4 steps, wrote
one step more, then filled the room with one name repeated and
ended on one digit whatever the task. The Transformer's run that
fell stopped at the corpus's longest form and guessed; its other
two wrote every chain of up to 12 steps whole. The loop teaches
only from paid attempts, so nothing makes the filler less likely.
`practice_that_stands.json` tests the other half of dopamine's
signal: attempts below their task's mean made less likely, as the
striatum's indirect pathway learns from a dopamine pause.

**The stack's reads counted bytes, so its memory ended at a line.**
Why did the stack's long chains thin out where the Transformer's
did not? Its state reads weigh every stored byte by its age, and
they learned to turn against bytes older than about 121, the
length of one line of the corpus. Asked to recall one binding a set
number of bytes back, the stack answered 0.985 near and 0.305 and
0.190 at 150 to 200 bytes; the Transformer stayed at 0.97 to 1.00
(`recall_over_distance.json`). The brain cuts experience into
events and recalls within them, so two of every read's four heads
now count lines instead of bytes: they reach only the present line
and weigh each byte by its place in it (`line_heads.json`). With
them the stack recalls a binding 150 to 200 bytes back in 0.965 and
0.975 of lines, against 0.195 and 0.29 without. Without any
practice it writes 0.975 of 8-step chains whole and 0.785 of
12-step chains, against 0.655 and 0.205, and it loses nothing on
what it was trained on. Practice with line heads is next.

**Tried and failed since v21, each by its mark.** Pools trained
with the energy's gradient: no gain in the stack, although in
floats such a pool beats the Transformer's LayerNorm by 0.008
(`pool_gradient.json`). Power-law cells at twice the rate: 0.013
worse; skip fibres: 0.001 better, under the mark
(`full_size_power_loop_variants.json`). Cubic cells: 0.03 behind
square ones on the bed (`cubic_cells_with_data.json`).

**Running as this is published.** The second seed of the power-law
stack; the gap's slope with data at full size, that stack and the
Transformer at 45,000, 90,000 and 180,000 updates
(`full_size_data_slope.json`); and practice that stands.

**Limits.** One seed at full size, of the stack and of the
Transformer; how far two seeds lie apart at full size is not known
yet. The win on test is on a measure chosen after the fact. The
pulses a byte are the trainer's count, not audited by a native
decoder. The practice tasks are synthetic lookups, not language.

## v21: the ceiling lifted at full length

v20 trained the looped stack for the full 90,000 updates and found
a quarter of its top cells' entries at the membrane's bound. A
membrane holds the charge of a neuron's largest count at its
resting threshold and cuts the rest, so a neuron at its bound
learns nothing through its charge. v21 trains the same loop under
the pool that lifts the ceiling (`--pool-lifts-ceiling`): every
population's inhibitory pool divides a neuron's charge before the
membrane bounds it. Nothing else changed, and two full-size pilots
chose that form before the run was queued. Every test was recorded
before its runs (`results/event_lm/knowledge_gap/`); the study is
[docs/SCALING_LAW.md](docs/SCALING_LAW.md).

**The lifted loop is 0.003 behind the Transformer on test.** After
90,000 updates at 6e-4 the stack scores 1.0817 bits per byte on
the enwik8 test and 1.1136 on validation, read as 16 unbroken
streams with fast output synapses on all 18 passes at a reach of
8,192 bytes (`full_size_lifted_loop_run.json`). That is 0.0222
better than v20's looped stack on test (paired 95% interval 0.0216
to 0.0228) and 0.0251 on validation, at the same 36.26M
parameters, the same pulses (506,746 a byte, 1.63 times the plain
stack's) and the same training time. The pilot's gain after 24,000
updates was 0.0095, so the gain did not fade with length: it more
than doubled. The Transformer at its best rate is still ahead, by
0.0028 on test (0.0017 to 0.0038) and 0.0088 on validation (0.0077
to 0.0098), a ninth and a quarter of what v20 left; against the
Transformer with fast output synapses of its own the distances are
0.0046 and 0.0092. The stack does not beat it.

**Its gain is knowledge.** Split byte by byte by what each byte's
context offers, the lifted loop is 0.034 bits per byte better than
v20's stack on new contexts, three quarters of the bytes, on test
and 0.037 on validation: more than its whole gain. Against the
Transformer the distance on new contexts falls from 0.078 to 0.044
on test and from 0.090 to 0.052 on validation, and the stack stays
far ahead on copies (0.087 near, 0.44 and 0.40 from 2 to 8 KB
back). It loses part of that where the eight bytes before ran
earlier and another byte followed them, about 5% of the bytes:
0.057 and 0.098 worse than v20's stack there.

**No membrane is held at a bound.** Of the top cells' entries
0.036% reach their largest count, where 26.7% of v20's sat at the
membrane's bound; the most at any level is 0.07%. The fast output
synapses now add little: 0.0009 on test, and on validation they
cost 0.0007, so the best validation score is the one without them,
1.1128. A scan of their threshold on v20's stack kept 2^24
(`fast_synapse_threshold_loop.json`).

**The run's form.** Three full-size pilots of 24,000 updates asked
what else the loop should take. A third pass is 0.0008 better for
1.39 times the pulses (`full_size_loop_passes.json`). Under the
lifting pool the Transformer's rate, 1.2e-3, is 0.0031 better,
where the Transformer gains 0.0075 from the same step
(`full_size_loop_variants.json`), and twice the pulse budget is
0.0005 worse at 1.51 times the pulses
(`full_size_loop_budget.json`). None gained the 0.005 that would
have changed the run, so it kept two passes, 6e-4 and 507,000
pulses a byte.

**Power-law cells keep their gain with data.** A cortical neuron's
rate grows with a power of about two of how far its membrane
stands past the threshold. A power-law cell
(`--granule-power 8`) fires round(k * k / 8) spikes for k
thresholds of charge, up to its cap: a plain cell's synapses, no
gate, no new parameters. On four seeds of the test bed it leads
plain cells by 0.0125 on the mean after 24,000 updates at equal
pulses, every seed ahead (`power_cells_with_data.json`). In a
full-size pilot the lifted loop with power-law cells is 0.0137
better than the same loop with plain cells (0.0129 to 0.0145), on
new contexts and far copies, with half as many cells firing at a
byte (`full_size_power_cells.json`), and on the looped test bed
0.0095 better (`power_cells_in_loop_screen.json`). The lifted loop
with power-law cells is training for the full 90,000 updates as
this release is published (`full_size_lifted_loop_second.json`).

**What the layout costs is the pool's division, and how it
trains.** In floats at full size, the stack's kind of unit costs
the Transformer 0.0077 after 24,000 updates
(`full_size_units_in_floats.json`), and the stack's pool in place
of every LayerNorm costs it 0.069
(`full_size_layout_in_floats.json`). A centred pool still costs
0.064 (`full_size_centring_in_floats.json`), and a pool that
divides by the root mean square costs only 0.002: the price is the
division by the mean absolute activity. In spikes, pools that hear
the energy fail on the test bed and at full size, where the lifted
loop with them is 0.004 behind (`energy_pool.json`). The trained
streams of both models are dense and close to Gaussian, and the
two divisions agree within a few percent of a constant
(`stream_loudness.json`), so the price lies in how the pool
trains. Pools that fire as before and pass the energy's gradient
are on test (`pool_gradient.json`).

**Tried and failed, each by its mark.** Weight decay on synapses
only: level on the bed and 0.0017 worse at full size
(`decay_synapses_only_with_data.json`,
`full_size_decay_synapses_only.json`). Phases for the loop's
passes: behind on three seeds (`loop_phases_with_data.json`). Sleep
alone: 0.0034 ahead on the mean of two seeds, under its mark
(`sleep_with_data.json`). The homeostatic gate at 96,000 updates:
behind again, by 0.0097 (`homeostatic_gate_at_length.json`), so no
gated cell is open. A charge grid four times finer: 0.0066 behind
(`charge_grid_at_best_rate.json`). A racing loop whose output sums
the evidence of every pass: summing costs 0.031, though a third of
the bytes settle before the last pass and the best pass for each
byte would beat the loop by 0.013 (`racing_loop_screen.json`).

**Sleep with gates that start closed passes its screen.** On the
laptop's seed it leads the bed by 0.0061 (0.0052 to 0.0070) after
24,000 updates, for twice the pulses
(`sleep_closed_gates_screen.json`); two server seeds and its loop
form wait in the queue (`sleep_closed_gates_with_data.json`).

**A stream update in 0.18 s for 0.81 s, with the same spikes.**
The stack trained about ten times slower than its Transformer (16
hours against 1.6), almost all of it in the stream stage, where
each read's race compared every byte with every cell of its table.
A read by age works a byte out only against the cells in its
reach, the same pulses spike for spike: a full-size stream update
takes 0.42 s for 0.81 s (`tests/test_read_by_age.py`). Fused reads
(`snnformer/race_kernel.py`, `--fused-reads`) run the whole race,
forward and backward, in four GPU kernels written in Triton: the
same spikes at every weight, the gradient within 7e-6 of each
tensor's largest entry, 0.18 s an update in 9 GiB of memory for
41. By these times a run of 90,000 updates trains in under five
hours where this release's took 16. Beside them
(`experiments/speed_options.py`): whole-number products on the
GPU's tensor cores, to the bit; the optimizer's steps over all
tensors at once; the window stage from a CUDA graph; compiled
cells. Exact products, the fused optimizer and the graph leave a
run's weights unchanged to the bit. Fused reads and compiled cells
round the gradient differently, and that makes another run: from
the same weights and bytes two reads hold different weights after
two updates, while each repeats itself to the bit and any trained
model scores the same bits by either read. This release's
checkpoint was trained without them.

**A second track: one model that reasons in writing and reads
telemetry.** The plan is one spiking model that writes and reasons
with chains of thought of its own and reads machine telemetry as a
second sense; non-spiking teachers may help train it, and what is
released stays purely spiking. Four first screens
(`results/event_lm/one_model/`). Trained on direct answers to
lookups along chains of 1 to 4 names, no model learned the task
(`hop_generalization.json`): the stack looks up one hop in every
line and two in most, the Transformer none. Practice with reward on
the answer alone made no model write a chain
(`practice_hops.json`): the stack wrote a name in 0.5% of its
attempts, so reward had nothing to reinforce. Trained on the same
lines with every chain written out, both models chain in writing
(0.995 and 0.991 of lines right over 1 to 4 hops) and neither
chains past what it read: at 8 hops both answer one line in ten
right (`written_chains.json`). And the stack's surprise as a fault
detector on a simulated pump fails its mark (AUC 0.715 for 0.80):
near certain for faults that add vibration, at chance for faults
that move the pump to a new steady state (`telemetry_screen.json`).
Practice from the models that write chains is on test
(`practice_written_chains.json`).

**Running as this is published.** The lifted loop with power-law
cells at full length; and the first measurement of the gap's slope
with data at full size, that stack and the Transformer at 45,000,
90,000 and 180,000 updates (`full_size_data_slope.json`).

**Limits.** One seed at full size. The pulses a byte are the
trainer's count, not audited by a native decoder. The fast output
synapses' threshold was not scanned on the lifted loop. The speed
work was checked for equal spikes and for gradients to rounding;
no full-length run has yet been trained with fused reads.

## v20: depth by recurrence at full length

v19 found that the distance to the Transformer is knowledge, and
its loop pilot found the first part that adds knowledge at full
size at equal parameters: running levels 2 to 8 twice through the
same synapses, as cortex iterates. v20 trains that looped stack
for the full 90,000 updates beside v19's pair, picks the loop's
form in full-size pilots, and screens two gated cells and sleep on
the test bed. Every test was recorded before its runs
(`results/event_lm/knowledge_gap/`); the study is
[docs/SCALING_LAW.md](docs/SCALING_LAW.md).

**The loop holds at full length.** After 90,000 updates at 6e-4
the looped stack scores 1.1039 bits per byte on the enwik8 test
and 1.1387 on validation, read as 16 unbroken streams with fast
output synapses on all 18 passes at a reach of 8,192 bytes. That
is 0.0090 better than v19's plain stack on test (paired 95%
interval 0.0084 to 0.0095) and 0.0086 on validation
(`full_size_loop_run.json`). The pilot's gain after 24,000
updates was 0.0096, so the gain has not faded with four times the
data, the streams, the knowledge cells and the fast synapses. The
Transformer is ahead by 0.025 on test (0.024 to 0.026) and 0.034
on validation, where it was ahead of the plain stack by 0.034 and
0.042. The loop adds no parameters; it fires 1.63 times the plain
stack's pulses, for which its budget was scaled, and every claim
of it says so.

![At full length the loop holds and narrows the gap to the Transformer](results/figures/event_lm_loop_full_length.png)

**Its gain is knowledge.** Split byte by byte by what each byte's
context offers, nine tenths of the loop's gain is on new contexts,
three quarters of the bytes: 0.012 bits per byte better than the
plain stack there on test and 0.010 on validation. Against the
Transformer the distance on new contexts falls from 0.090 to 0.078
on test and from 0.100 to 0.090 on validation, and the stack stays
far ahead on copies (0.08 near and 0.46 from 2 to 8 KB back).
Where the eight bytes before ran earlier and something else
followed them, about 5% of the bytes, the looped stack is about
0.05 worse than the plain one.

**Its membranes saturate, and the pool that lifts the ceiling
helps it.** The looped stack's top level sits at its bound twice
as often as the plain stack's: 27% of the top cells' entries
against 12%, and 11% of the top read's against 2.4%. In full-size
pilots of 24,000 updates (`full_size_loop_variants.json`) the loop
under the pool that lifts the ceiling, which divides a charge
before it is bounded, is 0.0095 better than the loop as it trained
here (0.0087 to 0.0104), 0.0145 on new contexts, and the loop at
the plain stack's pulse budget is 0.0063 worse: the loop needs its
pulses. On the test bed the loop's gain is constant with data on
two more seeds (`loop_with_data.json`: 0.014 and 0.028 over the
energy control after 24,000 updates), and skip fibres add nothing
to it (`skip_fibres_with_data.json`). The next full-length run
takes the loop under the lifting pool; two pilots decide whether
it also takes a third pass and the Transformer's rate.

**The gated cells: bounds are not the trouble, learning is.** In
floats the plain rectifying unit, the stack's cell, falls behind
with data, where the Transformer's gated units do not (v19). No
bounded gate falls behind with data in floats
(`bounded_gates_in_floats.json`): ramp, shunt, logistic and product
gates each cost a constant level and none grows. In spikes, a ramp
gate held in range as cortex holds a neuron, its synapses scaled
back to their first total and its rest moved by a local rule
toward a set share of open bytes (`--granule-gate-home`), is the
first spiking gate that catches up with data: 0.020 behind the
plain cell after 6,000 updates and 0.005 after 24,000, on two seeds
(`homeostatic_gates_with_data.json`), and from 0.027 to 0.017 on a
third. It misses the screen's mark, and a pair at 96,000 updates
decides whether it passes the plain cell
(`homeostatic_gate_at_length.json`). A gate that multiplies two
drives of either sign (`--granule-gate-signed`) falls behind with
data on three seeds (`signed_gates_with_data.json`): its product
lets in up to 2.2 times the value, its cells fire 22,200 pulses a
byte, and the reads are left with half their pulses.

**Sleep, built and on test.** At rest the hippocampus replays
recent experience, forward and in reverse, and memories are
rewritten in the light of what followed. Lee et al. (2026,
"Sample-Efficient In-Context Adaptation via Sleep") let a
Transformer re-run itself over every finished window of its
context and refine its stored keys and values through a learned
gate, and it reaches its loss with 2.1 to 2.4 times fewer context
tokens. v20 builds it in spikes (`--stack-sleep 64`): after every
64 bytes of a stream the stack replays the window that has just
ended. The stream as it left the top of the stack enters the
first level again, every level runs once more over the window, and
the replay's reads see the whole window, its later bytes too. Gate
cells in every read then decide, cell by cell and in quarters, how
much of the stored cell the replay rewrites. The spikes are the
same however the stream is cut, and with gates that keep every
cell it is the stack without sleep. After 6,000 updates on the
test bed it is not ahead: 0.004 behind the plain bed alone and
0.007 behind the loop on top of it, two seeds each. The runs of
24,000 updates decide (`sleep_with_data.json`). The replay doubles
the pulses a byte.

**Smaller results.** Reaches of 16,384 and 32,768 cells read worse
than 8,192 for v19's stack (`stream_reach_longer.json`), so 8,192
stays. An output whose every spike stands for 8 fine steps saves a
third of the pulses and costs 0.0016 after 24,000 updates on one
seed of the test bed (`output_step_with_data.json`).

**Limits.** One seed at full size. The pulses a byte are the
trainer's count, not audited by a native decoder. The looped stack
trained without the lifting pool that its pilots now prefer.

## v19: each model at its best rate

v18 found the learning rate at which each model trains best at
full size, in runs of 24,000 updates: the stack at 6e-4, the
Transformer at 1.2e-3. v19 trains both for the full 90,000
updates at those rates, asks why the rate pays one model and not
the other, and measures where the distance between them lies.
Every test was recorded before its runs
(`results/event_lm/knowledge_gap/`); the study is
[docs/SCALING_LAW.md](docs/SCALING_LAW.md).

**The best rate pays the Transformer, not the stack.** After
90,000 updates at 6e-4 the stack scores 1.1129 bits per byte on
the enwik8 test and 1.1473 on validation, read as 16 unbroken
streams with fast output synapses at a reach of 8,192 bytes. That
is its best score so far (v18: 1.1162 and 1.1491). The
Transformer at 1.2e-3 scores 1.0790 and 1.1048. The stack is
0.034 behind on test (paired 95% interval 0.033 to 0.035) and
0.042 behind on validation; at a shared rate of 3e-4, v17's gaps
were 0.010 and 0.019 (`full_size_best_rate_pair.json`). Paired
with its own run at 3e-4, the Transformer gains 0.028 from its
best rate and the stack 0.004. After 24,000 updates the two gains
were alike, 0.030 and 0.022, so with data the stack's gain from a
faster rate fades and the Transformer's stays. The stack is still
ahead wherever the text it has read offers a copy: by 0.07 bits
per byte on copies from the last 2,032 bytes and by 0.41 on copies
that only its reach finds. On new contexts, three quarters of the
bytes, it is 0.09 behind. The distance is knowledge, as in v17 and
v18.

![At each model's best rate the gap is still knowledge, and wider](results/figures/event_lm_gap_by_context_v19.png)

**The stack's code thins as the rate rises, and its silent cells
sink.** A census on trained weights
(`unit_density_by_rate.json`) counts the cells of a level that
fire at a byte and the Transformer's units whose gate is open. At
3e-4 the two are alike. Above it the stack's code thins: at full
size 30%, 16% and 7% of a level's cells fire at 3e-4, 6e-4 and
1.2e-3, and nine tenths of a level's spikes come from 20%, 11%
and 4% of its cells. Half of the Transformer's units carry nine
tenths of what a block passes on, at every rate and both sizes,
because a closed gate still lets a share through. No cell dies.
The silent cells are far from firing: at the stack's best rates
63% to 86% of cells and bytes stand more than eight thresholds
below zero charge, where the slope from which a cell learns is
zero. The stack stops gaining from a higher rate where its code
is thin. That need not be the cause: in floats, a plain unit's code
thins with the rate as well, and it still gains from the rate
(below).

![A higher learning rate thins the stack's code, not the Transformer's](results/figures/event_lm_unit_density.png)

**Four fifths of the distance is the layout, not the spikes.** A
float stand-in at full size (`full_size_spikes_late.json`; a
diagnostic, never a path) replaces the stack's racing reads and
threshold cells with softmax reads and real-valued cells, while
its stream still fires whole spikes. After 24,000 updates at 3e-4
the spiking stack is 0.036 behind the Transformer. The stand-in
closes 0.007 of that, and 0.029 is left; on new contexts the
stand-in is still 0.033 behind. On one pool, spikes in reads and cells cost 0.013;
against the model with its pool of fine quanta they cost 0.007,
between the record's two marks, so two half stand-ins split that
cost next. The rest belongs to the layout: to what a block
computes, not to the fact that it spikes.

![81% of the distance to the Transformer is the layout, not the spikes](results/figures/event_lm_distance_split.png)

**In floats, the stack's kind of unit is the one that falls
behind with data.** The largest priced step between the two
models is the unit: the Transformer's multiplies a smooth gate by
a value, the stack's cells add and fire above a threshold. A
diagnostic on the test bed's Transformer
(`plain_units_with_data.json`, floats only, never a path) trained
it with plain or gated, hard or smooth units for 6,000 and 24,000
updates on two seeds. Against its own units, the plain unit that
is shut below zero, the stack's threshold cell in floats, falls
behind as data grows: its gap grows by 0.029 on the mean of the
seeds and ends 0.011 and 0.018 behind, most of it on new contexts.
A smooth plain unit and a hard gated unit each end within about
0.005 on the mean (0.0020 and 0.0048), but their seeds disagree,
so their growth is unclear. A smoother cell and a cell that
multiplies both remain candidates in spikes; a second diagnostic
asks which gate. A census finds the hard gated unit's code as thin
as the plain one's, so a thin code alone does not decide it. And
a plain unit gains from a twice higher rate in floats (0.0046 and 0.0009
on the two seeds), where the spiking stack does not: the stack's
flat response to the rate is not its unit's shape.

![In floats, the plain hard unit falls behind as data grows](results/figures/event_lm_plain_units.png)

**The gated cells so far had bounded gates, and all fail.** A gate
dendrite as a shunt is 0.118 behind at full size, and 0.113 behind
under the pool that lifts the ceiling, where nothing saturates
(`full_size_lifted_gate.json`). A cell that gates itself is level
(0.001 behind, `full_size_unit_screen.json`). A gate that opens in
proportion keeps pace for 2,250 updates and then turns into a
switch: 0.65 and 0.80 behind after 24,000 updates on two seeds
(`proportional_gates_with_data.json`). The one gate form never
tried, a plain multiplying gate under the lifting pool, trains
stably in a check of 2,250 updates on the laptop but is 0.067
behind the plain lifted stack, so it earned no run. The other
spike form of a better unit, noisy thresholds, leads by 0.004 at
6,000 updates and by 0.002 at 24,000, a lead that shrinks with
data (`noisy_thresholds_with_data.json`). Every spiking gate so
far was bounded, and the gates of the float reference are not, so
a second diagnostic asks whether a bound on the gate costs a gated
unit its place as data grows (`bounded_gates_in_floats.json`,
training on the server's CPU).

![At 2,250 updates the gate that keeps pace is graded; the gates that lose are switches](results/figures/event_lm_gate_state.png)

**Depth by recurrence holds with data.** A looped stack runs some
levels several times with the same synapses. At full size, in
24,000-update runs at 6e-4, levels 2 to 8 run twice are 0.019
ahead of the plain stack given the same pulses, 0.026 on new
contexts (`full_size_loop_pilot.json`). On the test bed
(`loop_with_data.json`), levels 1 to 4 run three times are 0.025
ahead of the bed with the same pulses after 6,000 updates and
0.017 after 24,000 (0.013 on new contexts), and on two more
seeds the looped bed leads the plain bed by 0.020 and 0.025 after
24,000 updates. More pulses alone hurt in both places. The loop
costs 2.3 times the pulses and about twice the training time. It
is the one part that adds knowledge at equal parameters, and its
run of 90,000 updates at full size is half done
(`full_size_loop_run.json`).

![Full size: what each part is worth to the spiking stack](results/figures/event_lm_full_size_parts.png)

**Skip fibres give a small, steady gain, under their mark.**
Through skip fibres a level's read and cells also hear the stream
of earlier levels (`--stack-skips`, `skip_fibres_with_data.json`).
They lead the bed by about 0.005 at both lengths on both seeds,
for 0.2% to 0.3% more parameters and no more pulses. The mark was
0.010, so they go to no full-size pilot. On top of the loop they
lead by 0.007 on the first seed; that screen is read when its
runs are scored.

**A slope far below the threshold, built for v20.** A silent
cell can learn from how far its membrane stands below its
threshold, as voltage-dependent plasticity does
(`--granule-slope-tail`, `subthreshold_slope_with_data.json`). Its
spikes are unchanged, and its slope falls off as a power of its
distance below zero, so a cell eight thresholds down still learns
at a ninth. On the laptop's seed of the test bed, the form that
teaches everything the charge depends on is level with the base
after 6,000 updates (0.0002 behind) and 0.0095 behind after
24,000; the local form, which teaches only the cell's own synapses
and rest, is 0.0013 behind after 6,000 updates, and its long run
ends after this release. The server's two seeds and the full-size
pilots are next.

**The output's spikes were finer than the score needs.** An output
neuron fired one spike for every 1/128 bit of evidence. With one
spike for 1/16 bit, on trained weights with no weight changed, the
test bed fires 34% fewer pulses for 0.00004 bits per byte, and
v18's full-size stack fires 21% fewer (246,000 a byte for 310,000)
for 0.00002 on the first 131,072 test bytes
(`output_step_on_trained_weights.json`). That is a saving in
energy and no gain in bits.

![Output spikes of 1/16 bit: the same score for a fifth to a third fewer pulses](results/figures/event_lm_pulse_populations.png)

**The cell's shape holds with data.** A cell that hears every
channel's present byte and one delayed copy of it, the full
size's shape, leads the test bed's own shape by 0.015 after
24,000 updates on four seeds, and by 0.020 on new contexts at
every seed (`cell_shape_with_data.json`).

**Built and pre-registered for v20.** The bounded gates in floats
(`bounded_gates_in_floats.json`). The two half stand-ins and the
stand-in at the Transformer's rate (`full_size_spikes_late.json`).
The slope's two server seeds and its full-size pilot
(`subthreshold_slope_with_data.json`). Two pilots of loop variants
(`full_size_loop_variants.json`). The loop's run of 90,000
updates.

## v18: membranes that saturate, and a ceiling that lifts

v17 found that the stack's gap to the Transformer is knowledge,
and that on the test bed it reopens with more data. v18 asks
whether the stack was held back by the learning rate every earlier
release used, 3e-4, and finds what caps it at full size. Every
test was recorded before its runs
(`results/event_lm/knowledge_gap/`); the study is
[docs/SCALING_LAW.md](docs/SCALING_LAW.md).

**The test bed wanted a higher rate, but the flattening is not its
artefact.** At five peak rates and two seeds, both models do
better than at 3e-4: the stack at 1.2e-3, the Transformer at
2.4e-3. From 3e-4 to its best rate the stack gains 0.063 and the
Transformer 0.053. The difference, 0.010, is under the 0.020 that
the study set for calling the flattening an artefact of the rate.
At each model's best rate the gap is 0.048, and it is on new
contexts on both seeds (`learning_rate_with_data.json`).

![Both models wanted a higher learning rate](results/figures/event_lm_learning_rate.png)

**At full size the Transformer gains more from a higher rate.** In
complete 24,000-update runs the stack is best at 6e-4 and the
Transformer at 1.2e-3, and the gap at their best rates is 0.044,
wider than the 0.036 at 3e-4 (`full_size_rate_pilot.json`). On new
contexts the two gain alike from their best rates; the
Transformer's extra gain is on bytes it can copy.

**Because the stack's membranes saturate.** A membrane holds the
charge of a neuron's largest count at its resting threshold.
Charge beyond that is cut, so the neuron fires its largest count
whatever more it hears and learns nothing through its charge. At
1.2e-3, 2.7% of the full-size stack's read entries and 2.0% of its
cells' sit at the bound, up to 9.7% and 11.8% at the top level; at
3e-4 almost none do. The test bed barely saturates even at 2.4e-3,
which is why it never showed this
(`experiments/membrane_saturation.py`,
`membrane_saturation.json`).

![Saturated membranes cap the full-size stack's rate](results/figures/event_lm_membrane_saturation.png)

**The pool that lifts the ceiling removes it.** v15 built a pool
that divides the charge before the membrane bounds it, so a neuron
fires up to its largest count at every threshold. No release used
it, because all trained at 3e-4. At full size it removes the
saturation entirely, at the same energy (about 308,000 pulses a
byte in training), and turns the stack's loss at 1.2e-3 into a
level result with 6e-4 (1.3126 against 1.3134; the plain stack
scores 1.3475 at 1.2e-3). With each model at its best rate the
lifted stack is 0.036 behind, as at 3e-4: lifting removes the
widening that higher rates brought, and the gap that remains is
knowledge (`full_size_lifted_ceiling_pilot.json`).

![At full size, lifting the ceiling stops the stack breaking at a high rate](results/figures/event_lm_full_size_rate.png)

**Ruled out on the way.** The pulse budget does not hold the stack
back: its price falls to 0.00 to 0.02 by the end of every run at
both depths, where a budget that binds runs at 0.10 to 0.24. Nor
do noisy late weights, nor a code that grows denser with data: at
3e-4, 35% to 56% of a level's cells fire at a byte at 6,000 and
24,000 updates alike (`experiments/cell_activity.py`). Depth 22 saturates
no more than depth 11.

**Word heads at full size.** One word head in every read, a head
that reaches only the present word's cells and weighs each by its
place in the word, gains 0.0009 on test (paired 95% interval
0.0004 to 0.0013) and 0.0006 on validation (0.0001 to 0.0010) over
v17's stack. That is under the 0.003 its record asked, so word
heads do not join the design. The run is still the best spike-only
stack so far: 1.11615 on test and 1.14908 on validation, read as a
stream reaching 8,192 cells with taught output synapses
(`full_size_word_heads.json`, [model
card](models/spiking-byte-lm-v18/README.md)). Its gap to the
Transformer keeps v17's shape: better on bytes it can copy, by
0.10 within 2 KB and 0.40 from 2 to 8 KB back, and 0.057 behind on
new contexts.

**The unit, in spikes.** The largest step that v15's walk from the
Transformer to the stack priced was the unit: plain rectifying
units in place of gated ones cost 0.025. Two spike-only answers
were tested at full size: gated threshold cells, in which a gate
dendrite lets in a share of what a value dendrite collects (built
for an earlier pilot), and noisy thresholds, new in v18, in which
a cell's membrane carries logistic noise while training, so that
its average count is a smooth curve and the cell learns from that
curve's true slope; scoring stays exact. At full size, in one 24,000-update run each against the
plain stack at 6e-4 (`full_size_unit_screen.json`), gated cells at
equal parameters lose by 0.118, worse on every kind of context;
their gate dendrites sit at their bound a fifth of the time, so
the gate opens fully and the cell loses its range. Noisy
thresholds are 0.004 better at the same energy, all of it on near
copies and none on new contexts. The self-gated arm runs next, and
the test bed screens both forms by their slope over data.

**Still running** for v19: the full-size pair, each model at its
best rate (90,000 updates); the data-scaling curves from 6,000 to
96,000 updates at each model's best rate; quantal release and
stochastic firing at two rates; the test bed's screens of gated
cells and noisy thresholds by their slope; gated cells under the
lifted ceiling; and depth by recurrence, with the energy it takes
as an arm of its own.

**In since the release:** the full size's cell shape passes at four
seeds. After 24,000 updates it is 0.015 ahead of the test bed's own
shape and 0.020 better on new contexts, at every seed
(`cell_shape_with_data.json`).

## v17: the knowledge gap

The v17 study asks where the stack's gap to the Transformer lies,
and what moves it.

| Trained from scratch, 368.64M targets | Reads the text as | enwik8 test | enwik8 validation |
|---|---|---:|---:|
| 36.26M SNN stack, v16's with 128 knowledge cells a head, trained on streams | A stream, as trained | 1.13571 | 1.16518 |
| | A stream reaching 8,192 cells | 1.12097 | 1.15148 |
| | The same, with taught output synapses | **1.11702** | **1.14963** |
| 34.81M SNN stack, v16's release | A stream reaching 8,192 cells, taught output synapses | 1.11994 | 1.15138 |
| 34.75M full attention, windows of 2,048 bytes | Windows, 2,032 bytes of history before 16 scored | **1.10669** | **1.13033** |
| 69.36M SNN stack, 22 levels, the same pulse budget | A stream reaching 8,192 cells, taught output synapses | 1.09257 | 1.12302 |
| 69.37M full attention, 22 layers | Windows, 2,032 bytes of history before 16 scored | **1.07042** | **1.09299** |

The stack is 0.0103 behind on test (paired 95% interval 0.0092 to
0.0114) and 0.0193 behind on validation (0.0182 to 0.0205). The
knowledge cells missed their own mark, recorded before the run: at
the trained reach they gain 0.001 over v16's stack, where 0.005
was asked; with the longer reach and the fast synapses the gain is
0.003. At depth 22 the stack gains 0.027 over depth 11 and the
Transformer 0.036. At equal energy for the stack, the gap widens
with size, to 0.022 on test and 0.030 on validation.

| What a byte's context offers | Share of bytes | Stack minus Transformer, test | validation |
|---|---:|---:|---:|
| Copyable from the last 2 KB | 13 to 17% | **−0.091** | **−0.087** |
| Copyable only from 2 to 8 KB back | 4% | **−0.324** | **−0.307** |
| Seen before, another byte followed | 5% | −0.004 | −0.092 |
| New context | 74 to 77% | **+0.057** | **+0.066** |

`experiments/gap_by_piece.py` pairs two scorings of the same
streams piece by piece. From how the kinds of byte mix in each
piece, it fits the gap of each kind. A target's kind is what its
context offers: whether the eight bytes before it ran earlier in
the stream, how far back, and whether the byte that followed them
then is the target. The split above is v16's stack against its
Transformer.

![With four times the data the test bed's gap reopens](results/figures/event_lm_gap_with_data.png)

On the test bed, 5.25M parameters scored over 2 million validation
bytes as unbroken streams, the delayed stack trained on streams is
0.047 ahead of its Transformer after 6,000 updates. After 24,000
updates, with four times the data, it is 0.054 behind. Delayed
fibres, training on streams and knowledge cells all helped most
early.

A single knowledge cell a head copies random letters better at
all four seeds, by 1.37 bits a byte on average, and it is 0.017
better over long streams. The base's seeds copy at 2.60 to 4.04
bits a byte on the second showing; the sink's at 1.61 to 2.44.

These failed, each under its recorded rule:
- myelinated fibres that learn their delays: the delays move toward the present;
- a cerebellar sheet of four times the cells;
- word heads, which gain memory that fades with data;
- kept cells;
- fast synapses keyed by knowledge cells;
- branching dendrites, apical bursts and racing experts.

The full size's cell shape, a fibre for every channel's present
byte and one delayed copy, is better than the test bed's own at
6,000 updates on two devices, including on new contexts, and
still 0.016 better at 24,000.

[Knowledge-gap study](docs/KNOWLEDGE_GAP.md) ·
[Roadmap](docs/ROADMAP.md) ·
[Leaderboard](docs/LEADERBOARD.md) ·
[Model card](models/spiking-byte-lm-v17/README.md) ·
[Research handoff](docs/V17_HANDOFF.md) ·
[Checkpoint release](https://github.com/DanielTea/transformer-snn/releases/tag/spiking-byte-lm-v17)

## v16: long streams

The v16 study asks what v15's level score was worth. Its
Transformers train exactly as its stacks do, on windows of 512
bytes for 24,000 updates and on windows of 1,024 or 2,048 bytes
for the remaining 66,000, and read the test in windows of their
training length.

| Trained from scratch, 368.64M targets | Reads the text as | enwik8 test | enwik8 validation |
|---|---|---:|---:|
| 34.81M SNN stack, windows of 2,048 bytes | A stream, as trained | 1.13903 | 1.16870 |
| | A stream reaching 8,192 cells at levels 4 to 7, with taught output synapses | 1.13472 | 1.16660 |
| 34.75M full attention, windows of 2,048 bytes | Windows, 2,032 bytes of history before 16 scored | **1.10669** | **1.13033** |
| | With taught output synapses | 1.10321 | 1.12901 |
| 34.81M SNN stack, windows of 1,024 bytes (v15) | A stream, as trained | 1.15359 | 1.18278 |
| 34.75M full attention, windows of 1,024 bytes | Windows, 1,008 bytes of history before 16 scored | 1.11820 | 1.14104 |
| 34.81M SNN stack, windows of 2,048 bytes, slopes that learn | A stream, as trained | 1.13667 | 1.16624 |
| | A stream reaching 8,192 cells | 1.12349 | 1.15368 |
| | The same, with taught output synapses | **1.11994** | **1.15138** |
| | Second seed, reaching 8,192 cells | 1.12048 | 1.15193 |
| 34.75M full attention, windows of 2,048 bytes, second seed | Windows, 2,032 bytes of history before 16 scored | 1.10366 | 1.12873 |

![The stack and the Transformer at training windows of 512, 1,024 and 2,048 bytes, test and validation](results/figures/event_lm_long_streams.png)

Each read as it trained, the stack is 0.0323 behind on the test
at 2,048 bytes, with a paired 95% interval of 0.0314 to 0.0333,
and 0.0384 behind on validation (0.0374 to 0.0394). A stream lets
the stack read further than it trained at no cost. The stack whose
slopes stayed at zero gains nothing from it at 2,048 bytes; the one
whose slopes learned does: reaching 8,192 cells back with its fast
synapses learning, it is 0.0133 behind the Transformer on test, and
0.0167 when the Transformer has fast synapses too. Second seeds of
both models are each about 0.003 better and keep the gap; every
other row is one seed.

![On books the gain of the fast synapses grows as the square root of the text read](results/figures/event_lm_long_streams_reading.png)

Every head of a read now has a **forgetting slope**: a cell
beyond the window loses charge for every doubling of its age, so
its share of the read falls as a power of its age. Set at
scoring on v15's released stack, a slope of −1 at a reach of
16,384 cells gains 0.008 on both splits; learned in training,
79 of the 88 heads settle below −0.69, which bounds the far past,
at a median of −1.3. On 16 PG-19 books, the
gain of v15's fast output synapses grows from 0.004 bits per
byte in the first 8 KB to 0.028 after 128 to 256 KB, as the
square root of the text read; on enwik8 it stays level. Squared
and gated cells that multiply charges collapsed in training. A
study of fast synapses trained in the loop passed its go/no-go:
cells trained with a decaying rule make better keys for v15's
rule, by 0.013 bits per byte on the test bed, and the full-size
run is queued for the next release.

![The stack learns faster at first, the Transformer for longer](results/figures/event_lm_long_streams_scaling.png)

Along every axis that the weekend's runs measure, the
Transformer's curve is at least as steep as the stack's. The
stack leads for the first 7,500 updates and then falls behind;
doubling the depth gains the Transformer about 0.04 bits per
byte and the stack 0.02, though the deeper stack kept the same
budget of pulses per byte, so each level fired half as much. On
the small test bed that costs little: twelve levels gain 0.130
over six at the same budget and 0.152 with it doubled, while more
pulses alone make the stack worse, and a loop of tied levels is
worth about a third of a level of its own. Scaling this design
will not close the gap, and the next step has to change the
stack.
[How the two models scale](docs/LONG_STREAMS.md#how-the-two-models-scale)

Three ideas were tested this weekend under rules recorded before
their runs, and none holds. Fast synapses that consolidate into
slower stages forget the far past and did worse than v15's rule
([consolidating synapses](docs/CASCADE_SYNAPSES.md)). Shunting
gates that divide charges were stable but their gain did not
survive a second seed. Echoes of the output's error in place of
the backward pass failed, and on the way showed that the stack's
credit runs back in time along its reads
([echo learning](docs/ECHO_LEARNING.md)).

[Long-streams study](docs/LONG_STREAMS.md) ·
[Trained plasticity](docs/TRAINED_PLASTICITY.md) ·
[Leaderboard](docs/LEADERBOARD.md) ·
[Model card](models/spiking-byte-lm-v16/README.md) ·
[Research handoff](docs/V16_HANDOFF.md) ·
[Checkpoint release](https://github.com/DanielTea/transformer-snn/releases/tag/spiking-byte-lm-v16)

## v15: a stack that reads a stream

The v15 study drops the blocks of the decoders. Its **stack**
keeps the two parts that v13 and v14 added, state reads and
threshold cells, and gives them the layout of a Transformer: a
stream of 512 channels runs through eleven levels, each a state
read of 8 heads and then a layer of 1,188 threshold cells, and
both write their spikes back into the stream. A **pool** cell at
every level divides what a neuron hears by how loud the stream
is, in spikes, and takes the place of a normalisation. The cells
and the tags of the reads hear every channel now and once more
after one to three bytes. The stack has 34.81M parameters.

![The stack as it reads a stream: eleven levels of a state read and a layer of threshold cells](results/figures/event_lm_stream_form_circuit.png)

The stack reads a **stream**. It carries the tags and payloads
of its reads and the last bytes of its fibres from piece to
piece, and a stream cut anywhere fires exactly the spikes of one
pass over the whole of it. In a stream the reads can reach
further back than in training at no cost in synapses, and fast
output synapses that every arriving byte teaches learn the text
as it is read.

| Trained from scratch, 368.64M targets | Reads the text as | enwik8 test | enwik8 validation | PG-19 |
|---|---|---:|---:|---:|
| 34.81M SNN stack, long windows | A stream, the reads after levels 5 to 7 reach 4,096 cells | **1.14058** | 1.16936 | **1.42490** |
| | A stream, as trained | 1.15359 | 1.18278 | 1.44559 |
| 34.81M SNN stack, windows of 512 bytes | A stream, the reads after levels 4 to 7 reach 2,048 cells | 1.17095 | 1.19872 | 1.46178 |
| | A stream, as trained | 1.18518 | 1.20930 | 1.47610 |
| 34.75M full attention, standard shape | Windows, 496 bytes of history before 16 scored | 1.14388 | **1.16289** | 1.42649 |
| | 448 before 64 | 1.14533 | 1.16416 | 1.42786 |

The stack on long windows is 0.0033 bpb below the Transformer on
the test, with a paired 95% interval of −0.0048 to −0.0018, and
0.0065 above it on the validation split (+0.0050 to +0.0079).
With fast output synapses on both sides it scores 1.1339 on the
test against the Transformer's 1.1344, and 1.1649 against 1.1574
on validation. Every row is one seed, and the rows received
identical windows from random weights up to 24,000 updates.

![Validation loss over training in windows, and the stacks' distance to the Transformer](results/figures/event_lm_stream_form_training.png)

In windows from an empty state the stack trained on windows of
512 bytes follows the Transformer at about 0.048 bits per byte
to the end of training. What closes that distance is the stream.
The same weights of the released stack score 0.041 better as a
stream than in the warmed windows of v14, and a reach of 4,096
cells gains another 0.013. The long windows count as well: the
stack that read windows of 1,024 bytes from 24,000 updates is
0.030 ahead of the one that stayed on 512.

![From the Transformer to spikes, one change at a time](results/figures/event_lm_stream_form_ladder.png)

A walk from the Transformer to the spiking stack, one change per
run, places the rest of the distance. Whole spikes and synapses
on the 8-bit grid cost 0.0008, and age offsets in place of
rotary phases cost nothing. The plain units that threshold cells
compute in place of the Transformer's gated ones cost most,
0.025. Gated cells, the spike form of that unit, were built in
four variants and stalled. Training on streams with the state
handed on did not beat training on windows.
[Next pilots](docs/V15_HANDOFF.md#next-pilots)

[Stream-form study](docs/STREAM_FORM.md) ·
[Every training arm](docs/STREAM_FORM_ARMS.md) ·
[Leaderboard](docs/LEADERBOARD.md) ·
[Model card](models/spiking-byte-lm-v15/README.md) ·
[Research handoff](docs/V15_HANDOFF.md) ·
[Checkpoint release](https://github.com/DanielTea/transformer-snn/releases/tag/spiking-byte-lm-v15)

## v14: read clocks and the address rate

The v14 study changes how the read heads of the v13 decoder
learn where to look. A head addresses the earlier bytes with
charges: a gain on the match of probe and tag, and an offset for
every age. One unit of charge multiplies a cell's rate in the
race by e, so an address has to move much further than a synapse
does. With an **address rate** of 8 the stored charges count
eight times, and one update moves an address eight times as far.
The decoder keeps its 34.82M parameters.

The study also gives a head a second kind of age. Every head has
a **clock cell** whose spike is a beat, and the head counts the
beats that fell after each earlier byte. In a made-up language
whose words depend on the word before last, clock cells learn to
beat on the spaces, and a read then finds that word whatever the
lengths of the words between. At full size the clock cells did
not find the words. The decoder with them is the one released
because it scores best, but its lead of 0.0013 bpb is less than
a second run moves a decoder.

![A read head counts time in beats, and the test on the made-up language](results/figures/event_lm_read_clocks_language.png)

| Trained from scratch, 368.64M targets | Optimizer | enwik8 | PG-19 |
|---|---|---:|---:|
| 34.82M SNN, clock cells and address rate | Rotated, averaged | 1.24445 | 1.53378 |
| 34.82M SNN, address rate | Rotated, averaged | 1.24570 | 1.53600 |
| 34.82M SNN, spike reads and race cells | Rotated, averaged | 1.26226 | 1.55162 |
| 34.85M full attention, 256 wide | AdamW | 1.20684 | 1.48212 |
| 34.85M full attention, 256 wide | Rotated, averaged | 1.20407 | 1.48204 |
| 34.75M full attention, standard shape | AdamW, averaged | **1.18215** | **1.45701** |

Both Transformers are ahead. The released decoder is 0.038 bpb
behind the 256-wide one on enwik8 and 0.052 on 16 PG-19 books.
The Transformer of the standard shape is 512 wide with 11
layers, was trained for this study and is 0.062 and 0.077 ahead.
Every score in the table is in windows of 512 bytes that start
from an empty state, with 256 unscored bytes before the 256
scored ones. The six rows received identical windows from random
weights, and every arm is one seed. The decoders run in such
windows only. A window generator that is new in v14 writes text
with them, at one window per byte. The v14 weights are release
assets, and the model that runs from a clone is v11.

![Validation loss of the decoders and the Transformers, and the decoders' distance to the Transformer](results/figures/event_lm_read_clocks_training.png)

The faster addresses show from 6,000 updates on. The decoders
lead the 256-wide Transformer up to 9,000 updates. Both
Transformers take their lead by 30,000 updates, and it changes
little after that.

![The decoder's distance to the Transformer by the kind of target](results/figures/event_lm_read_clocks_gap.png)

On bytes that an earlier match in the window offers for copying,
the decoders are ahead of the Transformer. They are behind on
the other bytes, and furthest where four or more words are in
view. A linear readout finds more of the earlier words in the
decoder's stream than in the Transformer's, though that stream
is five and a half times as wide. The decoder holds the words
and makes less of them.

![What the streams know of the earlier words](results/figures/event_lm_read_clocks_words.png)

Beside its three decoders the study trained 43 pilots of other
layouts, and it counts none of them as a gain. Cells that answer
a byte with many of their number, on relay neurons that hear the
whole stream, were closing in on the race cells when the pilots
ended. They are the next step.
[Next pilots](docs/V14_HANDOFF.md#next-pilots)

[Read-clock study](docs/READ_CLOCKS.md) ·
[Every training arm](docs/READ_CLOCKS_ARMS.md) ·
[Leaderboard](docs/LEADERBOARD.md) ·
[Model card](models/spiking-byte-lm-v14/README.md) ·
[Research handoff](docs/V14_HANDOFF.md) ·
[Checkpoint release](https://github.com/DanielTea/transformer-snn/releases/tag/spiking-byte-lm-v14)

## v13: state reads and race cells

The v13 study trains a 34.82M-parameter spike-only decoder from
random weights for 368.64M targets. Three parts are new. A
**state read** lets the present byte address the stream's own
earlier states by what they hold. The states of the last 512
bytes race for a budget of spikes, and the ones that fire pass
their payload on. **Race cells** are clusters of 350 cells that
compete for 64 spikes per byte. They listen on **delayed
fibres** that bring the stream as it was up to three bytes ago.
All three are built from spike counts and trained by gradient
descent with the decoder.

![The decoder: a stream of 1,408 channels through eight blocks, with race cells after four of them and state reads after three](results/figures/event_lm_state_reads_circuit.png)

![A cluster of race cells on delayed fibres, and a state read](results/figures/event_lm_state_reads_mechanisms.png)

| Trained from scratch, 368.64M targets | Optimizer | enwik8 | PG-19 |
|---|---|---:|---:|
| 34.82M SNN, spike reads and race cells | Rotated, averaged | 1.26226 | 1.55162 |
| 34.65M SNN, spike reads | Rotated, averaged | 1.29908 | 1.57978 |
| 34.65M hybrid, soft reads | AdamW | 1.31975 | — |
| 34.85M SNN, surprise clock and read heads | AdamW | 1.46686 | — |
| 34.85M SNN, surprise clock | AdamW | 1.59726 | — |
| 34.85M full attention | AdamW | 1.20684 | 1.48212 |
| 34.85M full attention | Rotated, averaged | **1.20407** | **1.48204** |

The Transformer is still ahead, by 0.055 bpb on enwik8 and by
0.070 on 16 PG-19 books. Every score in this table is in windows
of 512 bytes that start from an empty state, with 256 unscored
bytes before the 256 scored ones. A dash marks a score that was
not measured. The v13 decoder runs in such windows only. It has
no stream form yet, so it has no streaming score, no native
integer run and no pulse audit. The seven rows
received identical windows from random weights. The rows marked
rotated were trained with a rotated optimizer, which runs Adam
in the eigenbasis of each weight matrix's gradient moments, and
are scored at the running average of their weights. Every arm
is one seed. The v13 weights are release assets, and the model
that runs from a clone is v11.

![Averaged validation loss over training: the decoders lead early, then the Transformer pulls away](results/figures/event_lm_state_reads_training.png)

For the first 6,000 updates the decoder learns faster than the
Transformer. At 9,000 the two are level, and from 12,000 on the
Transformer is ahead. At 84,000 updates the decoder is ahead on
bytes that an earlier match in the window offers for copying
and behind on the others. The first letter of a word shows
where the rest comes from. With one earlier word in view the
decoder predicts it as well as the Transformer, and with three
it is 0.070 bpb behind. The Transformer's lead is the use it
makes of the second to fifth word back.

![The gap by bytes in view, and for the first letter of a word by the words in view](results/figures/event_lm_state_reads_context.png)

The study ran 89 training arms in one night. Its page gives the
circuit, the controls, the probes behind each statement and the
changes that did not help.

![Validation loss of every arm by family](results/figures/event_lm_state_reads_iterations.png)

The next step that v13 named was fibres whose delay counts
words where a delayed fibre counts bytes. v14 ran them at full
size in pilots of 3,000 to 6,000 updates, and they did not help.
[v14](#v14-read-clocks-and-the-address-rate)

[State-read study](docs/STATE_READS.md) ·
[Every training arm](docs/STATE_READS_ARMS.md) ·
[Leaderboard](docs/LEADERBOARD.md) ·
[Model card](models/spiking-byte-lm-v13/README.md) ·
[Research handoff](docs/V13_HANDOFF.md) ·
[Checkpoint release](https://github.com/DanielTea/transformer-snn/releases/tag/spiking-byte-lm-v13)

## v12: coasting read heads

The v12 study trains the 34.85M decoder from random weights
beside **read heads on a tape of the stream**. A head follows a
passage one byte at a time, gains charge with every confirmed
byte and is removed by the first wrong one. With its heads the
decoder scores **1.414 bits per byte** on the full enwik8 test
split and 1.646 on 16 PG-19 books. The bundled v11 model scores
1.645 and 1.940.

![Held-out enwik8 and PG-19 loss of the four decoders trained from scratch](results/figures/event_lm_coasting_heads_leaderboard.png)

| Trained from scratch, 368.64M targets | enwik8 | PG-19 | State | Pulses per byte |
|---|---:|---:|---:|---:|
| 34.85M SNN, surprise clock | 1.61100 | 1.72052 | 43,796 B | 311,540 |
| 34.85M SNN, surprise clock and fast synapses | 1.48587 | 1.68629 | 68,388 B | 318,451 |
| 34.85M SNN, surprise clock and read heads | 1.41375 | 1.64562 | 179,028 B | 309,469 |
| 34.85M full attention | **1.20684** | **1.48212** | 6 MiB | — |

Full attention is far ahead at this training budget. The four
rows received identical windows from random weights. Every
score comes from the streaming evaluator, which carries the
tape, the key synapses and the heads from chunk to chunk. They
add 135,232 bytes of state per stream, and the head decoder
holds a 35th of the state full attention keeps for its 512-byte
context. Pulses are counted in the native integer decoder on
the first 4,096 bytes of the enwik8 test split, and 241 of the
head decoder's 309,469 are in the head circuit. Every arm is
one seed. The v12 weights are release assets, and the model
that runs from a clone is v11.

![Held-out loss against the state of one stream](results/figures/event_lm_coasting_heads_state.png)

The study first scored the heads in a replay over each
decoder's recorded outputs, with strengths refitted on enwik8
validation. The stream confirms it. With refitted strengths the
head decoder scores 1.41385 bpb both ways. The figure below is
that replay for nine decoders, each alone, with the v11 fast
synapses and with read heads.

![enwik8 test loss of each decoder alone, with fast synapses and with read heads](results/figures/event_lm_coasting_heads.png)

[Coasting-head study](docs/COASTING_HEADS.md) ·
[Leaderboard](docs/LEADERBOARD.md) ·
[Model card](models/spiking-byte-lm-v12/README.md) ·
[Research handoff](docs/V12_HANDOFF.md) ·
[Checkpoint release](https://github.com/DanielTea/transformer-snn/releases/tag/spiking-byte-lm-v12)

## v11: surprise clock and fast synapses

The model that runs from a clone is v11. Its checkpoint has
**25.79M parameters**, 11 spiking blocks and **53,432 bytes of
persistent tensor state per stream**. A population of forecast
neurons predicts every byte, and the spikes it gets wrong write
**fast synapses** that recall the byte which last followed the
same context. They lower held-out enwik8 loss from 1.790 to
**1.645 bits per byte** without a gradient step. The same error
spikes also clocked four added blocks. That clock did not lower
held-out loss and is published as a negative result with its
controls.

![Held-out enwik8 and PG-19 loss with the surprise clock and fast synapses](results/figures/event_lm_surprise_clock.png)

[Model card and circuit diagram](models/spiking-byte-lm-v11/README.md) ·
[Research handoff](docs/V11_HANDOFF.md) ·
[Checkpoint release](https://github.com/DanielTea/transformer-snn/releases/tag/spiking-byte-lm-v11)

## Run the bundled model

From the repository root, set up Python 3.12 and run greedy
generation with the native CPU integer decoder.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python experiments/generate_event_lm.py models/spiking-byte-lm-v11/weights.pt \
  --prompt 'The first chapter' --bytes 128
```

## Matched benchmark

Every row saw **140,902,400 cumulative training targets**.
The six trained models received identical 512-byte windows for
their final 9,830,400 targets. The two rows with fast synapses
add 24 fitted strengths to a trained decoder and change no
weight. The table shows bits per byte on the full enwik8 test
split and 16 pinned PG-19 test books. PG-19 temperatures are
fitted on eight separate calibration books.

<!-- event-lm-leaderboard:start -->

| Model | enwik8 | PG-19 calibrated | PG-19 raw | State |
|---|---:|---:|---:|---:|
| 25.79M SNN | 1.78973 | 1.97852 | 1.97852 | 28,584 B |
| 25.79M SNN with fast synapses | 1.64541 | 1.93961 | 1.93961 | 53,432 B |
| 34.85M SNN, every byte | 1.78633 | 1.96913 | 1.96913 | 40,714 B |
| 34.85M SNN, fixed stride | 1.78761 | 1.96842 | 1.96842 | 43,796 B |
| 34.85M SNN, surprise clock | 1.78973 | 1.97528 | 1.97528 | 43,796 B |
| 34.85M SNN, surprise clock and fast synapses | **1.64278** | **1.93610** | **1.93610** | 68,388 B |
| 25.79M full attention | 1.70433 | 1.93711 | 1.93711 | 6 MiB |
| 34.85M full attention | 1.70426 | 1.93678 | 1.93678 | 6 MiB |

<!-- event-lm-leaderboard:end -->

With fast synapses the 25.79M spiking decoder scores **1.64541
bpb** on enwik8, 0.0589 below the size-matched Transformer, and
1.93961 on PG-19, 0.0025 above it. Full attention reads 512
bytes and the synapses carry the whole stream. With the same
synapses on its output the Transformer scores 1.57753 on enwik8,
so the like-for-like gap is 0.0679 bpb. Without synapses it is
0.0854. The surprise clock scores 1.78973 on enwik8. The same
blocks score 1.78761 on a fixed stride and 1.78633 on every
byte. With fast synapses the three score 1.64278, 1.64281 and
1.64164, so the clock does not earn the lowest row of the table.
The study uses one continuation seed.

State counts persistent tensors per stream. The synapse strengths
are fitted on enwik8 validation and used unchanged on PG-19.
[Surprise study, controls and limits](docs/SURPRISE_CLOCK.md) ·
[Full leaderboard](docs/LEADERBOARD.md) ·
[Browser report](docs/report.html#surprise-clock)

## Byte transfer benchmarks

The fixed suite reports raw bits per byte on WikiText-103,
C4 validation, enwik9 beyond byte 100 million, and text8.
LAMBADA scores answer-byte loss and greedy word accuracy for
5,153 full passages and last-sentence prompts. Evaluations use
the same scoring windows. The preparation checks training text
for overlap, and the report also gives scores for Wikipedia
windows with zero hits. C4 validation and PG-19 test are
held-out in-domain evaluations for the mixed runs.

Every window starts from an empty stream, so the fast synapses
hold only what the window contains. With them the 25.79M SNN is
below its Transformer by 0.0743 bpb on WikiText-103, 0.0601 on
C4 and 0.0902 on text8, and 0.0221 above it on enwik9. From the
full LAMBADA passage it completes 174 of 5,153 final words.
Each Transformer completes one.

![Corpus loss and LAMBADA accuracy for the surprise study](results/figures/event_lm_byte_transfer_surprise_clock_140902k.png)

[Transfer scores](docs/LEADERBOARD.md#byte-transfer-at-the-same-training-exposure) ·
[Data, overlap audit and replay](docs/BYTE_TRANSFER.md)

## Resume the research

The v11 archive holds the eight study endpoints and both v10
parents. The manifest links weights, checksums, training
records, synapse fits and the exact training windows. The
handoff includes commands to replay each step of the study.
[Restore and replay](docs/V11_HANDOFF.md#restore-the-checkpoints) ·
[Exact manifest](models/spiking-byte-lm-v11/manifest.json) ·
[Result file guide](results/README.md)

The v12 archives hold the 21 checkpoints of the coasting-head
study. Its handoff has the restore commands, the launcher, the
streaming evaluation of a head decoder and the replay that fits
its strengths.
[Restore and replay](docs/V12_HANDOFF.md#restore-the-checkpoints) ·
[Exact manifest](models/spiking-byte-lm-v12/manifest.json)

The v13 archives hold the seven checkpoints of the state-read
study. Its handoff has the restore commands, the scoring in
warmed windows, the probes that run on the released files and
the next pilots.
[Restore and rescore](docs/V13_HANDOFF.md#restore-the-checkpoints) ·
[Exact manifest](models/spiking-byte-lm-v13/manifest.json)

The v14 archives hold the six checkpoints of the read-clock
study. Its handoff has the restore commands, the scoring in
warmed windows, the window generator, the diagnostics that run
on the released files and the next pilots.
[Restore and rescore](docs/V14_HANDOFF.md#restore-the-checkpoints) ·
[Exact manifest](models/spiking-byte-lm-v14/manifest.json)

The v15 archives hold the six checkpoints of the stream-form
study, and a fourth asset holds the bits of every scored piece.
Its handoff has the restore commands, the stream scorer, the
Transformer along the same streams, the paired intervals, the
stream generator and the next pilots.
[Restore and rescore](docs/V15_HANDOFF.md#restore-the-checkpoints) ·
[Exact manifest](models/spiking-byte-lm-v15/manifest.json)

The v16 archives hold the four checkpoints of the long-streams
study: the stack trained on windows of 2,048 bytes with
forgetting slopes that learn, the same stack with its slopes
held at zero, and the Transformers trained on windows of 2,048
and 1,024 bytes. A fifth asset holds the bits of every scored
piece. Its handoff has the restore commands, the scoring with a
longer reach, fixed slopes and fast synapses, and the forks that
trained the long windows.
[Restore and rescore](docs/V16_HANDOFF.md#restore-the-checkpoints) ·
[Exact manifest](models/spiking-byte-lm-v16/manifest.json)

The v17 archives hold the best spike-only stack so far, v16's with
128 knowledge cells a head and trained on streams, and the depth-22
pair. A fourth asset holds the bits of every scored piece. Its
handoff has the restore commands, the split of a gap by what each
byte's context offers, the test bed's recipe with every new flag,
and what not to try again.
[Restore and rescore](docs/V17_HANDOFF.md#restore-the-checkpoints) ·
[Exact manifest](models/spiking-byte-lm-v17/manifest.json)

The v18 archives hold v17's stack with a word head in every read,
and the full-size stack of the learning-rate pilots with the pool
that lifts the ceiling. A third asset holds the bits of every
scored piece, the full-size pilots' included. Its handoff has the
restore commands, the probe of membrane saturation, and the flags
of the lifted ceiling, noisy thresholds and gated cells.
[Restore and rescore](docs/V18_HANDOFF.md#restore-the-checkpoints) ·
[Exact manifest](models/spiking-byte-lm-v18/manifest.json)

The v19 archives hold the best spike-only stack so far, v17's
stack trained at its best rate, its Transformer trained at its own,
so that the pair can be rescored and split without training, and
the full-size looped stack of the loop pilot. A fourth asset holds
the bits of every full-size scoring in the release. Its handoff has
the restore commands, the scoring of both sides of the pair, the
paired split by context, and the flags of the loop, skip fibres,
the slope far below the threshold and the float reference's
control gates.
[Restore and rescore](docs/V19_HANDOFF.md#restore-the-checkpoints) ·
[Exact manifest](models/spiking-byte-lm-v19/manifest.json)

The v20 archive holds the best spike-only stack so far, v19's
stack with its levels 2 to 8 run twice through the same synapses
for 90,000 updates. A second asset holds the bits of every scoring
recorded since v19. Its handoff has the restore command, the
scoring of the looped stack and its pairing with v19's pair, the
saturation probe, and the flags of the homeostatic gate, the
signed gate and sleep.
[Restore and rescore](docs/V20_HANDOFF.md#restore-the-checkpoint) ·
[Exact manifest](models/spiking-byte-lm-v20/manifest.json)

The v21 archive holds the best spike-only stack by the registered
reading, v20's looped stack trained under the pool that lifts the
ceiling for 90,000 updates. A second asset holds the bits of every scoring
recorded since v20. Its handoff has the restore command, the
scoring of the lifted loop and its pairing with v20's stack and
v19's pair, the saturation probe, the flags of power-law cells,
energy pools and the racing loop, and the flags that train faster
with the same spikes.
[Restore and rescore](docs/V21_HANDOFF.md#restore-the-checkpoint) ·
[Exact manifest](models/spiking-byte-lm-v21/manifest.json)

The v22 archive holds v21's lifted loop with power-law cells after
90,000 updates: behind v21's stack read with fast output synapses,
the best spike-only stack read without them. A second asset holds
the bits of every scoring recorded since v21. Its handoff has the
restore command, the scoring of the stack with and without fast
output synapses and its pairing with v21's stack and v19's
Transformer, the runner of the second seed on the fast path, and
the flags of line heads and of the practice loop.
[Restore and rescore](docs/V22_HANDOFF.md#restore-the-checkpoint) ·
[Exact manifest](models/spiking-byte-lm-v22/manifest.json)

Regenerate the leaderboards, scientific figures and browser report
from the checked-in records.

```bash
python experiments/plot_event_lm_study.py
python experiments/plot_depth_inhibition.py
python experiments/plot_coasting_heads.py
python experiments/plot_state_reads.py
python experiments/plot_read_clocks.py
python experiments/plot_stream_form.py
python experiments/plot_saturation.py
python experiments/plot_full_size_rate.py
python experiments/plot_best_rate_pair.py
python experiments/plot_gap_by_context.py
python experiments/plot_distance_split.py
python experiments/plot_full_size_parts.py
python experiments/plot_plain_units.py
python experiments/plot_pair_curves.py
python experiments/plot_loop_full_length.py
python experiments/plot_full_size_line.py
python experiments/make_report.py docs/report.html
```

[v10 research handoff](docs/V10_HANDOFF.md) ·
[Historical ShuntFormer and Dendrite experiments](docs/SHUNTFORMER_HISTORY.md)

## License

Copyright 2026 control-f and Daniel Tremer. The code, documentation, figures,
results and released model weights are licensed under the
[GNU Affero General Public License, version 3 only](LICENSE) (AGPL-3.0-only).
You may use, study, change and share them, also commercially. Anyone who
distributes a changed version, or offers it to others over a network, must make
the full source of that version available under the same license and keep the
[NOTICE](NOTICE).

**Commercial licenses** that allow use without the AGPL's conditions, for
example inside closed-source products or services, are available from control-f.

Outside contributions are welcome under a contributor license agreement with
control-f, so that the work can still be licensed both ways. The datasets used to
train and evaluate the models (enwik8, PG-19 and C4) are not part of this work and
keep their own terms.
