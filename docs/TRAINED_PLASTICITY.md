# Trained plasticity

Fast output synapses that run inside every training window, so that the slow weights can learn to make the learning
that happens while reading useful. Four rules were trained on a small stack and scored as unbroken streams under a
go/no-go that was recorded before the deciding scorings.

In short: no rule trained in the loop beats v15's rule given after training when each is scored with its own rule.
But a stack trained with a decaying rule in the loop has cells that v15's rule uses better: with that rule while
reading, it beats the stack that never saw fast synapses by 0.0130 bits per byte [0.0122, 0.0138], along the whole
stream and on a long one. That passes the go/no-go by its second route, so the setting goes to a full-size run. One
seed per arm, and the best of seven settings: the effect of the whole family is nearer 0.009.

## Limits

- Every arm is one seed (20001), trained from scratch on the same data order. A difference between two arms is a
  difference between two runs, and the paired intervals below cover only the choice of the scored text.
- The test bed is small: a stack of 6 levels, 256 channels and 4.86 million parameters, trained on windows of 256
  bytes for 6,000 updates on a CPU (`results/event_lm/stream_form/small_test_bed.json`). Its numbers decide whether
  the idea earns a full-size run. They do not say what it would gain there.
- The stack's weights, the gains of the fast synapses included, are trained by gradient descent. "Learns without
  gradients" applies only to the rule that runs while a stream is read: in whole numbers, from every arriving byte,
  after that byte was predicted. Nothing takes a gradient on validation or test text.
- Everything here is scored on the enwik8 validation split, and the arms were compared there too. With a dozen arms
  scored on the same streams, the best of them is favoured by its selection. No test-split number is reported for
  the test bed; the full-size run is scored on both splits.
- The A-carry arm trains the whole stack on streams, with the state of its reads and cells handed on from window to
  window, so its difference from a is not the fast synapses alone.

## The question

v15 gave the released stack fast output synapses after training (`EventLM.teach`): every arriving byte teaches them
by a local rule in whole numbers, after it has been predicted, and they start silent in every stream. They gained
0.0067 bits per byte on the enwik8 test split and 0.020 on PG-19. The slow weights never saw them. Does training with
a rule in the loop make reading-time learning clearly stronger than v15's rule given after training, and does its
gain hold along a long stream?

## The rules

All rules share v15's parts. A fibre's activity is its cell's spikes less the cell's running mean, and the activities
of a level share 65,536 units at each byte. When the next byte arrives, the output neurons race for 256 spikes, the
arrived byte's neuron is taught all 256, and every fast synapse changes by its fibre's activity times what its neuron
was taught beyond what it fired. Each trained level has a gain g that starts at zero and sets its fast neurons'
threshold. In a stream every rule runs in whole numbers (`EventLM._teach` with `plastic_after`). In training the same
rule runs in every window (`EventLM._plastic_window`). The errors carry no gradient; the activities and the gains do,
through rounding that passes the gradient straight through, and the fast spikes count among the pulses that the
budget prices.

- **Rule A (base-error).** The race runs on the stack's own counts, without the fast spikes, so what a byte teaches
  does not depend on the fast synapses. The fast charge at byte t is then a sum over the earlier bytes,

      fast[t] = g · Σ_{s<t} (1 - 2^-k)^(t-1-s) · (a_s · a_t) · e_{s+1},

  with a_s the activity at byte s, e_{s+1} what the byte after s taught, and k an optional decay shift. A window
  finds it as one masked [T, T] product: a causal attention whose values are the errors on the bytes that came.
- **Rule B (chunked).** The race runs on the counts plus the fast spikes that the synapses held at the start of the
  byte's chunk of C bytes, and the synapses take a chunk's teaching at its end. A window goes chunk by chunk, with the
  gradient flowing through the synapses from chunk to chunk. With C = 1 it races, as v15's rule does, on the counts
  plus everything that the synapses fire. Its errors shrink as the synapses learn, so it stops growing on a passage
  that repeats.
- **N (normalized).** Rule A's charge divided by one plus the summed coincidence of the fibres' sizes of activity,
  which bounds the fast spikes by the gain times the largest error however long a stream runs.
- **A-carry.** Rule A trained on streams: stretches of 15 consecutive windows of 256 bytes, one window an update,
  in 4 groups that take turns. Every window starts from the synapses, means and last activity that the stream's
  earlier windows left, without their gradient.
- **Shuffled control.** Rule A or Rule B storing the error of a pseudo-random earlier byte (within the last 1,024)
  in place of the error of the byte that came. It keeps the compute and the gain's training and loses the memory.

`tests/test_trained_plasticity.py` checks that the window form and the stream rule fire identical spikes in 64-bit
floats (Rule A without a decay, Rule B with C = 1, 16 and 64, N, the shuffled controls, A-carry against the stream
rule), that no spike depends on a later byte, that a stream cut into pieces of 1, 7, 100 or 512 bytes (inside chunks)
fires and learns the same, that zero gains give the stack without fast synapses, that the gains are not weight
decayed, that the chunked rule stops growing on a 64-byte passage read 1,000 times while Rule A keeps growing, that
N stays within its bound, and that the stream rules refuse MPS, which has no 64-bit whole numbers.

## The test bed

Every arm adds flags to the test bed's stack command (`--lr 3e-4`, 6,000 updates, averaged weights). Scoring:
`experiments/eval_stream.py` on the validation split as 16 unbroken streams of about 312 KB, a lead of 8,192 bytes
and pieces of 512, paired piece by piece with `experiments/compare_stream_scores.py` and by quarter of every stream
with `experiments/stream_quarters.py`.

| arm | what | flags |
|---|---|---|
| a | the stack | none |
| b | a, with v15's rule given after training at scoring | `eval_stream.py --teach-levels 5` (b-last) or `0 1 2 3 4 5` (b-all) |
| c | Rule A on the last level | `--plastic-levels 5` |
| d | Rule A on all levels | `--plastic-levels 0,1,2,3,4,5` |
| e8, e10 | c with a decay | `--plastic-decay-shift 8` or `10` |
| f | control: c with shuffled errors | `--plastic-shuffle` |
| g | c scored with its rule off | `eval_stream.py --plastic-off` |
| B16, B64 | Rule B on the last level, chunks of 16 or 64 bytes | `--plastic-levels 5 --plastic-rule chunked --plastic-chunk 16` (or 64) |
| N | N on the last level | `--plastic-levels 5 --plastic-rule normalized` |
| carry | A-carry on the last level | `--plastic-levels 5 --stream 15 --stream-groups 4` |
| B16s | control: B16 with shuffled errors | B16's flags and `--plastic-shuffle` |

## The go/no-go

The first go/no-go asked whether c or d beat b by more than b's own gain over a. That bar was b's gain on this small
stack (0.0945), about 14 times v15's full-size gain, so it measured how weak the small stack is rather than whether
training helps. It was replaced at 22:54:24 UTC on 2026-11-20, before the scorings that decide, and the replacement
was committed with the scorings seen until then (`results/event_lm/trained_plasticity/arms.json`, "criteria"). A
candidate is an arm trained with a rule in the loop and scored with its own rule. It is a go if a candidate:

1. beats b by more than the paired 95% interval and by at least 0.010 bits per byte;
2. is stable along the stream: no quarter scores worse than a in that quarter, the gain over a in the last quarter is
   at least 0.8 times the gain in the first, and it still beats b on a long stream (a lead of 262,144 bytes);
3. has a control that fails: the arm with its stored values shuffled gains less than a fifth of its gain over a;
4. passes the tests.

A second route: if no candidate passes but trained cells scored with v15's rule beat b as in 1 and are stable as in 2,
the go is for that setting (train with a rule in the loop, run v15's rule while reading). Otherwise it is a no-go, and
no arm is added after the results.

v15's threshold for its rule was tuned on the cells of the full-size stack. As the analysis rules given with the
criteria ask, it was re-picked for every arm on the first million validation bytes before teach scores were compared
(`arms.json`, "teach_thresholds", recorded at 00:24 UTC on 2026-11-21), and b is a with v15's rule at its own
re-picked setting. Where the scan's best lay at the edge of its range, the range was extended until the best had a
worse neighbour on both sides.

## Results

All scores are bits per byte on the enwik8 validation split as 16 unbroken streams with a lead of 8,192 bytes.
Differences are paired over the same 10,016 pieces of 512 bytes, with 95% intervals. b is a with v15's rule on all
six levels at its re-picked threshold of 2^22: 1.81412.

### Every arm with its own rule

| arm | in windows | stream | gain over a | against b | by quarter of the stream, gain over a | rule off |
|---|---|---|---|---|---|---|
| a | 1.9835 | 1.95134 | | -0.1372 | | |
| c | 1.9337 | 22.223 | -20.27 | -20.41 | -13.83, -21.31, -23.11, -23.00 | 1.98415 |
| d | 1.9648 | 9.555 | -7.60 | -7.74 | -2.55, -6.58, -9.51, -11.87 | 1.93872 |
| e8 | 1.9347 | 1.85283 | +0.0985 [+0.0958, +0.1012] | -0.0387 [-0.0419, -0.0357] | +0.100, +0.104, +0.090, +0.101 | 1.98437 |
| e10 | 1.9307 | 1.92390 | +0.0274 [+0.0222, +0.0323] | -0.1098 | +0.025, +0.048, +0.047, -0.010 | 1.98130 |
| f (control) | 1.9793 | 1.94802 | +0.0033 [+0.0026, +0.0040] | -0.1339 | +0.005, +0.004, +0.005, -0.000 | 1.94702 |
| B16 | 1.9163 | 1.83466 | +0.1167 [+0.1119, +0.1217] | -0.0205 [-0.0221, -0.0189] | +0.131, +0.102, +0.095, +0.140 | 1.98762 |
| B64 | 1.9231 | 1.83833 | +0.1130 [+0.1083, +0.1180] | -0.0242 [-0.0258, -0.0225] | +0.128, +0.099, +0.090, +0.136 | 1.98639 |
| B16s (control) | 1.9744 | 1.93836 | +0.0130 [+0.0122, +0.0138] | -0.1242 | +0.014, +0.013, +0.012, +0.012 | 1.93752 |
| N | 2.1800 | 2.15236 | -0.2010 | -0.3382 | -0.206, -0.194, -0.193, -0.212 | 2.15242 |
| carry | 2.0338 | 8.818 | -6.87 | -7.00 | -2.33, -5.82, -8.51, -10.89 | 1.97496 |

"In windows" is the trainer's score of the averaged weights on 64 batches of 256-byte windows that start empty, with
the arm's rule running inside each window. No candidate beats b, so criterion 1 fails for every arm and the
candidate route is closed:

- Inside windows every trained rule except N gains over a: B16 most (0.067), then B64, e10, c and e8 (0.049 to
  0.060). The shuffled controls gain 0.004 (f) and 0.009 (B16s).
- On streams the gains, calibrated on windows of 256 bytes, carry over only where the rule limits itself. Without a
  decay Rule A fires its fast neurons ever harder: c reaches 22 bits per byte, d 9.6, and carry, trained on stretches
  of 15 windows, 8.8. A decay of about the window's length (e8) holds the gain level along the stream. The chunked
  rule holds it level without a decay, and B16 gains the most of all rules, 0.117 over a. It is still 0.021 behind
  v15's rule given after training.
- N's division leaves it no gain at all (its rule changes its score by less than 0.0001), and its stack trained
  worse than a.
- With its rule off, every stack trained with a rule on the last level is worse than a, by 0.024 to 0.036 (N by
  0.20), and the control f is 0.004 better. d, trained with Rule A on all six levels, is 0.0126 [0.0118, 0.0134]
  better than a with its rule off, in every quarter.
- A lead of 65,536 bytes changes none of the gains by more than 0.003 (v15's rule 0.0945 and 0.0927 on the last level,
  e8 0.0985 and 0.0987, B16 0.1167 and 0.1141, B64 0.1130 and 0.1106): on enwik8, whose articles are short, no rule
  accumulates or fades between 8 and 64 KB of reading.
- The trainer's own stream check runs streams of about 1.5 KB. There B64 scored 1.8062; over 312 KB it scores
  1.8383. A rule with a long memory has to be scored on long streams.

### Trained cells with v15's rule

Each stack with its own rule off and v15's rule on, at the setting re-picked for it on the first million bytes:

| cells | v15's rule | against b | after the first million, against b | gain from v15's rule | by quarter, gain over a |
|---|---|---|---|---|---|
| a (that is b) | 1.81412 | | | 0.1372 | +0.147, +0.122, +0.116, +0.165 |
| f (control) | 1.80997 | +0.0042 [+0.0035, +0.0049] | +0.0043 | 0.1371 | +0.151, +0.125, +0.121, +0.170 |
| c | 1.80585 | +0.0083 [+0.0075, +0.0091] | +0.0080 | 0.1783 | +0.157, +0.130, +0.124, +0.172 |
| e8 | 1.80532 | +0.0088 [+0.0080, +0.0096] | +0.0086 | 0.1791 | +0.157, +0.131, +0.124, +0.173 |
| e10 | 1.80110 | +0.0130 [+0.0122, +0.0138] | +0.0128 [+0.0119, +0.0138] | 0.1802 | +0.160, +0.134, +0.130, +0.178 |
| B16 | 1.80448 | +0.0096 [+0.0088, +0.0105] | +0.0092 | 0.1831 | +0.158, +0.131, +0.126, +0.173 |
| B64 | 1.80542 | +0.0087 [+0.0079, +0.0096] | +0.0083 | 0.1810 | +0.158, +0.130, +0.124, +0.173 |
| carry | 1.80816 | +0.0060 [+0.0050, +0.0069] | +0.0057 | 0.1668 | +0.156, +0.126, +0.122, +0.169 |
| d | 1.80876 | +0.0054 [+0.0047, +0.0061] | +0.0059 | 0.1300 | +0.154, +0.127, +0.121, +0.170 |

The gain from v15's rule is each stack's score with its rule off less its score with v15's rule. On cells trained
with a rule with real memory on the last level it is 0.178 to 0.183, against 0.137 on the cells of a, and the
control's cells take exactly what a's take (0.1371). Training with such a rule in the loop makes the cells better
keys for v15's rule, by about 0.04 bits per byte of its gain, and the memory is what does it. It also costs the stack
about 0.03 with no fast synapses, so the net over b is 0.008 to 0.013. Trained on streams (carry), the cells take
0.167. Trained with Rule A on all six levels (d), the stack is the best of all with no fast synapses, but its cells
take less from v15's rule than a's do (0.130), and the net is 0.005. At the threshold of 2^24 that v15 tuned on the full-size stack, the
trained cells looked like worse keys than a's (e8's cells with v15's rule on the last level scored 2.032 against
b-last's 1.857); with a threshold re-picked for each arm that reading reverses.

On a long stream, with a lead of 262,144 bytes before every scored part but the first, e10's cells with v15's
rule score 1.80429 against b's 1.81639: still ahead by 0.0121 [0.0112, 0.0129], and by 0.0129, 0.0126, 0.0118 and
0.0115 in the four quarters.

## Decision

**Go, by the second route**, at e10's setting: train with Rule A on the last level and a decay whose memory is four
times the training window, then read with v15's rule on all levels, at a threshold re-picked on the first million
validation bytes.

- Criterion 1: +0.0130 [+0.0122, +0.0138] over b, and +0.0128 [+0.0119, +0.0138] on the four million bytes after
  the first, where no threshold was picked.
- Criterion 2: no quarter worse than a (gains of 0.160, 0.134, 0.130 and 0.178), the last quarter's gain above 0.8
  times the first's, and ahead of b on the long stream.
- Criterion 4: the tests pass.

The candidate route is a no-go. No rule trained in the loop, scored with its own rule, beats v15's rule given after
training: the best, B16, is 0.0205 behind.

What a reader should weigh against the go:

- **Selection.** e10 is the best of seven trained-cell settings, and only it clears 0.010. The others land at
  0.0054 to 0.0096. With one seed per arm, part of e10's margin is its luck. The effect that training with a rule in
  the loop has on v15's rule is better estimated by all of them together, 0.0085 on average (0.0097 for the five
  trained with a rule on the last level), than by the best one.
- **The control's cells gain too.** f's cells beat b by 0.0042, through a base 0.0043 better than a's, not through a
  larger gain from v15's rule (0.1371 against 0.1372).
- **b moved.** When the criteria were recorded, b had been scored only at v15's threshold of 2^24 (1.84989). By that
  b, B16 (+0.0152) and B64 (+0.0116) with their own rules would have passed criterion 1. The analysis rules given
  with the criteria asked to re-pick the threshold for every arm, and the re-pick was recorded at 00:24 UTC, before
  B16 and B64 were scored at 03:50. It made b 0.036 stronger, and both fail against it. B16's shuffled control,
  trained before B16's score was known, fails as criterion 3 asks: it gains 0.0130 over a, a ninth of B16's 0.1167,
  and all of it from a better base (0.0138 with its rule off). B16's long stream was not scored.
- **The bases move from run to run.** With no fast synapses, the controls f and B16s are 0.004 and 0.014 better than
  a, d is 0.013 better and the other trained stacks 0.024 to 0.036 worse. Differences of 0.01 between the bases of
  single runs are within what the controls show.
- **The full-size setting is extrapolated.** The decay that keeps a memory of four windows is shift 13 for windows
  of 2,048 bytes; the test bed tested shift 10 for windows of 256.

## The full-size run

`l2048-plastic` forks the saved state from which `l2048-flat` started (`p-forget` at 24,000 updates) with
`l2048-flat`'s settings and `--plastic-levels 10 --plastic-decay-shift 13`, and trains to 90,000 updates on windows
of 2,048 bytes. It is scored with its own rule off and v15's rule on all 11 levels at a threshold re-picked on the
first million validation bytes, against `l2048-flat` with v15's rule at its own re-picked threshold, paired over the
same streams. The success mark was set with the study, before any arm trained: ahead by more than the paired
interval and by more than 0.005 on both enwik8 splits, a gain that does not shrink along the stream, and no loss on
PG-19.

<!-- FULL-SIZE RESULT -->

## Reproduce

Train an arm with the test bed's `stack` and `shared` commands from
`results/event_lm/stream_form/small_test_bed.json` (with `--threads 4`) and the arm's flags, for example e10:

```bash
python experiments/train_event_lm.py $STACK $SHARED --plastic-levels 5 --plastic-decay-shift 10 \
  --tag plastic_e10 --checkpoint checkpoints/plastic/e10/model.pt --results-dir checkpoints/plastic/e10/results
```

Score it as 16 unbroken streams of the validation split, with its own rule, with its rule off, and with v15's rule at
a threshold re-picked on the first million bytes (`--bytes 1000000`, thresholds 2^20 to 2^28):

```bash
python experiments/eval_stream.py --checkpoint checkpoints/plastic/e10/model.pt.averaged --enwik8 data/enwik8 \
  --split validation --streams 16 --lead 8192 --piece 512 --json e10.json --bits e10.npy
python experiments/eval_stream.py ... --plastic-off --json e10-off.json --bits e10-off.npy
python experiments/eval_stream.py ... --plastic-off --teach-levels 0 1 2 3 4 5 --teach-threshold 4194304 \
  --json e10-teach.json --bits e10-teach.npy
```

Pair two scorings of the same streams, overall and by quarter of the stream:

```bash
python experiments/compare_stream_scores.py --first e10-teach.json e10-teach.npy --second b.json b.npy --json pair.json
python experiments/stream_quarters.py --enwik8 data/enwik8 --split validation --lead 8192 \
  --reference b.json b.npy --arm e10-teach e10-teach.json e10-teach.npy --json quarters.json
```

The full-size arm forks the saved state from which `l2048-flat` started, then trains with `l2048-flat`'s flags and
the plastic ones:

```bash
python experiments/stack/fork_state.py checkpoints/v16/p-forget/model.pt.resume \
  checkpoints/v16/l2048-plastic/model.pt.resume --tag v16_l2048_plastic \
  --set ctx=2048 bs=2 read_reach=2048 read_forget_rate=0 plastic_levels=10 plastic_decay_shift=13
```

Every score of the test bed, with its paired intervals, is in `results/event_lm/trained_plasticity/arms.json`.
