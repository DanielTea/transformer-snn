# Training held charge with credit for actual writes

Each held-charge cell applies a sequence of interval projections. An
interval can move the charge to its lower or upper bound, or leave the
previous charge intact. The training rule in `--trace-write-credit` sends
a future state's gradient to the most recent bound that moved it. A
touch at equality preserves the earlier source. A cell without a new
write sends credit to its initial charge.

The scan computes the charge after each interval pair, then reconstructs
the intermediate states. Composing two clips uses two projected bounds.
The paired evaluation has linear total work and logarithmic depth in
the number of timesteps. A prefix maximum of actual write indices
selects the source for each output. Backpropagation follows those
selected writes. Native decoding still applies the integer clip once
per byte.
[Implementation](../snnformer/correction_memory.py) ·
[Parallel prefix scans](https://www.cs.cmu.edu/afs/cs.cmu.edu/project/scandal/public/papers/CMU-CS-90-190.html)

The original and new training rules produced identical forward
vocabulary counts with the same weights. When bounds tie, the grouping
of the original parallel clips can affect which bound receives a
gradient. The new rule matched explicit
chronological writes and their gradients on an enwik8 training sample
of eight 53-byte windows. The architecture and parameter count are
shared by the two training rules.

## Apple GPU timing

Six interleaved timings used the same five-block checkpoint and eight
512-byte enwik8 training windows. Median language-loss forward and
backward time was 0.498 seconds with the original scan and 0.278
seconds with write credit, a 1.79× speedup in this timing. The
measurement covers the language objective and excludes the optimizer
step, auxiliary forecast loss, and pulse-budget loss.
[Timing record](../results/event_lm/trace_write_credit/trace_training_pair.json)

```bash
PYTORCH_ENABLE_MPS_FALLBACK=0 python experiments/benchmark_trace_training.py \
  --enwik8 data/enwik8 --checkpoint models/spiking-byte-lm-v2/weights.pt \
  --batch-size 8 --context 512 --trials 6 --json checkpoints/trace-training-pair.json
```

The next-byte quality comparison uses the same starting weights, seed
711, optimizer settings, and 6,553,600 additional enwik8 targets as
the control in the opponent-steering study. Each arm uses the same
pulse target. Training time and held-out loss will be evaluated
separately. [Matched control protocol](OPPONENT_STEERING.md#paired-continuation)
