# Signed residual membranes after spike routing

Each residual population takes the synaptic sum of the routed spikes
and the direct residual. The v3 circuit rounds this bounded charge to
signed pulse counts for each byte. Here the residue from that firing
decision stays in the same neuron and can affect the next byte's
spikes. Content, steering, and vocabulary populations keep their
existing firing rules. The held-charge trace and forecast-error
bank remain available in the same model.

For firing threshold $\theta$, bounded synaptic charge $u_t$, and
phase $\phi_t \in [0,\theta-1]$, the block emits

$$
p_t = \left\lfloor \frac{u_t+\phi_t}{\theta} \right\rfloor,
\qquad
\phi_{t+1}=u_t+\phi_t-\theta p_t .
$$

The initial phase is $\theta/2$. Across any prefix, the total
bounded charge equals $\theta$ times the signed pulse total plus
the change in phase. One phase survives per residual neuron. At
the 64-tick threshold this takes one byte per neuron. The CPU
decoder uses the same recurrence one byte at a time. Training
takes differences of a causal integer charge scan and uses the
usual local straight-through gradient for firing.
[Decoder](../snnformer/event_lm.py)

Retention of membrane residue after firing is an established
spiking mechanism. This experiment places one signed residue
at the point where the spike paths meet and tests it alongside
held charge and the verified error bank on real text.
[Residual membrane potential neurons](https://arxiv.org/abs/2003.01811) ·
[Sigma-delta integrate-and-fire quantization](https://proceedings.mlr.press/v89/o-connor19a.html)

## Keep the stream state small

The 16× encoder bounds each axon count to 512 pulses in either
direction. The axon rings can store these counts in signed 16-bit
tensors and widen them before synaptic arithmetic. The compact
representation preserves every count. It is independent of the
new firing behavior. A differentiable chunk retains floating-point
history for its gradient.

On 4,096 real enwik8 validation bytes, compact axons matched every
CPU output count of the seven-block parent. CPU and MPS chunked
decoding also matched. Compact axons brought state from 31,692 B
to 19,536 B. Adding signed residual membranes used 22,224 B with
the same 3,257,088 parameters. The signed carry changed individual
output counts at the parent's weights. Its loss on 65,536
validation targets was 1.92806, compared with 1.92803 for the parent.
[Warm probe](../results/event_lm/residual_membrane/warm_membrane_probe.json)

## Paired continuation

Both arms start from the released seven-block held-charge model.
They use chronological write credit, 512-byte windows, 1,600
steps, a fresh optimizer, seed 711, and 6,553,600 additional
enwik8 targets. The candidate enables `--residual-membrane`
and `--compact-axon-state`. Both train to the 310,000-pulse
target. A separate 2,048-byte continuation evaluates longer
training windows on the same SNN and Transformer parents.
[Scale setup](SCALE_3260K.md) ·
[Three-step Metal smoke](../results/event_lm/residual_membrane/event_correction_code_s711_residual_membrane_smoke.json)

The signed-carry model scored 1.96628 bits per byte on the complete
enwik8 test split. The matched held-charge continuation scored
1.96430. On PG-19, signed carry scored 2.89838 versus 2.88668
for the matched control. Their raw losses were 2.99574 and
2.98308 bits per byte across the same 16 books. Resetting its
residual membranes with the trained weights lowered the
validation probe loss from 1.898439 to 1.898362.

The integer audit matched MPS counts and native chunked output on
4,096 test bytes. It measured 309,629 pulses per byte and 22,224
bytes of stream state.
[Training, test results, and reset intervention](../results/event_lm/residual_membrane) ·
[Equal-exposure enwik8 control](../results/event_lm/scale_3260k_context/enwik8_trace_ctx512_s711.json)
