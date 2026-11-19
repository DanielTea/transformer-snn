"""Turn a copy of the package into the diagnosis build of the stack study.

    git worktree add ../stack-diagnosis a9dc54e
    python experiments/stack/diagnosis_build.py ../stack-diagnosis

The script rewrites ``snnformer/event_lm.py`` of that copy, and only the text of commit ``a9dc54e``: every
replacement must match exactly once, or it stops. The copy then reads these switches from the environment when the
module is imported. With none set it computes what the commit computes.

  SNN_REAL=1    no rounding anywhere: weights off the grid, real-valued counts, softmax shares, exact slopes
  SNN_REALW=1   weights off the grid only: every population still fires whole spikes
  SNN_LN=1      every read and cell layer hears a normalised stream x resolution in place of the stream
  SNN_LN_OUT=1  the output synapses hear a normalised stream x resolution
  SNN_NORM=kind the normalisation of SNN_LN and SNN_LN_OUT: ln (LayerNorm, the default), or [c]rms[sg] / [c]l1[sg]:
                c centres, rms or l1 is the size divided by, sg stops the gradient through the size
  SNN_GAIN0=x   the reads' gains start at exp(x) (in nats; 0 is the model's own start)
  SNN_POOL=o,s  the threshold of a read's or cell layer's neurons is o quanta at rest (the model's own: 4) and the
                inhibitory pool fires s times as many spikes for the same activity (the model's own: 1)
  SNN_POOL_OUT=s the output neurons get a pool too: their threshold follows s * activity / (dim * resolution) in
                 steps of 1 / (8 * SNN_FINE) above a rest of one step
  SNN_FINE=F    the neurons of reads and cell layers count their threshold in quanta F times finer (4 * F to a
                threshold), and the pools may fire up to 4 * F times as many spikes; SNN_POOL then counts in fine
                quanta

The first six are float stand-ins and no spike mechanisms: a stack trained with one of them is not a spiking
network. They exist to place a gap. The last three are the fine pool as it was first tried. The package has it since
commit ``a73cb70`` as ``--pool-fine 16 --pool-rest 4 --pool-rate 32 --output-pool 128``, which is
``SNN_FINE=16 SNN_POOL=4,32 SNN_POOL_OUT=1`` with one rule changed: at an exact tie the output pool of the package
rounds its count up, where this build rounds it to the even number.
"""
import sys
from pathlib import Path

path = Path(sys.argv[1]) / "snnformer" / "event_lm.py"
text = path.read_text()


def swap(old, new, count=1):
    global text
    assert text.count(old) == count, (text.count(old), old[:80])
    text = text.replace(old, new)


swap('''def _weight_ticks(weight, quantized=True, *, charge_bits=_CHARGE_BITS):
    if not weight.is_floating_point():
        return weight
''', '''import os as _os

_REAL = _os.environ.get("SNN_REAL") == "1"
_REALW = _REAL or _os.environ.get("SNN_REALW") == "1"
_LN = _os.environ.get("SNN_LN") == "1"
_LN_OUT = _os.environ.get("SNN_LN_OUT") == "1"
_GAIN0 = float(_os.environ.get("SNN_GAIN0", "0"))
_NORM = _os.environ.get("SNN_NORM", "ln")
_POOL_REST, _POOL_GAIN = (float(part) for part in _os.environ.get("SNN_POOL", "4,1").split(","))
_POOL_OUT = float(_os.environ.get("SNN_POOL_OUT", "0"))
_FINE = int(_os.environ.get("SNN_FINE", "1"))


class _DiagNorm(nn.Module):
    """The normalisation that SNN_NORM names. A float stand-in."""

    def __init__(self, dim):
        super().__init__()
        self.ln = nn.LayerNorm(dim) if _NORM == "ln" else None

    def forward(self, x):
        if self.ln is not None:
            return self.ln(x)
        kind = _NORM
        if kind.startswith("c"):
            x, kind = x - x.mean(-1, keepdim=True), kind[1:]
        if kind.startswith("l1"):
            size = x.abs().mean(-1, keepdim=True) * 1.2533
        else:
            size = x.square().mean(-1, keepdim=True).sqrt()
        if kind.endswith("sg"):
            size = size.detach()
        return x / (size + 1e-5)


def _weight_ticks(weight, quantized=True, *, charge_bits=_CHARGE_BITS):
    if not weight.is_floating_point():
        return weight
    quantized = quantized and not _REALW
''')
swap('''    def forward(self, current, quanta=None, *, reference=None):
        bounded = current.clamp(-self.threshold * self.cap, self.threshold * self.cap)
        if reference is not None:
            charge = bounded.to(torch.int32)
''', '''    def forward(self, current, quanta=None, *, reference=None):
        bounded = current.clamp(-self.threshold * self.cap, self.threshold * self.cap)
        if _REAL and current.is_floating_point():
            if reference is not None:
                return bounded / self.threshold - reference
            if quanta is None:
                return bounded / self.threshold
            limit = self.cap * self.quanta_per_threshold / quanta
            value = bounded / (quanta * (self.threshold // self.quanta_per_threshold))
            return torch.maximum(torch.minimum(value, limit), -limit)
        if reference is not None:
            charge = bounded.to(torch.int32)
''')
swap('''        learned = smooth * F.gelu(current / (threshold * smooth))
        return counts.detach() + (learned - learned.detach())
''', '''        learned = smooth * F.gelu(current / (threshold * smooth))
        if _REAL:
            return learned
        return counts.detach() + (learned - learned.detach())
''')
swap('''            fired = rate + ((rate + phase).floor() - rate).detach()
        return fired / fired.sum(-1, keepdim=True).clamp_min(1), payload, fired.sum() + beats
''', '''            fired = rate if _REAL else rate + ((rate + phase).floor() - rate).detach()
        return fired / fired.sum(-1, keepdim=True).clamp_min(1), payload, fired.sum() + beats
''')
swap('''            packet = packet.detach().round() + (packet - packet.detach())
            if pulse_counts is not None:
''', '''            if not _REAL:
                packet = packet.detach().round() + (packet - packet.detach())
            if pulse_counts is not None:
''')
swap('''        quarters = self.rest.clamp(0, _GRANULE_REST_LIMIT) * _PACKET_UNITS
        return quarters.detach().round() + (quarters - quarters.detach())
''', '''        quarters = self.rest.clamp(0, _GRANULE_REST_LIMIT) * _PACKET_UNITS
        if _REAL:
            return quarters
        return quarters.detach().round() + (quarters - quarters.detach())
''')
# LayerNorm on what a read hears.
swap('''            self.gain = nn.Parameter(torch.zeros(self.heads, 1, 1))
        self.phases = cfg.read_phases
''', '''            self.gain = nn.Parameter(torch.full((self.heads, 1, 1), _GAIN0 / cfg.read_address_rate))
        self.norm = _DiagNorm(cfg.dim) if _LN else None
        self.phases = cfg.read_phases
''')
swap('''        if self.mode == "age":
            return None, None, self._unit(self.payload(spikes)) * self.read_resolution
        probe, tag = self.probe(spikes), self.tag(spikes)
''', '''        if self.norm is not None:
            spikes = self.norm(spikes) * self.resolution
        if self.mode == "age":
            return None, None, self._unit(self.payload(spikes)) * self.read_resolution
        probe, tag = self.probe(spikes), self.tag(spikes)
''')
# LayerNorm on what a cell layer hears.
swap('''        self.soft, self.slope = cfg.granule_soft, cfg.granule_slope
        self.pool = _IntegrateFire(_HIDDEN_CAP * _PACKET_UNITS, cfg.inhibition_bits, False)
        self.emit = _IntegrateFire(cap, self.bits, False)
''', '''        self.soft, self.slope = cfg.granule_soft, cfg.granule_slope
        self.pool = _IntegrateFire(_HIDDEN_CAP * _PACKET_UNITS, cfg.inhibition_bits, False)
        self.emit = _IntegrateFire(cap, self.bits, False)
        self.norm, self.norm_scale = (_DiagNorm(cfg.dim) if _LN else None), cfg.resolution
''')
swap('''        """The spike counts of all cells as [batch, bytes, clusters, cells]."""
        quanta = _PACKET_UNITS + self.pool(spikes.abs().sum(-1, keepdim=True, dtype=spikes.dtype))
''', '''        """The spike counts of all cells as [batch, bytes, clusters, cells]."""
        if self.norm is not None:
            spikes = self.norm(spikes) * self.norm_scale
        quanta = _PACKET_UNITS + self.pool(spikes.abs().sum(-1, keepdim=True, dtype=spikes.dtype))
''')
# LayerNorm before the output synapses.
swap('''        self.output = _Synapse(cfg.dim, _VOCAB, gain=0.8, continuous=cfg.continuous)
        self.output_emit = _IntegrateFire(output_cap, output_bits, cfg.carry, continuous=cfg.continuous)
''', '''        self.output = _Synapse(cfg.dim, _VOCAB, gain=0.8, continuous=cfg.continuous)
        self.output_emit = _IntegrateFire(output_cap, output_bits, cfg.carry, continuous=cfg.continuous)
        self.final_norm = _DiagNorm(cfg.dim) if _LN_OUT else None
''')
swap('''        current = self.output(spikes)
        counts = self.output_emit(current, reference=self._reference_counts(current))
        if self.fast_synapses is not None:
''', '''        if self.final_norm is not None:
            spikes = self.final_norm(spikes) * self.cfg.resolution
        current = self.output(spikes)
        counts = self.output_emit(current, reference=self._reference_counts(current))
        if self.fast_synapses is not None:
''')
# A stronger pool: a lower threshold at rest and more pool spikes for the same activity.
swap("quanta = _PACKET_UNITS + self.pool(spikes.abs().sum(-1, keepdim=True, dtype=spikes.dtype))",
     "quanta = _POOL_REST + self.pool(spikes.abs().sum(-1, keepdim=True, dtype=spikes.dtype) * _POOL_GAIN)",3)
# A pool at the output: the output neurons' threshold follows the stream's activity, in steps of an eighth.
swap('''        current = self.output(spikes)
        counts = self.output_emit(current, reference=self._reference_counts(current))
        if self.fast_synapses is not None:
''', '''        current = self.output(spikes)
        if _POOL_OUT:
            steps = 8 * _FINE
            activity = spikes.abs().sum(-1, keepdim=True, dtype=spikes.dtype) * (steps * _POOL_OUT / (self.cfg.dim * self.cfg.resolution))
            units = 1 + activity.detach().round() + (activity - activity.detach())
            current = current * (steps / units)
        counts = self.output_emit(current, reference=self._reference_counts(current))
        if self.fast_synapses is not None:
''')
# Finer quanta for the neurons that the pools inhibit, and room for the pools to fire that much more.
swap('''            self.fire = _IntegrateFire(cap if cfg.read_cap is None else cfg.read_cap, bits, False)
            self.pool = _IntegrateFire(_HIDDEN_CAP * _PACKET_UNITS, cfg.inhibition_bits, False)
''', '''            self.fire = _IntegrateFire(cap if cfg.read_cap is None else cfg.read_cap, bits, False,
                                       quanta_per_threshold=_PACKET_UNITS * _FINE)
            self.pool = _IntegrateFire(_HIDDEN_CAP * _PACKET_UNITS * _FINE, cfg.inhibition_bits, False)
''')
swap('''            self.fire = _ExcitatoryFire(cell_cap, self.bits, False)
''', '''            self.fire = _ExcitatoryFire(cell_cap, self.bits, False, quanta_per_threshold=_PACKET_UNITS * _FINE)
''')
swap('''        self.pool = _IntegrateFire(_HIDDEN_CAP * _PACKET_UNITS, cfg.inhibition_bits, False)
        self.emit = _IntegrateFire(cap, self.bits, False)
        self.norm, self.norm_scale''', '''        self.pool = _IntegrateFire(_HIDDEN_CAP * _PACKET_UNITS * _FINE, cfg.inhibition_bits, False)
        self.emit = _IntegrateFire(cap, self.bits, False)
        self.norm, self.norm_scale''')
# The turned probes and tags count in the same fine quanta.
swap('''            self.turned = _IntegrateFire(cap if cfg.read_cap is None else cfg.read_cap,
                                         bits + _PHASE_SPIKES.bit_length() - 1, False)
''', '''            self.turned = _IntegrateFire(cap if cfg.read_cap is None else cfg.read_cap,
                                         bits + _PHASE_SPIKES.bit_length() - 1, False,
                                         quanta_per_threshold=_PACKET_UNITS * _FINE)
''')
path.write_text(text)
print("patched", path)
