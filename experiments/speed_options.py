"""Ways for the trainer to run the same updates in less time, each behind a flag of train_event_lm.py.

  --fused-optimizer  the rotated optimizer's element-wise steps over all tensors at once (rotated_adam.RotatedAdam,
                     foreach): the same operations, a few dozen GPU kernels for thousands.
  --compile-cells    the cell layers' and the state reads' chains of element-wise operations compiled into fused
                     kernels (torch.compile). Their spikes are whole numbers computed in whole numbers, and came out
                     the same on the server's GPU; the gradient's rounding differs.
  --cuda-graph       an update on windows (the first stage) captured in one CUDA graph and replayed (GraphedWindows):
                     the same kernels, none launched one by one.
  --fused-reads      a stream's reads in the fused race kernels (snnformer/race_kernel.py): the same spikes, the
                     gradient to rounding.
  --exact-products   the products of whole spike counts and whole synapse ticks on the tensor cores, to the bit
                     (snnformer/whole_products.py).

Measured on an RTX PRO 6000 (2026-11-26, the full-size stack alone on the GPU, seconds an update, the fused optimizer
in every row):
  on windows (the first stage)   0.176 as it was; 0.167 with exact products; 0.157 from a CUDA graph (0.149 with
                                 exact products); 0.129 with compiled cells
  as a stream (the second)       0.416 with reads by age (0.405 with exact products); 0.179 with fused reads (0.173
                                 with exact products, 0.140 with compiled cells)
Exact products, the fused optimizer and the CUDA graph leave a run's weights as they were to the bit (the graph
against the same optimizer with its steps on the device). Fused reads and compiled cells fire the same spikes at the
same weights and round the gradient differently: a run with them repeats to the bit, and is another run than the
one without.
"""
from __future__ import annotations

import contextlib
import os

import torch
from torch.nn import functional as F


def compile_cells(model):
    """Compile every cell layer's cells and every state read's packets (probes, tags and payloads with their pools)
    for the model's device. Fused multiply-adds stay off, so that each operation rounds as it does uncompiled."""
    os.environ.setdefault("TRITON_DEFAULT_FP_FUSION", "0")
    import torch._dynamo

    config = torch._dynamo.config
    config.cache_size_limit = max(config.cache_size_limit, 256)
    config.accumulated_cache_size_limit = max(config.accumulated_cache_size_limit, 8192)
    for layer in dict.fromkeys(() if model.granule_layers is None else model.granule_layers.values()):
        layer.cells = torch.compile(layer.cells, dynamic=False)
    for read in dict.fromkeys(() if model.state_reads is None else model.state_reads.values()):
        read.packets = torch.compile(read.packets, dynamic=False)


class GraphedWindows:
    """Updates on windows (the first stage) captured in one CUDA graph and replayed.

    The update is the trainer's: forward with the pulse budget's price, backward, the gradient's clipping, the
    optimizer's device part (RotatedAdam.apply, capturable), synaptic scaling and the weight average. What changes
    from update to update reaches the graph through tensors it reads: the bytes, the optimizer's step sizes
    (RotatedAdam.prepare), the weight average's share, and the spike price, which the graph updates in place. The
    optimizer's basis refreshes, which decompose matrices, run after a replay, outside the graph; the weights' check
    of the exact range runs at every evaluation (check_weights) in place of every forward pass. The first ``WARMUP``
    updates run as the trainer runs them, on a side stream, so that the libraries set up their workspaces before
    the capture."""

    WARMUP = 3

    def __init__(self, model, optimizer, args, device, average, pulse_counts, spike_price):
        if device.type != "cuda":
            raise ValueError("--cuda-graph needs a CUDA device")
        self.model, self.optimizer, self.args, self.device = model, optimizer, args, device
        self.average, self.pulse_counts, self.spike_price = average, pulse_counts, spike_price
        self.side = torch.cuda.Stream(device)
        self.warmed, self.graph, self.outputs = 0, None, None
        self.inputs = torch.zeros((args.bs, args.ctx), dtype=torch.long, device=device)
        self.labels = torch.zeros_like(self.inputs)
        self.share = torch.zeros((), device=device)
        self.scaled = torch.zeros((), dtype=torch.int64, device=device)

    def ready(self):
        """Whether the next update on windows replays the graph (or captures it)."""
        return self.warmed >= self.WARMUP

    @contextlib.contextmanager
    def warm(self):
        """An update as the trainer runs it, on the side stream."""
        current = torch.cuda.current_stream(self.device)
        self.side.wait_stream(current)
        with torch.cuda.stream(self.side):
            yield
        current.wait_stream(self.side)
        self.warmed += 1

    def _update(self):
        pulse_counts = self.pulse_counts
        options = {}
        if pulse_counts is not None:
            pulse_counts.clear()
            options["memory_pulses"] = pulse_counts
        from train_event_lm import scoring_logits

        counts = self.model(self.inputs, **options)
        logits = scoring_logits(counts, True, self.args.spike_resolution)
        language_loss = F.cross_entropy(logits.flatten(0, 1), self.labels.flatten())
        loss, pulses, price = language_loss, None, None
        if pulse_counts is not None:
            pulses = torch.stack(pulse_counts).sum() / self.inputs.numel()
            gap = pulses / self.args.pulse_budget - 1.0
            loss = loss + self.spike_price * gap + 0.5 * gap.relu().square()
            price = (self.spike_price + 0.04 * gap.detach()).clamp_min(0)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.apply()
        if price is not None:
            self.spike_price.copy_(price)
        scaled = self.model.synaptic_scaling()
        if torch.is_tensor(scaled):
            self.scaled.add_(scaled)
        if self.average is not None:
            for mean, parameter in zip(self.average, self.model.parameters(), strict=True):
                mean.lerp_(parameter, self.share)
        return language_loss.detach(), None if pulses is None else pulses.detach()

    def run(self, inputs, labels, average_share):
        """One update on windows: the bytes, the optimizer's step sizes and the average's share into the tensors
        that the graph reads, the graph (captured at the first call), then the refreshes of the optimizer's bases.
        Returns the update's language loss and pulses a byte, tensors that the graph overwrites at every replay."""
        self.inputs.copy_(inputs, non_blocking=True)
        self.labels.copy_(labels, non_blocking=True)
        self.share.fill_(average_share)
        self.optimizer.prepare()
        if self.graph is None:
            self.model._check_weights = False
            self.optimizer.zero_grad(set_to_none=True)
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):
                self.outputs = self._update()
        self.graph.replay()
        self.optimizer.refresh_bases()
        return self.outputs

    def take_scaled(self):
        """The neurons that synaptic scaling shrank in the replays since the last call."""
        taken = self.scaled.clone()
        self.scaled.zero_()
        return taken

    def check_weights(self):
        """The check of the exact range that the forward passes skip under the graph."""
        if self.graph is not None and self.model.input.weight.is_floating_point():
            self.model._check_accumulator()
