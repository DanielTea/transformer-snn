"""Adam in the eigenbasis of each weight matrix's gradient statistics.

Adam scales every weight on its own. A direction that many weights of a
matrix share then moves by the sum of their steps, which is far for a wide
matrix. This optimizer keeps running second moments of the gradient's rows
and columns, rotates the gradient into their eigenbases and runs Adam
there, so a shared direction is one coordinate with one step. It follows
SOAP (Vyas et al., 2024, https://arxiv.org/abs/2409.11321). A side of a
matrix above ``largest`` keeps its own coordinates, and tensors that are
not matrices get Adam. With ``stacks`` a tensor of three dimensions is a
stack of matrices, each with its own bases.

With ``foreach`` a step takes each of its element-wise operations over all
tensors at once (torch._foreach_*), which launches a few dozen GPU kernels
where the loop over tensors launches thousands; each tensor gets the same
operations as in the loop, so on the CPU the two give the same bits.
``capturable`` (implies ``foreach``) keeps the learning rate's part of a
step on the device, in tensors that ``prepare`` fills, so that ``apply`` can
be captured in a CUDA graph and replayed: ``step`` is ``prepare``, ``apply``
and ``refresh``. Its update is multiplied by the step size and then added,
where the loop adds it with the step size as a factor: the same to rounding.
"""

from __future__ import annotations

import math

import torch


class RotatedAdam(torch.optim.Optimizer):
    def __init__(self, params, lr, betas=(0.9, 0.99), eps=1e-8, weight_decay=0.0, refresh=50, largest=4096,
                 stacks=False, foreach=False, capturable=False):
        if lr <= 0 or not 0 <= betas[0] < 1 or not 0 <= betas[1] < 1 or refresh < 1:
            raise ValueError("the learning rate and refresh interval must be positive and the betas in [0, 1)")
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay))
        self.refresh, self.largest, self.stacks = refresh, largest, stacks
        self.foreach, self.capturable = foreach or capturable, capturable
        # For each group, the tensors that a step updates, by the number of steps they have taken.
        self._live = None
        # With ``capturable``, each group's device tensors of its step sizes and of its weight decay's factor, kept
        # here and not in the groups, so that a saved state stays as the loop saves it.
        self._held = {}

    @staticmethod
    def _basis(moment):
        """Eigenvectors of a second moment, the largest eigenvalue first."""
        try:
            vectors = torch.linalg.eigh(moment + 1e-30 * torch.eye(moment.shape[-1], device=moment.device)).eigenvectors
        except RuntimeError:
            vectors = torch.linalg.eigh(moment.double().cpu()).eigenvectors.to(moment)
        return vectors.flip(-1)

    @staticmethod
    def _follow(moment, basis, second, side):
        """One power iteration keeps the basis on the moving moment and the squared gradients on their vectors."""
        order = torch.argsort((basis.mT @ moment @ basis).diagonal(dim1=-2, dim2=-1), descending=True)
        along = order.unsqueeze(-1) if side == 0 else order.unsqueeze(-2)
        second.copy_(torch.take_along_dim(second, along.expand_as(second), dim=side - 2))
        ordered = torch.take_along_dim(basis, order.unsqueeze(-2).expand_as(basis), dim=-1)
        return torch.linalg.qr(moment @ ordered).Q

    @staticmethod
    def _turn(value, rows, columns):
        """The value in the bases of the rows and columns that have one."""
        if rows is not None:
            value = rows.mT @ value
        return value if columns is None else value @ columns

    @staticmethod
    def _turn_back(value, rows, columns):
        if rows is not None:
            value = rows @ value
        return value if columns is None else value @ columns.mT

    def _start(self, parameter, gradient):
        """A tensor's state at its first gradient. True if that gradient only sets its bases."""
        state = self.state[parameter]
        state["step"] = 0
        state["first"] = torch.zeros_like(gradient)
        state["second"] = torch.zeros_like(gradient)
        matrix = gradient.ndim == 2 or (self.stacks and gradient.ndim == 3)
        sides = [matrix and 1 < size <= self.largest for size in gradient.shape[-2:]] + [False]
        state["rows"] = gradient @ gradient.mT if sides[0] else None
        state["columns"] = gradient.mT @ gradient if matrix and sides[1] else None
        state["row_basis"] = None if state["rows"] is None else self._basis(state["rows"])
        state["column_basis"] = None if state["columns"] is None else self._basis(state["columns"])
        return state["rows"] is not None or state["columns"] is not None

    @torch.no_grad()
    def prepare(self):
        """The host's part of a step: new tensors' states, every updated tensor's step count, and the step sizes.
        With ``capturable`` the step sizes go into device tensors, which ``apply`` reads."""
        live = []
        for index, group in enumerate(self.param_groups):
            beta1, beta2 = group["betas"]
            classes = {}
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                if not self.state[parameter] and self._start(parameter, parameter.grad):
                    # The first gradient only sets the bases.
                    continue
                state = self.state[parameter]
                state["step"] += 1
                classes.setdefault(state["step"], []).append(parameter)
            sizes = {step: -group["lr"] * math.sqrt(1 - beta2 ** step) / (1 - beta1 ** step) for step in classes}
            shrink = 1 - group["lr"] * group["weight_decay"]
            if self.capturable:
                device = next(iter(group["params"])).device
                held = self._held.setdefault(index, {"shrink": torch.zeros((), device=device), "by_rank": []})
                held["shrink"].fill_(shrink)
                # A class keeps its device tensor from step to step: its tensors all advance together.
                order = sorted(classes)
                tensors = held["by_rank"]
                while len(tensors) < len(order):
                    tensors.append(torch.zeros((), device=device))
                for rank, step in enumerate(order):
                    tensors[rank].fill_(sizes[step])
                sizes = {step: tensors[rank] for rank, step in enumerate(order)}
                shrink = held["shrink"]
            live.append((group, [(sizes[step], classes[step]) for step in sorted(classes)], shrink))
        self._live = live

    @torch.no_grad()
    def apply(self):
        """The device's part of a step, over the tensors that ``prepare`` found: element-wise operations over all
        of them at once, the turns into and out of the bases one tensor at a time."""
        for group, classes, shrink in self._live:
            beta1, beta2 = group["betas"]
            params = [parameter for _, members in classes for parameter in members]
            if not params:
                continue
            states = [self.state[parameter] for parameter in params]
            grads = [parameter.grad for parameter in params]
            firsts, seconds = [state["first"] for state in states], [state["second"] for state in states]
            turned = [self._turn(gradient, state["row_basis"], state["column_basis"])
                      for gradient, state in zip(grads, states)]
            torch._foreach_lerp_(firsts, grads, 1 - beta1)
            torch._foreach_mul_(seconds, beta2)
            torch._foreach_addcmul_(seconds, turned, turned, value=1 - beta2)
            denominators = torch._foreach_sqrt(seconds)
            torch._foreach_add_(denominators, group["eps"])
            ratios = torch._foreach_div([self._turn(first, state["row_basis"], state["column_basis"])
                                         for first, state in zip(firsts, states)], denominators)
            updates = [self._turn_back(ratio, state["row_basis"], state["column_basis"])
                       for ratio, state in zip(ratios, states)]
            if group["weight_decay"]:
                torch._foreach_mul_(params, shrink)
            start = 0
            for size, members in classes:
                part = slice(start, start + len(members))
                start += len(members)
                if self.capturable:
                    torch._foreach_mul_(updates[part], size)
                    torch._foreach_add_(params[part], updates[part])
                else:
                    torch._foreach_add_(params[part], updates[part], alpha=size)
            for side, product in (("rows", lambda g: g @ g.mT), ("columns", lambda g: g.mT @ g)):
                pairs = [(state[side], product(gradient)) for gradient, state in zip(grads, states)
                         if state[side] is not None]
                if pairs:
                    torch._foreach_lerp_([moment for moment, _ in pairs], [new for _, new in pairs], 1 - beta2)

    @torch.no_grad()
    def refresh_bases(self):
        """The bases of the tensors whose step count has reached a refresh, after their update: one power
        iteration each, written into the tensors that hold them (a CUDA graph keeps reading those)."""
        for group, classes, _ in self._live:
            for _, members in classes:
                for parameter in members:
                    state = self.state[parameter]
                    if state["step"] % self.refresh:
                        continue
                    for side, name, kind in ((0, "rows", "row_basis"), (1, "columns", "column_basis")):
                        if state[kind] is not None:
                            state[kind].copy_(self._follow(state[name], state[kind], state["second"], side))

    @torch.no_grad()
    def step(self):
        if self.foreach:
            self.prepare()
            self.apply()
            self.refresh_bases()
            return
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            for parameter in group["params"]:
                gradient = parameter.grad
                if gradient is None:
                    continue
                state = self.state[parameter]
                if not state:
                    state["step"] = 0
                    state["first"] = torch.zeros_like(gradient)
                    state["second"] = torch.zeros_like(gradient)
                    # A side of the matrix gets a basis when it is neither a single line nor above the limit.
                    matrix = gradient.ndim == 2 or (self.stacks and gradient.ndim == 3)
                    sides = [matrix and 1 < size <= self.largest for size in gradient.shape[-2:]] + [False]
                    state["rows"] = gradient @ gradient.mT if sides[0] else None
                    state["columns"] = gradient.mT @ gradient if matrix and sides[1] else None
                    state["row_basis"] = None if state["rows"] is None else self._basis(state["rows"])
                    state["column_basis"] = None if state["columns"] is None else self._basis(state["columns"])
                    if state["rows"] is not None or state["columns"] is not None:
                        # The first gradient only sets the bases.
                        continue
                state["step"] += 1
                step = state["step"]
                first, second = state["first"], state["second"]
                rows, columns = state["row_basis"], state["column_basis"]
                turned = self._turn(gradient, rows, columns)
                first.lerp_(gradient, 1 - beta1)
                second.mul_(beta2).addcmul_(turned, turned, value=1 - beta2)
                update = self._turn_back(self._turn(first, rows, columns) / (second.sqrt() + group["eps"]), rows, columns)
                if group["weight_decay"]:
                    parameter.mul_(1 - group["lr"] * group["weight_decay"])
                parameter.add_(update, alpha=-group["lr"] * math.sqrt(1 - beta2 ** step) / (1 - beta1 ** step))
                if rows is not None:
                    state["rows"].lerp_(gradient @ gradient.mT, 1 - beta2)
                    if step % self.refresh == 0:
                        state["row_basis"] = self._follow(state["rows"], rows, second, 0)
                if columns is not None:
                    state["columns"].lerp_(gradient.mT @ gradient, 1 - beta2)
                    if step % self.refresh == 0:
                        state["column_basis"] = self._follow(state["columns"], columns, second, 1)
