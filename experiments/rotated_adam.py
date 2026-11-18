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
"""

from __future__ import annotations

import math

import torch


class RotatedAdam(torch.optim.Optimizer):
    def __init__(self, params, lr, betas=(0.9, 0.99), eps=1e-8, weight_decay=0.0, refresh=50, largest=4096,
                 stacks=False):
        if lr <= 0 or not 0 <= betas[0] < 1 or not 0 <= betas[1] < 1 or refresh < 1:
            raise ValueError("the learning rate and refresh interval must be positive and the betas in [0, 1)")
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay))
        self.refresh, self.largest, self.stacks = refresh, largest, stacks

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

    @torch.no_grad()
    def step(self):
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
