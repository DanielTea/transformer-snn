"""CIFAR-10 / CIFAR-100 image classification with the small spiking ViT.

Data is held in memory as uint8 tensors and augmented on the fly (random crop
with 4-px padding + horizontal flip), so no DataLoader workers compete with the
training threads.  ``--train-frac`` trains on a class-balanced subset, for
sample-efficiency curves.
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from common import (
    Timer,
    add_common_args,
    build_cfg,
    cosine_lr,
    make_optimizer,
    run_name,
    save_result,
    setup,
)
from snnformer import SpikingViT, count_ops, count_params

ROOT = Path("/home/user/data")
MEAN = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1)
STD = torch.tensor([0.2470, 0.2435, 0.2616]).view(1, 3, 1, 1)


def load_cifar10():
    def read(f):
        d = pickle.load(open(ROOT / "cifar-10-batches-py" / f, "rb"), encoding="bytes")
        return d[b"data"].reshape(-1, 3, 32, 32), np.array(d[b"labels"])

    xs, ys = zip(*[read(f"data_batch_{i}") for i in range(1, 6)])
    xt, yt = read("test_batch")
    return (torch.from_numpy(np.concatenate(xs)), torch.from_numpy(np.concatenate(ys)),
            torch.from_numpy(xt), torch.from_numpy(yt))


def subset(x, y, frac, seed):
    if frac >= 1:
        return x, y
    g = torch.Generator().manual_seed(seed)
    keep = []
    for c in y.unique():
        idx = (y == c).nonzero().squeeze(1)
        idx = idx[torch.randperm(len(idx), generator=g)[: int(len(idx) * frac)]]
        keep.append(idx)
    keep = torch.cat(keep)
    return x[keep], y[keep]


def augment(x: torch.Tensor, gen: torch.Generator) -> torch.Tensor:
    """Random 32x32 crop from a 4-px reflect-padded image + random horizontal flip (float input)."""
    B = x.shape[0]
    x = F.pad(x, (4, 4, 4, 4), mode="reflect")
    i = torch.randint(0, 9, (B,), generator=gen)
    j = torch.randint(0, 9, (B,), generator=gen)
    ar = torch.arange(32)
    rows = (i[:, None] + ar)[:, None, :, None]  # [B,1,32,1]
    cols = (j[:, None] + ar)[:, None, None, :]  # [B,1,1,32]
    x = x[torch.arange(B)[:, None, None, None], torch.arange(3)[None, :, None, None], rows, cols]
    flip = torch.rand(B, generator=gen) < 0.5
    x[flip] = x[flip].flip(-1)
    return x


def prep(x_uint8: torch.Tensor) -> torch.Tensor:
    return (x_uint8.float() / 255.0 - MEAN) / STD


@torch.no_grad()
def evaluate(model, x, y, bs=500):
    model.eval()
    correct = 0
    for i in range(0, len(x), bs):
        correct += (model(prep(x[i : i + bs])).argmax(-1) == y[i : i + bs]).sum().item()
    model.train()
    return correct / len(x)


def main():
    p = argparse.ArgumentParser()
    add_common_args(p, vision=True)
    p.add_argument("--dim", type=int, default=128)
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--bs", type=int, default=128)
    p.add_argument("--train-frac", type=float, default=1.0)
    p.add_argument("--smoothing", type=float, default=0.1)
    args = p.parse_args()
    setup(args)
    cfg = build_cfg(args)
    xtr, ytr, xte, yte = load_cifar10()
    xtr, ytr = subset(xtr, ytr, args.train_frac, args.seed)
    model = SpikingViT(cfg, num_classes=10, dim=args.dim, depth=args.depth, heads=args.heads)
    opt = make_optimizer(model, args.lr, args.wd)
    gen = torch.Generator().manual_seed(args.seed)
    steps_per_epoch = len(xtr) // args.bs
    total = steps_per_epoch * args.epochs
    name = run_name(args, cfg) + (f"_frac{args.train_frac}" if args.train_frac < 1 else "")
    print(f"{name}: params={count_params(model)} train={len(xtr)}", flush=True)
    timer, curve, step = Timer(), [], 0
    for ep in range(args.epochs):
        perm = torch.randperm(len(xtr), generator=gen)
        run_loss = run_acc = 0.0
        for b in range(steps_per_epoch):
            for g in opt.param_groups:
                g["lr"] = cosine_lr(step, total, steps_per_epoch // 2, args.lr)
            idx = perm[b * args.bs : (b + 1) * args.bs]
            x = augment(prep(xtr[idx]), gen)
            y = ytr[idx]
            out = model(x)
            loss = F.cross_entropy(out, y, label_smoothing=args.smoothing)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            run_loss += loss.item()
            run_acc += (out.argmax(-1) == y).float().mean().item()
            step += 1
        acc = evaluate(model, xte, yte)
        curve.append({"epoch": ep + 1, "train_loss": run_loss / steps_per_epoch,
                      "train_acc": run_acc / steps_per_epoch, "test_acc": acc})
        print(f"{name} ep {ep+1} loss {run_loss/steps_per_epoch:.3f} train {run_acc/steps_per_epoch:.3f} "
              f"test {acc:.4f} ({timer():.0f}s)", flush=True)
    model.eval()
    with torch.no_grad(), count_ops(model, cfg.spike_levels, cfg.spike_unit) as c:
        model(prep(xte[:200]))
        c.samples += 200
    energy = c.summary()
    save_result("vision", name, {"args": vars(args), "cfg": cfg.to_dict(), "params": count_params(model),
                                 "test_acc": curve[-1]["test_acc"], "best_acc": max(e["test_acc"] for e in curve),
                                 "curve": curve, "energy_per_image": energy, "seconds": timer()})
    print(f"FINAL {name} test_acc={curve[-1]['test_acc']:.4f} energy={energy}")


if __name__ == "__main__":
    main()
