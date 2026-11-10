"""
Train and test a small ResNet on the shards written by extract.py.

    python train.py --data data --epochs 5
    python train.py --data data --epochs 5 --features board+skill            # + player ratings
    python train.py --data data --epochs 5 --features board+skill+time+history

--features picks the inputs (see dataset.py): board (always on), skill (Elo), time (clocks and
time control), history (previous positions as extra planes). ablate.py runs the comparisons.

Prints train/val metrics each epoch, saves the best model (by val loss) to --save,
then reloads it and reports test accuracy next to two simple baselines.
Finally it plots the train/val loss curves, saves them to --plot, and shows the figure.
"""
import argparse
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset import LichessPositions, parse_features

CLASSES = ["white win", "draw", "black win"]


class ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.c1 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.b1 = nn.BatchNorm2d(ch)
        self.c2 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.b2 = nn.BatchNorm2d(ch)

    def forward(self, x):
        h = F.relu(self.b1(self.c1(x)))
        h = self.b2(self.c2(h))
        return F.relu(x + h)


class ChessNet(nn.Module):
    def __init__(self, in_ch=18, ch=128, blocks=6, n_scalar=0):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, ch, 3, padding=1, bias=False), nn.BatchNorm2d(ch), nn.ReLU())
        self.blocks = nn.Sequential(*[ResBlock(ch) for _ in range(blocks)])
        self.squeeze = nn.Sequential(
            nn.Conv2d(ch, 32, 1, bias=False), nn.BatchNorm2d(32), nn.ReLU())
        self.fc = nn.Sequential(
            nn.Linear(32 * 64 + n_scalar, 256), nn.ReLU(),
            nn.Dropout(0.2), nn.Linear(256, 3))

    def forward(self, x, s):
        h = self.squeeze(self.blocks(self.stem(x))).flatten(1)
        return self.fc(torch.cat([h, s], dim=1))  # s may have 0 columns


def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_loss, n = 0.0, 0
    correct = np.zeros(3)
    count = np.zeros(3)
    for x, s, y in loader:
        x, s, y = x.to(device), s.to(device), y.to(device)
        logits = model(x, s)
        total_loss += F.cross_entropy(logits, y, reduction="sum").item()
        pred = logits.argmax(1)
        for c in range(3):
            m = y == c
            count[c] += m.sum().item()
            correct[c] += (pred[m] == c).sum().item()
        n += len(y)
    acc = correct.sum() / n
    per_class = correct / np.maximum(count, 1)
    return total_loss / n, acc, per_class


def baselines(ds, train_ds):
    """(1) always predict the most common training class; (2) predict the higher-rated player wins."""
    majority = np.bincount(train_ds.y, minlength=3).argmax()
    acc_major = (ds.y == majority).mean()
    diff = ds.elo[:, 0].astype(int) - ds.elo[:, 1].astype(int)
    pred = np.where(diff > 0, 0, np.where(diff < 0, 2, majority))
    acc_elo = (pred == ds.y).mean()
    return acc_major, acc_elo


def plot_losses(step_losses, steps_per_epoch, epoch_train, epoch_val, path, show=True, title="Training loss", quiet=False):
    """Visualize the per-step train loss (raw + smoothed) and per-epoch train/val loss, x-axis in epochs."""
    import matplotlib
    if not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    steps = np.asarray(step_losses)
    x = np.arange(1, len(steps) + 1) / steps_per_epoch
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(x, steps, color="tab:blue", alpha=0.2, linewidth=0.8, label="train (per step)")
    win = max(1, min(len(steps) // 20, steps_per_epoch))
    if len(steps) >= win > 1:
        smooth = np.convolve(steps, np.ones(win) / win, mode="valid")
        ax.plot(x[win - 1:], smooth, color="tab:blue", linewidth=2, label=f"train (avg of {win} steps)")
    epochs = np.arange(1, len(epoch_val) + 1)
    ax.plot(epochs, epoch_train, "s--", color="tab:blue", label="train (epoch mean)")
    ax.plot(epochs, epoch_val, "o-", color="tab:orange", linewidth=2, label="val")
    ax.set_xlabel("epoch")
    ax.set_ylabel("cross-entropy loss")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    if not quiet:
        print(f"\nsaved loss curve to {path}")
    if show and matplotlib.get_backend().lower() != "agg":
        plt.show()
    plt.close(fig)


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data")
    p.add_argument("--features", default="board",
                   help="inputs joined by '+': board, skill, time, history (board is always on)")
    p.add_argument("--history", type=int, default=8, help="previous positions used with 'history'")
    p.add_argument("--use-elo", action="store_true", help="same as adding 'skill' to --features")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--channels", type=int, default=128)
    p.add_argument("--blocks", type=int, default=6)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save", default="model.pt")
    p.add_argument("--plot", default="loss_curve.png", help="where to save the loss curve image")
    p.add_argument("--no-show", action="store_true", help="save the loss curve without opening a window")
    p.add_argument("--tasks", choices=("outcome", "all"), default="outcome",
                   help="all trains the five heads; requires shards extracted with --tasks all")
    p.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    p.add_argument("--task-weights", type=float, nargs=5, default=[1., .5, .25, .25, .1],
                   metavar=("OUTCOME", "MISTAKE", "MOVES", "TIME", "ENDING"))
    p.add_argument("--huber-delta", type=float, default=1.)
    p.add_argument("--select-by", choices=("outcome", "total"), default="outcome",
                   help="validation criterion for five-task checkpoint selection")
    p.add_argument("--skip-test", action="store_true",
                   help="reserve test evaluation for evaluate.py after model selection")
    p.add_argument("--quiet", action="store_true",
                   help="suppress progress messages; files are still saved and errors/warnings remain visible")
    a = p.parse_args(argv)
    if a.use_elo:
        a.features += "+skill"
    a.features = "+".join(parse_features(a.features))
    return a


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def run(a):
    """Train, test, plot. Returns a dict of test metrics and loss curves."""
    if getattr(a, "tasks", "outcome") == "all":
        from multitask_training import run_five_task
        return run_five_task(a)
    quiet = getattr(a, "quiet", False)

    def log(message):
        if not quiet:
            print(message, flush=True)

    seed_everything(a.seed)
    device = pick_device() if getattr(a, "device", "auto") == "auto" else torch.device(a.device)
    features = parse_features(a.features)
    log(f"device: {device}  features: {a.features}  seed: {a.seed}")

    ds_kw = dict(features=features, history=a.history)
    train_ds = LichessPositions(a.data, "train", **ds_kw)
    val_ds = LichessPositions(a.data, "val", **ds_kw)
    test_ds = None if getattr(a, "skip_test", False) else LichessPositions(a.data, "test", **ds_kw)
    log(f"positions: train {len(train_ds):,}  val {len(val_ds):,}")
    if not len(train_ds) or not len(val_ds) or a.epochs < 1:
        raise ValueError("Need nonempty train/validation partitions and --epochs >= 1")
    from pathlib import Path
    Path(a.save).parent.mkdir(parents=True, exist_ok=True)
    Path(a.plot).parent.mkdir(parents=True, exist_ok=True)

    kw = dict(batch_size=a.batch_size, num_workers=a.workers, pin_memory=device.type == "cuda",
              persistent_workers=a.workers > 0)
    train_dl = DataLoader(train_ds, shuffle=True, drop_last=False,
                          generator=torch.Generator().manual_seed(a.seed), **kw)
    val_dl = DataLoader(val_ds, **kw)
    test_dl = DataLoader(test_ds, **kw) if test_ds is not None else None

    model = ChessNet(in_ch=train_ds.in_channels, ch=a.channels, blocks=a.blocks,
                     n_scalar=train_ds.n_scalar).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    log(f"parameters: {n_params:,}  input planes: {train_ds.in_channels}  "
          f"scalar features: {train_ds.n_scalar}")

    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=a.lr, total_steps=a.epochs * len(train_dl))
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler(enabled=use_amp)

    best_val, best_epoch = float("inf"), 0
    step_losses, epoch_train, epoch_val = [], [], []
    for epoch in range(1, a.epochs + 1):
        model.train()
        t0, run_loss, run_correct, seen = time.time(), 0.0, 0, 0
        for step, (x, s, y) in enumerate(train_dl, 1):
            x, s, y = x.to(device), s.to(device), y.to(device)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                logits = model(x, s)
                loss = F.cross_entropy(logits, y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()

            step_losses.append(loss.item())
            run_loss += step_losses[-1] * len(y)
            run_correct += (logits.argmax(1) == y).sum().item()
            seen += len(y)
            if step % 200 == 0:
                log(f"  epoch {epoch} step {step}/{len(train_dl)} "
                    f"loss {run_loss / seen:.4f} acc {run_correct / seen:.4f}")

        val_loss, val_acc, val_pc = evaluate(model, val_dl, device)
        epoch_train.append(run_loss / seen)
        epoch_val.append(val_loss)
        log(f"epoch {epoch}: train loss {run_loss / seen:.4f} acc {run_correct / seen:.4f} | "
              f"val loss {val_loss:.4f} acc {val_acc:.4f} | {time.time() - t0:.0f}s")
        if val_loss < best_val:
            best_val, best_epoch = val_loss, epoch
            torch.save({"model": model.state_dict(), "args": vars(a),
                        "model_type": "outcome", "best_epoch": best_epoch,
                        "best_val_loss": best_val}, a.save)
            log(f"  saved best model to {a.save}")

    # Final test with the best checkpoint
    ckpt = torch.load(a.save, map_location=device, weights_only=True)
    ckpt["training_history"] = {"epoch_train": epoch_train, "epoch_val": epoch_val}
    torch.save(ckpt, a.save)
    model.load_state_dict(ckpt["model"])
    test_loss = test_acc = base_major = base_elo = None
    test_pc = [None] * 3
    if test_dl is not None:
        test_loss, test_acc, test_pc = evaluate(model, test_dl, device)
        base_major, base_elo = baselines(test_ds, train_ds)
        log(f"test: categorical cross-entropy {test_loss:.4f}; accuracy {test_acc:.4f}")
        log(f"baselines: majority accuracy {base_major:.4f}; higher Elo accuracy {base_elo:.4f}")
        for name, acc in zip(CLASSES, test_pc):
            log(f"  recall on {name:<10} {acc:.4f}")

    plot_losses(step_losses, len(train_dl), epoch_train, epoch_val, a.plot,
                show=not a.no_show, title=f"Training loss ({a.features})", quiet=quiet)

    return {
        "features": a.features, "seed": a.seed, "params": n_params,
        "test_loss": test_loss, "test_acc": test_acc,
        **{f"recall_{c.replace(' ', '_')}": r for c, r in zip(CLASSES, test_pc)},
        "best_val_loss": best_val, "best_epoch": best_epoch,
        "baseline_majority_acc": base_major, "baseline_elo_acc": base_elo,
        "epoch_train": epoch_train, "epoch_val": epoch_val,
    }


def main():
    run(parse_args())


if __name__ == "__main__":
    main()
