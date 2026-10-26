"""
Train and test a small ResNet on the shards written by extract.py.

    python train.py --data data --epochs 5
    python train.py --data data --epochs 5 --use-elo      # also feed player ratings to the model

Prints train/val metrics each epoch, saves the best model (by val loss) to --save,
then reloads it and reports test accuracy next to two simple baselines.
"""
import argparse
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset import LichessPositions

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
    def __init__(self, in_ch=18, ch=128, blocks=6, use_elo=False):
        super().__init__()
        self.use_elo = use_elo
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, ch, 3, padding=1, bias=False), nn.BatchNorm2d(ch), nn.ReLU())
        self.blocks = nn.Sequential(*[ResBlock(ch) for _ in range(blocks)])
        self.squeeze = nn.Sequential(
            nn.Conv2d(ch, 32, 1, bias=False), nn.BatchNorm2d(32), nn.ReLU())
        self.fc = nn.Sequential(
            nn.Linear(32 * 64 + (2 if use_elo else 0), 256), nn.ReLU(),
            nn.Dropout(0.2), nn.Linear(256, 3))

    def forward(self, x, elo):
        h = self.squeeze(self.blocks(self.stem(x))).flatten(1)
        if self.use_elo:
            h = torch.cat([h, elo], dim=1)
        return self.fc(h)


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
    for x, elo, y in loader:
        x, elo, y = x.to(device), elo.to(device), y.to(device)
        logits = model(x, elo)
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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--channels", type=int, default=128)
    p.add_argument("--blocks", type=int, default=6)
    p.add_argument("--use-elo", action="store_true", help="feed both players' ratings to the model")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--save", default="model.pt")
    a = p.parse_args()

    device = pick_device()
    print(f"device: {device}")

    train_ds = LichessPositions(a.data, "train")
    val_ds = LichessPositions(a.data, "val")
    test_ds = LichessPositions(a.data, "test")
    print(f"positions: train {len(train_ds):,}  val {len(val_ds):,}  test {len(test_ds):,}")

    kw = dict(batch_size=a.batch_size, num_workers=a.workers, pin_memory=device.type == "cuda",
              persistent_workers=a.workers > 0)
    train_dl = DataLoader(train_ds, shuffle=True, drop_last=True, **kw)
    val_dl = DataLoader(val_ds, **kw)
    test_dl = DataLoader(test_ds, **kw)

    model = ChessNet(ch=a.channels, blocks=a.blocks, use_elo=a.use_elo).to(device)
    print(f"parameters: {sum(p.numel() for p in model.parameters()):,}  use_elo={a.use_elo}")

    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=a.lr, total_steps=a.epochs * len(train_dl))
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler(enabled=use_amp)

    best_val = float("inf")
    for epoch in range(1, a.epochs + 1):
        model.train()
        t0, run_loss, run_correct, seen = time.time(), 0.0, 0, 0
        for step, (x, elo, y) in enumerate(train_dl, 1):
            x, elo, y = x.to(device), elo.to(device), y.to(device)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                logits = model(x, elo)
                loss = F.cross_entropy(logits, y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()

            run_loss += loss.item() * len(y)
            run_correct += (logits.argmax(1) == y).sum().item()
            seen += len(y)
            if step % 200 == 0:
                print(f"  epoch {epoch} step {step}/{len(train_dl)} "
                      f"loss {run_loss / seen:.4f} acc {run_correct / seen:.4f}", flush=True)

        val_loss, val_acc, val_pc = evaluate(model, val_dl, device)
        print(f"epoch {epoch}: train loss {run_loss / seen:.4f} acc {run_correct / seen:.4f} | "
              f"val loss {val_loss:.4f} acc {val_acc:.4f} | {time.time() - t0:.0f}s")
        if val_loss < best_val:
            best_val = val_loss
            torch.save({"model": model.state_dict(), "args": vars(a)}, a.save)
            print(f"  saved best model to {a.save}")

    # Final test with the best checkpoint
    ckpt = torch.load(a.save, map_location=device)
    model.load_state_dict(ckpt["model"])
    test_loss, test_acc, test_pc = evaluate(model, test_dl, device)
    base_major, base_elo = baselines(test_ds, train_ds)

    print("\n=== TEST RESULTS ===")
    print(f"model accuracy:              {test_acc:.4f}  (loss {test_loss:.4f})")
    print(f"baseline, majority class:    {base_major:.4f}")
    print(f"baseline, higher Elo wins:   {base_elo:.4f}")
    for name, acc in zip(CLASSES, test_pc):
        print(f"  recall on {name:<10} {acc:.4f}")


if __name__ == "__main__":
    main()