"""
Feature ablations: train the same network on the same positions with different inputs and
compare test loss, to measure what each factor adds to outcome prediction.

    python ablate.py --data data --epochs 5 --seeds 0 1 2
    python ablate.py --configs board board+time board+skill board+skill+time+history

Default configs are cumulative: board -> +skill -> +time -> +history. Each (config, seed) run
saves its checkpoint and loss curve under --out; results.csv gets one row per run and runs
already in it are skipped, so an interrupted ablation resumes where it stopped.
At the end it writes summary.csv and ablation.png.

The main metric is test cross-entropy (lower is better): it rewards calibrated probabilities,
while accuracy barely moves when a feature mostly sharpens confidence.
"""
import argparse
import csv
import gc
import json
import os

import numpy as np
import torch

import train
from dataset import parse_features

DEFAULT_CONFIGS = ["board", "board+skill", "board+skill+time", "board+skill+time+history"]
FIELDS = ["features", "seed", "params", "test_loss", "test_acc", "recall_white_win",
          "recall_draw", "recall_black_win", "best_val_loss", "best_epoch",
          "baseline_majority_acc", "baseline_elo_acc", "epoch_train", "epoch_val"]


def load_rows(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


def append_row(path, row):
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerow({k: json.dumps(v) if isinstance(v, list) else v for k, v in row.items()})


def summarize(rows, configs):
    """Mean and std over seeds per config, plus the loss change vs board-only and vs the previous config."""
    out = []
    for cfg in configs:
        rs = [r for r in rows if r["features"] == cfg]
        if not rs:
            continue
        loss = np.array([float(r["test_loss"]) for r in rs])
        acc = np.array([float(r["test_acc"]) for r in rs])
        out.append({"features": cfg, "seeds": len(rs),
                    "test_loss": loss.mean(), "test_loss_std": loss.std(ddof=1) if len(rs) > 1 else 0.0,
                    "test_acc": acc.mean(), "test_acc_std": acc.std(ddof=1) if len(rs) > 1 else 0.0,
                    "val_curve": np.mean([json.loads(r["epoch_val"]) for r in rs], axis=0)})
    ref = next((s["test_loss"] for s in out if s["features"] == "board"), None)
    for i, s in enumerate(out):
        s["d_loss_vs_board"] = s["test_loss"] - ref if ref is not None else float("nan")
        s["d_loss_vs_prev"] = s["test_loss"] - out[i - 1]["test_loss"] if i else float("nan")
    return out


def print_summary(summary):
    print("\n=== ABLATION SUMMARY (test set, mean ± std over seeds) ===")
    print(f"{'features':<28}{'seeds':>6}{'loss':>18}{'acc':>18}{'Δloss vs board':>16}{'Δloss vs prev':>15}")
    for s in summary:
        print(f"{s['features']:<28}{s['seeds']:>6}"
              f"{s['test_loss']:>11.4f} ± {s['test_loss_std']:.4f}"
              f"{s['test_acc']:>11.4f} ± {s['test_acc_std']:.4f}"
              f"{s['d_loss_vs_board']:>+16.4f}{s['d_loss_vs_prev']:>+15.4f}")


def write_summary(summary, path):
    keys = ["features", "seeds", "test_loss", "test_loss_std", "test_acc", "test_acc_std",
            "d_loss_vs_board", "d_loss_vs_prev"]
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(keys)
        for s in summary:
            w.writerow([s[k] for k in keys])


def plot_summary(summary, path, show):
    import matplotlib
    if not show or (not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY")):
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    names = [s["features"] for s in summary]
    loss = [s["test_loss"] for s in summary]
    err = [s["test_loss_std"] for s in summary]
    ax1.bar(range(len(names)), loss, yerr=err, capsize=4, color="tab:blue")
    ax1.set_xticks(range(len(names)), names, rotation=20, ha="right")
    lo, hi = min(loss) - max(err + [0]), max(loss) + max(err + [0])
    pad = max((hi - lo) * 0.5, 0.005)
    ax1.set_ylim(lo - pad, hi + pad)  # losses are close together; don't start the axis at 0
    for i, v in enumerate(loss):
        ax1.text(i, lo - pad * 0.9, f"{v:.4f}", ha="center", va="bottom", fontsize=9, color="white")
    ax1.set_ylabel("test cross-entropy (lower is better)")
    ax1.set_title("Test loss by feature set")
    ax1.grid(axis="y", alpha=0.3)

    for s in summary:
        ax2.plot(np.arange(1, len(s["val_curve"]) + 1), s["val_curve"], "o-", label=s["features"])
    ax2.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    ax2.set_xlabel("epoch")
    ax2.set_ylabel("val cross-entropy (mean over seeds)")
    ax2.set_title("Validation loss")
    ax2.grid(alpha=0.3)
    ax2.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    print(f"saved ablation plot to {path}")
    if show and matplotlib.get_backend().lower() != "agg":
        plt.show()
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default="data")
    p.add_argument("--configs", nargs="+", default=DEFAULT_CONFIGS,
                   help="feature sets to compare, e.g. board board+skill board+skill+time")
    p.add_argument("--seeds", nargs="+", type=int, default=[0])
    p.add_argument("--out", default="ablations")
    p.add_argument("--no-show", action="store_true", help="save the summary plot without opening a window")
    a, train_extra = p.parse_known_args()  # anything else (--epochs, --lr, ...) goes to train.py

    configs = ["+".join(parse_features(c)) for c in a.configs]
    os.makedirs(a.out, exist_ok=True)
    results = os.path.join(a.out, "results.csv")
    done = {(r["features"], int(r["seed"])) for r in load_rows(results)}

    for cfg in configs:
        for seed in a.seeds:
            if (cfg, seed) in done:
                print(f"skip {cfg} seed {seed}: already in {results}")
                continue
            tag = f"{cfg}_s{seed}"
            print(f"\n##### {tag} #####")
            args = train.parse_args(["--data", a.data, "--features", cfg, "--seed", str(seed),
                                     "--save", os.path.join(a.out, f"{tag}.pt"),
                                     "--plot", os.path.join(a.out, f"{tag}_loss.png"),
                                     "--no-show", *train_extra])
            append_row(results, train.run(args))
            gc.collect()
            torch.cuda.empty_cache()

    summary = summarize(load_rows(results), configs)
    print_summary(summary)
    write_summary(summary, os.path.join(a.out, "summary.csv"))
    plot_summary(summary, os.path.join(a.out, "ablation.png"), show=not a.no_show)


if __name__ == "__main__":
    main()
