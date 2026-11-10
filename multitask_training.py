"""Train all five heads with train-only normalization and validation selection.

Imported lazily by train.run after ChessNet is defined. Detailed per-task losses,
valid counts, selection decisions and optional test results are written beside
--save as .history.json. The return value retains the original ablate.py fields.
No optimizer resume or previous-checkpoint loading is performed.
"""
from contextlib import nullcontext
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import tempfile
import time
import warnings

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import parse_features
from multitask_data import MultiTaskPositions, TERMINATION_CLASSES, training_normalization
from multitask_model import DEFAULT_WEIGHTS, FiveTaskLoss, MultiTaskChessNet

TASKS = ("outcome", "mistake", "moves", "time", "termination")
OUTCOME_CLASSES = ("white_win", "draw", "black_win")


def _device(requested):
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if requested not in ("cpu", "cuda", "mps"):
        raise ValueError("--device must be auto, cpu, cuda, or mps")
    if requested == "cuda" and not torch.cuda.is_available():
        raise ValueError("--device cuda requested, but CUDA is not available")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise ValueError("--device mps requested, but MPS is not available")
    return torch.device(requested)


def _weights(values):
    if values is None:
        result = dict(DEFAULT_WEIGHTS)
    elif isinstance(values, dict):
        if set(values) != set(TASKS):
            raise ValueError("Task weights must contain exactly the five task names")
        result = {name: float(values[name]) for name in TASKS}
    else:
        if len(values) != len(TASKS):
            raise ValueError("--task-weights requires outcome mistake moves time termination")
        result = dict(zip(TASKS, map(float, values)))
    if any(not math.isfinite(v) or v < 0 for v in result.values()) or not any(result.values()):
        raise ValueError("Task weights must be finite, nonnegative, and not all zero")
    return result


def _valid_counts(dataset):
    return {name: int(np.asarray(dataset.masks[name], dtype=bool).sum()) for name in TASKS}


def _warn_class_support(dataset):
    for name, size in (("outcome", 3), ("mistake", 2), ("termination", len(TERMINATION_CLASSES))):
        observed = np.asarray(dataset.targets[name])[np.asarray(dataset.masks[name], dtype=bool)]
        counts = np.bincount(observed.astype(np.int64), minlength=size)
        absent = np.flatnonzero(counts == 0).tolist()
        if absent:
            warnings.warn(f"Training {name} labels omit classes {absent}; counts={counts.tolist()}. "
                          "The run can proceed, but unsupported classes cannot be learned reliably.")


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    return value


def _atomic_checkpoint(path, payload):
    """Replace only this run's checkpoint; the caller rejects existing output."""
    with tempfile.NamedTemporaryFile(prefix=f".{path.name}.", suffix=".tmp",
                                     dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
    try:
        torch.save(payload, temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _empty_totals():
    return {"sums": {name: 0.0 for name in TASKS},
            "counts": {name: 0 for name in TASKS},
            "correct": np.zeros(3, dtype=np.int64),
            "support": np.zeros(3, dtype=np.int64), "positions": 0}


def _accumulate(totals, components, counts, outputs, targets, masks):
    for name in TASKS:
        n = int(counts[name])
        if n:
            value = float(components[name].detach().item())
            if not math.isfinite(value):
                raise ValueError(f"Nonfinite {name} task loss")
            totals["sums"][name] += n * value
            totals["counts"][name] += n
    mask = masks["outcome"].bool()
    actual = targets["outcome"][mask].long()
    predicted = outputs["outcome"][mask].detach().argmax(dim=1)
    for label in range(3):
        selected = actual == label
        totals["support"][label] += int(selected.sum().item())
        totals["correct"][label] += int((predicted[selected] == label).sum().item())
    totals["positions"] += len(targets["outcome"])


def _finish(totals, weights):
    counts = totals["counts"]
    losses = {name: totals["sums"][name] / counts[name] if counts[name] else None for name in TASKS}
    # A missing task is not silently scored as zero in a global split summary.
    total = sum(weights[name] * losses[name] for name in TASKS) if all(counts.values()) else None
    outcome_n = int(totals["support"].sum())
    return {"positions": totals["positions"], "losses": losses, "counts": dict(counts),
            "weighted_total": float(total) if total is not None else None,
            "outcome_accuracy": float(totals["correct"].sum() / outcome_n) if outcome_n else None,
            "outcome_support": {name: int(n) for name, n in zip(OUTCOME_CLASSES, totals["support"])},
            "outcome_recall": {name: float(c / n) if n else None
                               for name, c, n in zip(OUTCOME_CLASSES, totals["correct"], totals["support"])}}


def _to_device(batch, device):
    board, scalars, targets, masks = batch
    return (board.to(device, non_blocking=True), scalars.to(device, non_blocking=True),
            {k: v.to(device, non_blocking=True) for k, v in targets.items()},
            {k: v.to(device, non_blocking=True) for k, v in masks.items()})


@torch.inference_mode()
def evaluate_five_task(model, loader, criterion, device):
    """Aggregate per-task sums/counts over the whole split, not batch means."""
    model.eval()
    totals = _empty_totals()
    for batch in loader:
        board, scalars, targets, masks = _to_device(batch, device)
        outputs = model(board, scalars)
        _, components, counts = criterion(outputs, targets, masks)
        _accumulate(totals, components, counts, outputs, targets, masks)
    return _finish(totals, criterion.weights)


def _baseline_accuracies(test_dataset, train_dataset):
    train_valid = np.asarray(train_dataset.masks["outcome"], dtype=bool)
    test_valid = np.asarray(test_dataset.masks["outcome"], dtype=bool)
    train_y = np.asarray(train_dataset.targets["outcome"])[train_valid].astype(np.int64)
    test_y = np.asarray(test_dataset.targets["outcome"])[test_valid].astype(np.int64)
    if not len(train_y) or not len(test_y):
        return None, None
    majority = int(np.bincount(train_y, minlength=3).argmax())
    rating = np.asarray(test_dataset.elo)[test_valid].astype(float)
    difference = rating[:, 0] - rating[:, 1]
    predicted = np.where(difference > 0, 0, np.where(difference < 0, 2, majority))
    return float(np.mean(test_y == majority)), float(np.mean(predicted == test_y))


def plot_training_history(history, path, show=False):
    """Write actual per-epoch task curves and their weighted total as PNG/SVG."""
    import matplotlib
    if not show or (not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY")):
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    png = Path(path).expanduser()
    if png.suffix.lower() != ".png":
        png = png.with_suffix(".png")
    svg = png.with_suffix(".svg")
    png.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(2, 3, figsize=(14, 8))
    titles = (("outcome", "Outcome CCE", "nats"),
              ("mistake", "Mistake BCE", "nats"),
              ("moves", "Remaining-length Huber", "standardized log-target units"),
              ("time", "Thinking-time Huber", "standardized log-target units"),
              ("termination", "Termination CCE", "nats"),
              ("weighted_total", "Weighted total (training objective)", "weighted task-loss units"))
    epochs = [row["epoch"] for row in history]
    for axis, (name, title, units) in zip(axes.flat, titles):
        has_line = False
        for split, label in (("train", "train (during updates)"), ("val", "validation")):
            values = [(row[split]["weighted_total"] if name == "weighted_total"
                       else row[split]["losses"][name]) for row in history]
            values = np.array([np.nan if v is None else v for v in values], dtype=float)
            if np.isfinite(values).any():
                axis.plot(epochs, values, "o-", label=label)
                has_line = True
        axis.set(title=title, xlabel="epoch", ylabel=units)
        axis.grid(alpha=0.25)
        if has_line:
            axis.legend(fontsize=8)
        else:
            axis.text(0.5, 0.5, "No valid split-level value", ha="center", transform=axis.transAxes)
    figure.suptitle("Measured five-task training and validation curves")
    figure.text(0.5, 0.01, "Each task uses its own valid-label count; missing task values create gaps. "
                "The weighted total is not the primary outcome metric.", ha="center", fontsize=9)
    figure.tight_layout(rect=(0, 0.04, 1, 0.96))
    try:
        figure.savefig(png, dpi=180)
        figure.savefig(svg)
        if show and matplotlib.get_backend().lower() != "agg":
            plt.show()
    finally:
        plt.close(figure)
    return [str(png), str(svg)]


def run_five_task(a):
    """Train five heads, select on validation, optionally test once, return legacy fields."""
    def log(message):
        if not getattr(a, "quiet", False):
            print(message, flush=True)

    if a.epochs < 1 or a.batch_size < 1 or a.workers < 0 or a.channels < 1 or a.blocks < 0:
        raise ValueError("epochs, batch size and channels must be positive; workers/blocks nonnegative")
    if not math.isfinite(float(a.lr)) or a.lr <= 0:
        raise ValueError("--lr must be finite and positive")
    delta = float(getattr(a, "huber_delta", 1.0))
    if not math.isfinite(delta) or delta <= 0:
        raise ValueError("--huber-delta must be finite and positive")
    select_by = getattr(a, "select_by", "outcome")
    if select_by not in ("outcome", "total"):
        raise ValueError("--select-by must be outcome or total")
    weights = _weights(getattr(a, "task_weights", None))
    save = Path(a.save).expanduser()
    history_path = Path(str(save) + ".history.json")
    if save.exists() or history_path.exists():
        raise FileExistsError("Choose a new --save path; five-task training does not resume or overwrite previous runs")
    if importlib.util.find_spec("matplotlib") is None:
        raise ImportError("Five-task training plots require matplotlib; install the project requirements")
    device = _device(getattr(a, "device", "auto"))
    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(a.seed)
        torch.backends.cudnn.benchmark = False
    features = parse_features(a.features)
    dataset_args = {"features": features, "history": a.history}
    train_dataset = MultiTaskPositions(a.data, "train", **dataset_args)
    val_dataset = MultiTaskPositions(a.data, "val", **dataset_args)
    if not len(train_dataset) or not len(val_dataset):
        raise ValueError("Training and validation splits must both contain positions")
    train_counts, val_counts = _valid_counts(train_dataset), _valid_counts(val_dataset)
    missing = [name for name, n in train_counts.items() if not n]
    if missing:
        raise ValueError(f"All five heads require valid training labels; missing {missing}. "
                         "Re-extract a labeled five-task dataset, not legacy outcome-only shards.")
    if not val_counts["outcome"]:
        raise ValueError("Validation outcome labels are required for checkpoint selection")
    if select_by == "total" and not all(val_counts.values()):
        raise ValueError("--select-by total requires valid validation labels for every task")
    _warn_class_support(train_dataset)
    normalization = training_normalization(train_dataset)
    criterion = FiveTaskLoss(normalization, weights=weights, delta=delta)
    model = MultiTaskChessNet(in_ch=train_dataset.in_channels, ch=a.channels,
                             blocks=a.blocks, n_scalar=train_dataset.n_scalar,
                             n_termination=len(TERMINATION_CLASSES)).to(device)
    params = sum(parameter.numel() for parameter in model.parameters())
    loader_args = {"batch_size": a.batch_size, "num_workers": a.workers,
                   "pin_memory": device.type == "cuda", "persistent_workers": a.workers > 0}
    train_loader = DataLoader(train_dataset, shuffle=True, drop_last=False,
                              generator=torch.Generator().manual_seed(a.seed), **loader_args)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_args)
    optimizer = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=a.lr, total_steps=a.epochs * len(train_loader))
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    manifest_path = Path(a.data).expanduser() / "manifest.json"
    manifest_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest() if manifest_path.is_file() else None
    save.parent.mkdir(parents=True, exist_ok=True)
    history, best_state, best_metrics = [], None, None
    best_selection, best_epoch = float("inf"), 0
    started = time.perf_counter()
    args_metadata = _jsonable(vars(a))
    loss_config = {"weights": dict(criterion.weights), "huber_delta": delta,
                   "reduction": "each task mean over its own valid labels"}

    def payload():
        return {"model_type": "five_task", "model": best_state, "args": args_metadata,
                "normalization": _jsonable(normalization), "loss_config": loss_config,
                "termination_classes": list(TERMINATION_CLASSES),
                "class_order": list(OUTCOME_CLASSES), "best_epoch": best_epoch,
                "best_val": best_selection, "best_val_loss": best_metrics["losses"]["outcome"],
                "selection_criterion": "outcome_cce" if select_by == "outcome" else "weighted_total",
                "best_val_metrics": best_metrics, "training_history": list(history),
                "dataset_manifest_sha256": manifest_hash,
                "dataset_manifest_path": str(manifest_path.resolve()) if manifest_hash else None,
                "architecture": {"in_ch": train_dataset.in_channels, "n_scalar": train_dataset.n_scalar,
                                 "channels": a.channels, "blocks": a.blocks,
                                 "n_termination": len(TERMINATION_CLASSES)}}

    log(f"Device: {device} | Parameters: {params:,} | Inputs: {a.features}")
    log(f"Positions: train {len(train_dataset):,}, validation {len(val_dataset):,} | "
        f"{len(train_loader):,} batches/epoch")
    log("Epoch   Train total   Val total   Train CCE   Val CCE   Val acc   Time")
    for epoch in range(1, a.epochs + 1):
        epoch_started = time.perf_counter()
        model.train()
        totals = _empty_totals()
        for batch in train_loader:
            board, scalars, targets, masks = _to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            context = torch.autocast(device_type="cuda", dtype=torch.float16) if use_amp else nullcontext()
            with context:
                outputs = model(board, scalars)
                loss, components, counts = criterion(outputs, targets, masks)
            if not bool(torch.isfinite(loss)):
                raise ValueError("Nonfinite weighted training loss")
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            _accumulate(totals, components, counts, outputs, targets, masks)
        train_metrics = _finish(totals, weights)
        val_metrics = evaluate_five_task(model, val_loader, criterion, device)
        selection = val_metrics["losses"]["outcome"] if select_by == "outcome" else val_metrics["weighted_total"]
        if selection is None or not math.isfinite(selection):
            raise ValueError("Validation selection metric is unavailable or nonfinite")
        history.append({"epoch": epoch, "train": train_metrics, "val": val_metrics,
                        "selection_value": selection, "seconds": time.perf_counter() - epoch_started})
        improved = selection < best_selection
        if improved:
            best_selection, best_epoch, best_metrics = float(selection), epoch, val_metrics
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
            _atomic_checkpoint(save, payload())
        def number(value):
            return f"{value:.4f}" if value is not None else "n/a"

        log(f"{epoch:>3}/{a.epochs:<3} {number(train_metrics['weighted_total']):>11} "
            f"{number(val_metrics['weighted_total']):>11} "
            f"{train_metrics['losses']['outcome']:>11.4f} "
            f"{val_metrics['losses']['outcome']:>9.4f} "
            f"{val_metrics['outcome_accuracy']:>8.1%} "
            f"{history[-1]['seconds']:>6.1f}s" + ("  best" if improved else ""))

    if best_state is None:
        raise RuntimeError("No valid validation-selected checkpoint was produced")
    # Preserve the best weights while adding the FULL final history, including later epochs.
    _atomic_checkpoint(save, payload())
    model.load_state_dict(best_state)
    test_metrics, majority_acc, elo_acc = None, None, None
    if not getattr(a, "skip_test", False):
        # Test is first loaded only after training and checkpoint selection are complete.
        test_dataset = MultiTaskPositions(a.data, "test", **dataset_args)
        if not len(test_dataset):
            raise ValueError("Test split contains no positions; use --skip-test for a validation-only run")
        test_loader = DataLoader(test_dataset, shuffle=False, **loader_args)
        test_metrics = evaluate_five_task(model, test_loader, criterion, device)
        majority_acc, elo_acc = _baseline_accuracies(test_dataset, train_dataset)
        log(f"Test: outcome CCE {test_metrics['losses']['outcome']:.4f} | "
            f"accuracy {test_metrics['outcome_accuracy']:.1%}")
    plot_paths = plot_training_history(history, a.plot, show=not a.no_show)
    details = {"model_type": "five_task", "args": args_metadata, "device": str(device),
               "torch_version": str(torch.__version__), "parameters": params,
               "normalization": _jsonable(normalization), "loss_config": loss_config,
               "training_history": history, "best_epoch": best_epoch,
               "best_val": best_selection, "best_val_metrics": best_metrics,
               "selection_criterion": select_by, "test": test_metrics,
               "baseline_majority_acc": majority_acc, "baseline_elo_acc": elo_acc,
               "plot_paths": plot_paths, "dataset_manifest_sha256": manifest_hash,
               "total_seconds": time.perf_counter() - started,
               "note": "Training normalization and validation selection only. Test was not used "
                       "for selection. Seeded execution does not guarantee identical results across hardware."}
    history_path.write_text(json.dumps(_jsonable(details), indent=2, allow_nan=False) + "\n")
    criterion_name = "outcome CCE" if select_by == "outcome" else "weighted total"
    log(f"Finished {a.epochs} epochs, {a.epochs * len(train_loader):,} batches. "
        f"Best: epoch {best_epoch} (validation {criterion_name} {best_selection:.4f}).")
    log(f"Checkpoint: {save}")
    log(f"History: {history_path}")
    log(f"Curves: {plot_paths[0]}")
    # Exact legacy field set keeps ablate.py's fixed CSV schema compatible.
    recalls = test_metrics["outcome_recall"] if test_metrics else dict.fromkeys(OUTCOME_CLASSES)
    return {"features": a.features, "seed": a.seed, "params": params,
            "test_loss": test_metrics["losses"]["outcome"] if test_metrics else None,
            "test_acc": test_metrics["outcome_accuracy"] if test_metrics else None,
            **{f"recall_{name}": recalls[name] for name in OUTCOME_CLASSES},
            "best_val_loss": best_metrics["losses"]["outcome"], "best_epoch": best_epoch,
            "baseline_majority_acc": majority_acc, "baseline_elo_acc": elo_acc,
            "epoch_train": [row["train"]["losses"]["outcome"] for row in history],
            "epoch_val": [row["val"]["losses"]["outcome"] for row in history]}
