"""Five heads sharing the current repository's 256-feature ChessNet backbone.

Outcome classes remain [White win, draw, Black win]. Regression heads emit
standardized log targets, not seconds/plies. Mistake outputs are raw logits.
"""
import math

import torch
from torch import nn
import torch.nn.functional as F

from train import ChessNet


DEFAULT_WEIGHTS = {"outcome": 1.0, "mistake": 0.5, "moves": 0.25,
                   "time": 0.25, "termination": 0.1}


class MultiTaskChessNet(nn.Module):
    def __init__(self, in_ch=18, ch=128, blocks=6, n_scalar=0, n_termination=3):
        super().__init__()
        if n_scalar < 0 or n_termination < 2:
            raise ValueError("n_scalar must be nonnegative and n_termination at least 2")
        self.n_scalar = n_scalar
        self.in_ch = in_ch
        self.backbone = ChessNet(in_ch=in_ch, ch=ch, blocks=blocks, n_scalar=n_scalar)
        self.backbone.fc[-1] = nn.Identity()
        self.heads = nn.ModuleDict({
            "outcome": nn.Linear(256, 3),
            "mistake": nn.Linear(256, 1),
            "moves": nn.Linear(256, 1),
            "time": nn.Linear(256, 1),
            "termination": nn.Linear(256, n_termination),
        })

    def forward(self, board, scalar=None):
        if board.ndim != 4 or tuple(board.shape[1:]) != (self.in_ch, 8, 8):
            raise ValueError("board must have shape (N, in_ch, 8, 8)")
        if scalar is None:
            if self.n_scalar:
                raise ValueError("this model requires scalar context features")
            scalar = board.new_empty((len(board), 0))
        if scalar.ndim != 2 or tuple(scalar.shape) != (len(board), self.n_scalar):
            raise ValueError("scalar features must have shape (N, n_scalar)")
        shared = self.backbone(board, scalar)
        outputs = {name: head(shared) for name, head in self.heads.items()}
        for name in ("mistake", "moves", "time"):
            outputs[name] = outputs[name].squeeze(-1)
        return outputs


class FiveTaskLoss(nn.Module):
    """Mean each task over its own explicit mask, then apply task weights.

    Empty *batch* masks contribute graph-connected zeros. Dataset-level training
    must separately ensure every enabled task has actual training labels.
    Normalization uses {'moves': {'mean': ..., 'std': ...}, 'time': {...}};
    these are training statistics of log1p(original-unit targets).
    """
    def __init__(self, normalization, weights=None, delta=1.0):
        super().__init__()
        self.delta = float(delta)
        if not math.isfinite(self.delta) or self.delta <= 0:
            raise ValueError("Huber delta must be finite and positive")
        self.weights = dict(DEFAULT_WEIGHTS)
        if weights is not None:
            if set(weights) - set(self.weights):
                raise ValueError("unknown task weight")
            self.weights.update(weights)
        self.weights = {name: float(value) for name, value in self.weights.items()}
        if any(not math.isfinite(value) or value < 0 for value in self.weights.values()):
            raise ValueError("task weights must be finite and nonnegative")
        self.normalization = {}
        for task in ("moves", "time"):
            if task not in normalization or not {"mean", "std"}.issubset(normalization[task]):
                raise ValueError(f"missing training normalization for {task}")
            mean, std = float(normalization[task]["mean"]), float(normalization[task]["std"])
            if not math.isfinite(mean) or not math.isfinite(std) or std <= 0:
                raise ValueError(f"{task} normalization needs a finite mean and positive finite std")
            self.normalization[task] = {"mean": mean, "std": std}

    def forward(self, outputs, targets, masks):
        required = set(self.weights)
        if any(required - set(mapping) for mapping in (outputs, targets, masks)):
            raise ValueError("outputs, targets, and masks must contain all five task names")
        components, counts = {}, {}
        batch_size = outputs["outcome"].shape[0]
        for task in self.weights:
            prediction, target, valid = outputs[task], targets[task], masks[task]
            categorical = task in ("outcome", "termination")
            valid_shape = (prediction.ndim == 2 and prediction.shape[1] >= 2) if categorical else prediction.ndim == 1
            if not valid_shape or prediction.shape[0] != batch_size:
                raise ValueError(f"invalid output shape for {task}")
            if task == "outcome" and prediction.shape[1] != 3:
                raise ValueError("outcome output must have three White/draw/Black logits")
            if tuple(target.shape) != (batch_size,) or tuple(valid.shape) != (batch_size,):
                raise ValueError(f"{task} targets and masks must have shape (N,)")
            if valid.dtype != torch.bool:
                raise ValueError(f"{task} mask must be boolean")
            if not bool(torch.isfinite(prediction).all()):
                raise ValueError(f"nonfinite {task} output")
            counts[task] = int(valid.sum().item())
            if not counts[task]:
                components[task] = prediction.sum() * 0.0
                continue
            observed, predicted = target[valid], prediction[valid]
            if not bool(torch.isfinite(observed).all()):
                raise ValueError(f"nonfinite selected {task} label")
            if categorical:
                if (not bool((observed == observed.long()).all())
                        or not bool(((observed >= 0) & (observed < prediction.shape[1])).all())):
                    raise ValueError(f"invalid selected {task} class label")
                loss = F.cross_entropy(predicted, observed.long(), reduction="mean")
            elif task == "mistake":
                if not bool(((observed == 0) | (observed == 1)).all()):
                    raise ValueError("selected mistake labels must be binary")
                loss = F.binary_cross_entropy_with_logits(predicted, observed.float(), reduction="mean")
            else:
                if not bool((observed >= 0).all()):
                    raise ValueError(f"selected {task} targets must be nonnegative original units")
                stats = self.normalization[task]
                standardized = (torch.log1p(observed.float()) - stats["mean"]) / stats["std"]
                loss = F.huber_loss(predicted, standardized, reduction="mean", delta=self.delta)
            if not bool(torch.isfinite(loss)):
                raise ValueError(f"nonfinite {task} loss")
            components[task] = loss
        total = sum(self.weights[task] * components[task] for task in self.weights)
        return total, components, counts
