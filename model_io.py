"""Safe checkpoint loading and inference inputs matching the current dataset."""
import hashlib
import json
import re
from pathlib import Path

import chess
import numpy as np
import torch

from dataset import FACTORS, parse_features
from extract import bitboards, encode_meta


def checkpoint_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_checkpoint(path, device="cpu"):
    """Return (eval-mode model, checkpoint), never enabling unsafe pickle loading.

    Architecture dimensions are inferred from tensor shapes. Saved feature names
    are checked against those shapes; shape alone distinguishes current scalar
    layouts (0=board, 2=skill, 6=time, 8=skill+time).
    """
    saved = torch.load(path, map_location=device, weights_only=True)
    if not isinstance(saved, dict) or not isinstance(saved.get("model"), dict):
        raise ValueError("checkpoint must contain a model state dictionary")
    args = saved.get("args", {})
    if not isinstance(args, dict):
        raise ValueError("checkpoint args must be a plain dictionary")
    state = saved["model"]
    five_task = "heads.outcome.weight" in state
    declared = saved.get("model_type")
    if declared not in (None, "outcome", "single_task", "five_task"):
        raise ValueError(f"unsupported checkpoint model_type: {declared}")
    if declared == "five_task" and not five_task:
        raise ValueError("five_task checkpoint is missing its output heads")
    prefix = "backbone." if five_task else ""
    required = [prefix + key for key in ("stem.0.weight", "squeeze.0.weight", "fc.0.weight")]
    if any(key not in state for key in required):
        raise ValueError("checkpoint does not match the repository ChessNet architecture")
    channels, in_ch = map(int, state[prefix + "stem.0.weight"].shape[:2])
    squeeze = int(state[prefix + "squeeze.0.weight"].shape[0])
    n_scalar = int(state[prefix + "fc.0.weight"].shape[1]) - squeeze * 64
    if in_ch < 18 or (in_ch - 18) % 12 or n_scalar not in (0, 2, 6, 8):
        raise ValueError("checkpoint has an unsupported board/scalar feature layout")
    history = (in_ch - 18) // 12
    pattern = re.compile(r"^" + re.escape(prefix) + r"blocks\.(\d+)\.")
    indices = [int(match.group(1)) for key in state if (match := pattern.match(key))]
    blocks = max(indices) + 1 if indices else 0
    spec = saved.get("features", args.get("features"))
    if spec is None:
        inferred = {0: "board", 2: "board+skill", 6: "board+time", 8: "board+skill+time"}[n_scalar]
        spec = inferred + ("+history" if history else "")
    if not isinstance(spec, str):
        spec = "+".join(spec)
    if args.get("use_elo", False):
        spec += "+skill"
    features = parse_features(spec)
    expected_scalar = 2 * ("skill" in features) + 6 * ("time" in features)
    if expected_scalar != n_scalar or (history and "history" not in features):
        raise ValueError("checkpoint feature names disagree with its model tensor shapes")
    if five_task:
        from multitask_model import MultiTaskChessNet

        n_termination = int(state["heads.termination.weight"].shape[0])
        model = MultiTaskChessNet(in_ch, channels, blocks, n_scalar, n_termination)
    else:
        from train import ChessNet

        n_termination = None
        model = ChessNet(in_ch=in_ch, ch=channels, blocks=blocks, n_scalar=n_scalar)
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    config = {"features": tuple(features), "history": history, "in_channels": in_ch,
              "n_scalar": n_scalar, "channels": channels, "blocks": blocks,
              "n_termination": n_termination, "model_type": "five_task" if five_task else "outcome"}
    saved = dict(saved)
    saved["inference_config"] = config
    model.inference_config = config
    return model, saved


def load_model(path, device="cpu"):
    """Compatibility wrapper; new callers should use load_checkpoint."""
    model, saved = load_checkpoint(path, device)
    return model, saved.get("args", {})


def calibration_temperature(report_path, checkpoint):
    report = json.loads(Path(report_path).read_text(encoding="utf-8"))
    if report.get("checkpoint_sha256") != checkpoint_sha256(checkpoint):
        raise ValueError("calibration report does not match this checkpoint")
    value = report.get("temperature")
    if value is None and isinstance(report.get("calibration"), dict):
        value = report["calibration"].get("temperature")
    if value is None:
        raise ValueError("calibration report contains no outcome temperature")
    value = float(value)
    if not np.isfinite(value) or value <= 0:
        raise ValueError("calibration temperature must be finite and positive")
    return value


def board_to_inputs(board, config, *, white_elo=None, black_elo=None,
                    white_clock=None, black_clock=None, base_seconds=None,
                    increment_seconds=None, history_verified=False, device="cpu"):
    """Encode a pre-move board with exactly dataset.py's planes/scalar ordering.

    Complete move stacks come from an original PGN. A bare FEN cannot reconstruct
    prior boards. history_verified allows the PGN's initial position (zero moves).
    """
    if not board.is_valid() or board.chess960:
        raise ValueError("a valid standard-chess board is required")
    config = config.get("inference_config", config)
    features = tuple(config["features"])
    history = int(config.get("history", 0))
    bb, metadata = bitboards(board), encode_meta(board, board.ply())
    pieces = np.unpackbits(bb.view(np.uint8), bitorder="little").reshape(12, 8, 8)
    extra = np.zeros((6, 8, 8), dtype=np.uint8)
    extra[0] = metadata[0]
    extra[1:5] = metadata[1:5, None, None]
    if metadata[5] < 8:
        extra[5, :, metadata[5]] = 1
    planes = [pieces, extra]
    if history:
        if board.root().fen() != chess.Board().fen() or (not board.move_stack and not history_verified):
            raise ValueError("history-trained checkpoint requires an original PGN/move stack; FEN alone is insufficient")
        past, stored = board.copy(stack=True), np.zeros((history, 12), dtype=np.uint64)
        for j in range(history):
            if not past.move_stack:
                break
            past.pop()
            stored[j] = bitboards(past)
        planes.append(np.unpackbits(stored.view(np.uint8), bitorder="little").reshape(12 * history, 8, 8))
    scalars = []
    if "skill" in features:
        ratings = _numeric_pair(white_elo, black_elo, "ratings", positive=True)
        scalars.append(ratings / np.float32(3000.0))
    if "time" in features:
        clocks = _numeric_pair(white_clock, black_clock, "pre-move clocks")
        control = _numeric_pair(base_seconds, increment_seconds, "base/increment seconds")
        base, increment = control
        scalars.extend([np.clip(clocks / np.maximum(base, np.float32(1.0)), 0, 2),
                        np.log1p(clocks) / np.float32(8.0),
                        np.array([np.log1p(base) / np.float32(8.0), increment / np.float32(60.0)], dtype=np.float32)])
    scalar = np.concatenate(scalars).astype(np.float32) if scalars else np.empty(0, dtype=np.float32)
    x = torch.from_numpy(np.concatenate(planes).astype(np.float32))[None].to(device)
    s = torch.from_numpy(scalar)[None].to(device)
    if x.shape[1] != config.get("in_channels", x.shape[1]) or s.shape[1] != config.get("n_scalar", s.shape[1]):
        raise ValueError("inference features do not match checkpoint dimensions")
    return x, s


def _numeric_pair(first, second, name, positive=False):
    if first is None or second is None:
        raise ValueError(f"both {name} values are required by this checkpoint")
    values = np.array([first, second], dtype=np.float32)
    if not np.isfinite(values).all() or (values <= 0 if positive else values < 0).any():
        raise ValueError(f"{name} must be finite and {'positive' if positive else 'nonnegative'}")
    return values
