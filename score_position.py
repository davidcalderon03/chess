"""Predict one position using current feature-aware or five-task checkpoints.

Use --fen for board/skill/time models. A history model requires --pgn FILE
--ply N so earlier boards are recovered from actual moves. Stockfish is optional
and remains a separate engine reference for the conversion-gap calculation.
"""
import argparse
import io
import json
from pathlib import Path

import chess
import chess.engine
import chess.pgn
import numpy as np
import torch

from evaluation_metrics import softmax
from model_io import (board_to_inputs, calibration_temperature, checkpoint_sha256,
                      load_checkpoint)
from scoring import difficulty_scores


def pgn_position(path, ply):
    """First game, before move at zero-based ply; inspect only earlier clocks."""
    if str(path).endswith(".zst"):
        import zstandard

        with open(path, "rb") as raw:
            with zstandard.ZstdDecompressor().stream_reader(raw) as binary:
                with io.TextIOWrapper(binary, encoding="utf-8") as text:
                    game = chess.pgn.read_game(text)
    else:
        with open(path, encoding="utf-8-sig") as text:
            game = chess.pgn.read_game(text)
    if game is None or game.errors:
        raise ValueError("PGN must contain a successfully parsed game")
    board = game.board()
    if board.fen() != chess.Board().fen():
        raise ValueError("PGN must start from the standard initial position to establish complete history")
    nodes = list(game.mainline())
    if ply is None or not 0 <= ply <= len(nodes):
        raise ValueError("--pgn requires --ply between 0 and the number of recorded plies")
    try:
        base, increment = map(float, game.headers.get("TimeControl", "").split("+"))
    except ValueError:
        base = increment = None
    clocks = [base, base]
    for node in nodes[:ply]:
        clocks[0 if board.turn else 1] = node.clock()
        board.push(node.move)
    context = {"white_elo": game.headers.get("WhiteElo"), "black_elo": game.headers.get("BlackElo"),
               "white_clock": clocks[0], "black_clock": clocks[1],
               "base_seconds": base, "increment_seconds": increment}
    return board, context


def stockfish_probabilities(board, engine_path, nodes=100_000):
    with chess.engine.SimpleEngine.popen_uci(engine_path) as engine:
        if "UCI_ShowWDL" not in engine.options:
            raise ValueError("engine must provide native UCI_ShowWDL")
        engine.configure({"UCI_ShowWDL": True, "Threads": 1, "Hash": 128})
        info = engine.analyse(board, chess.engine.Limit(nodes=nodes))
        if "wdl" not in info:
            raise ValueError("engine returned no native WDL; no centipawn fallback is used")
        wdl = info["wdl"].white()
        p = np.array([wdl.wins, wdl.draws, wdl.losses], dtype=float) / wdl.total()
        return p, {"engine_id": dict(engine.id), "path": str(engine_path), "nodes_limit": nodes,
                   "actual_nodes": info.get("nodes"), "depth": info.get("depth"), "threads": 1, "hash_mb": 128}


def conversion_gaps(engine, human, pov):
    gaps = difficulty_scores(np.asarray(engine)[None], np.asarray(human)[None], pov=pov)
    return {name: float(values[0]) for name, values in gaps.items()}


def auxiliary_predictions(outputs, checkpoint):
    if not isinstance(outputs, dict):
        return {}
    result = {"mistake_probability": float(torch.sigmoid(outputs["mistake"])[0].cpu())}
    normalization = checkpoint.get("normalization", {})
    for task, name in (("moves", "moves_remaining_plies"), ("time", "next_move_seconds")):
        stats = normalization.get(task, {})
        mean, std = stats.get("mean"), stats.get("std")
        if mean is None or std is None or not np.isfinite([mean, std]).all() or std <= 0:
            raise ValueError(f"five-task checkpoint lacks valid training normalization for {task}")
        with np.errstate(over="raise", invalid="raise"):
            try:
                value = np.expm1(float(mean) + float(std) * float(outputs[task][0].cpu()))
            except FloatingPointError as error:
                raise ValueError(f"{task} inverse prediction overflowed") from error
        result[name] = float(max(0, value))
    names = checkpoint.get("termination_classes", ["checkmate", "time_forfeit", "other_normal"])
    p = softmax(outputs["termination"].detach().cpu().numpy())[0]
    if len(names) != len(p):
        raise ValueError("termination class names do not match the head")
    result["termination_probabilities"] = dict(zip(names, map(float, p)))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--fen")
    source.add_argument("--pgn", "--history-pgn", dest="pgn", help="First recorded game; use --ply to select its position")
    parser.add_argument("--ply", type=int)
    parser.add_argument("--checkpoint", default="model.pt")
    parser.add_argument("--white-elo", type=float)
    parser.add_argument("--black-elo", type=float)
    parser.add_argument("--white-clock", type=float)
    parser.add_argument("--black-clock", type=float)
    parser.add_argument("--base-seconds", "--base", dest="base_seconds", type=float)
    parser.add_argument("--increment-seconds", "--increment", dest="increment_seconds", type=float)
    parser.add_argument("--pov", choices=["white", "black"], default="white")
    reference = parser.add_mutually_exclusive_group()
    reference.add_argument("--engine")
    reference.add_argument("--engine-wdl", type=float, nargs=3, metavar=("WHITE", "DRAW", "BLACK"))
    parser.add_argument("--nodes", type=int, default=100_000)
    calibration = parser.add_mutually_exclusive_group()
    calibration.add_argument("--temperature", type=float, default=1.0)
    calibration.add_argument("--calibration-report")
    parser.add_argument("--winning-threshold", type=float, default=0.8)
    parser.add_argument("--output")
    args = parser.parse_args(argv)
    try:
        if args.nodes <= 0 or not 0 <= args.winning_threshold <= 1:
            raise ValueError("nodes must be positive and winning-threshold in [0,1]")
        if args.fen and args.ply is not None:
            raise ValueError("--ply requires --pgn")
        board, context = pgn_position(args.pgn, args.ply) if args.pgn else (chess.Board(args.fen), {})
        if not board.is_valid() or board.is_game_over():
            raise ValueError("score a valid nonterminal standard-chess position")
        for field in ("white_elo", "black_elo", "white_clock", "black_clock", "base_seconds", "increment_seconds"):
            if getattr(args, field) is not None:
                context[field] = getattr(args, field)
        model, checkpoint = load_checkpoint(args.checkpoint)
        config = checkpoint["inference_config"]
        x, scalar = board_to_inputs(board, config, **context, history_verified=bool(args.pgn))
        temperature = calibration_temperature(args.calibration_report, args.checkpoint) if args.calibration_report else args.temperature
        with torch.inference_mode():
            outputs = model(x, scalar)
            logits = outputs["outcome"] if isinstance(outputs, dict) else outputs
            human = softmax(logits.cpu().numpy(), temperature)[0]
            auxiliary = auxiliary_predictions(outputs, checkpoint)
        engine = details = gaps = None
        if args.engine:
            engine, details = stockfish_probabilities(board, args.engine, args.nodes)
        elif args.engine_wdl is not None:
            engine = np.asarray(args.engine_wdl, dtype=float)
            details = {"source": "manual probabilities; plumbing inputs, not computed Stockfish results"}
        if engine is not None:
            gaps = conversion_gaps(engine, human, args.pov)
        result = {
            "fen": board.fen(), "pov": args.pov, "checkpoint": str(args.checkpoint),
            "checkpoint_sha256": checkpoint_sha256(args.checkpoint),
            "model_type": config["model_type"], "features": list(config["features"]),
            "history_positions": config["history"], "context": context,
            "temperature": float(temperature), "class_order": ["white_win", "draw", "black_win"],
            "human_wdl": human.tolist(), "auxiliary_predictions": auxiliary,
            "engine_wdl": engine.tolist() if engine is not None else None,
            "scores": gaps, "engine": details, "winning_threshold": args.winning_threshold,
            "qualifies_for_conversion_analysis": gaps["sf_win_probability"] >= args.winning_threshold if gaps else None,
            "limitations": ["Only checkpoint-listed feature groups affect this prediction.",
                            "A bare FEN has no repetition history; use an original PGN for research.",
                            "Stockfish WDL is an engine reference, not perfect-play or human ground truth.",
                            "Auxiliary heads are supervised predictions, not guarantees of future events."],
        }
        rendered = json.dumps(result, indent=2, allow_nan=False) + "\n"
        if args.output:
            path = Path(args.output)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(rendered, encoding="utf-8")
        print(rendered, end="")
    except (ValueError, OSError, RuntimeError, KeyError) as error:
        parser.exit(2, f"Position scoring failed: {error}\n")


if __name__ == "__main__":
    main()
