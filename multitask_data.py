"""Complete five-task labels and datasets, with explicit availability masks.

Outcome remains White win/draw/Black win. Features use only observations before
the sampled move. Engine-derived mistake labels and next-move clocks are targets,
never features. Regression normalization is fitted on training labels only.
"""
import argparse
from collections import Counter, deque
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import shutil
from urllib.parse import urlparse

import chess
import chess.engine
import chess.pgn
import numpy as np
import torch
import zstandard as zstd

from dataset import LichessPositions, parse_features
from extract import RESULT, accept, bitboards, encode_meta, parse_tc, split_of


TASKS = ("outcome", "mistake", "moves", "time", "termination")
TERMINATION_CLASSES = ("checkmate", "time_forfeit", "other_normal")
REFERENCE_LOSS_WEIGHTS = dict(outcome=1.0, mistake=0.5, moves=0.25, time=0.25, termination=0.1)
NEGATIVE_ENGINE_TOLERANCE = 0.01


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _identity(headers):
    site = headers.get("Site", "").strip()
    if not site or site == "?":
        return None
    parsed = urlparse(site)
    if parsed.hostname in ("lichess.org", "www.lichess.org"):
        identity = parsed.path.strip("/").split("/")[0]
        return identity or None
    return site


def _header_rejection(headers, args, identities):
    identity = _identity(headers)
    if identity is None:
        return "missing_site"
    if identity in identities:
        return "duplicate_game_id"
    if headers.get("Result") not in RESULT:
        return "unknown_result"
    if headers.get("Variant", "Standard").strip().lower() not in ("standard", "chess", "normal"):
        return "nonstandard_variant"
    if any(headers.get(key, "").strip().upper() == "BOT" for key in ("WhiteTitle", "BlackTitle")):
        return "bot"
    if not accept(headers, args):
        return "original_header_filter"
    base, increment = parse_tc(headers)
    ratings = [int(headers[key]) for key in ("WhiteElo", "BlackElo")]
    if base < 0 or increment < 0 or max(base, increment) > np.iinfo(np.int32).max:
        return "invalid_time_control"
    if min(ratings) < 0 or max(ratings) > np.iinfo(np.int16).max:
        return "invalid_rating"
    return None


def _builder(args, identities):
    class Builder(chess.pgn.GameBuilder):
        def end_headers(self):
            self.game._multitask_rejection = _header_rejection(self.game.headers, args, identities)
            return chess.pgn.SKIP if self.game._multitask_rejection else None

        def visit_result(self, result):
            self.game._multitask_result = result
            super().visit_result(result)

        def handle_error(self, error):
            self.game.errors.append(error)
    return Builder


def _termination(game, final_board):
    header = game.headers.get("Termination")
    if header == "Time forfeit":
        return 1, True
    if header == "Normal" and final_board.is_valid():
        return (0 if final_board.is_checkmate() else 2), True
    return 0, False


def _clock(node):
    value = node.clock()
    return float(value) if value is not None and np.isfinite(value) and value >= 0 else None


def _engine_expectation(info, mover):
    if "wdl" not in info:
        raise ValueError("Native Stockfish WDL is missing; no centipawn conversion is used")
    wdl = info["wdl"].pov(mover)
    if wdl.total() <= 0:
        raise ValueError("Native WDL has zero total")
    return (wdl.wins + .5 * wdl.draws) / wdl.total()


def _mistake_label(engine, board, move, nodes, threshold, counters, examples):
    """Compare two searches from the same original board and mover perspective."""
    result = dict(mistake_y=0.0, mistake_mask=False, mistake_delta=0.0,
                  mistake_best_expected_score=0.0, mistake_human_expected_score=0.0)
    try:
        best = engine.analyse(board, chess.engine.Limit(nodes=nodes), game=object())
        counters["engine_searches"] += 1
        best_value = _engine_expectation(best, board.turn)
        if best.get("pv") and best["pv"][0] == move:
            human_value = best_value
            counters["human_move_is_engine_best"] += 1
        else:
            human = engine.analyse(board, chess.engine.Limit(nodes=nodes),
                                   root_moves=[move], game=object())
            counters["engine_searches"] += 1
            human_value = _engine_expectation(human, board.turn)
        delta = best_value - human_value
        result.update(mistake_delta=float(delta), mistake_best_expected_score=float(best_value),
                      mistake_human_expected_score=float(human_value))
        if delta < -NEGATIVE_ENGINE_TOLERANCE:
            counters["mistake_masked_negative_engine_disagreement"] += 1
        else:
            result.update(mistake_y=float(max(0.0, delta) > threshold), mistake_mask=True)
    except (chess.engine.EngineError, TimeoutError, ValueError, OSError) as error:
        counters["mistake_engine_errors"] += 1
        if len(examples) < 10:
            examples.append({"type": type(error).__name__, "message": str(error)[:400]})
    return result


class _Writer:
    TYPES = {
        "bb": np.uint64, "meta": np.uint8, "elo": np.int16, "clk": np.float32,
        "clk_mask": np.bool_, "tc": np.int32, "hist": np.uint64, "y": np.uint8,
        "game_id": str, "ply": np.int32, "fen": str,
        "mistake_y": np.float32, "mistake_mask": np.bool_, "mistake_delta": np.float32,
        "mistake_best_expected_score": np.float32, "mistake_human_expected_score": np.float32,
        "remaining_plies": np.float32, "next_move_seconds": np.float32, "time_mask": np.bool_,
        "termination_y": np.int64, "termination_mask": np.bool_,
    }

    def __init__(self, out, shard_size, overwrite):
        self.out, self.shard_size = Path(out), shard_size
        self.manifest = self.out / "manifest.json"
        old = sorted(self.out.glob("*/shard_*.npz"))
        if (old or self.manifest.exists()) and not overwrite:
            raise FileExistsError(f"{self.out} already contains extraction outputs; choose a new --out")
        if overwrite:
            for path in old:
                path.unlink()
            self.manifest.unlink(missing_ok=True)
        self.buffers = {split: {key: [] for key in self.TYPES} for split in ("train", "val", "test")}
        self.indices = Counter()
        self.created = []
        for split in self.buffers:
            (self.out / split).mkdir(parents=True, exist_ok=True)

    def add(self, split, sample):
        for key in self.TYPES:
            self.buffers[split][key].append(sample[key])
        if len(self.buffers[split]["y"]) >= self.shard_size:
            self.flush(split)

    def flush(self, split):
        buffer = self.buffers[split]
        if not buffer["y"]:
            return
        path = self.out / split / f"shard_{self.indices[split]:05d}.npz"
        arrays = {key: np.asarray(buffer[key], dtype=dtype) for key, dtype in self.TYPES.items()}
        with path.open("xb") as destination:
            self.created.append(path)
            np.savez(destination, **arrays)
        self.indices[split] += 1
        self.buffers[split] = {key: [] for key in self.TYPES}

    def finish(self, manifest):
        for split in self.buffers:
            self.flush(split)
        with self.manifest.open("x", encoding="utf-8") as stream:
            self.created.append(self.manifest)
            json.dump(manifest, stream, indent=2, allow_nan=False)
            stream.write("\n")

    def cleanup(self):
        for path in self.created:
            path.unlink(missing_ok=True)


def extract_multitask(args):
    """Extract complete five-task shards and return their provenance manifest.

    Called by extract.py --tasks all. args includes its ordinary extraction
    options plus engine, engine_nodes, and mistake_threshold. No network access.
    """
    required_positive = ("max_games", "positions_per_game", "shard_size")
    if any(getattr(args, name) <= 0 for name in required_positive):
        raise ValueError("max-games, positions-per-game, and shard-size must be positive")
    if args.min_ply < 0 or args.history < 0:
        raise ValueError("min-ply/history must be nonnegative")
    engine_argument = getattr(args, "engine", None) or getattr(args, "engine_path", None)
    if not engine_argument:
        raise ValueError("--tasks all requires --engine pointing to a native Stockfish executable")
    nodes = int(getattr(args, "engine_nodes", 100_000))
    threshold = float(getattr(args, "mistake_threshold", .10))
    if nodes <= 0 or not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("engine-nodes must be positive; mistake-threshold must be in [0,1]")
    engine_path = Path(shutil.which(str(engine_argument)) or str(engine_argument)).expanduser().resolve()
    source = Path(args.input).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Missing source PGN: {source}")
    started = datetime.now(timezone.utc)
    print(f"Hashing source archive: {source}", flush=True)
    source_hash, engine_hash = _sha256(source), _sha256(engine_path)
    identities, errors = set(), []
    builder = _builder(args, identities)
    counts, rejected = Counter(), Counter()
    positions = Counter({split: 0 for split in ("train", "val", "test")})
    games = Counter({split: 0 for split in positions})
    label_counts = {task: Counter() for task in TASKS}
    writer = None
    try:
        with chess.engine.SimpleEngine.popen_uci(str(engine_path)) as engine:
            if "UCI_ShowWDL" not in engine.options or "stockfish" not in engine.id.get("name", "").lower():
                raise ValueError("Use Stockfish with native UCI_ShowWDL support")
            engine_config = {"UCI_ShowWDL": True, "Threads": 1, "Hash": 128}
            engine.configure(engine_config)
            engine_identity = dict(engine.id)
            writer = _Writer(args.out, args.shard_size, getattr(args, "overwrite", False))
            with source.open("rb") as raw:
                binary = zstd.ZstdDecompressor(max_window_size=2**31).stream_reader(raw) if str(source).endswith(".zst") else raw
                with io.TextIOWrapper(binary, encoding="utf-8", errors="strict") as stream:
                    while counts["kept_games"] < args.max_games:
                        game = chess.pgn.read_game(stream, Visitor=builder)
                        if game is None:
                            break
                        counts["parsed_games"] += 1
                        identity = _identity(game.headers)
                        reason = getattr(game, "_multitask_rejection", None)
                        if identity is not None:
                            identities.add(identity)
                        if reason is None and game.errors:
                            reason = "parse_error"
                        if reason is None and getattr(game, "_multitask_result", None) != game.headers.get("Result"):
                            reason = "incomplete_or_mismatched_result"
                        if reason is not None:
                            rejected[reason] += 1
                            continue
                        initial = game.board()
                        final = game.end().board()
                        if not initial.is_valid() or not final.is_valid():
                            rejected["invalid_board"] += 1
                            continue
                        if final.is_checkmate() and final.result() != game.headers["Result"]:
                            rejected["checkmate_result_mismatch"] += 1
                            continue
                        game_nodes = list(game.mainline())
                        n = len(game_nodes)
                        candidates = np.arange(args.min_ply, n - 1)
                        if not len(candidates):
                            rejected["too_short"] += 1
                            continue
                        split, code = split_of(game.headers)
                        rng = np.random.default_rng(code * 1_000_003 + counts["parsed_games"])
                        chosen = set(rng.choice(candidates, min(args.positions_per_game, len(candidates)), replace=False).tolist())
                        base, increment = parse_tc(game.headers)
                        term, term_mask = _termination(game, final)
                        elo = [int(game.headers["WhiteElo"]), int(game.headers["BlackElo"])]
                        board = initial
                        clocks, clock_valid = [0.0, 0.0], [False, False]
                        history = deque(maxlen=max(args.history, 1))
                        added = 0
                        for ply, node in enumerate(game_nodes):
                            if ply in chosen:
                                if not all(clock_valid):
                                    counts["positions_skipped_missing_current_clock"] += 1
                                elif not board.is_valid() or board.is_game_over():
                                    counts["positions_skipped_invalid_or_terminal"] += 1
                                else:
                                    mover = 0 if board.turn == chess.WHITE else 1
                                    future_clock = _clock(node)
                                    elapsed = clocks[mover] + increment - future_clock if future_clock is not None else None
                                    time_mask = elapsed is not None and np.isfinite(elapsed) and elapsed >= 0
                                    previous = np.zeros((args.history, 12), dtype=np.uint64)
                                    for index, old_board in enumerate(history if args.history else ()):
                                        previous[index] = old_board
                                    sample = dict(
                                        bb=bitboards(board), meta=encode_meta(board, ply), elo=elo,
                                        clk=clocks.copy(), clk_mask=clock_valid.copy(), tc=[base, increment],
                                        hist=previous, y=RESULT[game.headers["Result"]], game_id=identity,
                                        ply=ply, fen=board.fen(), remaining_plies=float(n - ply),
                                        next_move_seconds=float(elapsed) if time_mask else 0.0,
                                        time_mask=time_mask, termination_y=term, termination_mask=term_mask,
                                    )
                                    sample.update(_mistake_label(engine, board, node.move, nodes, threshold, counts, errors))
                                    writer.add(split, sample)
                                    positions[split] += 1
                                    added += 1
                                    for task, target, valid in (
                                        ("outcome", sample["y"], True), ("mistake", int(sample["mistake_y"]), sample["mistake_mask"]),
                                        ("moves", "observed", True), ("time", "observed", time_mask),
                                        ("termination", term, term_mask),
                                    ):
                                        label_counts[task]["valid" if valid else "masked"] += 1
                                        if valid:
                                            label_counts[task][str(target)] += 1
                            if ply >= max(chosen):
                                break
                            if args.history:
                                history.appendleft(bitboards(board))
                            observation = _clock(node)
                            side = 0 if board.turn == chess.WHITE else 1
                            clock_valid[side] = observation is not None
                            clocks[side] = observation if observation is not None else 0.0
                            board.push(node.move)
                        if added:
                            counts["kept_games"] += 1
                            games[split] += 1
                        else:
                            rejected["no_eligible_sampled_positions"] += 1
                        if counts["kept_games"] and counts["kept_games"] % 100 == 0:
                            print(f"Kept {counts['kept_games']:,} games / {sum(positions.values()):,} positions; "
                                  f"parsed {counts['parsed_games']:,}", flush=True)
            if not sum(positions.values()):
                raise ValueError("No sampled positions had both current clocks observed and valid")
            manifest = {
                "schema_version": 2, "tasks": list(TASKS),
                "created_utc": datetime.now(timezone.utc).isoformat(), "started_utc": started.isoformat(),
                "source": {"path": str(source), "sha256": source_hash, "bytes": source.stat().st_size},
                "class_order": ["white_win", "draw", "black_win"],
                "termination_classes": list(TERMINATION_CLASSES),
                "reference_loss_weights": REFERENCE_LOSS_WEIGHTS,
                "weights_note": "Reference proposal only; trainer records the weights actually used.",
                "task_labels": {
                    "outcome": "Actual recorded final result, fixed White perspective",
                    "mistake": "1 if best-minus-human native-engine expected score exceeds mistake_threshold; original mover POV",
                    "moves": "Remaining actual mainline plies: total plies minus sampled pre-move ply",
                    "time": "Prior observed same-mover post-move clock + increment - next post-move clock; seconds",
                    "termination": "Normal plus final checkmate -> checkmate; Time forfeit -> time_forfeit; other Normal -> other_normal",
                },
                "filters": {"min_elo": args.min_elo, "min_tc": args.min_tc,
                            "standard_only": True, "exclude_bots": True, "unique_game_ids": True,
                            "complete_error_free_pgn": True, "require_both_current_clocks_observed": True},
                "sampling": {"positions_per_game": args.positions_per_game, "min_ply": args.min_ply,
                             "history": args.history, "max_games": args.max_games,
                             "seed": "(CRC32(Site)%100)*1000003 + parsed_game_index", "split": "90/5/5 by CRC32(Site)%100"},
                "engine": {"path": str(engine_path), "sha256": engine_hash, "id": engine_identity,
                           "configuration": engine_config, "nodes_per_search": nodes,
                           "mistake_threshold": threshold, "negative_disagreement_tolerance": NEGATIVE_ENGINE_TOLERANCE,
                           "state_reset": "Unique game token / ucinewgame for each search", "native_wdl_only": True},
                "counts": dict(counts), "rejected_games": dict(rejected),
                "split_positions": dict(positions), "split_games": dict(games),
                "label_counts": {task: dict(values) for task, values in label_counts.items()},
                "engine_error_examples": errors,
                "missingness": "Masked targets contain numeric placeholders, never valid labels; use masks. First-move timing is excluded.",
                "limitations": ["Finite-search engine supervision is not perfect-play truth.",
                                "other_normal is not a resignation label; it includes draws and other normal endings.",
                                "No clock absence is replaced by a stale feature; sampled positions with unavailable current clocks are excluded."],
            }
            writer.finish(manifest)
    except BaseException:
        if writer is not None:
            writer.cleanup()
        raise
    print(f"Saved five-task dataset: {sum(positions.values()):,} positions; manifest: {writer.manifest}", flush=True)
    return manifest


class MultiTaskPositions(LichessPositions):
    """Existing board/context encoding plus aligned raw targets and availability masks."""
    def __init__(self, root, split="train", features=("board",), history=8):
        if isinstance(features, str):
            features = parse_features(features)
        super().__init__(root, split, features=features, history=history)
        self.root, self.split = Path(root), split
        required = ("game_id", "ply", "fen", "mistake_y", "mistake_mask", "remaining_plies",
                    "next_move_seconds", "time_mask", "termination_y", "termination_mask")
        buffers = {key: [] for key in required}
        for path in sorted((self.root / split).glob("shard_*.npz")):
            with np.load(path, allow_pickle=False) as arrays:
                missing = set(required) - set(arrays.files)
                if missing:
                    raise ValueError(f"{path} lacks five-task labels {sorted(missing)}; re-extract with --tasks all")
                for key in required:
                    buffers[key].append(arrays[key])
        values = {key: np.concatenate(parts) for key, parts in buffers.items()}
        if any(len(value) != len(self.y) for value in values.values()):
            raise ValueError("Multi-task labels and existing board/context rows are misaligned")
        self.game_ids = self.game_id = values["game_id"]
        self.fens = self.fen = values["fen"]
        self.ply = values["ply"]
        self.targets = {"outcome": self.y.astype(np.int64), "mistake": values["mistake_y"].astype(np.float32),
                        "moves": values["remaining_plies"].astype(np.float32),
                        "time": values["next_move_seconds"].astype(np.float32),
                        "termination": values["termination_y"].astype(np.int64)}
        self.masks = {"outcome": np.ones(len(self.y), dtype=bool),
                      "mistake": values["mistake_mask"].astype(bool),
                      "moves": np.ones(len(self.y), dtype=bool),
                      "time": values["time_mask"].astype(bool),
                      "termination": values["termination_mask"].astype(bool)}
        for task, target in self.targets.items():
            selected = self.masks[task]
            invalid = ~np.isfinite(target)
            if task in ("moves", "time"):
                invalid |= target < 0
            elif task == "mistake":
                invalid |= ~np.isin(target, [0, 1])
            else:
                invalid |= ~np.isin(target, [0, 1, 2])
            if invalid[selected].any():
                raise ValueError(f"Invalid observed {task} labels in {root}/{split}")
        self.termination_classes = TERMINATION_CLASSES
        manifest = self.root / "manifest.json"
        self.manifest = json.loads(manifest.read_text()) if manifest.exists() else None

    def __getitem__(self, index):
        x, scalars, _ = super().__getitem__(index)
        targets = {task: torch.as_tensor(values[index]) for task, values in self.targets.items()}
        masks = {task: torch.tensor(bool(values[index]), dtype=torch.bool) for task, values in self.masks.items()}
        return x, scalars, targets, masks


def training_normalization(dataset):
    """Training-only log1p mean/std, raw medians, and explicit zero-variance guards."""
    if getattr(dataset, "split", None) != "train":
        raise ValueError("Fit target normalization on the training split only")
    for task in TASKS:
        if not np.asarray(dataset.masks[task], dtype=bool).any():
            raise ValueError(f"Five-task training requires at least one valid training label for {task}")
    normalization = {}
    for task in ("moves", "time"):
        raw = np.asarray(dataset.targets[task], dtype=float)[dataset.masks[task]]
        logged = np.log1p(raw)
        standard_deviation = float(logged.std())
        guarded = not np.isfinite(standard_deviation) or standard_deviation < 1e-8
        normalization[task] = {"mean": float(logged.mean()), "std": 1.0 if guarded else standard_deviation,
                               "median": float(np.median(raw)), "n_valid": len(raw),
                               "std_guarded": bool(guarded), "transform": "log1p"}
    return normalization


compute_task_normalization = training_normalization
