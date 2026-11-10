"""
Stream a Lichess .pgn.zst file into sharded numpy arrays of (position, outcome) samples.

    pip install python-chess zstandard numpy
    python extract.py lichess_db_standard_rated_2024-01.pgn.zst --out data --max-games 2000000

Each sample is one board position sampled from a game, labelled with that game's final result.
Splits are assigned per GAME (hash of the game id), so positions from one game never
appear in both train and test. Games without [%clk] clock comments are skipped, so every
sample carries all features and ablations (see ablate.py) compare models on identical data.

Output layout:  data/{train,val,test}/shard_00000.npz  with arrays
    bb   (N, 12)    uint64   bitboards: white P N B R Q K, then black P N B R Q K (bit i = square i, a1=0)
    meta (N, 7)     uint8    [white_to_move, wK_castle, wQ_castle, bK_castle, bQ_castle, ep_file(8=none), ply]
    elo  (N, 2)     int16    [white_elo, black_elo]
    clk  (N, 2)     float32  [white, black] seconds left on the clock at this position
    tc   (N, 2)     int32    [base seconds, increment seconds]
    hist (N, H, 12) uint64   bitboards of the H previous positions, most recent first (zeros before ply 0)
    y    (N,)       uint8    0 = white wins, 1 = draw, 2 = black wins
"""
import argparse
import glob
import io
import os
import zlib
from collections import deque

import chess
import chess.pgn
import numpy as np
import zstandard as zstd

RESULT = {"1-0": 0, "1/2-1/2": 1, "0-1": 2}


def parse_tc(h):
    base, inc = h["TimeControl"].split("+")  # "-" (correspondence) raises ValueError
    return int(base), int(inc)


def accept(h, a):
    """Header-only filter, so rejected games are skipped without parsing their moves."""
    if h.get("Result") not in RESULT:
        return False
    if h.get("Termination") not in ("Normal", "Time forfeit"):  # drops Abandoned / Rules infraction
        return False
    try:
        we, be = int(h["WhiteElo"]), int(h["BlackElo"])
        base, inc = parse_tc(h)
    except (KeyError, ValueError):
        return False
    tc = base + 40 * inc  # estimated game duration in seconds
    return min(we, be) >= a.min_elo and tc >= a.min_tc


def make_builder(a):
    class FilteredBuilder(chess.pgn.GameBuilder):
        def end_headers(self):
            return None if accept(self.game.headers, a) else chess.pgn.SKIP

    return FilteredBuilder


def bitboards(board):
    bb = np.empty(12, np.uint64)
    for color in (chess.WHITE, chess.BLACK):
        off = 0 if color == chess.WHITE else 6
        for pt in range(1, 7):
            bb[off + pt - 1] = board.pieces_mask(pt, color)
    return bb


def encode_meta(board, ply):
    ep = chess.square_file(board.ep_square) if board.ep_square is not None else 8
    return np.array([
        board.turn == chess.WHITE,
        board.has_kingside_castling_rights(chess.WHITE),
        board.has_queenside_castling_rights(chess.WHITE),
        board.has_kingside_castling_rights(chess.BLACK),
        board.has_queenside_castling_rights(chess.BLACK),
        ep,
        min(ply, 255),
    ], dtype=np.uint8)


def sample_positions(game, k, min_ply, n_hist, base, rng):
    """Yields (bb, meta, clk, hist) for up to k random positions of the game."""
    nodes = list(game.mainline())  # nodes[i].move is the move played from the position at ply i
    n = len(nodes)
    # Last position is excluded: it's usually mate/resignation/flag, which leaks the label.
    candidates = np.arange(min_ply, n - 1)
    if len(candidates) == 0:
        return
    chosen = set(rng.choice(candidates, size=min(k, len(candidates)), replace=False).tolist())
    first, last = min(chosen), max(chosen)
    board = game.board()
    clk = [float(base), float(base)]  # white, black: clock after each side's latest move
    prev = deque(maxlen=max(n_hist, 1))
    for i, node in enumerate(nodes):
        if i in chosen:
            hist = np.zeros((n_hist, 12), np.uint64)
            for j, b in enumerate(prev if n_hist else ()):
                hist[j] = b
            yield bitboards(board), encode_meta(board, i), np.array(clk, np.float32), hist
        if i >= last:
            break
        if n_hist and i >= first - n_hist:
            prev.appendleft(bitboards(board))
        c = node.clock()
        if c is not None:
            clk[0 if board.turn == chess.WHITE else 1] = c
        board.push(node.move)


class ShardWriter:
    KEYS = ("bb", "meta", "elo", "clk", "tc", "hist", "y")

    def __init__(self, out, shard_size, overwrite=False):
        self.out, self.shard_size = out, shard_size
        old = sorted(glob.glob(os.path.join(out, "*", "shard_*.npz")))
        if old and not overwrite:
            raise SystemExit(f"{out} already has {len(old)} shards; new shards would overwrite only "
                             f"some of them and leave a mix of two datasets. Use a new --out, or "
                             f"--overwrite to delete the old shards first.")
        for f in old:
            os.remove(f)
        self.buf = {s: {k: [] for k in self.KEYS} for s in ("train", "val", "test")}
        self.count = {s: 0 for s in self.buf}
        for s in self.buf:
            os.makedirs(os.path.join(out, s), exist_ok=True)

    def add(self, split, **sample):
        b = self.buf[split]
        for k in self.KEYS:
            b[k].append(sample[k])
        if len(b["y"]) >= self.shard_size:
            self.flush(split)

    def flush(self, split):
        b = self.buf[split]
        if not b["y"]:
            return
        path = os.path.join(self.out, split, f"shard_{self.count[split]:05d}.npz")
        np.savez(path,
                 bb=np.stack(b["bb"]), meta=np.stack(b["meta"]),
                 elo=np.array(b["elo"], np.int16), clk=np.stack(b["clk"]),
                 tc=np.array(b["tc"], np.int32), hist=np.stack(b["hist"]),
                 y=np.array(b["y"], np.uint8))
        self.count[split] += 1
        for v in b.values():
            v.clear()

    def close(self):
        for s in self.buf:
            self.flush(s)


def split_of(h):
    code = zlib.crc32(h.get("Site", "").encode()) % 100
    return "train" if code < 90 else ("val" if code < 95 else "test"), code


def main():
    p = argparse.ArgumentParser()
    p.add_argument("input")
    p.add_argument("--out", default="data")
    p.add_argument("--max-games", type=int, default=1_000_000, help="accepted games to keep")
    p.add_argument("--positions-per-game", type=int, default=4)
    p.add_argument("--min-ply", type=int, default=8, help="skip the first N plies (opening noise)")
    p.add_argument("--min-elo", type=int, default=1200)
    p.add_argument("--min-tc", type=int, default=180, help="min base+40*inc seconds; 180 drops bullet")
    p.add_argument("--history", type=int, default=8, help="previous positions stored per sample")
    p.add_argument("--shard-size", type=int, default=500_000)
    p.add_argument("--overwrite", action="store_true", help="delete existing shards in --out first")
    p.add_argument("--tasks", choices=("outcome", "all"), default="outcome")
    p.add_argument("--engine", help="Stockfish executable, required for --tasks all")
    p.add_argument("--engine-nodes", type=int, default=100_000,
                   help="fixed search budget per engine analysis used for mistake labels")
    p.add_argument("--mistake-threshold", type=float, default=.10,
                   help="positive mistake if engine expected-score loss exceeds this value")
    a = p.parse_args()
    if a.tasks == "all":
        from multitask_data import extract_multitask
        extract_multitask(a)
        return

    writer = ShardWriter(a.out, a.shard_size, a.overwrite)
    Builder = make_builder(a)
    kept = seen = no_clock = 0

    with open(a.input, "rb") as f:
        reader = zstd.ZstdDecompressor(max_window_size=2**31).stream_reader(f)
        text = io.TextIOWrapper(reader, encoding="utf-8", errors="replace")
        while kept < a.max_games:
            game = chess.pgn.read_game(text, Visitor=Builder)
            if game is None:
                break
            seen += 1
            if seen % 100_000 == 0:
                print(f"parsed {seen:,} games, kept {kept:,}", flush=True)
            first = game.next()
            if first is None:  # skipped by header filter (or empty)
                continue
            if first.clock() is None:  # no [%clk] comments: can't build time features
                no_clock += 1
                continue
            h = game.headers
            split, code = split_of(h)
            rng = np.random.default_rng(code * 1_000_003 + seen)
            elo = (int(h["WhiteElo"]), int(h["BlackElo"]))
            tc = parse_tc(h)
            y = RESULT[h["Result"]]
            got = False
            for bb, meta, clk, hist in sample_positions(
                    game, a.positions_per_game, a.min_ply, a.history, tc[0], rng):
                writer.add(split, bb=bb, meta=meta, elo=elo, clk=clk, tc=tc, hist=hist, y=y)
                got = True
            kept += got

    writer.close()
    print(f"done: parsed {seen:,} games, kept {kept:,} (skipped {no_clock:,} without clocks); "
          f"shards: {writer.count}")


if __name__ == "__main__":
    main()
