"""
Stream a Lichess .pgn.zst file into sharded numpy arrays of (position, outcome) samples.

    pip install python-chess zstandard numpy
    python extract.py lichess_db_standard_rated_2024-01.pgn.zst --out data --max-games 2000000

Each sample is one board position sampled from a game, labelled with that game's final result.
Splits are assigned per GAME (hash of the game id), so positions from one game never
appear in both train and test.

Output layout:  data/{train,val,test}/shard_00000.npz  with arrays
    bb   (N, 12) uint64   bitboards: white P N B R Q K, then black P N B R Q K (bit i = square i, a1=0)
    meta (N, 7)  uint8    [white_to_move, wK_castle, wQ_castle, bK_castle, bQ_castle, ep_file(8=none), ply]
    elo  (N, 2)  int16    [white_elo, black_elo]
    y    (N,)    uint8    0 = white wins, 1 = draw, 2 = black wins
"""
import argparse
import io
import os
import zlib

import chess
import chess.pgn
import numpy as np
import zstandard as zstd

RESULT = {"1-0": 0, "1/2-1/2": 1, "0-1": 2}


def accept(h, a):
    """Header-only filter, so rejected games are skipped without parsing their moves."""
    if h.get("Result") not in RESULT:
        return False
    if h.get("Termination") not in ("Normal", "Time forfeit"):  # drops Abandoned / Rules infraction
        return False
    try:
        we, be = int(h["WhiteElo"]), int(h["BlackElo"])
        base, inc = h["TimeControl"].split("+")  # "-" (correspondence) raises ValueError
        tc = int(base) + 40 * int(inc)  # estimated game duration in seconds
    except (KeyError, ValueError):
        return False
    return min(we, be) >= a.min_elo and tc >= a.min_tc


def make_builder(a):
    class FilteredBuilder(chess.pgn.GameBuilder):
        def end_headers(self):
            return None if accept(self.game.headers, a) else chess.pgn.SKIP

        def visit_comment(self, comment):  # ignore [%clk]/[%eval] comments for speed
            pass

    return FilteredBuilder


def encode(board, ply):
    bb = np.empty(12, np.uint64)
    for color in (chess.WHITE, chess.BLACK):
        off = 0 if color == chess.WHITE else 6
        for pt in range(1, 7):
            bb[off + pt - 1] = board.pieces_mask(pt, color)
    ep = chess.square_file(board.ep_square) if board.ep_square is not None else 8
    meta = np.array([
        board.turn == chess.WHITE,
        board.has_kingside_castling_rights(chess.WHITE),
        board.has_queenside_castling_rights(chess.WHITE),
        board.has_kingside_castling_rights(chess.BLACK),
        board.has_queenside_castling_rights(chess.BLACK),
        ep,
        min(ply, 255),
    ], dtype=np.uint8)
    return bb, meta


def sample_positions(game, k, min_ply, rng):
    moves = list(game.mainline_moves())
    n = len(moves)
    # Last position is excluded: it's usually mate/resignation/flag, which leaks the label.
    candidates = np.arange(min_ply, n - 1)
    if len(candidates) == 0:
        return
    chosen = set(rng.choice(candidates, size=min(k, len(candidates)), replace=False).tolist())
    board = game.board()
    for i, mv in enumerate(moves):
        if i in chosen:
            yield encode(board, i)
        if i >= max(chosen):
            break
        board.push(mv)


class ShardWriter:
    def __init__(self, out, shard_size):
        self.out, self.shard_size = out, shard_size
        self.buf = {s: {"bb": [], "meta": [], "elo": [], "y": []} for s in ("train", "val", "test")}
        self.count = {s: 0 for s in self.buf}
        for s in self.buf:
            os.makedirs(os.path.join(out, s), exist_ok=True)

    def add(self, split, bb, meta, elo, y):
        b = self.buf[split]
        b["bb"].append(bb); b["meta"].append(meta); b["elo"].append(elo); b["y"].append(y)
        if len(b["y"]) >= self.shard_size:
            self.flush(split)

    def flush(self, split):
        b = self.buf[split]
        if not b["y"]:
            return
        path = os.path.join(self.out, split, f"shard_{self.count[split]:05d}.npz")
        np.savez(path,
                 bb=np.stack(b["bb"]), meta=np.stack(b["meta"]),
                 elo=np.array(b["elo"], np.int16), y=np.array(b["y"], np.uint8))
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
    p.add_argument("--shard-size", type=int, default=500_000)
    a = p.parse_args()

    writer = ShardWriter(a.out, a.shard_size)
    Builder = make_builder(a)
    kept = seen = 0

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
            if game.next() is None:  # skipped by header filter (or empty)
                continue
            h = game.headers
            split, code = split_of(h)
            rng = np.random.default_rng(code * 1_000_003 + seen)
            elo = (int(h["WhiteElo"]), int(h["BlackElo"]))
            y = RESULT[h["Result"]]
            n_before = kept
            for bb, meta in sample_positions(game, a.positions_per_game, a.min_ply, rng):
                writer.add(split, bb, meta, elo, y)
                kept = n_before + 1
            
    writer.close()
    print(f"done: parsed {seen:,} games, kept {kept:,}; shards: {writer.count}")


if __name__ == "__main__":
    main()