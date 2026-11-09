"""Rebuild a full python-chess Board from one stored sample (the format written by extract.py).

Stored per sample: bb (12,) uint64 piece bitboards, meta (7,) uint8
[white_to_move, wK, wQ, bK, bQ castling, ep_file (8 = none), ply].

NOT recoverable from the stored format (and what we do about it):
  * halfmove clock (50-move counter): set to 0. Stockfish then never sees a position as "close to a
    50-move draw". Its WDL estimate depends mainly on material, so the effect is small, but it is a
    known approximation.
  * move history / repetitions: gone. Positions are treated as fresh, so repetition draws are invisible.
  * exact en-passant square: only the file is stored. The rank is implied by the side to move
    (rank 6 if white to move, rank 3 if black), so this one IS fully recovered.
  * ply is capped at 255 by extract.py; for such samples the fullmove number is only a lower bound
    (flagged as "ply_saturated").
  * the starting position of the game: games that began from a custom FEN could break the
    "white to move <=> even ply" rule; those are flagged as "turn_ply_mismatch".
"""
from dataclasses import dataclass, field

import chess
import numpy as np

import config  # noqa: F401  (puts repo root on sys.path)

_CASTLE_SQUARES = (chess.H1, chess.A1, chess.H8, chess.A8)  # wK, wQ, bK, bQ


@dataclass
class Decoded:
    board: chess.Board
    ply: int
    issues: list = field(default_factory=list)  # empty list = clean

    @property
    def ok(self):
        return not any(i.startswith("illegal") for i in self.issues)


def decode_sample(bb, meta):
    """Return Decoded(board, ply, issues). Never raises on odd contents; problems go in `issues`."""
    bb = np.asarray(bb, dtype=np.uint64)
    meta = np.asarray(meta, dtype=np.uint8)
    issues = []
    board = chess.Board(None)  # empty board
    occupied = 0
    for i in range(12):
        color = chess.WHITE if i < 6 else chess.BLACK
        mask = int(bb[i])
        if mask & occupied:
            issues.append("illegal:overlapping_pieces")
        occupied |= mask
        for sq in chess.scan_forward(mask):
            board.set_piece_at(sq, chess.Piece(i % 6 + 1, color))
    white_to_move, ply = bool(meta[0]), int(meta[6])
    board.turn = chess.WHITE if white_to_move else chess.BLACK
    rights = 0
    for flag, sq in zip(meta[1:5], _CASTLE_SQUARES):
        if flag:
            rights |= chess.BB_SQUARES[sq]
    board.castling_rights = rights
    ep_file = int(meta[5])
    if ep_file < 8:
        board.ep_square = chess.square(ep_file, 5 if white_to_move else 2)
    board.halfmove_clock = 0
    board.fullmove_number = ply // 2 + 1
    if white_to_move != (ply % 2 == 0):
        issues.append("turn_ply_mismatch")
    if ply >= 255:
        issues.append("ply_saturated")
    status = board.status()
    if status != chess.STATUS_VALID:
        issues.append(f"illegal:status_{int(status)}")
    return Decoded(board, ply, issues)


def encode_board(board, ply):
    """Board back to the stored format (delegates to extract.encode so it can never drift)."""
    import extract
    return extract.encode(board, ply)


def roundtrip_matches(decoded, bb, meta):
    """True if re-encoding the rebuilt board gives back exactly the stored bb and meta."""
    bb2, meta2 = encode_board(decoded.board, decoded.ply)
    return bool(np.array_equal(bb2, np.asarray(bb, np.uint64))
                and np.array_equal(meta2, np.asarray(meta, np.uint8)))


def verify_roundtrip(bb_all, meta_all, n=5000, seed=0):
    """Check n random samples. Returns (n_checked, n_mismatch, n_with_issues, first_bad_indices)."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(bb_all), size=min(n, len(bb_all)), replace=False)
    bad, flagged = [], 0
    for i in idx:
        d = decode_sample(bb_all[i], meta_all[i])
        flagged += bool(d.issues)
        if not roundtrip_matches(d, bb_all[i], meta_all[i]):
            bad.append(int(i))
    return len(idx), len(bad), flagged, bad[:10]
