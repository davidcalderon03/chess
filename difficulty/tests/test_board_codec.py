import chess
import numpy as np

from board_codec import decode_sample, encode_board, roundtrip_matches, verify_roundtrip
from data_utils import load_split


def random_game_positions(n_games=40, seed=1):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_games):
        b = chess.Board()
        for ply in range(int(rng.integers(0, 120))):
            moves = list(b.legal_moves)
            if not moves:
                break
            b.push(moves[int(rng.integers(len(moves)))])
            out.append((b.copy(), ply + 1))
    return out


def test_roundtrip_on_random_games():
    for board, ply in random_game_positions():
        bb, meta = encode_board(board, ply)
        d = decode_sample(bb, meta)
        assert roundtrip_matches(d, bb, meta)
        assert d.board.turn == board.turn and d.board.castling_rights == board.castling_rights


def test_roundtrip_on_real_test_split():
    try:
        d = load_split("test")
    except FileNotFoundError:
        import pytest
        pytest.skip("no data")
    checked, mismatches, flagged, _ = verify_roundtrip(d["bb"], d["meta"], 3000)
    assert checked == 3000 and mismatches == 0 and flagged == 0


def test_en_passant_rank_and_clock_reset():
    b = chess.Board()
    for m in ("e2e4", "a7a6", "e4e5", "d7d5"):
        b.push_uci(m)
    dec = decode_sample(*encode_board(b, 4))
    assert dec.board.ep_square == chess.D6          # rank implied by side to move
    assert dec.board.halfmove_clock == 0            # documented: not recoverable


def test_illegal_positions_are_flagged_not_raised():
    bb = np.zeros(12, np.uint64)
    meta = np.array([1, 0, 0, 0, 0, 8, 0], np.uint8)  # empty board
    dec = decode_sample(bb, meta)
    assert not dec.ok
    bb[0] = np.uint64(1); bb[6] = np.uint64(1)         # white pawn and black pawn on same square
    assert "illegal:overlapping_pieces" in decode_sample(bb, meta).issues
