import os
import shutil

import chess
import pytest

import config
from scoring import difficulty_scores, expected_score, rating_grid

HAVE_ENGINE = os.path.exists(config.STOCKFISH_PATH) or shutil.which("stockfish")


def test_formula_sign_and_values():
    # Engine says 90% win, 10% draw -> score 0.95. Model says 50/20/30 -> score 0.6.
    d, dw = difficulty_scores(0.9, 0.1, 0.5, 0.2)
    assert d == pytest.approx(0.35) and dw == pytest.approx(0.4)      # positive: humans do worse
    d, dw = difficulty_scores(0.2, 0.2, 0.5, 0.2)
    assert d < 0 and dw < 0                                           # humans do better than engine
    d, dw = difficulty_scores(0.4, 0.2, 0.4, 0.2)
    assert d == 0 and dw == 0
    assert expected_score(0.0, 1.0) == 0.5


def test_rating_grid():
    assert rating_grid(1300, 1700, 100) == [1300, 1400, 1500, 1600, 1700]


needs_all = pytest.mark.skipif(not (HAVE_ENGINE and os.path.exists(config.MODEL_PATH)),
                               reason="needs Stockfish and model.pt")


@pytest.fixture(scope="module")
def tools():
    from outcome_model import load_outcome_model
    from stockfish_wdl import StockfishWDL
    with StockfishWDL(nodes=50_000) as sf:
        yield sf, load_outcome_model()


@needs_all
def test_completely_equal_position_has_small_difficulty(tools):
    from scoring import difficulty_for_boards
    sf, model = tools
    boards = [chess.Board(), chess.Board("rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2")]
    for r in difficulty_for_boards(boards, 1600, sf, model):
        assert r["status"] == "ok" and abs(r["sf"][0] - r["sf"][2]) < 0.1   # engine: roughly level
        assert abs(r["difficulty"]) < 0.15


@needs_all
def test_simple_winning_endgame(tools):
    from scoring import difficulty_for_boards
    sf, model = tools
    # Queen up, either side to move: engine is certain; the side to move is the winner / the loser.
    win_stm, lose_stm = difficulty_for_boards(
        [chess.Board("4k3/8/8/8/8/8/8/3QK3 w - - 0 1"), chess.Board("4k3/8/8/8/8/8/8/3QK3 b - - 0 1")],
        1600, sf, model)
    assert win_stm["sf"][0] > 0.95 and lose_stm["sf"][2] > 0.95
    # humans cannot beat a certain engine result, so the side that should win is not "easier than the engine"
    assert win_stm["difficulty"] > -0.02
    # the losing side to move: difficulty (engine score - model score) should be <= ~0, not hard to "play"
    assert lose_stm["difficulty"] < 0.1


@needs_all
def test_mirrored_position_gives_similar_difficulty(tools):
    from scoring import difficulty_for_boards
    sf, model = tools
    b = chess.Board("r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4")
    a, m = difficulty_for_boards([b, b.mirror()], 1600, sf, model)
    assert a["sf"][0] == pytest.approx(m["sf"][0], abs=0.03)
    assert abs(a["difficulty"] - m["difficulty"]) < 0.15
