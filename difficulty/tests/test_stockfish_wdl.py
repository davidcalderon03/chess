import os

import chess
import pytest

import config
from stockfish_wdl import StockfishWDL

pytestmark = pytest.mark.skipif(not os.path.exists(config.STOCKFISH_PATH) and not __import__("shutil").which("stockfish"),
                                reason="Stockfish not installed")


@pytest.fixture(scope="module")
def sf():
    with StockfishWDL(nodes=20_000) as e:
        yield e


def test_perspective_is_side_to_move(sf):
    white_up_queen = "4k3/8/8/8/8/8/8/3QK3"
    a = sf.evaluate(chess.Board(white_up_queen + " w - - 0 1"))   # winning side to move
    b = sf.evaluate(chess.Board(white_up_queen + " b - - 0 1"))   # losing side to move
    assert a.win > 0.95 and b.loss > 0.95
    black_up_queen = "3qk3/8/8/8/8/8/8/4K3"
    c = sf.evaluate(chess.Board(black_up_queen + " b - - 0 1"))
    assert c.win > 0.95


def test_probabilities_sum_to_one_and_repeatable(sf):
    b = chess.Board()
    r1, r2 = sf.evaluate(b), sf.evaluate(b)
    assert abs(r1.win + r1.draw + r1.loss - 1) < 1e-9
    assert (r1.win, r1.draw, r1.nodes) == (r2.win, r2.draw, r2.nodes)


def test_edge_cases_do_not_crash(sf):
    assert sf.evaluate(chess.Board("7k/6Q1/6K1/8/8/8/8/8 b - - 0 1")).status == "checkmate"
    r = sf.evaluate(chess.Board("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1"))
    assert r.status == "stalemate" and r.draw == 1.0
    assert sf.evaluate(chess.Board("8/8/8/8/8/8/8/8 w - - 0 1")).status == "illegal"
    assert sf.evaluate(chess.Board("8/8/8/8/8/8/k7/K7 w - - 0 1")).status in ("draw_rule", "illegal")
    m = sf.evaluate(chess.Board("6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1"))
    assert m.status == "ok" and m.mate_in == 1


def test_missing_binary_gives_clear_error():
    with pytest.raises(FileNotFoundError, match="Stockfish not found"):
        StockfishWDL(path="/nonexistent/stockfish").__enter__()
