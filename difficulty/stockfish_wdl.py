"""Stockfish win/draw/loss probabilities from the SIDE TO MOVE's point of view.

    with StockfishWDL() as sf:
        r = sf.evaluate(board)      # r.status, r.win, r.draw, r.loss, r.score, r.mate_in, ...

Search budget is a fixed NODE count (config.STOCKFISH_NODES), 1 thread, and a fresh game state per
position, so repeated runs give the same numbers. Note these are Stockfish's *engine-strength* WDL
estimates (its UCI_ShowWDL model), not the outcome of human games.

status values:
  ok              normal evaluation (check mate_in for forced mates: >0 side to move mates, <0 gets mated)
  checkmate       side to move is already mated      -> win 0, draw 0, loss 1 (exact, no search)
  stalemate       side to move has no moves, no check -> 0, 1, 0 (exact)
  draw_rule       insufficient material              -> 0, 1, 0 (exact)
  illegal         position is not a legal chess position (not searched; probabilities None)
  engine_error    Stockfish crashed or returned no WDL twice in a row (probabilities None)
"""
import time
from dataclasses import dataclass
from typing import Optional

import chess
import chess.engine

import config


@dataclass
class EngineResult:
    status: str
    win: Optional[float] = None    # probabilities, side to move's point of view, sum to 1
    draw: Optional[float] = None
    loss: Optional[float] = None
    mate_in: Optional[int] = None  # signed, side to move's point of view
    cp: Optional[int] = None       # centipawns, side to move's point of view (None for mate scores)
    depth: Optional[int] = None
    nodes: Optional[int] = None
    seconds: float = 0.0
    detail: str = ""

    @property
    def score(self):
        """Expected score = win + 0.5 * draw."""
        return None if self.win is None else self.win + 0.5 * self.draw


class StockfishWDL:
    def __init__(self, path=None, nodes=None, hash_mb=None, threads=None):
        self.path = path or config.STOCKFISH_PATH
        self.nodes = nodes or config.STOCKFISH_NODES
        self.hash_mb = hash_mb or config.STOCKFISH_HASH_MB
        self.threads = threads or config.STOCKFISH_THREADS
        self.engine = None
        self.name = None

    def __enter__(self):
        self._start()
        return self

    def __exit__(self, *exc):
        self.close()

    def _start(self):
        try:
            self.engine = chess.engine.SimpleEngine.popen_uci(self.path)
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"Stockfish not found at '{self.path}'. Install it and set $STOCKFISH_PATH "
                f"or pass --stockfish.") from e
        self.name = self.engine.id.get("name", "unknown")
        self.engine.configure({"Threads": self.threads, "Hash": self.hash_mb, "UCI_ShowWDL": True})

    def close(self):
        if self.engine is not None:
            try:
                self.engine.quit()
            except Exception:
                pass
            self.engine = None

    def evaluate(self, board):
        t0 = time.time()
        res = self._evaluate(board)
        res.seconds = time.time() - t0
        return res

    def _evaluate(self, board):
        status = board.status()
        if status != chess.STATUS_VALID:
            return EngineResult("illegal", detail=f"python-chess status flags {int(status)}")
        if board.is_checkmate():
            return EngineResult("checkmate", 0.0, 0.0, 1.0, mate_in=0)
        if board.is_stalemate():
            return EngineResult("stalemate", 0.0, 1.0, 0.0)
        if board.is_insufficient_material():
            return EngineResult("draw_rule", 0.0, 1.0, 0.0, detail="insufficient material")
        last = ""
        for attempt in range(2):
            try:
                if self.engine is None:
                    self._start()
                # game=object() makes python-chess send ucinewgame => clean hash => repeatable.
                info = self.engine.analyse(board, chess.engine.Limit(nodes=self.nodes), game=object())
                wdl = info.get("wdl")
                if wdl is None:
                    last = "no wdl in engine output"
                    continue
                w, d, l = wdl.pov(board.turn)
                total = float(w + d + l)
                score = info["score"].pov(board.turn)
                return EngineResult(
                    "ok", w / total, d / total, l / total,
                    mate_in=score.mate(), cp=score.score(),
                    depth=info.get("depth"), nodes=info.get("nodes"))
            except (chess.engine.EngineTerminatedError, chess.engine.EngineError, OSError) as e:
                last = f"{type(e).__name__}: {e}"
                self.close()  # restart on next attempt
        return EngineResult("engine_error", detail=last)
