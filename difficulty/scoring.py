"""Position difficulty = how much worse humans do than Stockfish says they should.

    difficulty       = stockfish_score - model_score          (score = win + 0.5 * draw)
    difficulty_win   = stockfish_win   - model_win            (win chance only)

Both from the SIDE TO MOVE's point of view. Positive = humans of that rating do WORSE than the engine
expects (the position is hard to play); negative = humans do better than the engine's number.
"""
import numpy as np


def expected_score(win, draw):
    return np.asarray(win) + 0.5 * np.asarray(draw)


def difficulty_scores(sf_win, sf_draw, model_win, model_draw):
    """Return (difficulty, difficulty_win). Inputs are arrays or scalars, same shape."""
    d = expected_score(sf_win, sf_draw) - expected_score(model_win, model_draw)
    dw = np.asarray(sf_win) - np.asarray(model_win)
    return d, dw


def rating_grid(lo, hi, step):
    return list(range(int(lo), int(hi) + 1, int(step)))


def difficulty_for_boards(boards, rating, sf, model, opp_rating=None):
    """Convenience for single positions / notebooks.

    boards: list of chess.Board (ply taken from the fullmove number and turn); sf: StockfishWDL;
    model: OutcomeModel. Returns a list of dicts with Stockfish + model WDL and both difficulties
    (None where Stockfish status is not "ok").
    """
    import chess
    from outcome_model import samples_from_boards
    plies = [2 * (b.fullmove_number - 1) + (0 if b.turn == chess.WHITE else 1) for b in boards]
    pred = model.predict(samples_from_boards(list(zip(boards, plies))), rating, opp_rating)
    out = []
    for k, b in enumerate(boards):
        r = sf.evaluate(b)
        row = {"status": r.status, "sf": (r.win, r.draw, r.loss),
               "model": (float(pred.win[k]), float(pred.draw[k]), float(pred.loss[k])),
               "difficulty": None, "difficulty_win": None}
        if r.status == "ok":
            d, dw = difficulty_scores(r.win, r.draw, pred.win[k], pred.draw[k])
            row["difficulty"], row["difficulty_win"] = float(d), float(dw)
        out.append(row)
    return out
