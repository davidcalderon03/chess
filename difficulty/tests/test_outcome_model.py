import os

import chess
import numpy as np
import pytest

import config
from outcome_model import ModelAdapter, OutcomeModel, load_outcome_model, samples_from_boards


class FakeAdapter(ModelAdapter):
    """Always says: white wins 70%, draw 20%, black wins 10%. Needs clocks (to test defaults)."""
    name = "fake"
    required_context = ("white_clock_s", "time_control")

    def __init__(self, uses_ratings=True):
        self.uses_ratings = uses_ratings
        self.seen = None

    def predict_white_wdl(self, samples, white_elo, black_elo, context):
        self.seen = (white_elo.copy(), black_elo.copy(), dict(context))
        return np.tile([0.7, 0.2, 0.1], (len(samples), 1)), {"moves_left": np.full(len(samples), 30.0)}


def two_positions():
    return samples_from_boards([(chess.Board(), 0), (chess.Board("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"), 1)])


def test_side_to_move_flip():
    m = OutcomeModel(FakeAdapter(), "fake.pt")
    p = m.predict(two_positions(), 1500)
    assert np.allclose(p.win, [0.7, 0.1]) and np.allclose(p.loss, [0.1, 0.7]) and np.allclose(p.draw, 0.2)
    assert np.allclose(p.score, [0.8, 0.2])


def test_ratings_mapped_to_correct_colors():
    a = FakeAdapter()
    OutcomeModel(a, "fake.pt").predict(two_positions(), stm_rating=1800, opp_rating=1400)
    white_elo, black_elo, _ = a.seen
    assert list(white_elo) == [1800, 1400] and list(black_elo) == [1400, 1800]  # 2nd position: black to move


def test_defaults_recorded_and_overridable():
    m = OutcomeModel(FakeAdapter(), "fake.pt")
    p = m.predict(two_positions(), 1500)
    assert p.defaults_used == {"white_clock_s": config.DEFAULT_CONTEXT["white_clock_s"],
                               "time_control": config.DEFAULT_CONTEXT["time_control"]}
    p2 = m.predict(two_positions(), 1500, context={"white_clock_s": 30.0})
    assert list(p2.defaults_used) == ["time_control"] and p2.extras["moves_left"][0] == 30.0


def test_unrated_model_warns_loudly(capsys):
    with pytest.warns(UserWarning, match="WITHOUT player ratings"):
        m = OutcomeModel(FakeAdapter(uses_ratings=False), "norating.pt")
    assert "WITHOUT player ratings" in capsys.readouterr().out
    assert m.predict(two_positions(), 1500).rating_sensitive is False


@pytest.mark.skipif(not os.path.exists(config.MODEL_PATH), reason="model.pt missing")
def test_real_checkpoint_loads_and_outputs_probabilities():
    m = load_outcome_model()
    assert m.adapter.name == "chessnet" and m.rating_sensitive
    p = m.predict(two_positions(), 1600)
    assert np.allclose(p.win + p.draw + p.loss, 1, atol=1e-5)
    assert p.defaults_used == {}
    # Wiring check ONLY: the rating input reaches the network (output changes at all). It does NOT
    # assert the model is strongly rating-sensitive: for this checkpoint it is not (see reports/rating_sweep.md).
    assert not np.allclose(m.predict(two_positions(), 1300).win, m.predict(two_positions(), 2000).win,
                           rtol=0, atol=1e-7)
