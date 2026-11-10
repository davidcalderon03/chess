"""Position difficulty from separate engine and human W/D/L distributions.

Columns stay White win / draw / Black win. Gaps are signed; *_pp denotes
percentage points. Held-out prediction metrics live in evaluation_metrics.py.
"""
import numpy as np


def _matrix(values, name):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 3 or len(values) == 0:
        raise ValueError(f"{name} must have nonempty shape (N, 3)")
    if not np.isfinite(values).all():
        raise ValueError(f"{name} must contain finite values")
    return values



def validate_probabilities(probabilities):
    """Return a validated float64 (N, 3) array; never silently normalize it."""
    probabilities = _matrix(probabilities, "probabilities")
    if ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError("probabilities must lie in [0, 1]")
    if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-6, rtol=0):
        raise ValueError("each probability row must sum to 1")
    return probabilities



def difficulty_scores(engine_probabilities, human_probabilities, pov="white"):
    """Return signed Stockfish-minus-human gaps for the declared focal player.

    Both inputs retain White W/D/L column order. Positive win gaps mean humans
    are less likely to convert than the engine reference. Negative gaps remain
    visible. For conversion analysis, separately select engine-winning positions.
    """
    engine = validate_probabilities(engine_probabilities)
    human = validate_probabilities(human_probabilities)
    if engine.shape != human.shape:
        raise ValueError("engine and human probability arrays must have the same shape")
    if pov not in ("white", "black"):
        raise ValueError("pov must be 'white' or 'black'")
    win, loss = (0, 2) if pov == "white" else (2, 0)
    win_gap = engine[:, win] - human[:, win]
    expected_gap = (engine[:, win] + 0.5 * engine[:, 1]) - (
        human[:, win] + 0.5 * human[:, 1]
    )
    return {
        "win_gap": win_gap,
        "win_gap_pp": 100.0 * win_gap,
        "expected_score_gap": expected_gap,
        "expected_score_gap_pp": 100.0 * expected_gap,
        "sf_win_probability": engine[:, win].copy(),
        "human_win_probability": human[:, win].copy(),
    }


def difficulty_evaluation(engine_probabilities, human_probabilities, labels, *,
                          pov="white", winning_threshold=0.8, bins=10):
    """Evaluate matched engine/human forecasts and describe held-out conversion.

    ``labels`` retain the repository's White/draw/Black order. ``pov`` is either
    one declared focal color or a length-N array of those colors, chosen before
    observing game results. Gaps remain descriptive scores, separate from the
    outcome metrics and the conversion check. No engine probability is a human
    outcome label, and no scalar expected score is expanded into a fictitious WDL.
    """
    from evaluation_metrics import (binary_metrics, categorical_metrics,
                                    expected_score_metrics)

    engine = validate_probabilities(engine_probabilities)
    human = validate_probabilities(human_probabilities)
    if engine.shape != human.shape:
        raise ValueError("engine and human probability arrays must have the same shape")
    if not np.isfinite(winning_threshold) or not 0.5 < winning_threshold <= 1:
        raise ValueError("winning_threshold must be finite and in (0.5, 1]")
    if not isinstance(bins, (int, np.integer)) or isinstance(bins, (bool, np.bool_)) or bins < 1:
        raise ValueError("bins must be a positive integer")
    n = len(engine)
    focal = np.full(n, pov) if isinstance(pov, str) else np.asarray(pov)
    if focal.shape != (n,) or not np.isin(focal, ["white", "black"]).all():
        raise ValueError("pov must be 'white', 'black', or an (N,) array of those colors")

    classes = ["white_win", "draw", "black_win"]
    # These helpers also validate the complete observed label vector.
    matched_outcome = {
        "human": categorical_metrics(human, labels, classes, bins),
        "stockfish": categorical_metrics(engine, labels, classes, bins),
    }
    actual = np.asarray(labels).astype(np.int64)
    positions = {name: np.empty(n, dtype=float) for name in
                 ("win_gap_pp", "expected_score_gap_pp", "sf_win_probability", "human_win_probability")}
    for color in ("white", "black"):
        select = focal == color
        if select.any():
            gaps = difficulty_scores(engine[select], human[select], pov=color)
            for name in positions:
                positions[name][select] = gaps[name]
    focal_black = focal == "black"
    focal_win = actual == np.where(focal_black, 2, 0)
    engine_winning = positions["sf_win_probability"] >= winning_threshold
    gap = positions["win_gap_pp"] / 100.0
    band_index = np.minimum(np.floor((gap + 1) / 2 * bins).astype(int), bins - 1)
    positions.update(focal_win=focal_win, engine_winning=engine_winning,
                     pov_black=focal_black, band_index=band_index)

    human_conversion = binary_metrics(positions["human_win_probability"], focal_win.astype(int),
                                       bins=bins, mask=engine_winning)
    engine_conversion = binary_metrics(positions["sf_win_probability"], focal_win.astype(int),
                                        bins=bins, mask=engine_winning)
    edges = np.linspace(-1.0, 1.0, bins + 1)
    bands = []
    for index in range(bins):
        use = engine_winning & (band_index == index)
        count = int(use.sum())
        bands.append({
            "lower": float(edges[index]), "upper": float(edges[index + 1]), "count": count,
            "mean_gap_pp": float(positions["win_gap_pp"][use].mean()) if count else None,
            "observed_win_fraction": float(focal_win[use].mean()) if count else None,
            "mean_human_win_probability": float(positions["human_win_probability"][use].mean()) if count else None,
            "mean_engine_win_probability": float(positions["sf_win_probability"][use].mean()) if count else None,
        })
    matched_expected = {
        "human": expected_score_metrics(human[:, 0] + .5 * human[:, 1], actual, bins),
        "stockfish": expected_score_metrics(engine[:, 0] + .5 * engine[:, 1], actual, bins),
    }
    summary = {
        "status": "evaluated", "n": n, "winning_threshold": float(winning_threshold),
        "pov_policy": pov if isinstance(pov, str) else "per_position",
        "focal_color_counts": {color: int((focal == color).sum()) for color in ("white", "black")},
        "gap_descriptive": {
            "mean_win_gap_pp": float(positions["win_gap_pp"].mean()),
            "median_win_gap_pp": float(np.median(positions["win_gap_pp"])),
            "mean_expected_score_gap_pp": float(positions["expected_score_gap_pp"].mean()),
            "median_expected_score_gap_pp": float(np.median(positions["expected_score_gap_pp"])),
        },
        "matched_outcome_metrics": matched_outcome,
        "matched_expected_score_metrics": matched_expected,
        "loss_probability_floor": 1e-15,
        "cross_entropy_convention": "Matched probability-based CCE clips observed event probabilities at 1e-15 for both distributions; the main model report uses stable logits.",
        "conversion": {
            "n": int(engine_winning.sum()),
            "definition": "Focal player's final win (draw/loss=0), only where engine focal win probability meets the declared threshold",
            "human_win_brier_score": human_conversion["brier_score"],
            "stockfish_win_brier_score": engine_conversion["brier_score"],
            "bins": bands,
            "bin_definition": "Fixed equal-width win-gap bins on [-1, 1]; left-inclusive, last right endpoint inclusive",
        },
        "notes": [
            "Signed gap means are descriptive, not prediction-error metrics; lower is not automatically better.",
            "Stockfish is a comparison baseline against observed human results, not the true human label.",
            "Conversion bands are unadjusted exploratory summaries, not independent proof of difficulty or a causal effect.",
            "Gap residual MSE against engine-minus-observed score equals human expected-score MSE and is not counted as a new metric.",
            "Position-weighted summaries may contain several positions from the same game; no independence-based confidence intervals are added here.",
        ],
    }
    return summary, positions

