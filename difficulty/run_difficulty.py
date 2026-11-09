"""Build the difficulty table: one row per (position, rating).

    difficulty/.venv/bin/python difficulty/run_difficulty.py                     # uses results/stockfish_test.csv
    difficulty/.venv/bin/python difficulty/run_difficulty.py --checkpoint other.pt --out results/difficulty_other.csv
    difficulty/.venv/bin/python difficulty/run_difficulty.py --rating-min 1400 --rating-max 1800 --rating-step 200
    difficulty/.venv/bin/python difficulty/run_difficulty.py --opp-offset -200   # opponent 200 points weaker

Stockfish numbers come from results/stockfish_test.csv (run_stockfish.py). Only positions with
sf_status == "ok" get rows; checkmate / stalemate / illegal ones are left out (nothing to play).
Column meanings: see README.md ("Output table").
"""
import argparse
import csv
import json
import os
import warnings

import numpy as np

import config
from board_codec import decode_sample
from data_utils import load_split
from scoring import difficulty_scores, expected_score, rating_grid
from outcome_model import SampleBatch, load_outcome_model

COLUMNS = ["pos_id", "fen", "ply", "move_number", "side_to_move", "rating_stm", "rating_opp",
           "sf_win", "sf_draw", "sf_loss", "sf_score", "sf_mate_in",
           "model_win", "model_draw", "model_loss", "model_score",
           "difficulty", "difficulty_win",
           "checkpoint", "model_kind", "rating_sensitive", "sf_nodes", "stockfish", "defaults_used"]


def default_grid():
    """Trusted rating range from rating_support.json; falls back (loudly) to config.FALLBACK_GRID."""
    p = config.RATING_SUPPORT_JSON
    if os.path.exists(p):
        s = json.load(open(p))
        if s.get("trusted_min") is not None:
            return s["trusted_min"], s["trusted_max"], s["step"]
    warnings.warn(f"{p} not found: using fallback grid {config.FALLBACK_GRID}, which may include "
                  f"ratings the training data does not cover. Run rating_support.py first.")
    return config.FALLBACK_GRID[0], config.FALLBACK_GRID[1], config.GRID_STEP


def main():
    lo0, hi0, step0 = default_grid()
    p = argparse.ArgumentParser()
    p.add_argument("--stockfish-csv", default=str(config.STOCKFISH_CSV))
    p.add_argument("--data", default=config.DATA_DIR)
    p.add_argument("--checkpoint", default=config.MODEL_PATH)
    p.add_argument("--out", default=str(config.DIFFICULTY_CSV))
    p.add_argument("--rating-min", type=int, default=lo0)
    p.add_argument("--rating-max", type=int, default=hi0)
    p.add_argument("--rating-step", type=int, default=step0)
    p.add_argument("--opp-offset", type=int, default=0,
                   help="opponent rating = side-to-move rating + offset (default 0: same rating)")
    p.add_argument("--device", default="cpu")
    a = p.parse_args()

    with open(a.stockfish_csv, newline="") as f:
        sf_rows = [r for r in csv.DictReader(f)]
    settings_path = a.stockfish_csv + ".settings.json"
    sf_meta = json.load(open(settings_path)) if os.path.exists(settings_path) else {}
    ok = [r for r in sf_rows if r["sf_status"] == "ok"]
    print(f"{len(sf_rows)} Stockfish rows, {len(ok)} usable (status ok), {len(sf_rows) - len(ok)} left out")

    data = load_split("test", a.data)
    idx = np.array([int(r["idx"]) for r in ok])
    samples = SampleBatch(data["bb"][idx], data["meta"][idx])
    # Safety: the rebuilt FEN must describe the same board as the stored sample at this idx.
    for r, i in list(zip(ok, idx))[:50]:
        assert decode_sample(data["bb"][i], data["meta"][i]).board.fen() == r["fen"], \
            "stockfish csv does not match the data split (different --data / seed?)"

    model = load_outcome_model(a.checkpoint, a.device)
    grid = rating_grid(a.rating_min, a.rating_max, a.rating_step)
    print(f"checkpoint {a.checkpoint} ({model.adapter.name}); ratings {grid}; opp offset {a.opp_offset}")
    sf = {k: np.array([float(r[k]) for r in ok]) for k in ("sf_win", "sf_draw", "sf_loss")}

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    n_rows = 0
    with open(a.out, "w", newline="") as f:
        wr = csv.DictWriter(f, COLUMNS)
        wr.writeheader()
        for R in grid:
            pred = model.predict(samples, stm_rating=R, opp_rating=R + a.opp_offset)
            diff, diff_w = difficulty_scores(sf["sf_win"], sf["sf_draw"], pred.win, pred.draw)
            used = json.dumps(pred.defaults_used)
            for k, r in enumerate(ok):
                wr.writerow({
                    "pos_id": r["pos_id"], "fen": r["fen"], "ply": r["ply"],
                    "move_number": r["fullmove"], "side_to_move": r["stm"],
                    "rating_stm": R, "rating_opp": R + a.opp_offset,
                    "sf_win": f"{sf['sf_win'][k]:.4f}", "sf_draw": f"{sf['sf_draw'][k]:.4f}",
                    "sf_loss": f"{sf['sf_loss'][k]:.4f}",
                    "sf_score": f"{expected_score(sf['sf_win'][k], sf['sf_draw'][k]):.4f}",
                    "sf_mate_in": "" if r["sf_mate_in"] in ("", "None") else r["sf_mate_in"],
                    "model_win": f"{pred.win[k]:.4f}", "model_draw": f"{pred.draw[k]:.4f}",
                    "model_loss": f"{pred.loss[k]:.4f}", "model_score": f"{pred.score[k]:.4f}",
                    "difficulty": f"{diff[k]:.4f}", "difficulty_win": f"{diff_w[k]:.4f}",
                    "checkpoint": os.path.basename(a.checkpoint), "model_kind": pred.model_kind,
                    "rating_sensitive": pred.rating_sensitive, "sf_nodes": sf_meta.get("nodes", ""),
                    "stockfish": sf_meta.get("stockfish", ""), "defaults_used": used})
                n_rows += 1
    print(f"wrote {n_rows:,} rows ({len(ok)} positions x {len(grid)} ratings) to {a.out}")

    # short summary
    with open(a.out, newline="") as f:
        rows = list(csv.DictReader(f))
    print("\nmean difficulty by rating (score version / win version):")
    for R in grid:
        d = np.array([float(r["difficulty"]) for r in rows if int(r["rating_stm"]) == R])
        dw = np.array([float(r["difficulty_win"]) for r in rows if int(r["rating_stm"]) == R])
        print(f"  {R}: {d.mean():+.4f} / {dw.mean():+.4f}   (std {d.std():.3f}, "
              f"{(abs(d) < 0.05).mean() * 100:.0f}% within +-0.05 of zero)")


if __name__ == "__main__":
    main()
