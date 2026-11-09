"""Sanity check: does the model's predicted result change sensibly with rating?

Takes 20 test positions (8 where Stockfish says the side to move is clearly winning, 8 clearly losing,
4 balanced), runs the model at ratings 1200..2400, and reports how the side to move's expected score
(win + 0.5 draw) changes. Expectation: stronger players convert winning positions more often, so the
score in winning positions should RISE with rating (and fall for the clearly losing side to move).

    difficulty/.venv/bin/python difficulty/sanity_rating_sweep.py
"""
import argparse
import csv
import os

import numpy as np

import config
from data_utils import load_split
from outcome_model import SampleBatch, load_outcome_model

REPORT = config.DIFF_DIR / "reports" / "rating_sweep.md"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stockfish-csv", default=str(config.STOCKFISH_CSV))
    p.add_argument("--checkpoint", default=config.MODEL_PATH)
    p.add_argument("--rmin", type=int, default=1200)
    p.add_argument("--rmax", type=int, default=2400)
    p.add_argument("--step", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    rows = [r for r in csv.DictReader(open(a.stockfish_csv)) if r["sf_status"] == "ok"]
    sc = np.array([float(r["sf_score"]) for r in rows])
    rng = np.random.default_rng(a.seed)
    groups = {"winning": (np.where(sc >= 0.95)[0], 8), "losing": (np.where(sc <= 0.05)[0], 8),
              "balanced": (np.where((sc > 0.4) & (sc < 0.6))[0], 4)}
    chosen = {g: rng.choice(ix, size=min(n, len(ix)), replace=False) for g, (ix, n) in groups.items()}
    allk = np.concatenate(list(chosen.values()))
    data = load_split("test")
    idx = np.array([int(rows[k]["idx"]) for k in allk])
    samples = SampleBatch(data["bb"][idx], data["meta"][idx])
    model = load_outcome_model(a.checkpoint)
    ratings = list(range(a.rmin, a.rmax + 1, a.step))
    scores = np.stack([model.predict(samples, R).score for R in ratings], 1)   # (20, n_ratings)

    lines = [f"# Rating sweep (checkpoint {os.path.basename(a.checkpoint)}, 20 test positions)", "",
             f"Side to move's model expected score (win + 0.5 draw) at ratings {ratings}. "
             f"Ratings outside 1300-2000 are outside what the (sample) training data covers well.", "",
             "| group | n | mean score at lowest rating | at highest rating | change | positions moving the 'right' way |",
             "|---|---|---|---|---|---|"]
    pos = 0
    summary = {}
    for g, ks in chosen.items():
        block = scores[pos:pos + len(ks)]
        pos += len(ks)
        change = block[:, -1] - block[:, 0]
        right = (change > 0) if g == "winning" else (change < 0) if g == "losing" else np.abs(change) < 0.02
        summary[g] = (change.mean(), right.sum(), len(ks))
        lines.append(f"| {g} | {len(ks)} | {block[:, 0].mean():.4f} | {block[:, -1].mean():.4f} | "
                     f"{change.mean():+.4f} | {right.sum()}/{len(ks)} |")
    lines += ["", "## Per position (score at each rating; Stockfish score for reference)", "",
              "| pos_id | group | SF score | " + " | ".join(map(str, ratings)) + " |",
              "|---|---|---|" + "---|" * len(ratings)]
    pos = 0
    for g, ks in chosen.items():
        for j, k in enumerate(ks):
            lines.append(f"| {rows[k]['pos_id']} | {g} | {sc[k]:.2f} | "
                         + " | ".join(f"{v:.3f}" for v in scores[pos + j]) + " |")
        pos += len(ks)
    total_range = scores.max(1) - scores.min(1)
    lines += ["", f"Largest change in any single position's score across the whole sweep: "
                  f"{total_range.max():.4f}; median {np.median(total_range):.4f}."]
    w_gain = summary["winning"][0]
    verdict = ("The model does NOT respond to rating in a meaningful way: predicted results barely move "
               "from 1200 to 2400, so it does not show stronger players converting winning positions more often."
               if abs(w_gain) < 0.02 else
               ("Stronger ratings convert winning positions more often, as expected." if w_gain > 0
                else "Stronger ratings do WORSE in winning positions, the opposite of expectation."))
    lines += ["", "## Verdict", "", verdict]
    os.makedirs(REPORT.parent, exist_ok=True)
    REPORT.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nsaved {REPORT}")


if __name__ == "__main__":
    main()
