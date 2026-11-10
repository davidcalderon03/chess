"""Which ratings does the TRAINING data cover well enough to trust?

For a rating R we count training positions where BOTH players are within +-BIN_HALF_WIDTH of R
(because difficulty at "rating R" feeds the model [R, R]). A rating is trusted if that count is at
least --min-count. The trusted range is the longest contiguous run of trusted ratings.

    difficulty/.venv/bin/python difficulty/rating_support.py     # writes results/rating_support.json
"""
import argparse
import json

import config
from data_utils import load_split


def rating_counts(elo, step, half_width):
    lo, hi = int(elo.min()), int(elo.max())
    grid = list(range(lo // step * step, hi // step * step + step + 1, step))
    w, b = elo[:, 0].astype(int), elo[:, 1].astype(int)
    counts = {r: int(((abs(w - r) <= half_width) & (abs(b - r) <= half_width)).sum()) for r in grid}
    return counts


def trusted_range(counts, min_count):
    """Longest contiguous run of ratings whose count >= min_count; (None, None) if there is none."""
    best, cur = (None, None), []
    for r in sorted(counts):
        cur = cur + [r] if counts[r] >= min_count else []
        if cur and (best[0] is None or cur[-1] - cur[0] > best[1] - best[0]):
            best = (cur[0], cur[-1])
    return best


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=config.DATA_DIR)
    p.add_argument("--step", type=int, default=config.GRID_STEP)
    p.add_argument("--half-width", type=int, default=config.BIN_HALF_WIDTH)
    p.add_argument("--min-count", type=int, default=config.MIN_COUNT_PER_BIN)
    p.add_argument("--out", default=str(config.RATING_SUPPORT_JSON))
    a = p.parse_args()

    d = load_split("train", a.data)
    elo = d["elo"]
    counts = rating_counts(elo, a.step, a.half_width)
    lo, hi = trusted_range(counts, a.min_count)
    out = {"data": a.data, "n_train_positions": int(len(elo)),
           "min_rating_seen": int(elo.min()), "max_rating_seen": int(elo.max()),
           "step": a.step, "half_width": a.half_width, "min_count": a.min_count,
           "counts": counts, "trusted_min": lo, "trusted_max": hi}
    config.RESULTS_DIR.mkdir(exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1)
    print(f"{len(elo):,} train positions; ratings seen {elo.min()}..{elo.max()}")
    print(f"positions with BOTH players within +-{a.half_width} of R (need >= {a.min_count:,}):")
    for r, c in counts.items():
        print(f"  {r}: {c:>8,} {'trusted' if c >= a.min_count else '-'}")
    print(f"trusted range: {lo}..{hi}   (saved {a.out})")


if __name__ == "__main__":
    main()
