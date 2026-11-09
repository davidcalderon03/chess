"""Explore the difficulty table: by rating, by game stage, hardest/easiest positions, spread.

    difficulty/.venv/bin/python difficulty/explore_difficulty.py

Reads results/difficulty.csv; writes plots and an HTML page with board images to difficulty/reports/
and prints a numbers summary (JSON). (Deliberately does NOT compare against thinking time, tablebases or puzzle ratings.)

Game stage (documented so it can be changed): material M = N+B (3 each) + R (5) + Q (9) for BOTH sides
(pawns and kings excluded; start position = 62).
    opening    fullmove <= 12 and M >= 50
    endgame    M <= 20
    middlegame everything else
"""
import argparse
import csv
import json
import os

import chess
import chess.svg
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import config

REPORTS = config.DIFF_DIR / "reports"
VAL = {chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9, chess.PAWN: 1}
STAGES = ["opening", "middlegame", "endgame"]


def features(fen, stm):
    b = chess.Board(fen)
    me = chess.WHITE if stm == "white" else chess.BLACK
    mat = lambda c: sum(VAL[p.piece_type] for p in b.piece_map().values()
                        if p.color == c and p.piece_type != chess.KING)
    nonpawn = sum(VAL[p.piece_type] for p in b.piece_map().values() if p.piece_type not in (chess.PAWN, chess.KING))
    return {"pieces": len(b.piece_map()), "in_check": b.is_check(), "nonpawn_material": nonpawn,
            "material_balance": mat(me) - mat(not me)}   # side to move minus opponent, pawn units


def stage_of(fullmove, nonpawn):
    if fullmove <= 12 and nonpawn >= 50:
        return "opening"
    return "endgame" if nonpawn <= 20 else "middlegame"


def perm_p(group, everyone, n=5000, seed=0):
    """Two-sided permutation p-value: is mean(group) unusual vs random groups of the same size?"""
    rng = np.random.default_rng(seed)
    obs = abs(group.mean() - everyone.mean())
    sims = np.array([abs(rng.choice(everyone, len(group), replace=False).mean() - everyone.mean())
                     for _ in range(n)])
    return float((sims >= obs).mean())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--table", default=str(config.DIFFICULTY_CSV))
    p.add_argument("--rating", type=int, default=1600, help="rating used for hardest/easiest lists")
    p.add_argument("--top", type=int, default=25)
    a = p.parse_args()
    rows = list(csv.DictReader(open(a.table)))
    for r in rows:
        r["rating"] = int(r["rating_stm"])
        r["d"] = float(r["difficulty"]); r["dw"] = float(r["difficulty_win"])
        r["sf"] = float(r["sf_score"]); r["ms"] = float(r["model_score"])
    ratings = sorted({r["rating"] for r in rows})
    os.makedirs(REPORTS, exist_ok=True)
    out = {"n_rows": len(rows), "ratings": ratings}

    # features per position (rating-independent)
    feat = {}
    for r in rows:
        if r["pos_id"] not in feat:
            f = features(r["fen"], r["side_to_move"])
            f["stage"] = stage_of(int(r["move_number"]), f["nonpawn_material"])
            feat[r["pos_id"]] = f
    npos = len(feat)

    # 1. mean difficulty by rating
    mean_d = [np.mean([r["d"] for r in rows if r["rating"] == R]) for R in ratings]
    mean_w = [np.mean([r["dw"] for r in rows if r["rating"] == R]) for R in ratings]
    se = [np.std([r["d"] for r in rows if r["rating"] == R]) / np.sqrt(npos) for R in ratings]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.errorbar(ratings, mean_d, yerr=[1.96 * s for s in se], marker="o", label="score version (win + 0.5 draw)")
    ax.plot(ratings, mean_w, marker="s", label="win-chance version")
    ax.axhline(0, color="gray", lw=0.8)
    ax.set_xlabel("rating (both players)"); ax.set_ylabel("mean difficulty")
    ax.set_title(f"Mean difficulty by rating ({npos} positions)\nflat: the model barely reacts to rating")
    ax.legend(); fig.tight_layout(); fig.savefig(REPORTS / "difficulty_by_rating.png", dpi=130); plt.close(fig)
    out["mean_difficulty_by_rating"] = dict(zip(ratings, np.round(mean_d, 4).tolist()))
    out["mean_difficulty_win_by_rating"] = dict(zip(ratings, np.round(mean_w, 4).tolist()))
    out["change_first_to_last_rating"] = round(mean_d[-1] - mean_d[0], 5)

    # 2. by stage across ratings
    stage_n = {s: sum(f["stage"] == s for f in feat.values()) for s in STAGES}
    fig, ax = plt.subplots(figsize=(7, 4.2))
    out["by_stage"] = {}
    for s in STAGES:
        ys = [np.mean([r["d"] for r in rows if r["rating"] == R and feat[r["pos_id"]]["stage"] == s])
              for R in ratings]
        ax.plot(ratings, ys, marker="o", label=f"{s} (n={stage_n[s]})")
        out["by_stage"][s] = {"n_positions": stage_n[s], "mean_by_rating": dict(zip(ratings, np.round(ys, 4).tolist()))}
    ax.axhline(0, color="gray", lw=0.8)
    ax.set_xlabel("rating (both players)"); ax.set_ylabel("mean difficulty")
    ax.set_title("Mean difficulty by game stage"); ax.legend(); fig.tight_layout()
    fig.savefig(REPORTS / "difficulty_by_stage.png", dpi=130); plt.close(fig)

    # 4. spread
    at = [r for r in rows if r["rating"] == a.rating]
    d = np.array([r["d"] for r in at]); dw = np.array([r["dw"] for r in at])
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    axs[0].hist(d, bins=40, range=(-1, 1), color="tab:blue"); axs[0].set_title(f"difficulty (score), rating {a.rating}")
    axs[1].hist(dw, bins=40, range=(-1, 1), color="tab:orange"); axs[1].set_title("difficulty (win chance only)")
    for ax in axs:
        ax.axvline(0, color="k", lw=0.8); ax.set_xlabel("difficulty"); ax.set_ylabel("positions")
    fig.tight_layout(); fig.savefig(REPORTS / "difficulty_spread.png", dpi=130); plt.close(fig)
    sf_arr = np.array([r["sf"] for r in at])
    out["spread_at_rating"] = {
        "rating": a.rating, "n": len(d), "mean": round(float(d.mean()), 4), "std": round(float(d.std()), 4),
        "percentiles_5_25_50_75_95": np.round(np.percentile(d, [5, 25, 50, 75, 95]), 3).tolist(),
        "share_within_0.05_of_zero": round(float((abs(d) < 0.05).mean()), 3),
        "share_within_0.10_of_zero": round(float((abs(d) < 0.10).mean()), 3),
        "share_positive": round(float((d > 0).mean()), 3),
        "share_stockfish_decided_(score<=0.05_or_>=0.95)": round(float(((sf_arr <= 0.05) | (sf_arr >= 0.95)).mean()), 3),
        "corr_difficulty_vs_stockfish_score": round(float(np.corrcoef(d, sf_arr)[0, 1]), 3)}

    # 3. hardest / easiest at the chosen rating
    order = np.argsort(d)
    hard, easy = [at[i] for i in order[::-1][:a.top]], [at[i] for i in order[:a.top]]
    fmat = lambda grp, k: np.array([feat[r["pos_id"]][k] for r in grp], float)
    allf = {k: np.array([feat[r["pos_id"]][k] for r in at], float) for k in ("pieces", "in_check", "material_balance")}
    cmp_rows = []
    for k in ("pieces", "in_check", "material_balance"):
        row = {"feature": k, "all": round(float(allf[k].mean()), 3)}
        for name, grp in (("hardest", hard), ("easiest", easy)):
            v = fmat(grp, k)
            row[name] = round(float(v.mean()), 3)
            row[f"{name}_perm_p"] = round(perm_p(v, allf[k]), 3)
        cmp_rows.append(row)
    for name, grp in (("hardest", hard), ("easiest", easy)):
        cmp_rows.append({"feature": f"stage counts ({name})",
                         **{s: sum(feat[r['pos_id']]['stage'] == s for r in grp) for s in STAGES}})
        cmp_rows.append({"feature": f"stm white ({name})", "n": sum(r["side_to_move"] == "white" for r in grp)})
        cmp_rows.append({"feature": f"mean Stockfish score ({name})", "v": round(float(np.mean([r["sf"] for r in grp])), 3)})
        cmp_rows.append({"feature": f"mean model score ({name})", "v": round(float(np.mean([r["ms"] for r in grp])), 3)})
    out["hardest_vs_easiest"] = cmp_rows

    def card(r, i):
        b = chess.Board(r["fen"])
        svg = chess.svg.board(b, size=240, coordinates=False)
        return (f"<div class=c>{svg}<div class=t>#{i} {r['pos_id']}<br>{r['side_to_move']} to move, move {r['move_number']}<br>"
                f"difficulty {float(r['difficulty']):+.2f} | Stockfish {r['sf']:.2f} | model {r['ms']:.2f}</div></div>")
    html = ["<!doctype html><meta charset=utf-8><title>Hardest and easiest positions</title>",
            "<style>body{font:14px sans-serif;margin:20px}.g{display:flex;flex-wrap:wrap;gap:14px}.c{width:240px}"
            ".t{font-size:12px;color:#444}</style>",
            f"<h1>Hardest and easiest positions at rating {a.rating}</h1>",
            "<p>Difficulty = Stockfish expected score minus model expected score, side to move. "
            "Positive = humans do worse than the engine says.</p>",
            f"<h2>{a.top} hardest (most positive)</h2><div class=g>" + "".join(card(r, i + 1) for i, r in enumerate(hard)) + "</div>",
            f"<h2>{a.top} easiest (most negative)</h2><div class=g>" + "".join(card(r, i + 1) for i, r in enumerate(easy)) + "</div>"]
    (REPORTS / f"hardest_easiest_{a.rating}.html").write_text("\n".join(html))
    with open(REPORTS / f"hardest_easiest_{a.rating}.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["list", "rank", "pos_id", "fen", "side_to_move", "move_number", "sf_score", "model_score", "difficulty"])
        for name, grp in (("hardest", hard), ("easiest", easy)):
            for i, r in enumerate(grp, 1):
                w.writerow([name, i, r["pos_id"], r["fen"], r["side_to_move"], r["move_number"], r["sf"], r["ms"], r["difficulty"]])

    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
