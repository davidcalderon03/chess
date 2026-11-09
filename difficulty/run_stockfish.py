"""Run Stockfish over a random sample of the TEST split (never train/val) and save the WDL results.

    difficulty/.venv/bin/python difficulty/run_stockfish.py --n 1000

* Saves one CSV row per position as it goes (flushed to disk), so Ctrl-C / crashes lose nothing.
* Re-running with the same settings resumes: positions already in the CSV are skipped.
* Different search settings (nodes, seed, n-source data) than the existing file => refuses to mix.
* Bad positions are written with a status (illegal, checkmate, ...) instead of crashing; see stockfish_wdl.py.
"""
import argparse
import csv
import json
import os
import time
from collections import Counter

import chess
import numpy as np

import config
from board_codec import decode_sample, roundtrip_matches
from data_utils import load_split
from stockfish_wdl import StockfishWDL

FIELDS = ["idx", "pos_id", "fen", "ply", "fullmove", "stm", "white_elo", "black_elo", "result",
          "decode_issues", "roundtrip_ok", "sf_status", "sf_win", "sf_draw", "sf_loss", "sf_score",
          "sf_mate_in", "sf_cp", "sf_depth", "sf_nodes", "sf_seconds"]
SPLIT = "test"  # hard-coded on purpose: the model has never trained on these positions


def choose_indices(n_total, n, seed):
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n_total, size=min(n, n_total), replace=False))


def read_done(path):
    if not os.path.exists(path):
        return set()
    with open(path, newline="") as f:
        return {int(r["idx"]) for r in csv.DictReader(f)}


def make_row(i, data, sf):
    bb, meta = data["bb"][i], data["meta"][i]
    d = decode_sample(bb, meta)
    row = {"idx": int(i), "pos_id": f"{SPLIT}:{data['shard'][i]}:{int(data['row'][i])}",
           "fen": d.board.fen(), "ply": d.ply, "fullmove": d.board.fullmove_number,
           "stm": "white" if d.board.turn == chess.WHITE else "black",
           "white_elo": int(data["elo"][i][0]), "black_elo": int(data["elo"][i][1]),
           "result": ["white", "draw", "black"][int(data["y"][i])],
           "decode_issues": ";".join(d.issues), "roundtrip_ok": roundtrip_matches(d, bb, meta)}
    r = sf.evaluate(d.board)
    row.update(sf_status=r.status, sf_win=r.win, sf_draw=r.draw, sf_loss=r.loss, sf_score=r.score,
               sf_mate_in=r.mate_in, sf_cp=r.cp, sf_depth=r.depth, sf_nodes=r.nodes,
               sf_seconds=round(r.seconds, 4))
    return row


def summarize(path):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        print("no rows"); return
    secs = np.array([float(r["sf_seconds"]) for r in rows])
    print(f"\n=== Stockfish summary: {len(rows)} positions ({path}) ===")
    print(f"time per position: mean {secs.mean():.3f}s, median {np.median(secs):.3f}s, "
          f"p95 {np.percentile(secs, 95):.3f}s; total {secs.sum() / 60:.1f} min")
    print("status counts:", dict(Counter(r["sf_status"] for r in rows)))
    print("decode issues:", dict(Counter(i for r in rows for i in r["decode_issues"].split(";") if i)) or "none")
    print("roundtrip mismatches:", sum(r["roundtrip_ok"] != "True" for r in rows))
    ok = [r for r in rows if r["sf_status"] == "ok"]
    if ok:
        w = np.array([float(r["sf_win"]) for r in ok])
        s = np.array([float(r["sf_score"]) for r in ok])
        mates = sum(r["sf_mate_in"] not in ("", "None") for r in ok)
        print(f"evaluated {len(ok)}; forced mates found: {mates}")
        print("Stockfish win prob (side to move) percentiles 5/25/50/75/95:",
              np.round(np.percentile(w, [5, 25, 50, 75, 95]), 3))
        print("Stockfish expected score percentiles 5/25/50/75/95:    ",
              np.round(np.percentile(s, [5, 25, 50, 75, 95]), 3))
        hist, edges = np.histogram(s, bins=10, range=(0, 1))
        print("expected score histogram (0..1 in tenths):", hist.tolist())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=config.DATA_DIR)
    p.add_argument("--n", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--nodes", type=int, default=config.STOCKFISH_NODES)
    p.add_argument("--stockfish", default=config.STOCKFISH_PATH)
    p.add_argument("--out", default=str(config.STOCKFISH_CSV))
    p.add_argument("--summary-only", action="store_true")
    a = p.parse_args()
    if a.summary_only:
        summarize(a.out); return

    data = load_split(SPLIT, a.data)
    wanted = choose_indices(len(data["y"]), a.n, a.seed)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    meta_path = a.out + ".settings.json"
    with StockfishWDL(a.stockfish, a.nodes) as sf:
        settings = {"split": SPLIT, "seed": a.seed, "nodes": a.nodes, "stockfish": sf.name,
                    "hash_mb": sf.hash_mb, "threads": sf.threads, "data": a.data,
                    "n_split_positions": int(len(data["y"]))}
        if os.path.exists(meta_path) and os.path.exists(a.out):
            old = json.load(open(meta_path))
            diff = {k: (old.get(k), v) for k, v in settings.items() if old.get(k) != v and k != "data"}
            if diff:
                raise SystemExit(f"existing {a.out} was made with different settings {diff}; "
                                 f"use another --out or delete it")
        json.dump(settings, open(meta_path, "w"), indent=1)

        done = read_done(a.out)
        todo = [int(i) for i in wanted if int(i) not in done]
        print(f"{sf.name}, nodes={a.nodes}: {len(done)} already done, {len(todo)} to do "
              f"(of {len(wanted)} requested from {len(data['y']):,} test positions)")
        new_file = not os.path.exists(a.out) or os.path.getsize(a.out) == 0
        t0 = time.time()
        with open(a.out, "a", newline="") as f:
            wr = csv.DictWriter(f, FIELDS)
            if new_file:
                wr.writeheader()
            for k, i in enumerate(todo, 1):
                wr.writerow(make_row(i, data, sf))
                f.flush()
                if k % 50 == 0:
                    os.fsync(f.fileno())
                    el = time.time() - t0
                    print(f"  {k}/{len(todo)}  {el / k:.2f}s/pos  eta {(len(todo) - k) * el / k / 60:.1f} min",
                          flush=True)
            os.fsync(f.fileno())
    summarize(a.out)


if __name__ == "__main__":
    main()
