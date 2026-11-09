"""Light shard loader that needs only numpy (no torch). Reads the files written by extract.py."""
import glob
import os

import numpy as np

import config


def shard_files(split, root=None):
    root = root or config.DATA_DIR
    files = sorted(glob.glob(os.path.join(root, split, "shard_*.npz")))
    if not files:
        raise FileNotFoundError(f"no shards in {os.path.join(root, split)} (set CHESS_DATA or --data)")
    return files


def load_split(split, root=None):
    """Return dict of concatenated arrays bb, meta, elo, y plus `shard` and `row` locating each sample."""
    parts = {k: [] for k in ("bb", "meta", "elo", "y")}
    shard, row = [], []
    for f in shard_files(split, root):
        z = np.load(f)
        n = len(z["y"])
        for k in parts:
            parts[k].append(z[k])
        shard += [os.path.basename(f)] * n
        row += list(range(n))
    out = {k: np.concatenate(v) for k, v in parts.items()}
    out["shard"] = np.array(shard)
    out["row"] = np.array(row)
    return out
