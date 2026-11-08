"""PyTorch Dataset over the shards written by extract.py."""
import glob

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

FACTORS = ("board", "skill", "time", "history")


def parse_features(spec):
    """'board+skill+time' -> ('board', 'skill', 'time'). The board is always included."""
    parts = [s.strip() for s in spec.replace(",", "+").split("+") if s.strip()]
    bad = set(parts) - set(FACTORS)
    if bad:
        raise ValueError(f"unknown feature(s) {sorted(bad)}; choose from {FACTORS}")
    return tuple(f for f in FACTORS if f == "board" or f in parts)


class LichessPositions(Dataset):
    """Returns (x, s, y):
        x float32 (18 + 12*H, 8, 8): 12 piece planes, side-to-move, 4 castling rights,
                  en-passant file, then (with "history") 12 piece planes for each of the
                  H previous positions, most recent first
        s float32 (n_scalar,): scalar features fed to the model head
                  "skill": [white, black] Elo / 3000
                  "time":  [white, black] clock as a fraction of the base time (clipped to 2),
                           [white, black] log1p(clock seconds) / 8,
                           log1p(base seconds) / 8, increment / 60
        y int64: 0 white wins / 1 draw / 2 black wins
    Board is in White's perspective (rank index = row, file index = column, a1 = [0, 0]).
    """

    def __init__(self, root, split="train", features=("board",), history=8):
        files = sorted(glob.glob(f"{root}/{split}/shard_*.npz"))
        if not files:
            raise FileNotFoundError(f"no shards in {root}/{split}")
        self.features = tuple(features)
        self.history = history if "history" in self.features else 0
        need = {"bb", "meta", "elo", "y"}
        if "time" in self.features:
            need |= {"clk", "tc"}
        if self.history:
            need.add("hist")

        parts = [np.load(f) for f in files]
        formats = {frozenset(p.files) for p in parts}
        if len(formats) > 1:
            raise ValueError(f"shards in {root}/{split} come from different extract.py versions "
                             f"(mixed datasets); re-run extract.py with --overwrite")
        missing = need - set(parts[0].files)
        if missing:
            raise KeyError(f"shards in {root}/{split} lack {sorted(missing)}; "
                           f"re-run extract.py to build features {self.features}")
        cat = {k: np.concatenate([p[k] for p in parts]) for k in need}
        self.bb, self.meta, self.elo, self.y = cat["bb"], cat["meta"], cat["elo"], cat["y"]
        if self.history:
            stored = cat["hist"].shape[1]
            if self.history > stored:
                raise ValueError(f"--history {self.history} but shards only store {stored} positions")
            self.hist = np.ascontiguousarray(cat["hist"][:, :self.history])

        scalars = []
        if "skill" in self.features:
            scalars.append(self.elo.astype(np.float32) / 3000.0)
        if "time" in self.features:
            clk = cat["clk"].astype(np.float32)
            base = cat["tc"][:, :1].astype(np.float32)
            inc = cat["tc"][:, 1:].astype(np.float32)
            scalars += [np.clip(clk / np.maximum(base, 1.0), 0, 2), np.log1p(clk) / 8,
                        np.log1p(base) / 8, inc / 60]
        self.scalars = (np.concatenate(scalars, 1) if scalars
                        else np.zeros((len(self.y), 0), np.float32)).astype(np.float32)

    @property
    def in_channels(self):
        return 18 + 12 * self.history

    @property
    def n_scalar(self):
        return self.scalars.shape[1]

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        planes = np.unpackbits(self.bb[i].view(np.uint8), bitorder="little").reshape(12, 8, 8)
        m = self.meta[i]
        extra = np.zeros((6, 8, 8), np.uint8)
        extra[0] = m[0]          # white to move
        extra[1:5] = m[1:5, None, None]  # castling rights
        if m[5] < 8:
            extra[5, :, m[5]] = 1  # en-passant file
        stack = [planes, extra]
        if self.history:
            stack.append(np.unpackbits(self.hist[i].view(np.uint8), bitorder="little")
                         .reshape(12 * self.history, 8, 8))
        x = torch.from_numpy(np.concatenate(stack).astype(np.float32))
        return x, torch.from_numpy(self.scalars[i]), int(self.y[i])


if __name__ == "__main__":
    ds = LichessPositions("data", "train", features=FACTORS)
    print(len(ds), "training positions")
    x, s, y = ds[0]
    print(x.shape, s, y)
    dl = DataLoader(ds, batch_size=256, shuffle=True, num_workers=4)
    xb, sb, yb = next(iter(dl))
    print(xb.shape, sb.shape, yb.shape, "class balance:", np.bincount(ds.y) / len(ds))
