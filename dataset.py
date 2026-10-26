"""PyTorch Dataset over the shards written by extract.py."""
import glob

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class LichessPositions(Dataset):
    """Returns (x, elo, y):
        x   float32 (18, 8, 8): 12 piece planes, side-to-move, 4 castling rights, en-passant file
        elo float32 (2,): [white, black] scaled by 1/3000 (drop it if you want a position-only model)
        y   int64: 0 white wins / 1 draw / 2 black wins
    Board is in White's perspective (rank index = row, file index = column, a1 = [0, 0]).
    """

    def __init__(self, root, split="train"):
        files = sorted(glob.glob(f"{root}/{split}/shard_*.npz"))
        if not files:
            raise FileNotFoundError(f"no shards in {root}/{split}")
        parts = [np.load(f) for f in files]
        self.bb = np.concatenate([p["bb"] for p in parts])      # (N, 12) uint64
        self.meta = np.concatenate([p["meta"] for p in parts])  # (N, 7) uint8
        self.elo = np.concatenate([p["elo"] for p in parts])    # (N, 2) int16
        self.y = np.concatenate([p["y"] for p in parts])        # (N,) uint8

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
        x = torch.from_numpy(np.concatenate([planes, extra]).astype(np.float32))
        elo = torch.from_numpy(self.elo[i].astype(np.float32) / 3000.0)
        return x, elo, int(self.y[i])


if __name__ == "__main__":
    ds = LichessPositions("data", "train")
    print(len(ds), "training positions")
    x, elo, y = ds[0]
    print(x.shape, elo, y)
    dl = DataLoader(ds, batch_size=256, shuffle=True, num_workers=4)
    xb, eb, yb = next(iter(dl))
    print(xb.shape, eb.shape, yb.shape, "class balance:", np.bincount(ds.y) / len(ds))