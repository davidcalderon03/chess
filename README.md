# Chess outcome prediction from Lichess games

Predict the result of a chess game (white win / draw / black win) from a board position, using the [Lichess open database](https://database.lichess.org/).

Files:

- `extract.py` streams a `.pgn.zst` file into sharded numpy arrays
- `dataset.py` is the PyTorch `Dataset` that reads those shards
- `train.py` trains and tests a small ResNet

## 1. Setup

```bash
conda create -n chess python=3.12 -y
conda activate chess
pip install torch numpy python-chess zstandard
```

## 2. Get the data

Lichess publishes one file per month. Start with an old month (small and quick); recent months are tens of GB. Check the site for available months.

```bash
wget -c https://database.lichess.org/standard/lichess_db_standard_rated_2026-08.pgn.zst
```

Do not decompress it. `extract.py` reads the `.zst` directly.

## 3. Extract the dataset

```bash
python extract.py lichess_db_standard_rated_2026-08.pgn.zst --out data --max-games 200000
```

This writes `data/train/`, `data/val/`, and `data/test/` shards. Splits are assigned per game, so positions from one game never appear in two splits.

Useful options:

| Arg | Default | Meaning |
|---|---|---|
| `--max-games` | 1,000,000 | Stop after keeping this many games |
| `--positions-per-game` | 4 | Random positions sampled per game |
| `--min-ply` | 8 | Skip the first N half-moves |
| `--min-elo` | 1200 | Both players must be at least this rating |
| `--min-tc` | 180 | Minimum base + 40 x increment (seconds); drops bullet |

Sanity check:

```bash
python dataset.py
```

## 4. Train and test

```bash
python train.py --data data --epochs 5
python train.py --data data --epochs 5 --use-elo   # also feed player ratings to the model
```

The best checkpoint (by validation loss) is saved to `model.pt`. At the end the script reports test accuracy next to two baselines: always predicting the most common class, and predicting that the higher-rated player wins. The model should beat the Elo baseline to be learning something from the board.

## Notes

- Use at least ~100k games, or the val/test splits may be too small.
- `dataset.py` loads all shards into RAM, which is fine for a few million positions.
- Clock times and Stockfish evals in the PGN are ignored by `extract.py`.