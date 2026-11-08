# Chess outcome prediction from Lichess games

Predict the result of a chess game (white win / draw / black win) from a board position, using the [Lichess open database](https://database.lichess.org/).

Files:

- `extract.py` streams a `.pgn.zst` file into sharded numpy arrays
- `dataset.py` is the PyTorch `Dataset` that reads those shards
- `train.py` trains and tests a small ResNet on a chosen feature set
- `ablate.py` runs feature ablations (board, + skill, + time, + history) and compares them

## 1. Setup

```bash
conda create -n chess python=3.12 -y
conda activate chess
pip install torch numpy python-chess zstandard matplotlib
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

Each sample stores the position, both ratings, both clocks (from the `[%clk]` comments), the time control, and the previous `--history` positions. Games without clock comments are skipped, so every feature set trains and tests on exactly the same positions. Shards made by older versions of `extract.py` lack these fields: re-run it before using `time` or `history`.

Useful options:

| Arg | Default | Meaning |
|---|---|---|
| `--max-games` | 1,000,000 | Stop after keeping this many games |
| `--positions-per-game` | 4 | Random positions sampled per game |
| `--min-ply` | 8 | Skip the first N half-moves |
| `--min-elo` | 1200 | Both players must be at least this rating |
| `--min-tc` | 180 | Minimum base + 40 x increment (seconds); drops bullet |
| `--history` | 8 | Previous positions stored per sample (for the `history` feature) |

Sanity check:

```bash
python dataset.py
```

## 4. Train and test

```bash
python train.py --data data --epochs 5
python train.py --data data --epochs 5 --features board+skill
python train.py --data data --epochs 5 --features board+skill+time+history
```

`--features` chooses the model inputs, joined with `+`:

| Feature | Input |
|---|---|
| `board` | Always on. 18 planes: pieces, side to move, castling rights, en-passant file |
| `skill` | Both players' Elo, fed to the head |
| `time` | Both clocks (fraction of base time and log seconds) plus base time and increment, fed to the head |
| `history` | Piece planes for the previous `--history` positions (default 8), as extra input planes |

`--use-elo` still works and is the same as adding `skill`. Use `--seed` for repeatable runs.

The best checkpoint (by validation loss) is saved to `model.pt`. At the end the script reports test accuracy next to two baselines: always predicting the most common class, and predicting that the higher-rated player wins. The model should beat the Elo baseline to be learning something from the board.

Finally it plots the loss curves (per-step and smoothed train loss, plus per-epoch train and val loss), saves the figure to `loss_curve.png` (change with `--plot`), and opens it in a window if a display is available.

## 5. Ablations

`ablate.py` trains the same network on the same positions with different feature sets. It measures how much each factor adds to outcome prediction, extending Anderson, Kleinberg and Mullainathan (KDD 2016) from hand-built features on endgames to learned features on whole games.

```bash
python ablate.py --data data --epochs 5 --seeds 0 1 2
python ablate.py --data data --configs board board+time board+skill board+skill+time+history
```

The default configs are cumulative: `board`, `board+skill`, `board+skill+time`, `board+skill+time+history`. Any other `train.py` option (`--epochs`, `--lr`, `--channels`, ...) is passed through.

Outputs in `--out` (default `ablations/`):

- `results.csv`: one row per (config, seed) with test loss, accuracy, per-class recall and the val curve. Runs already listed are skipped, so an interrupted ablation resumes where it stopped. Delete the file (or use a new `--out`) after changing hyperparameters.
- `summary.csv`: mean and std over seeds, with the change in test loss against `board` and against the previous config.
- `ablation.png`: test loss per config, and val loss curves.
- A checkpoint and loss curve per run.

Compare configs by test cross-entropy, not accuracy. A feature that mostly sharpens the model's confidence can lower the loss a lot while barely moving accuracy. Use at least 3 seeds before reading much into small differences. Cumulative order also matters: what `time` adds after `skill` can differ from what it adds alone, so add configs like `board+time` to check.

## Notes

- Use at least ~100k games, or the val/test splits may be too small.
- `dataset.py` loads all shards into RAM, which is fine for a few million positions.
- Stockfish evals in the PGN are ignored by `extract.py`.
- Clocks partly encode how the game ends: a player near zero often loses on time ("Time forfeit" games are kept). That is real signal, but it is part of what the `time` feature measures.