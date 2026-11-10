# Chess outcome prediction from Lichess games

Predict the result of a chess game (white win / draw / black win) from a board position, using the [Lichess open database](https://database.lichess.org/).

Files:

- `extract.py` streams a `.pgn.zst` file into sharded numpy arrays
- `dataset.py` is the PyTorch `Dataset` that reads those shards
- `train.py` trains and tests a small ResNet on a chosen feature set
- `ablate.py` runs feature ablations (board, + skill, + time, + history) and compares them
- `multitask_data.py`, `multitask_model.py`, and `multitask_training.py` provide five-task labels, heads/loss, and training
- `evaluate.py` reports the five held-out losses and their weighted total
- `model_io.py` restores checkpoints and the matching board/context encoding
- `scoring.py` and `score_position.py` calculate engine–human difficulty gaps for individual positions
- `requirements.txt` lists the Python dependencies; `tests/` contains optional unit tests

## 1. Setup

```bash
conda create -n chess python=3.12 -y
conda activate chess
python -m pip install -r requirements.txt
```

## 2. Get the data

Lichess publishes one file per month. Start with an old month (small and quick); recent months are tens of GB. Check the site for available months.

```bash
curl -L -C - -O https://database.lichess.org/standard/lichess_db_standard_rated_2026-08.pgn.zst
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

## 6. Five-task training and evaluation

The commands above still train outcome-only models by default. To train all five tasks, replay your source `.pgn.zst` into a **new** directory: existing outcome-only shards do not contain the other labels. Five-task extraction also requires a Stockfish executable with native `UCI_ShowWDL` support; substitute your archive and engine paths below.

Install Stockfish separately from the Python dependencies. On macOS with [Homebrew](https://brew.sh/):

```bash
brew install stockfish
command -v stockfish
```

Use the path printed by the second command for `--engine`, or use `--engine stockfish` when it is on your PATH. See the [Homebrew package](https://formulae.brew.sh/formula/stockfish). Without Homebrew, or on Windows/Linux, download and extract the executable for your platform from the [official Stockfish downloads](https://stockfishchess.org/download/), then pass its full path to `--engine`.

```bash
python extract.py /path/to/games.pgn.zst --tasks all \
  --out data_five --max-games 1000 --positions-per-game 2 \
  --engine /path/to/stockfish --engine-nodes 100000
python train.py --data data_five --tasks all --features board+skill+time \
  --epochs 5 --batch-size 128 --channels 64 --blocks 3 --workers 0 \
  --save checkpoints/five_seed0.pt --plot reports/five_training.png \
  --no-show --skip-test
python evaluate.py --checkpoint checkpoints/five_seed0.pt --data data_five \
  --split test --out reports/five_test
open reports/five_test/report.html
```

The tasks are final outcome, next-move mistake, remaining plies, next-move seconds, and ending category. The objective is outcome categorical cross-entropy (CCE) + 0.50 × mistake binary cross-entropy (BCE) + 0.25 × length Huber + 0.25 × time Huber + 0.10 × ending CCE. Regression targets use training-only standardized `log1p` transforms; unavailable labels are masked. Ending categories are **checkmate / time forfeit / other normal ending**; PGN `Normal` does not reliably identify resignation.

`--skip-test` reserves test games for `evaluate.py`. The best five-task checkpoint is selected by validation outcome CCE by default, and stores all five heads plus normalization, loss settings, dataset identity, and training history. Use a new checkpoint path for each five-task run.

Training prints one compact row per epoch; full task losses and counts are saved in `<checkpoint>.history.json`. Add `--quiet` to suppress progress output. `--no-show` saves the curves without opening a plot window. A 1,000-game pilot has only about 2,000 positions, so training can finish in seconds on a GPU.

The evaluator uses the same five losses as training: outcome CCE, mistake BCE, remaining-length Huber, thinking-time Huber, and termination CCE. It averages each loss over valid held-out labels and reports their weighted total. Regression normalization and loss settings come from the checkpoint. Results are saved as `metrics.json`, `metrics.csv`, and a simple `report.html` table. Add `--quiet` to suppress terminal output.

Existing outcome-only checkpoints can also be evaluated:

```bash
python evaluate.py --checkpoint model.pt --data data --out reports/outcome
```

Use `score_position.py --help` for individual-position human/engine comparisons. Difficulty is a derived engine–human probability gap; prediction metrics compare the model with observed human labels.

The longer step-by-step explanation is in [docs/TRAINING_AND_EVALUATION.md](docs/TRAINING_AND_EVALUATION.md).

Optional unit tests check label alignment, masks, loss formulas, and checkpoint compatibility. Training and evaluation do not run these tests:

```bash
python -m unittest discover -s tests -p 'test_*.py' -v
```

## Notes

- Use at least ~100k games, or the val/test splits may be too small.
- `dataset.py` loads all shards into RAM, which is fine for a few million positions.
- Stockfish evals in the PGN are ignored by `extract.py`.
- Clocks partly encode how the game ends: a player near zero often loses on time ("Time forfeit" games are kept). That is real signal, but it is part of what the `time` feature measures.
