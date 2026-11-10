# Position difficulty

Owner: Arnav. Everything in this folder is new; no file outside `difficulty/` was changed
(`extract.py`, `dataset.py`, `train.py` and the main README are imported from, never edited).

## Working rules (for people and future AI sessions)

1. Work on branch `arnav/difficulty`. All difficulty code lives in `difficulty/`; only add new files.
2. Never edit `extract.py`, `dataset.py`, `train.py` or the main README. Importing from them is fine.
   If something there blocks the difficulty work, stop and tell Arnav.
3. Ablation tools belong to David. Validation against tablebases, Lichess puzzle ratings and thinking time belongs to Mary.
4. Do not commit or push unless Arnav asks.

## What "difficulty" means

For a position and a rating R:

    difficulty      = Stockfish expected score  -  model expected score        (score = win + 0.5 * draw)
    difficulty_win  = Stockfish win chance      -  model win chance

Both are from the **side to move's** point of view. **Positive = humans of rating R do worse than the engine says
they should** (a hard position to play). Negative = humans do better than the engine's number. The model is asked
"what happens if two players rated R play on from here?" (opponent rating can be changed with `--opp-offset`).

## Current state: read this first

The checkpoint `model.pt` is rating-aware but **almost ignores rating** (see `reports/rating_sweep.md` and
`reports/FINDINGS.md`): predictions move by under 0.02 between rating 1200 and 2400, so difficulty is nearly the same at
every rating. The code is correct; the model is the limiting part. Treat rating-based conclusions with great caution.

## Setup

```bash
cd difficulty
uv venv .venv --python 3.12
uv pip install --python .venv/bin/python -r requirements.txt     # numpy<2 is required by torch 2.2.2
```
Stockfish: put the binary at `difficulty/bin/stockfish`, or set `STOCKFISH_PATH`, or pass `--stockfish`.
Data: `$CHESS_DATA`, else the repo's `data/` folder, else `difficulty/data_cache/data`.
(`bin/`, `data_cache/`, `results/` and `.venv/` are git-ignored; nothing large is committed.)

## How to run, start to finish (from the repo root)

```bash
PY=difficulty/.venv/bin/python
$PY difficulty/rating_support.py                  # 1. which ratings the TRAIN data covers -> results/rating_support.json
$PY difficulty/run_stockfish.py --n 1000          # 2. Stockfish WDL on random TEST positions -> results/stockfish_test.csv (resumable)
$PY difficulty/run_difficulty.py                  # 3. the difficulty table -> results/difficulty.csv
$PY difficulty/sanity_rating_sweep.py             # 4. does the model react to rating? -> reports/rating_sweep.md
$PY difficulty/explore_difficulty.py              # 5. plots + hardest/easiest -> reports/
$PY -m pytest difficulty/tests -q                 # tests (18)
```

| File | What it does |
|---|---|
| `config.py` | All settings (Stockfish path/nodes, data and model paths, rating-grid rules, default inputs for future models) |
| `board_codec.py` | Rebuilds a full chess board from one stored sample; checks it by re-encoding (exact on 5,000 real samples) |
| `stockfish_wdl.py` | Stockfish win/draw/loss for the side to move, fixed node budget, handles checkmate/stalemate/illegal/crashes |
| `outcome_model.py` | Loads a checkpoint, predicts win/draw/loss for the side to move at chosen ratings; adapter registry for new model types |
| `scoring.py` | The difficulty formulas (+ `difficulty_for_boards` for single positions) |
| `data_utils.py` | numpy-only shard loader |
| `rating_support.py`, `run_stockfish.py`, `run_difficulty.py`, `sanity_rating_sweep.py`, `explore_difficulty.py` | The scripts above |
| `reports/` | Committed results: `FINDINGS.md` (plain-language write-up), plots, `rating_sweep.md`, hardest/easiest positions |
| `tests/` | pytest suite |

## Settings

| Setting | Where | Default |
|---|---|---|
| Stockfish location | `$STOCKFISH_PATH`, `--stockfish` | `stockfish` on PATH, else `difficulty/bin/stockfish` |
| Search budget | `$STOCKFISH_NODES`, `run_stockfish.py --nodes` | 200,000 nodes, 1 thread, 64 MB hash, fresh state per position (repeatable) |
| Data folder | `$CHESS_DATA`, `--data` | repo `data/`, else `difficulty/data_cache/data` |
| Checkpoint | `$CHESS_MODEL`, `run_difficulty.py --checkpoint` | repo `model.pt` |
| Number / seed of test positions | `run_stockfish.py --n --seed` | 1000 / 0 |
| Rating grid | `run_difficulty.py --rating-min --rating-max --rating-step` | the trusted range in `rating_support.json` (positions where both players are within 100 of R, at least 5,000 training positions); falls back with a warning to 1200..2400 |
| Opponent rating | `run_difficulty.py --opp-offset` | 0 (same rating for both players) |
| Defaults for future inputs (clocks etc.) | `config.DEFAULT_CONTEXT` | both clocks 600 s, time control "600+0", no history |

Only the **test split** is used (hard-coded in `run_stockfish.py`).

## Output table: `results/difficulty.csv` (one row per position and rating)

| Column | Meaning |
|---|---|
| `pos_id` | Position identifier `test:<shard file>:<row in shard>`, stable for a given extraction |
| `fen` | Standard text description of the position (FEN). Move counters: halfmove clock is 0 (not stored in the data) |
| `ply` | Half-moves played before this position (0 = start) |
| `move_number` | Full-move number from the FEN (`ply // 2 + 1`) |
| `side_to_move` | `white` or `black`. All probabilities below are from this side's point of view |
| `rating_stm` | Rating given to the side to move |
| `rating_opp` | Rating given to the opponent (= `rating_stm` unless `--opp-offset` was used) |
| `sf_win`, `sf_draw`, `sf_loss` | Stockfish's win/draw/loss chances (sum to 1) from its WDL model at the node budget |
| `sf_score` | `sf_win + 0.5 * sf_draw` |
| `sf_mate_in` | Blank unless Stockfish sees a forced mate: positive = side to move mates in N, negative = gets mated in N |
| `model_win`, `model_draw`, `model_loss` | Model's chances for humans of the given ratings (converted from White's view to the side to move's) |
| `model_score` | `model_win + 0.5 * model_draw` |
| `difficulty` | `sf_score - model_score`. Positive = humans do worse than the engine expects |
| `difficulty_win` | `sf_win - model_win` (win-chance-only version) |
| `checkpoint` | Checkpoint file name used |
| `model_kind` | Adapter that ran the model (`chessnet` for the current model) |
| `rating_sensitive` | `False` if the checkpoint was trained without ratings (then difficulty cannot vary with rating) |
| `sf_nodes` | Stockfish search budget used (nodes) |
| `stockfish` | Stockfish version string |
| `defaults_used` | JSON of inputs filled with default values (clocks, time control...). `{}` for the current model |

Positions where Stockfish status is not `ok` (checkmate, stalemate, illegal, engine error) are in
`results/stockfish_test.csv` (with `sf_status`) but have no rows in the difficulty table.
`results/stockfish_test.csv` extra columns: `idx` (row in the whole test split), `result` (actual game result),
`decode_issues`, `roundtrip_ok`, `sf_cp`, `sf_depth`, `sf_seconds`.

## For David (ablations): difficulty for each model

Stockfish numbers do not depend on the model, so run step 2 once and reuse it. Then, per ablation checkpoint:

```bash
$PY difficulty/run_difficulty.py --checkpoint path/to/ablation_model.pt --out difficulty/results/difficulty_<name>.csv
```
* Use a different `--save` name in `train.py` for each ablation; the default overwrites `model.pt`.
* A board-only checkpoint (`use_elo=False`) works but prints a loud warning and has `rating_sensitive = False`.
* A model with new inputs (clocks, history, extra outputs) needs one small adapter class in `outcome_model.py`
  (instructions in its docstring); the rest of the code is unchanged. Missing inputs get the defaults in
  `config.DEFAULT_CONTEXT`, recorded in `defaults_used`.
* Run `rating_support.py` and `run_stockfish.py --data <real data>` on the team's real data first. The reports here
  come from a small 2013 sample.

## For Mary (validation): what to use

* **File:** `results/difficulty.csv` (regenerate with the steps above; it is not committed).
* **Columns:** `pos_id`, `fen`, `rating_stm`, `difficulty` (main), `difficulty_win`, `sf_score`, `model_score`, `sf_mate_in`,
  `move_number`, `side_to_move`. Filter on `rating_sensitive` and `checkpoint` to know which model made it.
* **Tablebases:** count pieces from `fen`; positions with 7 or fewer pieces are in scope. Exclude rows with `sf_mate_in`
  set if you want to avoid trivial mates.
* **Puzzle ratings:** the scripts only read positions from the dataset's test split. For arbitrary FENs use
  `scoring.difficulty_for_boards(boards, rating, StockfishWDL(), load_outcome_model())` (no command-line mode yet).
* **Thinking time:** needs real thinking-time data and a model with clocks; neither exists in the current code
  (`extract.py` drops clock times).
* Only the test split is used, so the model has not trained on these positions.

## Known limitations

See `reports/FINDINGS.md`. In short: the current model hardly reacts to rating; Stockfish is an engine whose probabilities are
very decisive (difficulty tracks its certainty, correlation 0.89); the model learned from whole-game results, not move
difficulty; no clocks, time control or history; rebuilt positions lose the 50-move counter and repetition history;
the reports use a small 2013 sample; ratings below 1200 have no data.
